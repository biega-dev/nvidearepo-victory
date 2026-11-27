//! An indexed refresh belongs to the caller that has to submit it next.
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

pub(super) struct Batch<F> {
    work: F,
    count: usize,
    next: AtomicUsize,
    chunk: usize,
    state: Mutex<State>,
    done: Condvar,
}

#[derive(Default)]
struct State {
    active: usize,
    closed: bool,
    panic: Option<Box<dyn Any + Send>>,
}

impl<F: Fn(usize)> Batch<F> {
    pub(super) fn new(count: usize, chunk: usize, work: F) -> Self {
        Self {
            work, count, chunk, next: AtomicUsize::new(0),
            state: Mutex::new(State::default()), done: Condvar::new(),
        }
    }

    fn drain(&self) {
        loop {
            let start = self.next.fetch_add(self.chunk, Ordering::Relaxed);
            if start >= self.count { return; }
            for i in start..start.saturating_add(self.chunk).min(self.count) {
                (self.work)(i);
            }
        }
    }

    pub(super) fn help(&self) {
        {
            let mut state = self.state.lock().unwrap();
            if state.closed { return; }
            state.active += 1;
        }
        let result = catch_unwind(AssertUnwindSafe(|| self.drain()));
        let mut state = self.state.lock().unwrap();
        if let Err(panic) = result { state.panic.get_or_insert(panic); }
        state.active -= 1;
        if state.closed && state.active == 0 { self.done.notify_one(); }
    }

    /// Once the caller has drained all unclaimed indices, only helpers already
    /// inside this batch may be reading its sources. Acquire the lock after the
    /// last helper exits before allowing submission of any upload bytes.
    pub(super) fn finish(&self) -> std::time::Duration {
        let result = catch_unwind(AssertUnwindSafe(|| self.drain()));
        let waiting = std::time::Instant::now();
        let mut state = self.state.lock().unwrap();
        state.closed = true;
        while state.active != 0 { state = self.done.wait(state).unwrap(); }
        let panic = result.err().or_else(|| state.panic.take());
        drop(state);
        if let Some(panic) = panic { resume_unwind(panic); }
        waiting.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn every_index_finishes_once_even_when_helpers_arrive_after_the_caller() {
        for n in [0, 1, 7, 129, 5503] {
            let seen: Vec<_> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            let batch = Batch::new(n, 5, |i: usize| { seen[i].fetch_add(1, Ordering::Relaxed); });
            std::thread::scope(|s| {
                for _ in 0..4 { s.spawn(|| batch.help()); }
                batch.finish();
                assert!(seen.iter().all(|v| v.load(Ordering::Relaxed) == 1));
                batch.help();
            });
            assert!(seen.iter().all(|v| v.load(Ordering::Relaxed) == 1));
        }
    }

    #[test]
    fn panicking_helper_reports_to_caller_and_late_tickets_do_not_hold_sources() {
        let batch = Arc::new(Batch::new(2, 2, |i| { assert_ne!(i, 0); }));
        let late = Arc::downgrade(&batch);
        batch.help();
        assert!(catch_unwind(AssertUnwindSafe(|| batch.finish())).is_err());
        drop(batch);
        assert!(late.upgrade().is_none());
    }

    #[test]
    fn returning_to_the_submission_waits_for_work_that_a_helper_claimed() {
        use std::sync::Barrier;
        let started = Barrier::new(2);
        let release = Barrier::new(2);
        let batch = Batch::new(1, 2, |_| { started.wait(); release.wait(); });
        std::thread::scope(|s| {
            s.spawn(|| batch.help());
            started.wait();
            let submit = s.spawn(|| batch.finish());
            while !batch.state.lock().unwrap().closed { std::thread::yield_now(); }
            assert!(!submit.is_finished());
            release.wait();
            submit.join().unwrap();
        });
    }
}
