//! Positive answers to shadow refresh polls, on the CPU only. A poll during
//! preparation reuses metadata. No title bytes move and no GPU work changes.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU64, Ordering};

use super::PAGE;

/// VICTORY_REFRESH_MEMO=1 reuses a prior poll of these exact bytes or a
/// containing range. Unset by default. Producers and readers use the same
/// startup value throughout the run.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_REFRESH_MEMO"))
}

/// The journal has one writer. Every dirty page, including every known
/// alias, passes through Journal::note with an increasing asking number.
/// Neighbours and hash collisions can only send a poll back to the journal.
const GROUP: usize = 16;
const STRIPES: usize = 8192;
const MAX_GROUPS: usize = 8;
pub(super) struct Writes([AtomicU64; STRIPES]);
pub(super) static WRITES: Writes = Writes([const { AtomicU64::new(0) }; STRIPES]);

impl Writes {
    pub(super) fn record(&self, page: usize, asked: u64) {
        self.0[page / GROUP % STRIPES].store(asked, Ordering::Release);
    }

    fn since(&self, from: usize, to: usize, asked: u64) -> bool {
        let first = from / PAGE / GROUP;
        let last = (to - 1) / PAGE / GROUP;
        last - first >= MAX_GROUPS || (first..=last).any(|g| self.0[g % STRIPES].load(Ordering::Acquire) > asked)
    }
}

/// Identifies the currently published ahead answer. Zero throughout a
/// change, including when the runtime changes a mapping under an asking.
pub(super) static AHEAD: AtomicU64 = AtomicU64::new(0);

pub(super) fn set_ahead(asked: u64) {
    if enabled() {
        AHEAD.store(asked, Ordering::Release);
    }
}

// Refreshed under the journal's read lock once per worker and asking, and
// only when a prior poll can save the other locks. Keep individual successful
// kernel queries distinct. Two adjacent queries cannot vouch for their union.
fn ahead_covers(from: usize, to: usize, standing: u64) -> bool {
    thread_local! {
        static COVER: RefCell<(u64, super::AskedRanges)> = const { RefCell::new((0, super::AskedRanges::EMPTY)) };
    }
    COVER.with_borrow_mut(|(saved, ranges)| {
        if *saved != standing {
            let j = super::journal();
            if AHEAD.load(Ordering::Acquire) != standing {
                return false;
            }
            ranges.ranges.clone_from(&j.ahead.ranges);
            ranges.indexed = j.ahead.indexed;
            *saved = standing;
        }
        ranges.covers(from / PAGE, to.div_ceil(PAGE)) && AHEAD.load(Ordering::Acquire) == standing
    })
}

// A remapping can replace a write-watched allocation at the same address.
// Require a full poll before reusing that address under a new asking.
static TOPOLOGY: AtomicU64 = AtomicU64::new(0);

pub(super) fn mapping_changed() {
    if enabled() {
        AHEAD.store(0, Ordering::Release);
        TOPOLOGY.fetch_add(1, Ordering::Release);
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CopyKey {
    id: u64,
    pub(crate) revision: u64,
    topology: u64,
}

impl CopyKey {
    pub(crate) fn new(id: u64, revision: u64) -> Self {
        Self { id, revision, topology: TOPOLOGY.load(Ordering::Acquire) }
    }
}

#[derive(Clone, Copy, Default)]
struct Answer {
    copy: CopyKey,
    from: usize,
    to: usize,
    asked: u64,
    // Zero for guarded bytes. A later kernel asking needs its own coverage.
    ahead: u64,
}

impl Answer {
    fn holds(&self, copy: CopyKey, from: usize, to: usize, writes: &Writes, known: impl FnOnce(u64) -> bool) -> bool {
        if self.copy != copy || self.from > from || to > self.to || !known(self.ahead) {
            return false;
        }
        !writes.since(from, to, self.asked)
    }
}

// One submitter and several refresh workers can poll the same copy. Each
// keeps its own bounded table, allocated on its first poll. The stable copy
// id distinguishes a new shadow at a recycled CPU or title address.
const ENTRIES: usize = 1024;
thread_local! {
    static ANSWERS: Box<[Cell<Answer>]> = (0..ENTRIES).map(|_| Cell::default()).collect();
}

fn index(copy: CopyKey, from: usize, base: usize) -> usize {
    // Small buffers usually have one polled range. Consecutive copy ids
    // get different slots even when the title also placed them in order.
    (copy.id as usize).wrapping_add((from / PAGE - base) / GROUP) % ENTRIES
}

pub(crate) fn holds(copy: CopyKey, from: usize, to: usize, base: usize) -> bool {
    let hit = ANSWERS.with(|slots| slots[index(copy, from, base)].get().holds(copy, from, to, &WRITES, |previous| {
        if previous == 0 {
            return super::published(from, to);
        }
        let standing = AHEAD.load(Ordering::Acquire);
        standing != 0 && (standing == previous || ahead_covers(from, to, standing))
    }));
    hit && copy.topology == TOPOLOGY.load(Ordering::Acquire)
}

pub(super) fn remember(copy: CopyKey, from: usize, to: usize, base: usize, asked: u64, ahead: u64) {
    ANSWERS.with(|slots| slots[index(copy, from, base)].set(Answer { copy, from, to, asked, ahead }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::watch::epochs::Epochs;

    #[test]
    fn hits_match_the_page_journal_across_partial_copies_writes_and_remaps() {
        let mut key = CopyKey::new(19, 0);
        let mut prior = Answer::default();
        let mut oracle = Epochs::new(false);
        let writes = Writes([const { AtomicU64::new(0) }; STRIPES]);
        let mut full = [0; 40];
        let mut partial = (0, 0, 0);
        let mut ahead = 0;
        let mut guarded = true;
        let mut pending = false;
        let mut hits = 0;
        let mut clock = 1;
        let mut rand = 17u32;
        for step in 0..24_000 {
            rand = rand.wrapping_mul(1664525).wrapping_add(1013904223);
            let page = (rand as usize >> 10) % (full.len() - 2);
            let (from, to) = (page * PAGE + 8, (page + 2) * PAGE - 8);
            clock += 1;
            match step % 12 {
                0 => { // A cold copy at a recycled address.
                    full.fill(0);
                    partial = (0, 0, 0);
                    key.id += 1;
                }
                1 => { // A partial copy vouches for exactly its bytes.
                    partial = (from, to, clock);
                    key.revision += 1;
                }
                2 => { // Full copies include zero-journal pages on first touch.
                    full.fill(clock);
                    key.revision += 1;
                }
                3 | 4 => { // A write recorded at both views of a physical page.
                    for p in [page, (page + 19) % full.len()] {
                        oracle.record(p, clock);
                        writes.record(p, clock);
                    }
                }
                5 | 6 => { // A shader write is declared, then landed.
                    pending = !pending;
                    key.revision += 1;
                }
                7 | 8 => { // Change what the same address shows.
                    guarded = !guarded;
                    ahead = 0;
                    key.topology += 1;
                }
                9 => ahead = clock,
                10 => ahead = 0,
                _ => { // An unrelated page hits the same dirty stripe.
                    writes.record(page + STRIPES * GROUP, clock);
                }
            }
            // Probe the old entry as well as these addresses, including one
            // byte beyond a previously accepted partial range.
            let old = (prior.from, prior.to);
            for (a, b) in [old, (from, to), (from, to), (from, to + 1), (to, to + 8)] {
                if a == b { continue; }
                let pages = a / PAGE..b.div_ceil(PAGE);
                let ranged = if partial.0 <= a && b <= partial.1 { partial.2 } else { 0 };
                let exact = !pending && (guarded || ahead != 0) && oracle.unchanged(pages.start, &full[pages], ranged);
                let hit = prior.holds(key, a, b, &writes, |source| if source == 0 { guarded } else { ahead != 0 });
                assert!(!hit || exact, "step {step}, bytes {a}..{b}");
                hits += usize::from(hit);
                if !hit && exact {
                    prior = Answer { copy: key, from: a, to: b, asked: clock, ahead: if guarded { 0 } else { ahead } };
                }
            }
        }
        assert!(hits > 10_000);
    }

    #[test]
    fn unobserved_bytes_and_wide_or_colliding_stripes_fall_back() {
        let copy = CopyKey::new(3, 0);
        let writes = Writes([const { AtomicU64::new(0) }; STRIPES]);
        let end = (MAX_GROUPS + 1) * GROUP * PAGE;
        let answer = Answer { copy, from: 8, to: end, asked: 40, ahead: 1 };
        let hit = |a, b| answer.holds(copy, a, b, &writes, |_| true);
        assert!(hit(8, PAGE));
        assert!(!hit(7, PAGE));
        assert!(!hit(8, end));
        writes.record(STRIPES * GROUP, 41);
        assert!(!hit(8, PAGE));
        assert!(hit(GROUP * PAGE, GROUP * PAGE + 1));
    }
}
