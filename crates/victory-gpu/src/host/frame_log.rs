use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

pub(super) struct FrameLog {
    file: BufWriter<File>,
    start: Instant,
    last: Instant,
    drawn: u64,
    frame: u64,
}

impl FrameLog {
    pub(super) fn open(now: Instant, drawn: u64) -> Option<Self> {
        let path = std::env::var_os("VICTORY_BENCH_CSV")?;
        let mut file = BufWriter::new(File::create(path).ok()?);
        writeln!(file, "frame,elapsed_ms,frame_ms,draws").ok()?;
        Some(Self { file, start: now, last: now, drawn, frame: 0 })
    }

    pub(super) fn present(&mut self, now: Instant, drawn: u64) {
        self.frame += 1;
        let elapsed = now.duration_since(self.start).as_secs_f64() * 1000.0;
        let ms = now.duration_since(self.last).as_secs_f64() * 1000.0;
        let _ = writeln!(self.file, "{},{elapsed:.3},{ms:.3},{}", self.frame, drawn.saturating_sub(self.drawn));
        if self.frame.is_multiple_of(16) {
            let _ = self.file.flush();
        }
        self.last = now;
        self.drawn = drawn;
    }
}
