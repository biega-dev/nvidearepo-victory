use super::{Copy, RUNS_AT_A_TIME, short};

pub(super) type Layout<'a> = (u64, Vec<Copy>, Vec<(usize, &'a [(usize, usize)], usize)>);

/// The original separate passes for copy commands, scatter words and jobs.
pub(super) fn separate<'a>(
    resources: impl Iterator<Item = (usize, &'a [(usize, usize)])> + Clone,
) -> Layout<'a> {
    let mut end = 0;
    let mut copies = Vec::with_capacity(resources.size_hint().0);
    let short_run = |&&(a, b): &&(usize, usize)| short(a, b - a);
    for (_, ranges) in resources.clone() {
        let mut runs = Vec::new();
        for &(a, b) in ranges.iter().filter(|r| !short_run(r)) {
            runs.push((a, end, b - a));
            end += ((b - a) as u64).next_multiple_of(16);
        }
        let words: usize = ranges
            .iter()
            .filter(short_run)
            .map(|&(a, b)| (b - a) / 4)
            .sum();
        copies.push(Copy {
            runs,
            pairs: (end, words as u32),
        });
        end += (words as u64 * 8).next_multiple_of(16);
    }
    let mut jobs = Vec::new();
    for ((source, ranges), copy) in resources.zip(&copies) {
        let mut offset = copy.pairs.0 as usize;
        for chunk in ranges.chunks(RUNS_AT_A_TIME) {
            let words: usize = chunk
                .iter()
                .filter(short_run)
                .map(|&(a, b)| (b - a) / 4)
                .sum();
            if words != 0 {
                jobs.push((source, chunk, offset));
                offset += words * 8;
            }
        }
    }
    (end, copies, jobs)
}

/// Walk each dirty interval once. The copy commands precede the packed
/// dwords in upload memory. Each worker receives a disjoint part of that
/// dword stream and can fill it without a shared append position.
pub(super) fn plan<'a>(
    resources: impl Iterator<Item = (usize, &'a [(usize, usize)])>,
) -> Layout<'a> {
    let mut end = 0u64;
    let mut copies = Vec::with_capacity(resources.size_hint().0);
    let mut jobs = Vec::new();
    for (source, ranges) in resources {
        let mut runs = Vec::new();
        let mut words = 0usize;
        let first_job = jobs.len();
        for chunk in ranges.chunks(RUNS_AT_A_TIME) {
            let first_word = words;
            for &(lo, hi) in chunk {
                let len = hi - lo;
                if short(lo, len) {
                    words += len / 4;
                } else {
                    runs.push((lo, end, len));
                    end += (len as u64).next_multiple_of(16);
                }
            }
            if words > first_word {
                jobs.push((source, chunk, first_word * 8));
            }
        }
        for (_, _, offset) in &mut jobs[first_job..] {
            *offset += end as usize;
        }
        copies.push(Copy {
            runs,
            pairs: (end, words as u32),
        });
        end += (words as u64 * 8).next_multiple_of(16);
    }
    (end, copies, jobs)
}

pub(super) const BULK_BYTES: usize = 1 << 20;

/// A long contiguous copy and several small copies can share one worker.
/// A descriptor crossing the byte budget is split without copying its data.
pub(super) fn bulk_parts(moves: &[(usize, usize, usize)]) -> Vec<Vec<(usize, usize, usize)>> {
    let mut parts = Vec::new();
    let mut part = Vec::new();
    let mut room = BULK_BYTES;
    for &(source, destination, len) in moves {
        let mut done = 0;
        while done < len {
            let n = room.min(len - done);
            part.push((source + done, destination + done, n));
            done += n;
            room -= n;
            if room == 0 {
                parts.push(std::mem::take(&mut part));
                room = BULK_BYTES;
            }
        }
    }
    if !part.is_empty() {
        parts.push(part);
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bulk_jobs_cover_each_byte_once_across_unaligned_boundaries() {
        let mib = 1 << 20;
        let moves = [(50, 19, 0), (11, 7, 3), (4 * mib, 8 * mib, 2 * mib)];
        let parts = bulk_parts(&moves);
        assert_eq!(parts, [
            vec![(11, 7, 3), (4 * mib, 8 * mib, mib - 3)],
            vec![(5 * mib - 3, 9 * mib - 3, mib)],
            vec![(6 * mib - 3, 10 * mib - 3, 3)],
        ]);
        assert!(bulk_parts(&[]).is_empty());
    }

    #[test]
    fn mixed_uploads_have_the_original_layout_and_disjoint_worker_ranges() {
        let mut seed = 21u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as usize
        };
        for count in [0, 1, 53, RUNS_AT_A_TIME, RUNS_AT_A_TIME + 1, 10_031] {
            let buffers: Vec<Vec<_>> = [0, count, 3, count / 2]
                .into_iter()
                .map(|n| {
                    (0..n)
                        .map(|i| {
                            let lo = i * 2048 + next() % 2;
                            (lo, lo + [4, 8, 13, 256, 260, 1024][next() % 6])
                        })
                        .collect()
                })
                .collect();
            let (size, copies, jobs) =
                plan(buffers.iter().enumerate().map(|(i, v)| (i, v.as_slice())));
            let mut expected = 0;
            let mut actual_jobs = jobs.into_iter();
            for (i, (copy, ranges)) in copies.iter().zip(&buffers).enumerate() {
                let mut runs = Vec::new();
                for &(a, b) in ranges.iter().filter(|&&(a, b)| !short(a, b - a)) {
                    runs.push((a, expected, b - a));
                    expected += ((b - a) as u64).next_multiple_of(16);
                }
                assert_eq!(copy.runs, runs);
                let mut words = 0;
                for chunk in ranges.chunks(RUNS_AT_A_TIME) {
                    let n: usize = chunk
                        .iter()
                        .filter(|&&(a, b)| short(a, b - a))
                        .map(|&(a, b)| (b - a) / 4)
                        .sum();
                    if n > 0 {
                        let (src, actual, offset) = actual_jobs.next().unwrap();
                        assert_eq!(
                            (src, actual, offset),
                            (i, chunk, expected as usize + words * 8)
                        );
                        words += n;
                    }
                }
                assert_eq!(copy.pairs, (expected, words as u32));
                expected += (words as u64 * 8).next_multiple_of(16);
            }
            assert!(actual_jobs.next().is_none());
            assert_eq!(size, expected);
        }
    }
}
