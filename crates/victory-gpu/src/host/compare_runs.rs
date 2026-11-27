//! The card hashes whole pieces, even for two disjoint declarations within
//! one piece. Share that result and keep the original byte limits to land.

use crate::shadow::CardSpan;
use std::sync::OnceLock;

/// With `VICTORY_CARD_COMPARE_RUNS=1` the submitting thread records one
/// compare for consecutive pieces of the same copy. The copy and its hashes
/// stay on the card. One flag and folded hash (4 bytes in mapped PC memory)
/// serves the relay for each distinct piece. No title data moves here.
pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_COMPARE_RUNS").is_some_and(|v| v == "1"))
}

/// The declared byte ranges of one copy are sorted and disjoint, so their
/// piece numbers and ends are nondecreasing. Never include a missing piece:
/// storing its hash would hide a later change from another flight.
pub(super) fn touching(a: CardSpan, b: CardSpan) -> bool {
    a.first <= b.first && b.first <= a.first + a.pieces
}

/// Places another declaration in a dispatch's results. Only its result
/// address changes: its original byte span is what the relay fetches.
/// A run uses at most the sum of the slots already reserved for its ranges.
pub(super) fn extend(run: &mut CardSpan, base: u32, next: CardSpan, slot: &mut u32) -> u32 {
    let offset = next.first - run.first;
    *slot = base + offset;
    let pieces = run.pieces.max(offset + next.pieces);
    let overlap = next.pieces - (pieces - run.pieces);
    run.pieces = pieces;
    run.to = run.to.max(next.to);
    overlap
}

/// Record hash passes for the sorted, disjoint CPU-side ranges returned by
/// `card_dirty`. The bytes have already been sent. The GPU hashes exactly the
/// same pieces; the submitting thread can skip over short writes within one
/// piece without building and sorting another list of offsets.
pub(super) fn rehash_sorted(
    mut ranges: &[(usize, usize)],
    span: impl Fn(usize, usize) -> CardSpan,
    mut record: impl FnMut(CardSpan),
) {
    let Some((&(first, last), rest)) = ranges.split_first() else { return };
    let mut run = span(first, last);
    ranges = rest;
    while let Some((&(first, last), rest)) = ranges.split_first() {
        let limit = run.end_of_pieces();
        if last <= limit {
            // Only search when several ranges fit. A sparse range costs one
            // comparison and the number of ranges covered by a piece varies.
            if ranges.get(3).is_some_and(|&(_, end)| end <= limit) {
                let count = 4 + ranges[4..].partition_point(|&(_, end)| end <= limit);
                run.to = ranges[count - 1].1 as u32;
                ranges = &ranges[count..];
            } else {
                run.to = last as u32;
                ranges = rest;
            }
            continue;
        }
        let next = span(first, last);
        if next.first <= run.first + run.pieces {
            run.pieces = (next.first - run.first + next.pieces).max(run.pieces);
            run.to = next.to;
        } else {
            record(run);
            run = next;
        }
        ranges = rest;
    }
    record(run);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(from: u32, to: u32) -> CardSpan {
        const BLOCK: u32 = 1 << 16;
        let skew = 16;
        let first = (from + skew) / BLOCK;
        CardSpan { from, to, first, pieces: (to + skew).div_ceil(BLOCK) - first, skew }
    }

    #[test]
    fn rehashing_sorted_offsets_records_the_same_dispatches() {
        const BLOCK: usize = 1 << 16;
        let mut sparse = 0;
        let mut input = Vec::new();
        for count in [0, 1, 2, 17, 8000] {
            input.clear();
            for n in 0..count {
                sparse += if n % 91 == 0 { BLOCK * 4 } else { 8 };
                let len = if n % 64 == 0 { BLOCK + 15 } else { 4 };
                input.push((sparse, sparse + len));
                sparse += len;
            }
            for skew in [0, 1, 16, BLOCK - 1] {
                let span = |lo: usize, hi: usize| {
                    let first = (lo + skew) / BLOCK;
                    CardSpan { from: lo as u32, to: hi as u32, first: first as u32,
                        pieces: ((hi + skew).div_ceil(BLOCK) - first) as u32, skew: skew as u32 }
                };
                let expected = crate::shadow::joined_piece_runs(input.iter().map(|&(a, b)| span(a, b)).collect());
                let mut out = Vec::new();
                rehash_sorted(&input, span, |r| out.push(r));
                let commands = |spans: &[CardSpan]| spans.iter().map(|s| (s.first, s.pieces, s.skew)).collect::<Vec<_>>();
                assert_eq!(commands(&out), commands(&expected));
            }
        }
    }

    #[test]
    fn rehashing_near_the_end_of_a_four_gib_copy_respects_the_skew_and_clip() {
        const BLOCK: usize = 1 << 16;
        let end = u32::MAX as usize;
        let ranges = [(end - 3 * BLOCK, end - 2 * BLOCK), (end - 4, end - 2), (end, end + 32)];
        for size in [end - 2 * BLOCK, end, end + 1] {
            for skew in [0, BLOCK - 1] {
                let span = |lo: usize, hi: usize| {
                    let hi = hi.min(size);
                    let lo = lo.min(hi - 1);
                    let first = (lo + skew) / BLOCK;
                    CardSpan { from: lo as u32, to: hi as u32, first: first as u32,
                        pieces: ((hi + skew).div_ceil(BLOCK) - first) as u32, skew: skew as u32 }
                };
                let expected = crate::shadow::joined_piece_runs(ranges.iter().map(|&(a, b)| span(a, b)).collect());
                let mut out = Vec::new();
                rehash_sorted(&ranges, span, |r| out.push((r.first, r.pieces, r.skew)));
                assert_eq!(out, expected.iter().map(|r| (r.first, r.pieces, r.skew)).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn declarations_keep_their_bytes_and_share_only_the_same_pieces() {
        let declarations = [span(4, 12), span(24, 28), span(36, 131_050), span(131_054, 131_070), span(196_610, 196_620)];
        let separate = span(400_000, 400_010);
        assert!(!touching(*declarations.last().unwrap(), separate));
        let base = 23;
        let mut slots = [base; 5];
        let mut run = declarations[0];
        let mut saved = 0;
        for i in 1..declarations.len() {
            assert!(touching(declarations[i - 1], declarations[i]));
            saved += extend(&mut run, base, declarations[i], &mut slots[i]);
        }
        assert_eq!(run.pieces, 4);
        assert_eq!(saved + run.pieces, declarations.iter().map(|d| d.pieces).sum::<u32>());
        assert_eq!(slots, [23, 23, 23, 24, 26]);
        // Any combination of changed pieces, including pieces shared by
        // multiple declarations: each fetch clips to its own byte bounds.
        for flags in 0..1u32 << run.pieces {
            let from_slots: Vec<_> = declarations.iter().zip(slots).flat_map(|(d, at)| {
                (0..d.pieces).filter_map(move |p| {
                    (flags & (1 << (at + p - base)) != 0).then_some((d.from, d.to, d.first + p))
                })
            }).collect();
            let expected: Vec<_> = declarations.iter().flat_map(|d| {
                (d.first..d.first + d.pieces).filter_map(move |p| (flags & (1 << p) != 0).then_some((d.from, d.to, p)))
            }).collect();
            assert_eq!(from_slots, expected);
        }
    }
}
