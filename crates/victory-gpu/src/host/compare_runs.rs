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
