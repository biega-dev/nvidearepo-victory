//! Last write-watch asking for each 64 KiB page of title memory.

use std::collections::BTreeMap;

/// This is CPU metadata under the journal's existing lock. No title or GPU
/// bytes are copied. Only blocks with a recorded write occupy memory.
pub(super) enum Epochs {
    Tree(BTreeMap<usize, u64>),
    Blocks(BTreeMap<usize, Box<Block>>),
}

const PAGES: usize = u64::BITS as usize;

pub(super) struct Block {
    pages: [u64; PAGES],
    present: u64,
    newest: u64,
}

impl Epochs {
    pub(super) fn new(blocks: bool) -> Self {
        if blocks { Self::Blocks(BTreeMap::new()) } else { Self::Tree(BTreeMap::new()) }
    }

    pub(super) fn record(&mut self, page: usize, epoch: u64) -> u64 {
        match self {
            Self::Tree(tree) => tree.insert(page, epoch).unwrap_or(0),
            Self::Blocks(blocks) => {
                let block = blocks.entry(page / PAGES).or_insert_with(|| Box::new(Block {
                    pages: [0; PAGES], present: 0, newest: 0,
                }));
                let i = page % PAGES;
                block.newest = block.newest.max(epoch);
                block.present |= 1 << i;
                std::mem::replace(&mut block.pages[i], epoch)
            }
        }
    }

    pub(super) fn records(&self, first: usize, last: usize) -> Vec<(usize, u64)> {
        let mut records = Vec::new();
        if first >= last { return records; }
        match self {
            Self::Tree(tree) => records.extend(tree.range(first..last).map(|(&p, &v)| (p, v))),
            Self::Blocks(blocks) => {
                for (&key, block) in blocks.range(first / PAGES..=(last - 1) / PAGES) {
                    let base = key * PAGES;
                    let mut bits = block.between(first, last, base);
                    while bits != 0 {
                        let i = bits.trailing_zeros() as usize;
                        records.push((base + i, block.pages[i]));
                        bits &= bits - 1;
                    }
                }
            }
        }
        records
    }

    /// A partial page is current only inside a previously copied range, or
    /// if the whole page was copied. A block older than that range needs no
    /// per-page look. An unrelated newer write falls back to the exact pages.
    pub(super) fn unchanged(&self, first: usize, taken: &[u64], ranged: u64) -> bool {
        if taken.is_empty() { return true; }
        if ranged == 0 && taken.contains(&0) { return false; }
        let last = first + taken.len();
        match self {
            Self::Tree(tree) => tree.range(first..last).all(|(&p, &v)| v <= taken[p - first].max(ranged)),
            Self::Blocks(blocks) if taken.len() == 1 => {
                blocks.get(&(first / PAGES)).is_none_or(|b| b.pages[first % PAGES] <= taken[0].max(ranged))
            }
            Self::Blocks(blocks) => blocks.range(first / PAGES..=(last - 1) / PAGES).all(|(&key, block)| {
                if block.newest <= ranged { return true; }
                let base = key * PAGES;
                let mut bits = block.between(first, last, base);
                while bits != 0 {
                    let i = bits.trailing_zeros() as usize;
                    if block.pages[i] > taken[base + i - first].max(ranged) { return false; }
                    bits &= bits - 1;
                }
                true
            }),
        }
    }
}

impl Block {
    fn between(&self, first: usize, last: usize, base: usize) -> u64 {
        let lo = first.saturating_sub(base);
        let hi = (last - base).min(PAGES);
        self.present & (u64::MAX << lo) & (u64::MAX >> (PAGES - hi))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_queries_match_pages_for_partial_copies_aliases_and_sparse_holes() {
        let (mut tree, mut blocks) = (Epochs::new(false), Epochs::new(true));
        for (page, epoch) in [(0, 2), (1, 7), (63, 5), (64, 8), (65, 9), (127, 12), (129, 8), (129, 3), (10_000_000, 6)] {
            assert_eq!(blocks.record(page, epoch), tree.record(page, epoch));
        }
        assert_eq!(blocks.records(0, 10_000_001), tree.records(0, 10_000_001));
        for first in [0, 1, 62, 63, 64, 65, 126, 128, 130, 9_999_999] {
            for len in [0, 1, 2, 63, 64, 65, 128] {
                assert_eq!(blocks.records(first, first + len), tree.records(first, first + len));
                for ranged in [0, 3, 8, 12] {
                    for mut taken in [vec![0; len], vec![3; len], vec![12; len]] {
                        if len > 1 { taken[len - 1] = 7; }
                        assert_eq!(blocks.unchanged(first, &taken, ranged), tree.unchanged(first, &taken, ranged));
                    }
                }
            }
        }
    }
}
