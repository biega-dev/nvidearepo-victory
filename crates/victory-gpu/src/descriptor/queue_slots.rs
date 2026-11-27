use super::Waiting;
use indexmap::IndexMap;
use std::collections::BTreeMap;

/// Entries of one texture queue, accessed only under its lock. The dense
/// representation keeps the slot and its one Waiting together for the scan.
/// An isolated census of 10,240 slots on a Ryzen 3700X took 28 microseconds
/// at the median with dense entries and 57 with the tree.
pub(super) enum QueueSlots {
    Tree(BTreeMap<u32, Waiting>),
    Dense(IndexMap<u32, Waiting>),
}

impl Default for QueueSlots {
    fn default() -> Self {
        // VICTORY_TEXTURE_QUEUE_DENSE=1 uses contiguous CPU metadata for the
        // queue scan. Texture contents and the bytes sent to the card stay put.
        static DENSE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        Self::new(*DENSE.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_QUEUE_DENSE").is_some_and(|v| v == "1")))
    }
}

impl QueueSlots {
    pub(super) fn new(dense: bool) -> Self {
        if dense { Self::Dense(IndexMap::new()) } else { Self::Tree(BTreeMap::new()) }
    }

    pub(super) fn insert(&mut self, slot: u32, value: Waiting) {
        match self {
            Self::Tree(m) => m.insert(slot, value),
            Self::Dense(m) => m.insert(slot, value),
        };
    }

    pub(super) fn remove(&mut self, slot: &u32) -> Option<Waiting> {
        match self {
            Self::Tree(m) => m.remove(slot),
            Self::Dense(m) => m.swap_remove(slot),
        }
    }

    pub(super) fn get(&self, slot: &u32) -> Option<&Waiting> {
        match self {
            Self::Tree(m) => m.get(slot),
            Self::Dense(m) => m.get(slot),
        }
    }

    pub(super) fn get_mut(&mut self, slot: &u32) -> Option<&mut Waiting> {
        match self {
            Self::Tree(m) => m.get_mut(slot),
            Self::Dense(m) => m.get_mut(slot),
        }
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Tree(m) => m.len(),
            Self::Dense(m) => m.len(),
        }
    }

    pub(super) fn visit(&self, mut f: impl FnMut(u32, &Waiting)) {
        match self {
            Self::Tree(m) => m.iter().for_each(|(&slot, w)| f(slot, w)),
            Self::Dense(m) => m.iter().for_each(|(&slot, w)| f(slot, w)),
        }
    }
}
