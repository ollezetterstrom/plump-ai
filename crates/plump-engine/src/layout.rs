//! Single-source-of-truth feature layout (§5.2) and structural suit
//! permutation table (§5.4).
//!
//! The Rust const table is the authority. Python imports it via the PyO3
//! bridge and asserts `net.fc0.num_embeddings == N_FEATURES`. Compile-time
//! assertions guarantee the blocks are contiguous, non-overlapping, and sum
//! exactly to `N_FEATURES`.

use crate::cards::{N_CARDS, N_RANKS, N_SUITS};

/// How a block's slots transform under a suit permutation (§3.3, §5.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuitRule {
    /// Seat/role/scalar slots; identity under suit permutation.
    Agnostic,
    /// Card-indexed slots: 52 slots per group, optionally with a stride for
    /// contiguous repeated groups (e.g. per-seat × per-card blocks).
    CardIndexed { stride: usize },
    /// Suit-indexed slots: 4 slots per group, optionally strided.
    SuitIndexed { stride: usize },
}

/// One contiguous feature block in the layout.
#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub name: &'static str,
    pub offset: usize,
    pub size: usize,
    pub suit_rule: SuitRule,
}

/// Authoritative feature layout. Offsets are contiguous by construction and
/// checked at compile time below.
pub const LAYOUT: &[Block] = &[
    Block {
        name: "my_hand",
        offset: 0,
        size: 52,
        suit_rule: SuitRule::CardIndexed { stride: 1 },
    },
    Block {
        name: "played_by_rel",
        offset: 52,
        size: 312,
        suit_rule: SuitRule::CardIndexed { stride: 1 },
    }, // 6*52
    Block {
        name: "trick_cards",
        offset: 364,
        size: 312,
        suit_rule: SuitRule::CardIndexed { stride: 1 },
    }, // pos*52
    Block {
        name: "led_suit",
        offset: 676,
        size: 5,
        suit_rule: SuitRule::SuitIndexed { stride: 1 },
    }, // 4 suits + none
    Block {
        name: "my_voids",
        offset: 681,
        size: 4,
        suit_rule: SuitRule::SuitIndexed { stride: 1 },
    },
    Block {
        name: "other_voids",
        offset: 685,
        size: 24,
        suit_rule: SuitRule::SuitIndexed { stride: 1 },
    }, // 6*4
    Block {
        name: "unseen_by_suit",
        offset: 709,
        size: 56,
        suit_rule: SuitRule::SuitIndexed { stride: 14 },
    }, // per-suit count 0..13
    Block {
        name: "bids_rel",
        offset: 765,
        size: 78,
        suit_rule: SuitRule::Agnostic,
    }, // 6*13 = bid 0..=10, "unknown", 1 reserved
    Block {
        name: "tricks_rel",
        offset: 843,
        size: 66,
        suit_rule: SuitRule::Agnostic,
    },
    Block {
        name: "need_rel",
        offset: 909,
        size: 126,
        suit_rule: SuitRule::Agnostic,
    }, // 6*(bid-tricks, -10..10)
    Block {
        name: "role_flags",
        offset: 1035,
        size: 30,
        suit_rule: SuitRule::Agnostic,
    },
    Block {
        name: "round_ctx",
        offset: 1065,
        size: 40,
        suit_rule: SuitRule::Agnostic,
    },
    Block {
        name: "match_ctx",
        offset: 1105,
        size: 120,
        suit_rule: SuitRule::Agnostic,
    },
];

/// Total number of scalar features.
pub const N_FEATURES: usize = 1225;

/// Number of suit permutations of 4 elements.
pub const N_PERMUTATIONS: usize = 24;

/// Compile-time check: blocks are contiguous, non-empty, and sum to `N_FEATURES`.
const _: () = {
    assert!(LAYOUT.len() == 13, "layout must have exactly 13 blocks");
    let mut expected: usize = 0;
    let mut i: usize = 0;
    while i < LAYOUT.len() {
        let b = &LAYOUT[i];
        assert!(b.offset == expected, "block not contiguous");
        assert!(b.size > 0, "zero-size block");
        expected += b.size;
        i += 1;
    }
    assert!(expected == N_FEATURES, "layout does not sum to N_FEATURES");
};

/// All 24 permutations of `[0, 1, 2, 3]` (Heap's algorithm).
pub fn all_permutations_of_4() -> Vec<[u8; 4]> {
    let mut out = Vec::with_capacity(N_PERMUTATIONS);
    let mut p = [0u8, 1, 2, 3];
    let mut c = [0usize; 4];
    out.push(p);
    let mut i = 0usize;
    while i < 4 {
        if c[i] < i {
            if i.is_multiple_of(2) {
                p.swap(0, i);
            } else {
                p.swap(c[i], i);
            }
            out.push(p);
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
    debug_assert_eq!(out.len(), N_PERMUTATIONS);
    out
}

/// Maps an index `k` inside a block under a suit permutation `perm`.
///
/// Slot groups whose size is an exact multiple of the group width are
/// permuted; any trailing slots ("none" led-suit marker, extra scalars) map
/// to themselves.
fn map_index(b: &Block, k: usize, perm: &[u8; 4]) -> usize {
    match b.suit_rule {
        SuitRule::Agnostic => k,
        SuitRule::CardIndexed { stride } => {
            let group = N_CARDS * stride;
            let n_full = b.size / group;
            if k < n_full * group {
                let outer = k / group;
                let c = k % N_CARDS;
                let s = perm[c / N_RANKS] as usize;
                outer * group + s * N_RANKS + (c % N_RANKS)
            } else {
                k
            }
        }
        SuitRule::SuitIndexed { stride } => {
            let group = N_SUITS * stride;
            let n_full = b.size / group;
            if k < n_full * group {
                let outer = k / group;
                let s = (k / stride) % N_SUITS;
                let inner = k % stride;
                outer * group + (perm[s] as usize) * stride + inner
            } else {
                k
            }
        }
    }
}

/// Precomputed permutation table: `table[perm * N_FEATURES + idx]` is the
/// index `idx` maps to under permutation `perm`. Applying an augmentation is a
/// single lookup and is provably identity on agnostic blocks.
pub fn build_perm_table() -> Vec<i16> {
    let mut t = vec![0i16; N_PERMUTATIONS * N_FEATURES];
    for (pi, perm) in all_permutations_of_4().into_iter().enumerate() {
        for b in LAYOUT {
            for k in 0..b.size {
                let mapped = map_index(b, k, &perm);
                debug_assert!(mapped < b.size, "permutation escaped its block");
                t[pi * N_FEATURES + b.offset + k] = (b.offset + mapped) as i16;
            }
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_match_plan() {
        let expect = [
            (0, 52),
            (52, 312),
            (364, 312),
            (676, 5),
            (681, 4),
            (685, 24),
            (709, 56),
            (765, 78),
            (843, 66),
            (909, 126),
            (1035, 30),
            (1065, 40),
            (1105, 120),
        ];
        for (b, (off, size)) in LAYOUT.iter().zip(expect.iter()) {
            assert_eq!(b.offset, *off, "offset of {}", b.name);
            assert_eq!(b.size, *size, "size of {}", b.name);
        }
        assert_eq!(
            LAYOUT.last().unwrap().offset + LAYOUT.last().unwrap().size,
            N_FEATURES
        );
    }

    #[test]
    fn twenty_four_permutations() {
        let perms = all_permutations_of_4();
        assert_eq!(perms.len(), 24);
        for p in &perms {
            let mut sorted = *p;
            sorted.sort_unstable();
            assert_eq!(sorted, [0, 1, 2, 3], "not a permutation");
        }
    }

    #[test]
    fn identity_permutation_is_identity() {
        let t = build_perm_table();
        // First generated permutation is the identity [0,1,2,3].
        for (i, &v) in t.iter().take(N_FEATURES).enumerate() {
            assert_eq!(v, i as i16);
        }
    }

    #[test]
    fn each_permutation_bijective_on_full_index_space() {
        let t = build_perm_table();
        for pi in 0..N_PERMUTATIONS {
            let row = &t[pi * N_FEATURES..(pi + 1) * N_FEATURES];
            let mut seen = vec![false; N_FEATURES];
            for &v in row {
                assert!((0..N_FEATURES as i16).contains(&v), "index out of range");
                assert!(!seen[v as usize], "duplicate mapping");
                seen[v as usize] = true;
            }
            assert!(seen.iter().all(|&x| x), "non-bijective mapping");
        }
    }

    #[test]
    fn agnostic_blocks_are_identity() {
        let t = build_perm_table();
        for pi in 0..N_PERMUTATIONS {
            for b in LAYOUT {
                if b.suit_rule == SuitRule::Agnostic {
                    for k in 0..b.size {
                        let src = pi * N_FEATURES + b.offset + k;
                        assert_eq!(
                            t[src],
                            (b.offset + k) as i16,
                            "agnostic block {} moved",
                            b.name
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn card_permutation_swaps_suit_blocks() {
        let t = build_perm_table();
        let pi = all_permutations_of_4()
            .iter()
            .position(|p| *p == [1, 0, 2, 3])
            .unwrap();
        // my_hand: card index (suit=1, rank=5) == 1*13+5 == 18 must map to
        // (suit=0, rank=5) == 5.
        let src = pi * N_FEATURES + 18;
        assert_eq!(t[src], 5);
        // Suit 0 rank 5 (== 5) must map to suit 1 rank 5 (== 18).
        assert_eq!(t[pi * N_FEATURES + 5], 18);
    }

    #[test]
    fn led_suit_none_slot_is_invariant() {
        let t = build_perm_table();
        for pi in 0..N_PERMUTATIONS {
            // led_suit block starts at 676; slot 4 is "none".
            assert_eq!(t[pi * N_FEATURES + 676 + 4], (676 + 4) as i16);
        }
    }
}
