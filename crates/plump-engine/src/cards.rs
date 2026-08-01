//! Card and set representation.
//!
//! A card is a `u8` in `0..52` with the **canonical index**
//! `card = suit * 13 + rank`, where `suit` is `0..4` and `rank` is `0..13`.
//! A set of cards is a `u64` with bit `card` set.
//!
//! This convention is used **everywhere** — action indices, belief targets,
//! the layout registry, and the suit-permutation table. The v1 convention
//! (`suit = c >> 4`, `rank = c & 0xF`) was mutually inconsistent with
//! `suit * 13 + rank` and is not used here.

pub type Card = u8;
pub type CardSet = u64;

pub const N_SUITS: usize = 4;
pub const N_RANKS: usize = 13;
pub const N_CARDS: usize = 52;

/// Maximum supported number of players (layout/state slots are fixed-width at
/// this size; seats `P..MAX_SEATS` are simply never emitted).
pub const MAX_SEATS: usize = 6;

/// Suit of a card, `0..4`.
#[inline]
pub const fn suit(c: Card) -> u8 {
    c / 13
}

/// Rank of a card within its suit, `0..13`.
#[inline]
pub const fn rank(c: Card) -> u8 {
    c % 13
}

/// Canonical card index from suit and rank.
#[inline]
pub const fn card(suit: u8, rank: u8) -> Card {
    suit * 13 + rank
}

/// Constructs a card, asserting in debug builds that both fields are in range.
#[inline]
pub fn new_card(suit: u8, rank: u8) -> Card {
    debug_assert!(suit < N_SUITS as u8, "suit out of range");
    debug_assert!(rank < N_RANKS as u8, "rank out of range");
    card(suit, rank)
}

/// Bit for a single card.
#[inline]
pub const fn bit(c: Card) -> CardSet {
    1u64 << c
}

/// One-bit mask per suit over the whole deck.
pub const SUIT_MASK: [CardSet; 4] = [
    0x0000_0000_0000_1FFF, // suit 0: bits 0..=12
    0x0000_0000_03FF_E000, // suit 1: bits 13..=25
    0x0000_007F_FC00_0000, // suit 2: bits 26..=38
    0x000F_FF80_0000_0000, // suit 3: bits 39..=51
];

/// All 52 bits set.
pub const FULL_DECK: CardSet = (1u64 << 52) - 1;

/// Popcount of a card set.
#[inline]
pub fn popcount(s: CardSet) -> u32 {
    s.count_ones()
}

/// Maps a card through a suit permutation (rank preserved).
#[inline]
pub fn relabel_card(c: Card, perm: &[u8; 4]) -> Card {
    card(perm[suit(c) as usize], rank(c))
}

/// Maps every card in a set through a suit permutation (§5.4).
#[inline]
pub fn relabel_set(s: CardSet, perm: &[u8; 4]) -> CardSet {
    let mut out: CardSet = 0;
    let mut m = s;
    while m != 0 {
        let c = m.trailing_zeros() as u8;
        m &= m - 1;
        out |= bit(relabel_card(c, perm));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suit_rank_roundtrip() {
        for s in 0..N_SUITS as u8 {
            for r in 0..N_RANKS as u8 {
                let c = card(s, r);
                assert_eq!(suit(c), s);
                assert_eq!(rank(c), r);
            }
        }
    }

    #[test]
    fn canonical_encoding_is_card_id() {
        for c in 0..N_CARDS as u8 {
            assert_eq!(card(suit(c), rank(c)), c);
        }
    }

    #[test]
    fn suit_masks_are_disjoint_and_complete() {
        let mut seen: CardSet = 0;
        for m in SUIT_MASK {
            assert_eq!(m.count_ones(), N_RANKS as u32);
            assert_eq!(seen & m, 0, "suit masks overlap");
            seen |= m;
        }
        assert_eq!(seen, FULL_DECK);
    }

    #[test]
    fn bit_and_popcount() {
        assert_eq!(bit(0), 1);
        assert_eq!(bit(51), 1u64 << 51);
        let s = bit(3) | bit(9) | bit(3);
        assert_eq!(popcount(s), 2);
    }

    #[test]
    fn relabel_permutes_suit_keeps_rank() {
        let perm = [1u8, 0, 3, 2];
        let c = card(2, 5);
        let rc = relabel_card(c, &perm);
        assert_eq!(suit(rc), 3);
        assert_eq!(rank(rc), 5);
        let s = bit(c) | bit(0); // suit2 r5 + suit0 r0
        let rs = relabel_set(s, &perm);
        assert_eq!(rs, bit(card(3, 5)) | bit(card(1, 0)));
        assert_eq!(rs.count_ones(), 2);
    }

    #[test]
    fn relabel_set_followed_by_same_perm_is_identity() {
        let perm = [2u8, 3, 0, 1];
        let s = bit(0) | bit(13) | bit(51);
        let back = relabel_set(relabel_set(s, &perm), &perm);
        assert_eq!(back, s);
    }
}
