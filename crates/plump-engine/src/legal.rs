//! Legal-action masks (§3.4).
//!
//! The engine returns a per-row `u16` bid mask and a per-row `u64` play mask;
//! Python unpacks bits into `[B, 11]` / `[B, 52]` bool tensors. Shapes are
//! static; only contents vary, so masking is natively batchable.

use crate::cards::{CardSet, SUIT_MASK};

/// Returns the set of legal plays given the hand and, if present, the led suit.
///
/// Follow-suit if possible; otherwise the whole hand is legal (void).
#[inline]
pub fn legal_plays(hand: CardSet, led: Option<u8>) -> CardSet {
    match led {
        None => hand,
        Some(s) => {
            debug_assert!(s < 4, "led suit out of range");
            let f = hand & SUIT_MASK[s as usize];
            if f != 0 {
                f
            } else {
                hand
            }
        }
    }
}

/// Returns an 11-bit mask over bids `0..=10` (bit `b` set => bid `b` legal).
///
/// Fixes two v1 bugs:
/// - `n_cards - sum_others` on `u8` underflows when `sum_others > n_cards`;
///   handled explicitly (no bid is forbidden then).
/// - The forbidden-bid rule applies **only to the dealer** (last bidder).
#[inline]
pub fn legal_bids(n_cards: u8, sum_others: u8, is_dealer: bool) -> u16 {
    debug_assert!(n_cards <= 10, "n_cards out of supported range");
    let mut m: u16 = (1u16 << (n_cards + 1)) - 1; // bids 0..=n_cards
    if is_dealer && sum_others <= n_cards {
        // Forbidden bid may be negative or out of range; then nothing is forbidden.
        let forbidden = n_cards - sum_others;
        m &= !(1u16 << forbidden);
    }
    debug_assert!(m != 0, "legal bid mask must never be empty");
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn suit0_high() -> CardSet {
        1u64 << 12 // A of suit 0
    }

    #[test]
    fn play_follow_suit() {
        let hand = suit0_high() | (1u64 << 13); // one spade, one diamond
        let legal = legal_plays(hand, Some(0));
        assert_eq!(legal, suit0_high());
    }

    #[test]
    fn play_void_free_choice() {
        let hand = suit0_high() | (1u64 << 13);
        // No card of suit 3 in hand => void, whole hand is legal.
        let legal = legal_plays(hand, Some(3));
        assert_eq!(legal, hand);
    }

    #[test]
    fn play_no_led_whole_hand() {
        let hand = (1u64 << 3) | (1u64 << 40);
        assert_eq!(legal_plays(hand, None), hand);
    }

    #[test]
    fn bid_mask_full_for_non_dealer() {
        // 10 cards, others already bid 0..=10 in some sum; non-dealer never constrained.
        assert_eq!(legal_bids(10, 10, false), 0x7FF);
        assert_eq!(legal_bids(10, 0, false), 0x7FF);
    }

    #[test]
    fn bid_no_underflow_when_sum_exceeds_cards() {
        // v1 bug: `10 - 12` on u8 would wrap/panic. Here nothing is forbidden.
        assert_eq!(legal_bids(10, 12, true), 0x7FF);
        assert_eq!(legal_bids(4, 5, true), 0x1F);
    }

    #[test]
    fn bid_forbidden_only_for_dealer() {
        // sum_others = 7 with 10 cards => forbidden bid = 3 for the dealer.
        assert_eq!(legal_bids(10, 7, true), 0x7FF & !(1u16 << 3));
        assert_eq!(legal_bids(10, 7, false), 0x7FF);
    }

    #[test]
    fn bid_forbidden_never_out_of_range() {
        // sum_others == n_cards => forbidden bid = 0 (still in range, cleared).
        assert_eq!(legal_bids(10, 10, true), 0x7FF & !1);
        // sum_others == n_cards + 1 handled by the underflow guard above.
    }

    #[test]
    fn bid_mask_bounds() {
        assert_eq!(legal_bids(0, 0, false), 0b1); // only bid 0
        assert_eq!(legal_bids(1, 0, false), 0b11); // bids 0..=1
                                                   // Dealer with sum_others = 0 must not bid 10 (forbidden == n_cards, in range).
        assert_eq!(legal_bids(10, 0, true), 0x7FF & !(1u16 << 10));
        // Dealer whose forbidden bid would be 11 (out of range) is unconstrained.
        assert_eq!(legal_bids(10, 1, true), 0x7FF & !(1u16 << 9));
    }
}
