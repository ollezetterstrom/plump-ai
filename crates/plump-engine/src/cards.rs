//! cards.rs — pure constants, mirrors plump/env/cards.py

pub const SUITS: [char; 4] = ['H', 'S', 'D', 'C'];

pub fn suit_index(s: char) -> usize {
    match s {
        'H' => 0,
        'S' => 1,
        'D' => 2,
        'C' => 3,
        _ => panic!("bad suit"),
    }
}

pub fn card_index(suit: char, rank: u8) -> usize {
    suit_index(suit) * 13 + (rank - 2) as usize
}

pub fn index_to_card(idx: usize) -> (char, u8) {
    let suit = SUITS[idx / 13];
    let rank = (idx % 13 + 2) as u8;
    (suit, rank)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        for s in SUITS {
            for r in 2..15 {
                assert_eq!(index_to_card(card_index(s, r)), (s, r));
            }
        }
    }
}
