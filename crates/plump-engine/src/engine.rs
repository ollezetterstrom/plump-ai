use rand::seq::SliceRandom;
use rand::Rng;

/// Vocabulary size of the token stream (0..70).
pub const VOCAB: usize = 70;
/// Maximum token sequence length per decision.
pub const MAX_SEQ: usize = 96;
/// Size of the dense feature vector per decision.
pub const FEAT_DIM: usize = 296;

pub const PAD_TOK: u8 = 0;
pub const BOS_TOK: u8 = 1;
pub const BID_BASE: u8 = 2; // 2..=12 (bids / round_cards 0..=10)
pub const PLAYER_BASE: u8 = 13; // 13..=16 (relative player 0..=3)
pub const CARD_BASE: u8 = 17; // 17..=68 (card_idx 0..=51)

pub const NUM_PLAYERS: usize = 4;
pub const NUM_CARDS: usize = 52;
pub const SUIT_MASK: u64 = 0x1FFF;

/// The 20-round Swedish Plump schedule: 10 -> 1, then 1 -> 10.
pub const ROUND_SCHEDULE: [u8; 20] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

pub const FULL_DECK: u64 = (1u64 << 52) - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Bidding,
    Playing,
    Terminal,
}

#[inline]
pub fn suit_of(card_idx: u8) -> u8 {
    card_idx / 13
}

#[inline]
pub fn rank_of(card_idx: u8) -> u8 {
    card_idx % 13 + 2
}

#[inline]
pub fn card_index(suit: u8, rank: u8) -> u8 {
    debug_assert!(suit < 4);
    debug_assert!((2..=14).contains(&rank));
    suit * 13 + (rank - 2)
}

#[inline]
pub fn suit_bits(suit: u8) -> u64 {
    SUIT_MASK << (suit as usize * 13)
}

/// Display points used on the physical scoreboard.
#[inline]
pub fn display_points(tricks_won: u8, bid: i8) -> i32 {
    if bid < 0 {
        return 0;
    }
    if tricks_won as i32 == bid as i32 {
        if bid == 0 {
            5
        } else {
            10 + bid as i32
        }
    } else {
        0
    }
}

/// Dense RL reward for one player's round result.
#[inline]
pub fn terminal_reward(tricks_won: u8, bid: i8) -> f32 {
    if bid < 0 {
        return 0.0;
    }
    if tricks_won as i32 == bid as i32 {
        if bid == 0 {
            5.0
        } else {
            10.0 + bid as f32
        }
    } else {
        -2.0 * (tricks_won as f32 - bid as f32).abs()
    }
}

/// One complete Plump round for four players.
///
/// The struct is deliberately `Copy` so the search can clone worlds for free.
#[derive(Clone, Copy, Debug)]
pub struct FastEnv {
    pub hands: [u64; 4],
    pub played: u64,
    pub void_mask: [u8; 4],
    pub bids: [i8; 4],
    pub tricks_won: [u8; 4],
    pub table: [(u8, u8); 4], // (player, card_idx)
    pub table_len: u8,
    pub led_suit: u8, // 255 if none, else 0..4
    pub round_cards: u8,
    pub current_trick: u8,
    pub start_bidder: u8,
    pub bids_made: u8,
    pub current_player: u8,
    pub phase: Phase,
    pub history_events: [(u8, u8, u8); 44], // (kind: 0=bid, 1=play; abs_player; payload)
    pub history_len: u8,
}

impl Default for FastEnv {
    fn default() -> Self {
        Self {
            hands: [0; 4],
            played: 0,
            void_mask: [0; 4],
            bids: [-1; 4],
            tricks_won: [0; 4],
            table: [(0, 0); 4],
            table_len: 0,
            led_suit: 255,
            round_cards: 0,
            current_trick: 0,
            start_bidder: 0,
            bids_made: 0,
            current_player: 0,
            phase: Phase::Bidding,
            history_events: [(0, 0, 0); 44],
            history_len: 0,
        }
    }
}

impl FastEnv {
    /// Deal a fresh round. `start_bidder` is the player directly after the
    /// dealer/shuffler and bids first.
    pub fn new_round(&mut self, round_cards: u8, start_bidder: u8, rng: &mut impl Rng) {
        assert!(
            (1..=10).contains(&round_cards),
            "round_cards must be 1..=10"
        );
        assert!(start_bidder < 4, "start_bidder must be 0..=3");

        let mut deck: [u8; NUM_CARDS] = std::array::from_fn(|i| i as u8);
        deck.shuffle(rng);

        self.hands = [0; 4];
        for p in 0..NUM_PLAYERS {
            for i in 0..round_cards as usize {
                let card = deck[p * round_cards as usize + i];
                self.hands[p] |= 1u64 << card;
            }
        }

        self.played = 0;
        self.void_mask = [0; 4];
        self.bids = [-1; 4];
        self.tricks_won = [0; 4];
        self.table = [(0, 0); 4];
        self.table_len = 0;
        self.led_suit = 255;
        self.round_cards = round_cards;
        self.current_trick = 0;
        self.start_bidder = start_bidder;
        self.bids_made = 0;
        self.current_player = start_bidder;
        self.phase = Phase::Bidding;
        self.history_events = [(0, 0, 0); 44];
        self.history_len = 0;
    }

    /// Offset of `player` in the bidding order (0 = first bidder).
    #[inline]
    pub fn bidding_offset(&self, player: usize) -> usize {
        (player + 4 - self.start_bidder as usize) % 4
    }

    /// 11-bit mask of legal bids (bits 0..=10).
    pub fn legal_bids_mask(&self, player: usize) -> u16 {
        if self.phase != Phase::Bidding || self.current_player as usize != player {
            return 0;
        }
        let all: u16 = (1u16 << (self.round_cards as u16 + 1)) - 1;
        if self.bidding_offset(player) == 3 {
            let sum: i32 = (0..4)
                .filter(|&p| p != player)
                .map(|p| self.bids[p] as i32)
                .sum();
            let forbidden = self.round_cards as i32 - sum;
            if (0..=self.round_cards as i32).contains(&forbidden) {
                return all & !(1u16 << forbidden);
            }
        }
        all
    }

    /// 52-bit mask of cards `player` may legally play.
    #[inline]
    pub fn legal_cards_mask(&self, player: usize) -> u64 {
        if self.phase != Phase::Playing || self.current_player as usize != player {
            return 0;
        }
        let hand = self.hands[player];
        if self.table_len == 0 || self.led_suit >= 4 {
            return hand;
        }
        let in_suit = hand & suit_bits(self.led_suit);
        if in_suit != 0 {
            in_suit
        } else {
            hand
        }
    }

    /// Either a legal card mask or a legal bid mask, depending on phase.
    pub fn legal_mask(&self) -> u64 {
        match self.phase {
            Phase::Bidding => self.legal_bids_mask(self.current_player as usize) as u64,
            Phase::Playing => self.legal_cards_mask(self.current_player as usize),
            Phase::Terminal => 0,
        }
    }

    /// Earliest highest bidder in bidding order leads trick 1.
    pub fn get_leader(&self) -> usize {
        let mut best_bid = i32::MIN;
        let mut leader = self.start_bidder as usize;
        for offset in 0..4 {
            let p = (self.start_bidder as usize + offset) % 4;
            let bid = self.bids[p] as i32;
            if bid > best_bid {
                best_bid = bid;
                leader = p;
            }
        }
        leader
    }

    #[inline]
    fn push_history(&mut self, kind: u8, player: u8, payload: u8) {
        if (self.history_len as usize) < self.history_events.len() {
            self.history_events[self.history_len as usize] = (kind, player, payload);
            self.history_len += 1;
        }
    }

    /// Record a bid. After the 4th bid the round switches to trick play and
    /// the earliest highest bidder leads.
    pub fn step_bid(&mut self, bid: u8) {
        assert_eq!(self.phase, Phase::Bidding, "step_bid in {:?}", self.phase);
        let p = self.current_player as usize;
        debug_assert!(
            self.legal_bids_mask(p) & (1u16 << bid) != 0,
            "illegal bid {bid} by player {p}"
        );
        assert!(bid as u8 <= self.round_cards, "bid exceeds round_cards");

        self.bids[p] = bid as i8;
        self.push_history(0, p as u8, bid);
        self.bids_made += 1;

        if self.bids_made == 4 {
            self.phase = Phase::Playing;
            self.current_player = self.get_leader() as u8;
            self.current_trick = 0;
            self.table_len = 0;
            self.led_suit = 255;
        } else {
            self.current_player = ((p + 1) % 4) as u8;
        }
    }

    /// Play one card. Returns `Some(winner)` when the trick completes.
    pub fn step_play(&mut self, card_idx: u8) -> Option<usize> {
        assert_eq!(self.phase, Phase::Playing, "step_play in {:?}", self.phase);
        assert!(card_idx < 52, "card index out of range");
        let p = self.current_player as usize;
        let bit = 1u64 << card_idx;
        debug_assert!(
            self.hands[p] & bit != 0,
            "player {p} does not hold card {card_idx}"
        );
        debug_assert!(
            self.legal_cards_mask(p) & bit != 0,
            "illegal play {card_idx} by player {p}"
        );

        let card_suit = suit_of(card_idx);
        if self.table_len == 0 {
            self.led_suit = card_suit;
        } else if card_suit != self.led_suit {
            self.void_mask[p] |= 1u8 << self.led_suit;
        }

        self.hands[p] &= !bit;
        self.table[self.table_len as usize] = (p as u8, card_idx);
        self.table_len += 1;
        self.push_history(1, p as u8, card_idx);

        if self.table_len == 4 {
            let winner_pos = self.trick_winner_position();
            let winner = self.table[winner_pos].0 as usize;
            self.tricks_won[winner] += 1;

            for i in 0..4 {
                self.played |= 1u64 << self.table[i].1;
            }

            self.current_trick += 1;
            self.table_len = 0;
            self.led_suit = 255;
            self.current_player = winner as u8;

            if self.current_trick == self.round_cards {
                self.phase = Phase::Terminal;
            }
            Some(winner)
        } else {
            self.current_player = ((p + 1) % 4) as u8;
            None
        }
    }

    /// Index (0..4) into `table` of the highest card of the led suit.
    fn trick_winner_position(&self) -> usize {
        let led = self.led_suit;
        let mut best_pos = 0usize;
        let mut best_rank = self.table[0].1 % 13;
        for i in 1..4 {
            let card = self.table[i].1;
            if suit_of(card) == led && (card % 13) > best_rank {
                best_rank = card % 13;
                best_pos = i;
            }
        }
        best_pos
    }

    /// Winners of every completed trick, derived from the public event log.
    pub fn trick_winners(&self) -> Vec<usize> {
        let mut winners = Vec::with_capacity(self.round_cards as usize);
        let mut trick: Vec<(u8, u8)> = Vec::with_capacity(4);
        for &(kind, player, card) in &self.history_events[..self.history_len as usize] {
            if kind != 1 {
                continue;
            }
            trick.push((player, card));
            if trick.len() == 4 {
                let led = suit_of(trick[0].1);
                let mut best = 0usize;
                let mut best_rank = trick[0].1 % 13;
                for i in 1..4 {
                    let card = trick[i].1;
                    if suit_of(card) == led && (card % 13) > best_rank {
                        best_rank = card % 13;
                        best = i;
                    }
                }
                winners.push(trick[best].0 as usize);
                trick.clear();
            }
        }
        winners
    }

    /// `true` when a player has publicly shown out of `suit`.
    #[inline]
    pub fn is_void(&self, player: usize, suit: u8) -> bool {
        self.void_mask[player] & (1u8 << suit) != 0
    }

    /// Encode the round from `viewer`'s perspective.
    ///
    /// * tokens: `[BOS, BID_BASE + round_cards, (PLAYER_BASE+rel, event)...]`
    /// * features: 296 floats, see docs/ARCHITECTURE.md
    /// * `opp_hands_out`: ground truth 3x52 one-hot for relative opponents
    ///   `(viewer+1)`, `(viewer+2)`, `(viewer+3)`.
    pub fn encode_decision(
        &self,
        viewer: usize,
        toks_out: &mut [i64; MAX_SEQ],
        feats_out: &mut [f32; FEAT_DIM],
        opp_hands_out: &mut [f32; 156],
    ) {
        assert!(viewer < 4, "viewer must be 0..=3");
        toks_out.fill(PAD_TOK as i64);
        feats_out.fill(0.0);
        opp_hands_out.fill(0.0);

        toks_out[0] = BOS_TOK as i64;
        toks_out[1] = BID_BASE as i64 + self.round_cards as i64;

        let mut pos = 2usize;
        for &(kind, abs_p, payload) in &self.history_events[..self.history_len as usize] {
            if pos + 2 > MAX_SEQ {
                break;
            }
            let rel_p = (abs_p as usize + 4 - viewer) % 4;
            toks_out[pos] = PLAYER_BASE as i64 + rel_p as i64;
            toks_out[pos + 1] = if kind == 0 {
                BID_BASE as i64 + payload as i64
            } else {
                CARD_BASE as i64 + payload as i64
            };
            pos += 2;
        }

        // 0..52 own hand (hidden on the forehead in a 1-card round).
        if self.round_cards >= 2 {
            let mut hand = self.hands[viewer];
            while hand != 0 {
                let card = hand.trailing_zeros() as usize;
                feats_out[card] = 1.0;
                hand &= hand - 1;
            }
            // 52..208 current trick cards played by relative opponents.
            for idx in 0..self.table_len as usize {
                let abs_p = self.table[idx].0 as usize;
                if abs_p != viewer {
                    let rel_opp = (abs_p + 4 - viewer) % 4 - 1;
                    let card = self.table[idx].1;
                    feats_out[52 + rel_opp * 52 + card as usize] = 1.0;
                }
            }
        } else {
            // 52..208 faced-up cards on the three opponents' foreheads.
            for i in 0..3 {
                let p = (viewer + i + 1) % 4;
                let mut hand = self.hands[p];
                while hand != 0 {
                    let card = hand.trailing_zeros() as usize;
                    feats_out[52 + i * 52 + card] = 1.0;
                    hand &= hand - 1;
                }
            }
        }

        // 208..260 completed tricks.
        let mut played = self.played;
        while played != 0 {
            let card = played.trailing_zeros() as usize;
            feats_out[208 + card] = 1.0;
            played &= played - 1;
        }

        // 260..276 public void mask, relative seats.
        for abs_p in 0..4 {
            let rel_p = (abs_p + 4 - viewer) % 4;
            for suit in 0..4 {
                if self.void_mask[abs_p] & (1u8 << suit) != 0 {
                    feats_out[260 + rel_p * 4 + suit] = 1.0;
                }
            }
        }

        // 276..288 bids / tricks / remaining need, relative seats.
        for abs_p in 0..4 {
            let rel_p = (abs_p + 4 - viewer) % 4;
            let bid = self.bids[abs_p];
            feats_out[276 + rel_p] = if bid >= 0 { bid as f32 / 10.0 } else { -0.1 };
            feats_out[280 + rel_p] = self.tricks_won[abs_p] as f32 / 10.0;
            feats_out[284 + rel_p] = if bid >= 0 {
                (bid as i32 - self.tricks_won[abs_p] as i32) as f32 / 10.0
            } else {
                0.0
            };
        }

        // 288..292 own position in the bidding order.
        let order_pos = (viewer + 4 - self.start_bidder as usize) % 4;
        feats_out[288 + order_pos] = 1.0;

        // 292..296 led suit (one-hot, all zero when the trick is not led).
        if self.led_suit < 4 {
            feats_out[292 + self.led_suit as usize] = 1.0;
        }

        // Ground-truth opponent hands for the belief auxiliary head.
        for i in 0..3 {
            let p = (viewer + i + 1) % 4;
            let mut hand = self.hands[p];
            while hand != 0 {
                let card = hand.trailing_zeros() as usize;
                opp_hands_out[i * 52 + card] = 1.0;
                hand &= hand - 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn empty_env(round_cards: u8, start_bidder: u8, phase: Phase) -> FastEnv {
        let mut env = FastEnv::default();
        env.round_cards = round_cards;
        env.start_bidder = start_bidder;
        env.current_player = start_bidder;
        env.phase = phase;
        env
    }

    fn encode(env: &FastEnv, viewer: usize) -> ([i64; MAX_SEQ], [f32; FEAT_DIM], [f32; 156]) {
        let mut toks = [0i64; MAX_SEQ];
        let mut feats = [0.0f32; FEAT_DIM];
        let mut opp = [0.0f32; 156];
        env.encode_decision(viewer, &mut toks, &mut feats, &mut opp);
        (toks, feats, opp)
    }

    #[test]
    fn rotating_start_bidder_and_early_tie_break() {
        // start_bidder rotates 0,1,2,3 and the first bidder is that seat.
        for start in 0..4u8 {
            let mut rng = StdRng::seed_from_u64(start as u64);
            let mut env = FastEnv::default();
            env.new_round(5, start, &mut rng);
            assert_eq!(env.start_bidder, start);
            assert_eq!(env.current_player, start);
        }

        // start=2 => bidding order 2, 3, 0, 1. Bids in order 5, 7, 7, 5:
        // player 3 reached 7 first, so they lead despite player 0 also bidding 7.
        let mut env = empty_env(10, 2, Phase::Bidding);
        env.bids = [7, 5, 5, 7];
        assert_eq!(env.get_leader(), 3);

        // Unique highest bid.
        let mut env = empty_env(10, 1, Phase::Bidding);
        env.bids = [2, 4, 9, 4];
        assert_eq!(env.get_leader(), 2);

        // Earliest among all-equal bids is the first bidder.
        let mut env = empty_env(10, 3, Phase::Bidding);
        env.bids = [6, 6, 6, 6];
        assert_eq!(env.get_leader(), 3);
    }

    #[test]
    fn last_bidder_cannot_make_bid_sum_equal_round_cards() {
        for start in 0..4u8 {
            let mut env = empty_env(5, start, Phase::Bidding);
            // First three bidders bid 2, 1, 1 => sum 4, forbidden dealer bid = 1.
            for bid in [2u8, 1, 1] {
                let p = env.current_player as usize;
                assert_eq!(env.bidding_offset(p) < 3, true);
                env.bids[p] = bid as i8;
                env.bids_made += 1;
                env.current_player = ((p + 1) % 4) as u8;
            }
            let dealer = env.current_player as usize;
            assert_eq!(env.bidding_offset(dealer), 3);
            assert_eq!(
                env.legal_bids_mask(dealer),
                0b11_1101, // 0..=5 minus bit 1
                "start_bidder={start}"
            );

            // A sum equal to round_cards forbids bidding 0.
            let mut env = empty_env(5, start, Phase::Bidding);
            for bid in [2u8, 2, 1] {
                let p = env.current_player as usize;
                env.bids[p] = bid as i8;
                env.bids_made += 1;
                env.current_player = ((p + 1) % 4) as u8;
            }
            let dealer = env.current_player as usize;
            assert_eq!(env.legal_bids_mask(dealer), 0b11_1110);

            // Overshooting the round total forbids nothing.
            let mut env = empty_env(5, start, Phase::Bidding);
            for bid in [5u8, 5, 3] {
                let p = env.current_player as usize;
                env.bids[p] = bid as i8;
                env.bids_made += 1;
                env.current_player = ((p + 1) % 4) as u8;
            }
            let dealer = env.current_player as usize;
            assert_eq!(env.legal_bids_mask(dealer), 0b11_1111);
        }

        // First three bidders always see the full range.
        for start in 0..4u8 {
            let env = empty_env(7, start, Phase::Bidding);
            let first = start as usize;
            assert_eq!(env.legal_bids_mask(first), 0b1111_1111);
        }
    }

    #[test]
    fn pannplump_hides_own_card_and_shows_foreheads() {
        // 1-card round: viewer 0 holds card 51 but must not see it; the three
        // opponent cards are fully visible.
        let mut env = empty_env(1, 0, Phase::Bidding);
        env.hands = [1u64 << 51, 1u64 << 10, 1u64 << 20, 1u64 << 30];
        let (toks, feats, opp) = encode(&env, 0);

        assert_eq!(toks[0], BOS_TOK as i64);
        assert_eq!(toks[1], BID_BASE as i64 + 1);
        assert!(feats[0..52].iter().all(|&v| v == 0.0));
        assert_eq!(feats[52 + 10], 1.0);
        assert_eq!(feats[52 + 52 + 20], 1.0);
        assert_eq!(feats[52 + 104 + 30], 1.0);
        let visible: f32 = feats[52..208].iter().sum();
        assert_eq!(visible, 3.0);
        assert_eq!(opp[0 * 52 + 10], 1.0);
        assert_eq!(opp[1 * 52 + 20], 1.0);
        assert_eq!(opp[2 * 52 + 30], 1.0);

        // From viewer 1's perspective the relative opponent order rotates.
        let (_, feats1, opp1) = encode(&env, 1);
        assert!(feats1[0..52].iter().all(|&v| v == 0.0));
        assert_eq!(feats1[52 + 20], 1.0);
        assert_eq!(feats1[52 + 52 + 30], 1.0);
        assert_eq!(feats1[52 + 104 + 51], 1.0);
        assert_eq!(opp1[0 * 52 + 20], 1.0);
        assert_eq!(opp1[1 * 52 + 30], 1.0);
        assert_eq!(opp1[2 * 52 + 51], 1.0);

        // 2+ cards: own hand is visible, forehead block is empty.
        let mut env = empty_env(2, 0, Phase::Bidding);
        env.hands = [0b11, 0b1100, 0b11_0000, 0b11_0000_00];
        let (_, feats, _) = encode(&env, 0);
        assert_eq!(feats[0], 1.0);
        assert_eq!(feats[1], 1.0);
        assert_eq!(feats[0..52].iter().sum::<f32>(), 2.0);
        assert!(feats[52..208].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn legal_cards_void_tracking_and_trick_winner() {
        let mut env = empty_env(3, 0, Phase::Playing);
        // Player 0 leads the 5 of hearts and also holds the 2 of spades.
        let lead_card = card_index(0, 5);
        let spade = card_index(1, 2); // 2 of spades
        env.hands = [1u64 << lead_card | 1u64 << spade, 0, 0, 0];
        env.current_player = 0;

        // With no led suit, every card is legal.
        assert_eq!(env.legal_cards_mask(0), env.hands[0]);

        env.step_play(lead_card);
        assert_eq!(env.led_suit, 0);

        // Player 1 is void in hearts and holds only diamonds => all legal.
        let p1_card = card_index(2, 14);
        env.hands[1] = 1u64 << p1_card;
        env.current_player = 1;
        assert_eq!(env.legal_cards_mask(1), env.hands[1]);
        env.step_play(p1_card);
        assert!(env.is_void(1, 0));

        // Player 2 must follow hearts even though a higher card exists elsewhere.
        let p2_low_heart = card_index(0, 4);
        let p2_high_heart = card_index(0, 13);
        let p2_spade = card_index(1, 14);
        env.hands[2] = (1u64 << p2_low_heart) | (1u64 << p2_high_heart) | (1u64 << p2_spade);
        env.current_player = 2;
        assert_eq!(
            env.legal_cards_mask(2),
            (1u64 << p2_low_heart) | (1u64 << p2_high_heart)
        );
        // The off-suit ace is illegal while holding hearts.
        env.step_play(p2_high_heart);

        // Player 3 follows with a lower heart; player 2 wins.
        let p3_heart = card_index(0, 9);
        env.hands[3] = 1u64 << p3_heart;
        env.current_player = 3;
        let winner = env.step_play(p3_heart);
        assert_eq!(winner, Some(2));
        assert_eq!(env.tricks_won[2], 1);
        assert_eq!(env.current_player, 2);
        assert_eq!(env.current_trick, 1);
        assert_eq!(env.led_suit, 255);
        assert_eq!(
            env.played,
            (1u64 << lead_card) | (1u64 << p1_card) | (1u64 << p2_high_heart) | (1u64 << p3_heart)
        );
    }

    #[test]
    fn full_round_reaches_terminal_and_scores() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut env = FastEnv::default();
        env.new_round(4, 1, &mut rng);

        while env.phase != Phase::Terminal {
            match env.phase {
                Phase::Bidding => {
                    let p = env.current_player as usize;
                    let mask = env.legal_bids_mask(p);
                    let bid = mask.trailing_zeros() as u8;
                    env.step_bid(bid);
                }
                Phase::Playing => {
                    let p = env.current_player as usize;
                    let mask = env.legal_cards_mask(p);
                    assert!(mask != 0);
                    let card = mask.trailing_zeros() as u8;
                    env.step_play(card);
                }
                Phase::Terminal => break,
            }
        }

        assert_eq!(env.current_trick, 4);
        assert_eq!(env.tricks_won.iter().map(|&x| x as u32).sum::<u32>(), 4);
        assert_eq!(env.history_len, 4 + 16);
        for p in 0..4 {
            let pts = display_points(env.tricks_won[p], env.bids[p]);
            let rew = terminal_reward(env.tricks_won[p], env.bids[p]);
            if env.tricks_won[p] as i8 == env.bids[p] {
                assert_eq!(
                    pts,
                    if env.bids[p] == 0 {
                        5
                    } else {
                        10 + env.bids[p] as i32
                    }
                );
                assert_eq!(
                    rew,
                    if env.bids[p] == 0 {
                        5.0
                    } else {
                        10.0 + env.bids[p] as f32
                    }
                );
            } else {
                assert_eq!(pts, 0);
                assert_eq!(
                    rew,
                    -2.0 * (env.tricks_won[p] as f32 - env.bids[p] as f32).abs()
                );
            }
        }
    }

    #[test]
    fn normal_round_table_cards_encoded_in_feats() {
        let mut env = empty_env(2, 0, Phase::Playing);
        // Player 0 leads 5 of hearts (suit 0, rank 5 -> card_index(0, 7) = 5)
        let c0 = 5u8;
        // Player 1 plays 10 of hearts (suit 0, rank 10 -> card_index(0, 12) = 10)
        let c1 = 10u8;
        env.hands = [1u64 << c0, 1u64 << c1, 1u64 << 20, 1u64 << 30];
        env.step_play(c0);
        env.step_play(c1);

        assert_eq!(env.table_len, 2);
        assert_eq!(env.current_player, 2);

        // From viewer 2's perspective:
        // abs_p 0: rel_opp = (0 + 4 - 2) % 4 - 1 = 1. Feat index: 52 + 1 * 52 + 5 = 109.
        // abs_p 1: rel_opp = (1 + 4 - 2) % 4 - 1 = 2. Feat index: 52 + 2 * 52 + 10 = 166.
        let (_, feats, _) = encode(&env, 2);
        assert_eq!(feats[52 + 1 * 52 + 5], 1.0);
        assert_eq!(feats[52 + 2 * 52 + 10], 1.0);
        let table_sum: f32 = feats[52..208].iter().sum();
        assert_eq!(table_sum, 2.0);
    }

    #[test]
    fn trick_winner_lowest_rank_led() {
        let mut env = empty_env(1, 0, Phase::Playing);
        // Card index 0 is 2 of hearts (suit 0, rank 2, rank index 0).
        let c0 = card_index(0, 2);
        let c1 = card_index(1, 14); // Ace of spades (off-suit)
        let c2 = card_index(2, 14); // Ace of diamonds (off-suit)
        let c3 = card_index(3, 14); // Ace of clubs (off-suit)
        env.hands = [1u64 << c0, 1u64 << c1, 1u64 << c2, 1u64 << c3];

        env.step_play(c0);
        env.step_play(c1);
        env.step_play(c2);
        let winner = env.step_play(c3);

        // Player 0's 2 of hearts must win because other cards did not follow suit.
        assert_eq!(winner, Some(0));
        assert_eq!(env.tricks_won[0], 1);
        assert_eq!(env.trick_winners(), vec![0]);
    }

    #[test]
    fn scoring_constants_match_house_rules() {
        assert_eq!(display_points(0, 0), 5);
        assert_eq!(display_points(1, 1), 11);
        assert_eq!(display_points(3, 3), 13);
        assert_eq!(display_points(2, 1), 0);
        assert_eq!(terminal_reward(0, 0), 5.0);
        assert_eq!(terminal_reward(4, 4), 14.0);
        assert_eq!(terminal_reward(2, 1), -2.0);
        assert_eq!(terminal_reward(0, 3), -6.0);
    }
}
