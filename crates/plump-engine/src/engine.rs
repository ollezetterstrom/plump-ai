//! engine.rs — mirrors plump/env/engine.py, pure, no deps beyond rand.

use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::cards::{card_index, SUITS};

#[derive(Clone, Debug)]
pub struct PlumpEnv {
    pub num_players: usize,
    pub hands: Vec<Vec<(char, u8)>>,
    pub bids: Vec<i8>,
    pub tricks_won: Vec<u8>,
    pub table: Vec<(usize, (char, u8))>,
    pub led_suit: Option<char>,
    pub round_cards: usize,
    pub current_trick: usize,
    pub played: Vec<(char, u8)>,
    pub void_matrix: Vec<[bool; 4]>,
    pub history: Vec<(String, usize, String)>, // stub for tokenizer
}

impl PlumpEnv {
    pub fn new() -> Self {
        Self {
            num_players: 4,
            hands: vec![],
            bids: vec![],
            tricks_won: vec![],
            table: vec![],
            led_suit: None,
            round_cards: 0,
            current_trick: 0,
            played: vec![],
            void_matrix: vec![[false; 4]; 4],
            history: vec![],
        }
    }

    pub fn new_round(&mut self, round_cards: usize) {
        self.round_cards = round_cards;
        self.current_trick = 0;
        self.table.clear();
        self.led_suit = None;
        self.played.clear();
        self.hands = vec![vec![]; 4];
        self.bids = vec![-1; 4];
        self.tricks_won = vec![0; 4];
        self.void_matrix = vec![[false; 4]; 4];
        self.history.clear();
        let mut deck: Vec<(char, u8)> = SUITS.iter().flat_map(|&s| (2..15).map(move |r| (s, r))).collect();
        deck.shuffle(&mut thread_rng());
        for i in 0..round_cards * 4 {
            self.hands[i % 4].push(deck[i]);
        }
    }

    pub fn can_follow(&self, player: usize) -> bool {
        if let Some(led) = self.led_suit {
            self.hands[player].iter().any(|(s, _)| *s == led)
        } else {
            false
        }
    }

    pub fn is_legal(&self, player: usize, card: (char, u8)) -> bool {
        if !self.hands[player].contains(&card) {
            return false;
        }
        match self.led_suit {
            None => true,
            Some(led) if card.0 == led => true,
            _ => !self.can_follow(player),
        }
    }

    pub fn legal_cards(&self, player: usize) -> Vec<(char, u8)> {
        self.hands[player].iter().copied().filter(|&c| self.is_legal(player, c)).collect()
    }

    pub fn legal_bids(&self, player: usize) -> Vec<i8> {
        let made = self.bids.iter().filter(|&&b| b != -1).count();
        let is_last = made == 3;
        let sum: i32 = self.bids.iter().filter(|&&b| b != -1).map(|&b| b as i32).sum();
        let forbidden = self.round_cards as i32 - sum;
        (0..=self.round_cards as i8)
            .filter(|&b| !(is_last && b as i32 == forbidden))
            .collect()
    }

    pub fn get_leader(&self) -> usize {
        if self.bids.iter().any(|&b| b == -1) {
            return 0;
        }
        let max = *self.bids.iter().max().unwrap();
        self.bids.iter().position(|&b| b == max).unwrap_or(0)
    }

    pub fn play_card(&mut self, player: usize, card: (char, u8)) {
        if self.led_suit.is_none() {
            self.led_suit = Some(card.0);
        } else if Some(card.0) != self.led_suit {
            let idx = match self.led_suit.unwrap() {
                'H' => 0,
                'S' => 1,
                'D' => 2,
                'C' => 3,
                _ => 0,
            };
            self.void_matrix[player][idx] = true;
        }
        self.hands[player].retain(|&c| c != card);
        self.table.push((player, card));
        self.history.push(("play".into(), player, format!("{}{}", card.0, card.1)));
    }

    pub fn resolve_trick(&mut self) -> (usize, Vec<(usize, (char, u8))>) {
        assert!(!self.table.is_empty());
        let led = self.table[0].1 .0;
        let mut winner = self.table[0].0;
        let mut best = self.table[0].1 .1;
        for &(p, (s, r)) in &self.table[1..] {
            if s == led && r > best {
                best = r;
                winner = p;
            }
        }
        self.tricks_won[winner] += 1;
        let res = (winner, self.table.clone());
        for &(_, c) in &self.table {
            self.played.push(c);
        }
        self.table.clear();
        self.led_suit = None;
        self.current_trick += 1;
        res
    }
}

impl Default for PlumpEnv {
    fn default() -> Self {
        Self::new()
    }
}
