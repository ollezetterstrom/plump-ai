//! Round scoring for Plump.
//!
//! Defaults follow the most common Swedish convention: a made bid scores
//! `make_bonus + bid` (`10 + bid`, the protocol's "1"-prefix form), a made
//! **0-bid is a special case** scoring `zero_bid_bonus` (the common "05" rule,
//! since a 0-bid is the safest), and a miss scores `-miss_penalty * |tricks - bid|`
//! with `miss_penalty == 0` by default (a "plump": 0 points, no negatives).
//! All parameters are configurable.

/// Scoring parameters for a Plump variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScoringConfig {
    /// Fixed bonus for making the bid exactly.
    pub make_bonus: i32,
    /// Penalty per trick of difference when missing the bid (0 = flat 0).
    pub miss_penalty: i32,
    /// Special-case score for a made bid of 0; `None` falls back to
    /// `make_bonus + 0`.
    pub zero_bid_bonus: Option<i32>,
    /// Lowest legal bid (inclusive).
    pub min_bid: u8,
    /// Highest legal bid (inclusive).
    pub max_bid: u8,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            make_bonus: 10,
            miss_penalty: 0,
            zero_bid_bonus: Some(5),
            min_bid: 0,
            max_bid: 10,
        }
    }
}

impl ScoringConfig {
    /// Round score for a bid and a final trick count.
    #[inline]
    pub fn score(&self, bid: u8, tricks: u8) -> i32 {
        debug_assert!(bid <= self.max_bid, "bid out of range");
        debug_assert!(tricks <= self.max_bid, "tricks out of range");
        if bid == tricks {
            if bid == 0 {
                self.zero_bid_bonus.unwrap_or(self.make_bonus)
            } else {
                self.make_bonus + i32::from(bid)
            }
        } else {
            -self.miss_penalty * (i32::from(bid) - i32::from(tricks)).abs()
        }
    }

    /// `score_table[bid][tricks]` over `min_bid..=max_bid`, used by the
    /// factored value head (§4.2).
    pub fn score_table(&self) -> Vec<Vec<i32>> {
        let n = self.max_bid + 1;
        let mut t = vec![vec![0i32; n as usize]; n as usize];
        for bid in 0..=self.max_bid {
            for tricks in 0..=self.max_bid {
                t[bid as usize][tricks as usize] = self.score(bid, tricks);
            }
        }
        t
    }

    /// Unique achievable round scores across all `(bid, tricks)`, sorted.
    /// This is the atom support for the categorical value head (§4.2).
    pub fn score_atoms(&self) -> Vec<i32> {
        let mut atoms: Vec<i32> = Vec::new();
        for bid in 0..=self.max_bid {
            for tricks in 0..=self.max_bid {
                let s = self.score(bid, tricks);
                if !atoms.contains(&s) {
                    atoms.push(s);
                }
            }
        }
        atoms.sort_unstable();
        atoms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_scoring_examples() {
        let c = ScoringConfig::default();
        assert_eq!(c.score(3, 3), 13); // made it: 10 + bid
        assert_eq!(c.score(10, 10), 20);
        assert_eq!(c.score(0, 0), 5); // special case: the "05" rule
        assert_eq!(c.score(3, 2), 0); // miss: no negatives by default
        assert_eq!(c.score(3, 5), 0); // over by 2: still 0
        assert_eq!(c.score(0, 2), 0); // a 0-bid miss is 0 too
    }

    #[test]
    fn made_zero_bid_special_case() {
        // None keeps the old behavior: a made 0-bid is make_bonus + 0.
        let c = ScoringConfig {
            zero_bid_bonus: None,
            ..Default::default()
        };
        assert_eq!(c.score(0, 0), 10);
    }

    #[test]
    fn miss_is_always_zero_with_default_penalty() {
        let c = ScoringConfig::default();
        for bid in 0..=10u8 {
            for tricks in 0..=10u8 {
                if bid != tricks {
                    assert_eq!(c.score(bid, tricks), 0);
                }
            }
        }
    }

    #[test]
    fn per_trick_miss_penalty_restores_negatives() {
        let c = ScoringConfig {
            miss_penalty: 1,
            ..Default::default()
        };
        assert_eq!(c.score(3, 2), -1);
        assert_eq!(c.score(3, 5), -2);
        assert_eq!(c.score(0, 0), 5); // zero-bid special case unaffected
        for bid in 1..=9u8 {
            for d in 1..=bid.min(10 - bid) {
                assert_eq!(c.score(bid, bid + d), -i32::from(d));
                assert_eq!(c.score(bid, bid - d), -i32::from(d));
            }
        }
    }

    #[test]
    fn atoms_cover_table() {
        let c = ScoringConfig::default();
        let atoms = c.score_atoms();
        // Achievable scores: 0 (any miss), 5 (made 0-bid), 11..=20 (made bids 1..=10).
        assert_eq!(atoms, vec![0, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
        for row in c.score_table() {
            for s in row {
                assert!(atoms.contains(&s));
            }
        }
    }

    #[test]
    fn custom_make_bonus() {
        let c = ScoringConfig {
            make_bonus: 5,
            miss_penalty: 1,
            zero_bid_bonus: None,
            ..Default::default()
        };
        assert_eq!(c.score(4, 4), 9);
        assert_eq!(c.score(4, 2), -2);
        assert_eq!(c.score(0, 0), 5); // None -> make_bonus + 0
    }
}
