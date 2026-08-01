//! Round scoring for Plump.
//!
//! The default variant scores `make_bonus + bid` points for making the bid
//! exactly, and `-miss_penalty * |tricks - bid|` otherwise. The exact rule is
//! configuration-driven; the default matches §4.2 of the plan.

/// Scoring parameters for a Plump variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScoringConfig {
    /// Fixed bonus for making the bid exactly.
    pub make_bonus: i32,
    /// Penalty per trick of difference when missing the bid.
    pub miss_penalty: i32,
    /// Lowest legal bid (inclusive).
    pub min_bid: u8,
    /// Highest legal bid (inclusive).
    pub max_bid: u8,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            make_bonus: 10,
            miss_penalty: 1,
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
            self.make_bonus + i32::from(bid)
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
        assert_eq!(c.score(3, 2), -1); // under by 1
        assert_eq!(c.score(3, 5), -2); // over by 2
        assert_eq!(c.score(0, 0), 10);
        assert_eq!(c.score(10, 10), 20);
    }

    #[test]
    fn symmetric_around_bid() {
        let c = ScoringConfig::default();
        // Both directions must stay on the "miss" side (never collapse onto bid).
        for bid in 1..=9u8 {
            for d in 1..=bid.min(10 - bid) {
                assert_eq!(c.score(bid, bid + d), -c.miss_penalty * i32::from(d));
                assert_eq!(c.score(bid, bid - d), -c.miss_penalty * i32::from(d));
            }
        }
    }

    #[test]
    fn atoms_cover_table() {
        let c = ScoringConfig::default();
        let atoms = c.score_atoms();
        assert_eq!(atoms[0], -10);
        assert_eq!(*atoms.last().unwrap(), 20);
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
            miss_penalty: 2,
            ..Default::default()
        };
        assert_eq!(c.score(4, 4), 9);
        assert_eq!(c.score(4, 2), -4);
    }
}
