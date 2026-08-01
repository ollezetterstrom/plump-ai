//! Deterministic PRNG for dealing (§3.6 "the engine already deals in bulk").
//!
//! Self-contained SplitMix64 — no external dependency, fully reproducible from
//! a `u64` seed. Bias from the `% n` range reduction is negligible for dealing.

/// SplitMix64 state.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next pseudo-random `u64`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform index in `0..n` (modulo reduction; fine for card dealing).
    #[inline]
    pub fn next_below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0, "next_below(0)");
        (self.next_u64() % n as u64) as usize
    }

    /// Fisher–Yates shuffle in place.
    pub fn shuffle<T>(&mut self, a: &mut [T]) {
        for i in (1..a.len()).rev() {
            let j = self.next_below(i + 1);
            a.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_from_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let (x, y) = (a.next_u64(), b.next_u64());
        assert_ne!(x, y);
    }

    #[test]
    fn shuffle_is_permutation() {
        let mut rng = Rng::new(7);
        let mut v: Vec<u8> = (0..52).collect();
        rng.shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..52).collect::<Vec<_>>());
    }

    #[test]
    fn next_below_stays_in_range() {
        let mut rng = Rng::new(3);
        for _ in 0..10_000 {
            assert!(rng.next_below(13) < 13);
        }
    }
}
