//! PyO3 bridge: `import plump._engine`.
//!
//! Exposes the Rust layout registry, scoring, and legal masks so Python can
//! assert against the single source of truth (§5.2) and reuse the engine's
//! legality/scoring logic bit-for-bit (§10).

use plump_engine::cards::FULL_DECK;
use plump_engine::layout::{build_perm_table, LAYOUT, N_FEATURES, N_PERMUTATIONS};
use plump_engine::legal::{legal_bids as engine_legal_bids, legal_plays as engine_legal_plays};
use plump_engine::scoring::ScoringConfig;
use pyo3::prelude::*;

/// Number of scalar features in the layout.
#[pyfunction]
fn n_features() -> usize {
    N_FEATURES
}

/// Number of suit permutations (24).
#[pyfunction]
fn n_permutations() -> usize {
    N_PERMUTATIONS
}

/// `[(name, offset, size)]` for every block of the authoritative layout.
#[pyfunction]
fn layout() -> Vec<(String, usize, usize)> {
    LAYOUT
        .iter()
        .map(|b| (b.name.to_string(), b.offset, b.size))
        .collect()
}

/// The full 52-bit deck mask.
#[pyfunction]
fn full_deck() -> u64 {
    FULL_DECK
}

/// Round score for `(bid, tricks)` under the default variant.
#[pyfunction]
fn score(bid: u8, tricks: u8) -> i32 {
    ScoringConfig::default().score(bid, tricks)
}

/// Unique achievable round scores (value-head atom support).
#[pyfunction]
fn score_atoms() -> Vec<i32> {
    ScoringConfig::default().score_atoms()
}

/// 11-bit legal-bid mask.
#[pyfunction]
fn legal_bids(n_cards: u8, sum_others: u8, is_dealer: bool) -> u16 {
    engine_legal_bids(n_cards, sum_others, is_dealer)
}

/// 52-bit legal-play mask.
#[pyfunction]
#[pyo3(signature = (hand, led=None))]
fn legal_plays(hand: u64, led: Option<u8>) -> u64 {
    engine_legal_plays(hand, led)
}

/// Precomputed `24 * N_FEATURES` suit-permutation table.
#[pyfunction]
fn permutation_table() -> Vec<i16> {
    build_perm_table()
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(n_features, m)?)?;
    m.add_function(wrap_pyfunction!(n_permutations, m)?)?;
    m.add_function(wrap_pyfunction!(layout, m)?)?;
    m.add_function(wrap_pyfunction!(full_deck, m)?)?;
    m.add_function(wrap_pyfunction!(score, m)?)?;
    m.add_function(wrap_pyfunction!(score_atoms, m)?)?;
    m.add_function(wrap_pyfunction!(legal_bids, m)?)?;
    m.add_function(wrap_pyfunction!(legal_plays, m)?)?;
    m.add_function(wrap_pyfunction!(permutation_table, m)?)?;
    Ok(())
}
