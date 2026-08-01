//! PyO3 bridge: `import plump._engine`.
//!
//! Exposes the Rust layout registry, scoring, legal masks, the batch encoder,
//! and the lockstep `Rollout` driver so Python can assert against the single
//! source of truth (§5.2) and run full 16 k-game rollouts with pinned buffers
//! (§7): buffers are allocated once in Python (optionally `pin_memory()`-ed),
//! and Rust writes into them in place — zero copies per step.

use numpy::{PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArray2};
use plump_engine::cards::FULL_DECK;
use plump_engine::encode::K_MAX;
use plump_engine::layout::{build_perm_table, LAYOUT, N_FEATURES, N_PERMUTATIONS};
use plump_engine::legal::{legal_bids as engine_legal_bids, legal_plays as engine_legal_plays};
use plump_engine::rollout::Rollout as EngineRollout;
use plump_engine::scoring::ScoringConfig;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Number of scalar features in the layout.
#[pyfunction]
fn n_features() -> usize {
    N_FEATURES
}

/// Hard cap on active features per row.
#[pyfunction]
fn k_max() -> usize {
    K_MAX
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

/// Lockstep rollout of `batch` games (§7). Buffers are Python-owned and
/// written in place, so the host→device copy can be a pinned `torch.Tensor`
/// view of the same memory.
#[pyclass]
struct Rollout {
    inner: EngineRollout,
}

#[pymethods]
impl Rollout {
    #[new]
    #[pyo3(signature = (n_players, n_cards, batch, seed=0))]
    fn new(n_players: u8, n_cards: u8, batch: usize, seed: u64) -> Self {
        Self {
            inner: EngineRollout::new(n_players, n_cards, batch, seed, ScoringConfig::default()),
        }
    }

    #[getter]
    fn n_players(&self) -> u8 {
        self.inner.n_players
    }

    #[getter]
    fn n_cards(&self) -> u8 {
        self.inner.n_cards
    }

    #[getter]
    fn batch(&self) -> usize {
        self.inner.batch
    }

    #[getter]
    fn step_index(&self) -> usize {
        self.inner.step
    }

    #[getter]
    fn rounds_dealt(&self) -> usize {
        self.inner.rounds_dealt
    }

    #[getter]
    fn round_over(&self) -> bool {
        self.inner.round_over()
    }

    #[getter]
    fn decisions_per_round(&self) -> usize {
        self.inner.decisions_per_round()
    }

    /// Encodes the current snapshot into `idx` (`[B, K_MAX]` i16) and `len`
    /// (`[B]` u16).
    fn encode(
        &self,
        py: Python<'_>,
        mut idx: PyReadwriteArray2<'_, i16>,
        mut len: PyReadwriteArray1<'_, u16>,
    ) -> PyResult<()> {
        let idx_slice = idx
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("idx must be a C-contiguous array"))?;
        let len_slice = len
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("len must be a C-contiguous array"))?;
        py.allow_threads(|| self.inner.encode(idx_slice, len_slice));
        Ok(())
    }

    /// Advances every unfinished game by one decision.
    fn step(&mut self, py: Python<'_>, actions: PyReadonlyArray1<'_, u8>) -> PyResult<()> {
        let a = actions.as_array();
        let slice = a
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("actions must be a C-contiguous array"))?;
        py.allow_threads(|| self.inner.step(slice));
        Ok(())
    }

    /// Fills the per-game decision-snapshot arrays: legal bid masks (u16),
    /// legal play masks (u64), actor seat (u8), and is-bid flag (u8).
    #[allow(clippy::too_many_arguments)]
    fn snapshot(
        &self,
        py: Python<'_>,
        mut legal_bid: PyReadwriteArray1<'_, u16>,
        mut legal_play: PyReadwriteArray1<'_, u64>,
        mut actor: PyReadwriteArray1<'_, u8>,
        mut is_bid: PyReadwriteArray1<'_, u8>,
    ) -> PyResult<()> {
        let b = self.inner.batch;
        let lb = legal_bid
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("legal_bid must be C-contiguous"))?;
        let lp = legal_play
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("legal_play must be C-contiguous"))?;
        let ac = actor
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("actor must be C-contiguous"))?;
        let ib = is_bid
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("is_bid must be C-contiguous"))?;
        if lb.len() != b || lp.len() != b || ac.len() != b || ib.len() != b {
            return Err(PyValueError::new_err("snapshot arrays must have length B"));
        }
        py.allow_threads(|| {
            for g in 0..b {
                lb[g] = self.inner.legal_bid(g);
                lp[g] = self.inner.legal_play(g);
                ac[g] = self.inner.actor(g);
                ib[g] = u8::from(self.inner.is_bid(g));
            }
        });
        Ok(())
    }

    /// Re-deals a new round; returns the round number.
    fn reset(&mut self, seed: u64) -> usize {
        self.inner.reset(seed)
    }

    /// Round scores for game `g` as a list.
    fn round_scores(&self, g: usize) -> Vec<i32> {
        self.inner.round_scores(g).to_vec()
    }
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(n_features, m)?)?;
    m.add_function(wrap_pyfunction!(k_max, m)?)?;
    m.add_function(wrap_pyfunction!(n_permutations, m)?)?;
    m.add_function(wrap_pyfunction!(layout, m)?)?;
    m.add_function(wrap_pyfunction!(full_deck, m)?)?;
    m.add_function(wrap_pyfunction!(score, m)?)?;
    m.add_function(wrap_pyfunction!(score_atoms, m)?)?;
    m.add_function(wrap_pyfunction!(legal_bids, m)?)?;
    m.add_function(wrap_pyfunction!(legal_plays, m)?)?;
    m.add_function(wrap_pyfunction!(permutation_table, m)?)?;
    m.add_class::<Rollout>()?;
    Ok(())
}
