# Master Plan v2: A Strong Plump AI on a Consumer GPU
### (Revised to resolve all identified technical, algorithmic, game-theoretic, and systems defects)

---

## 0. Issue → Fix Traceability

| # | Issue | Resolution | Section |
|---|-------|-----------|---------|
| 1.1 | Per-player trajectory contamination | Per-seat trajectory attribution; terminal-only Monte-Carlo returns per seat; **no cross-seat GAE** | §4.1 |
| 1.2 | Terminal reward discontinuity / MSE instability | **Distributional (categorical) value head** over the finite score support + trick-count factorization + CE loss | §4.2 |
| 1.3 | Opponent-pool batch divergence | **Static-shape dense evaluation**: one frozen opponent per iteration (K=2 forwards), optional stacked-weights `vmap` for K>2. Never dynamic re-batching | §4.4 |
| 1.4 | Self-play conventions / blind spots | Multi-lineage population, per-game suit relabeling, opponent-noise injection, held-out lineage eval, exploitability probe | §4.5 |
| 2.1 | Strategy fusion in determinized bidding | Bidding never uses double-dummy. Uses **policy-rollout trick distributions** + a learned **DD→actual calibration** | §6.2 |
| 2.2 | Invalid worlds from marginal logits | **Constrained sampler**: capacity-aware sequential assignment + swap-MCMC against belief weights. Validity is structural, not probabilistic | §6.1 |
| 2.3 | Non-locality / no information hiding | Search used only as a **policy-improvement operator distilled back** into the imperfect-info net; hard blend cap; IS-MCTS roadmap | §6.4 |
| 2.4 | Public-history violations during sampling | Engine maintains hard `possible[seat]` bitmasks (voids, played, dead-stock cardinality); sampler cannot violate them | §6.1, §3.3 |
| 3.1 | Shared 63-action head overlap | **Two separate heads** (bid: 11, play: 52). Phase selects the head. Cross-phase actions are structurally impossible | §5.1 |
| 3.2 | Dynamic dealer forbidden-bid mask | Per-row bitmask computed in Rust (`u16` bid mask, `u64` play mask); row-wise masking is natively batchable; wrap/out-of-range handled | §3.4 |
| 3.3 | Permutation distortion of suit-agnostic features | Structural augmentation via a **precomputed 24 × N_FEATURES permutation table** derived from the layout registry; suit-agnostic blocks map to identity | §5.4 |
| 4.1 | EmbeddingBag index collisions | **Single source-of-truth layout registry** in Rust with compile-time non-overlap assertions, exported to Python, debug range checks | §5.2 |
| 4.2 | Dead-stock / undealt ambiguity | Explicit per-card public-knowledge class + unseen-count features; belief head has an `UNDEALT` class with impossibility masking | §5.3 |
| 4.3 | Relative seat inconsistency | All encoding is actor-relative (`rel = (abs − actor) mod P`), performed inside the engine at every decision; property-tested | §3.5 |
| 5.1 | Dynamic leader shifts in lockstep | Lockstep holds on *decision count*, not seat identity; an `actor[]` vector carries seat identity; nothing else needs to be uniform | §3.2 |
| 5.2 | CPU encoding bottleneck | Fixed-width pinned pre-allocated buffers, rayon, zero per-step allocation, **double-buffered pipelining**, plus a validated **GPU encoder Plan B** | §3.6, §7.2 |
| 5.3 | Trajectory memory bloat | Pre-allocated fixed-shape `int16`/`int8`/`uint64` GPU tensors; index-permutation shuffling; no Python lists of tensors | §7.3 |
| 6 | Round-level vs. match-level suboptimality | Match-context features from day one; Stage-B fine-tune against a **match-level utility** with ΔV_match rewards over a ladder simulator | §8 |

---

## 1. Executive Summary (revised)

We build a Plump agent that trains in hours on one consumer GPU. Relative to v1, the architecture changes in six substantive ways:

1. **Credit assignment is per-seat.** The environment is treated as *P* interleaved single-agent trajectories per game, not one trajectory.
2. **The value function is distributional and factorized** through a trick-count distribution, because Plump's score function is a step function of `|tricks − bid|`.
3. **The policy has two disjoint heads** (bid / play), eliminating an entire class of masking bugs.
4. **All batch shapes are static.** Opponent mixing, masking, and encoding never produce variable-sized kernels.
5. **Search is a policy-improvement operator, not an inference-time oracle.** PIMC produces targets that are distilled into the imperfect-information network; bidding never consumes double-dummy values directly.
6. **World sampling is constraint-first.** Belief logits only *weight* samples drawn from a provably legal support.

---

## 2. High-Level Design Decisions (revised)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Rust engine + PyTorch | unchanged |
| Episode unit | One **round** for Stage A; one **match (ladder)** for Stage B | Resolves horizon mismatch (§8) |
| Return estimator | Terminal per-seat Monte-Carlo return, `γ=1` within round | No bootstrapping across foreign states |
| Value loss | Categorical cross-entropy over score atoms | Handles reward discontinuity |
| Policy heads | Disjoint bid head (11) + play head (52) | Phase safety |
| Opponent mixing | One frozen opponent per iteration (static K=2) | GPU-friendly |
| Search | PIMC + policy rollouts, distilled (ExIt) | Mitigates strategy fusion |
| World sampling | Constrained + MCMC-refined | Legality by construction |
| Augmentation | Layout-table-driven S₄ permutation | Cannot corrupt agnostic features |

---

## 3. Rust Engine (revised)

### 3.1 Card & set representation

```rust
pub type Card = u8;              // 0..52,  card = suit*13 + rank
pub type CardSet = u64;          // bit `card` set

#[inline] pub const fn suit(c: Card) -> u8 { c / 13 }
#[inline] pub const fn rank(c: Card) -> u8 { c % 13 }

pub const SUIT_MASK: [CardSet; 4] = [
    0x0000_0000_0000_1FFF,
    0x0000_0000_03FF_E000,
    0x0000_007F_FC00_0000,
    0x000F_FF80_0000_0000,
];
pub const FULL_DECK: CardSet = (1u64 << 52) - 1;
```

> **Fix (v1 bug):** v1 defined `suit(c) = c >> 4`, `rank(c) = c & 0xF` while simultaneously claiming `i = suit*13 + rank`. These are mutually inconsistent and would have silently corrupted every suit-based feature and the entire symmetry layer. We standardize on `suit*13 + rank` everywhere, including action indices, belief targets, and the permutation table. A `debug_assert` in `Card::new` enforces `rank < 13`.

### 3.2 Lockstep invariant — precisely stated

**What is uniform across a batch:** `(n_players P, n_cards C)`, hence the total decision count per round is exactly

```
D = P            (bids)
  + P * C        (plays)
```

**What is *not* uniform:** which seat acts at step *t*, which seat leads, and the trick winners. This is fine and requires no vectorization gymnastics:

- The engine exposes `actor: Vec<u8>` (absolute seat acting now, per game).
- The encoder always emits features **from that actor's perspective** (§3.5).
- The learner/opponent routing and the per-seat trajectory attribution use `actor[]` as the key.

There is therefore **no requirement that games agree on seat identity at step *t***, only on *step count*. Issue 5.1 dissolves once the invariant is stated at the right granularity. A debug assertion verifies `steps_taken == D` at round end for every game.

### 3.3 Public-knowledge state (new)

The engine maintains, per game, hard constraint structures that *any* world sampler must respect:

```rust
pub struct PublicKnowledge {
    pub possible: [CardSet; 6],   // cards seat p could still hold (voids + played + own hand removed)
    pub remaining_hand_size: [u8; 6],
    pub undealt_count: u8,        // 52 - P*C  (dead stock size), constant per round
    pub undealt_possible: CardSet,// cards that could be in dead stock
    pub played_by: [CardSet; 6],
    pub all_played: CardSet,
}
```

Updated incrementally on every play:
- when a seat fails to follow suit `s`, clear `SUIT_MASK[s]` from `possible[seat]`;
- when a card is played, clear it from every `possible[*]` and from `undealt_possible`;
- the actor's own hand is removed from all other seats' `possible`.

`PublicKnowledge` is the *only* input to the world sampler's support. Belief logits can reweight but never expand it. In debug builds, `assert_world_valid()` re-checks every sampled world against this struct.

### 3.4 Legal actions (revised)

```rust
pub fn legal_plays(hand: CardSet, led: Option<u8>) -> CardSet {
    match led {
        None => hand,
        Some(s) => { let f = hand & SUIT_MASK[s as usize]; if f != 0 { f } else { hand } }
    }
}

/// Returns an 11-bit mask over bids 0..=10 (bit b set => bid b legal).
pub fn legal_bids(n_cards: u8, sum_others: u8, is_dealer: bool) -> u16 {
    let mut m: u16 = (1u16 << (n_cards + 1)) - 1;   // bids 0..=n_cards
    if is_dealer {
        // forbidden bid may be negative-or-out-of-range; then no bid is forbidden.
        if sum_others <= n_cards {
            let forbidden = n_cards - sum_others;
            m &= !(1u16 << forbidden);
        }
    }
    debug_assert!(m != 0);
    m
}
```

Two things v1 got wrong and are now fixed:

- `n_cards - sum_others` on `u8` **underflows and panics/wraps** when `sum_others > n_cards`. Handled explicitly.
- The forbidden-bid rule applies **only to the dealer** (last bidder). v1's `legal_bids` had no such guard.

Regarding issue 3.2: a per-row dynamic mask is not a batching problem. The engine returns `bid_mask: Vec<u16>` and `play_mask: Vec<u64>`; Python unpacks bits into a `[B, 11]` / `[B, 52]` bool tensor with a fixed-shape bit-unpack kernel. Shapes are static; only contents vary.

### 3.5 Actor-relative encoding (fix 4.3)

Every observation is built after rotating absolute seats into actor-relative offsets:

```rust
#[inline] fn rel(abs: u8, actor: u8, p: u8) -> u8 { (abs + p - actor) % p }
```

Consequences enforced by the layout:
- Seat-indexed blocks (`played_by`, `bids`, `tricks_won`, `voids`) are always written at `rel` offsets, so slot 0 is *always* "me".
- Role information that would otherwise be lost is encoded **as explicit flags**, not inferred from seat index: `is_dealer[rel]`, `is_leader[rel]`, `position_in_trick` one-hot, `bid_order_position` one-hot, `n_players` one-hot.
- Slots `rel >= P` are simply never emitted (sparse encoding, so unused slots contribute nothing — no padding embeddings).

**Property test:** for a random state and a random cyclic relabeling of absolute seats, the encoded index multiset for a fixed logical actor is identical.

### 3.6 Encoder performance contract (fix 5.2)

```rust
pub const K_MAX: usize = 128;   // hard cap on active features; asserted in debug

pub fn encode_batch(
    batch: &Batch,
    out_idx: &mut [i16],   // pre-allocated, len = n * K_MAX, pinned
    out_len: &mut [u16],   // pre-allocated, len = n
) {
    out_idx.par_chunks_mut(K_MAX)
        .zip(out_len.par_iter_mut())
        .enumerate()
        .for_each(|(g, (slot, len))| { *len = encode_one(batch, g, slot) as u16; });
}
```

Rules:
- **Zero allocation per step.** All buffers live for the lifetime of the `Batch` and are pinned (`cudaHostAlloc` via `cudapy`/`torch.from_numpy(...).pin_memory()` once).
- **Fixed width `K_MAX`**, so the host→device copy is a single contiguous `16384 × 128 × 2 B = 4 MB` transfer. At 55 steps/round that is 230 MB/iteration — irrelevant against PCIe 4.0.
- **Bit-iteration only.** Each active feature comes from `while s != 0 { let c = s.trailing_zeros(); s &= s-1; push(base + c); }`. No branching over all 52 cards (v1's `for c in 0..52 { if bit }` loop was 52× more work than necessary).
- **Double buffering:** the engine owns two `(idx, len)` buffer pairs. A worker thread encodes buffer *B* for step *t+1* while the GPU consumes buffer *A* for step *t*. `step()` and `encode()` are decoupled via a channel.

**Plan B (kept in reserve, validated by differential test):** if profiling shows the CPU encoder is still the critical path, ship the raw SoA state instead — `6 × u64` hands + `6 × u64` played + ~16 bytes of scalars ≈ **112 bytes/game** (1.8 MB/step) — and expand into features with a Triton kernel that consumes the *same* layout registry. A `test_encoder_parity` test asserts the Rust and Triton encoders produce identical index sets on 10⁶ random states.

---

## 4. Learning Algorithm (heavily revised)

### 4.1 Per-seat trajectory attribution (fix 1.1)

A round of one game produces `D = P + P·C` decisions, of which each seat owns exactly `1 + C`. We store, for every transition:

```
game_id : int32
seat    : int8      # absolute seat of the actor
step    : int16
```

**Returns.** Rewards in Plump are terminal-only at the round level. Therefore, for seat *p*:

```
R_p = score_p(bid_p, tricks_p)            # scalar, known at round end
G(s_t) = R_{seat(t)}      for every decision t owned by seat(t)
```

No discounting, no bootstrapping, no GAE across steps. This is exact (it is the Monte-Carlo return of that seat's own trajectory) and it **structurally cannot mix seats**, because the return is a per-seat scalar broadcast to that seat's own transitions only.

```python
# returns: [B*D]  built by a single gather, no loops
ret = round_scores[game_id, seat]        # round_scores: [B, P]
adv = ret - value_pred                   # value_pred is V for *that* seat's own state
adv = (adv - adv.mean()) / (adv.std() + 1e-8)
```

**Why not GAE?** GAE would require a value function evaluated at *the same agent's* next decision point, i.e. skipping over `P−1` foreign decisions. That is legitimate (it is the "agent-centric MDP" view) but with `C ≤ 10` the trajectory is at most 11 decisions long and terminal-only-reward, so the bias/variance tradeoff is dominated by plain MC + a good baseline. We therefore use MC returns and invest the variance budget in the baseline instead (§4.2, §4.3).

**Additional variance reduction (optional, cheap):** *duplicate-deal baselining.* Deal each hand `m = 4` times into adjacent batch slots with independent action sampling. The leave-one-out mean of `R_p` across the `m` replicas of the same deal is an unbiased, deal-conditioned baseline that removes essentially all card-luck variance:

```
adv = R_p^{(i)} - (1/(m-1)) * Σ_{j≠i} R_p^{(j)}
```

This is free (the engine already deals in bulk) and is the single highest-leverage variance reduction available in a card game.

### 4.2 Distributional, factorized value head (fix 1.2)

Plump's score is a step function, e.g. `score(b, t) = 10 + b if t == b else -|t - b|` (exact rule is config-driven). Under MSE the value target jumps by ~11 points when a trick flips hands, which is precisely the regime where MSE is worst-behaved.

Two changes:

**(a) Categorical value head.** The set of achievable round scores is small and finite (at most ~25 atoms for `C ≤ 10`). We enumerate them once into `SCORE_ATOMS: [f32; A]` and train:

```python
self.value_head = nn.Linear(H, A)          # logits over score atoms
loss_v = F.cross_entropy(value_logits, atom_index_of(R_p))
V = (F.softmax(value_logits, -1) * SCORE_ATOMS).sum(-1)   # used as the PPO baseline
```

CE on a discrete support is bounded-gradient and completely insensitive to the step discontinuity — the discontinuity becomes a *class boundary*, which is exactly what classification losses are for.

**(b) Trick-count factorization.** The `trickdist_head` (11-way distribution over the actor's final trick count) is not merely auxiliary — post-bid it *analytically determines* the value:

```python
# after the actor has bid:
V_factored = (p_tricks * score_table[bid]).sum(-1)     # score_table: [11, 11]
```

We train `V` with CE **and** add a consistency loss `‖V − V_factored‖²` (post-bid states only). This gives the value head a strong, well-shaped learning signal and makes bid calibration directly inspectable. During the bid phase the factored form is unavailable (bid not chosen yet), so the categorical head stands alone there.

**Reward scaling:** advantages are normalized per-minibatch; raw scores are never fed to a regression loss.

### 4.3 Policy loss

```python
# heads are disjoint; select per-row by phase
logits = torch.where(is_bid[:, None], bid_logits_padded, play_logits)   # [B, 52] with bid in 0..10
logits = logits.masked_fill(~legal, -1e30)
logp   = log_softmax(logits, -1).gather(-1, action[:, None]).squeeze(-1)
ratio  = (logp - old_logp).exp()
policy_loss = -torch.min(ratio*adv, ratio.clamp(1-eps, 1+eps)*adv).mean()

loss = policy_loss
     + c_v * value_ce
     + c_c * value_consistency
     + c_t * trick_ce
     + c_b * belief_ce
     - c_e * entropy            # entropy computed over legal actions only
```

- **Entropy floor:** `c_e` is annealed but never below `5e-3`, and we monitor per-phase entropy separately (bid entropy collapses much faster than play entropy).
- **Approx-KL early stopping** per epoch (`target_kl = 0.02`) — with terminal-only rewards and normalized advantages, PPO can take very large steps early.
- **Auxiliary losses are gradient-scaled, not head-detached**, since the shared trunk genuinely benefits from the belief signal.

### 4.4 Opponent pool with static shapes (fix 1.3)

**Default (K = 2, fully static, CUDA-graph capturable):**

- At the start of each iteration, sample **one** frozen checkpoint `θ_old` from the pool (plus a small probability of a scripted heuristic bot).
- Assign per game a learner-seat set `L_g ⊆ {0..P−1}` at deal time (e.g. with probability 0.5 all seats are the learner → pure self-play; otherwise exactly one seat is `θ_old`).
- At each step, run **two full-batch forwards**: `θ_cur` and `θ_old`, both on the entire `[B, K_MAX]` observation tensor. Select with `torch.where(is_learner_actor[:, None], cur_logits, old_logits)`.
- Only transitions where `is_learner_actor` is true are written into the PPO buffer (a boolean mask on a fixed-size buffer, compacted once at iteration end).

Cost: 2× inference. Inference is ~30–35 % of iteration wall time, so the overhead is ~15 %. Shapes are constant across every step and every iteration → the whole rollout step is CUDA-graph capturable.

**Scaling to K > 2 (optional):** stack the pool's weights and evaluate them as one batched GEMM:

```python
params, buffers = torch.func.stack_module_state(pool)         # [K, ...]
logits_all = torch.vmap(fmodel)(params, buffers, obs)         # [K, B, A]
logits = logits_all.gather(0, policy_id[None, :, None].expand(1, B, A)).squeeze(0)
```

Still one kernel, still static shapes. **We never dynamically group games by policy id** — that was the source of the fragmentation, and it buys nothing when the network is this small.

### 4.5 Countering self-play conventions (fix 1.4)

Five concrete mechanisms, all cheap:

1. **Per-game suit relabeling.** Every game in every batch is dealt under an independently sampled `σ ∈ S₄` applied *to the deal itself*. Since Plump has no trump, any suit-specific convention ("lead clubs to signal") is destroyed in expectation — the agent cannot condition on suit identity because suit identity carries no cross-game signal. This is stronger than post-hoc augmentation and costs one permutation per deal.
2. **Multi-lineage population.** Train `n = 3` independently seeded lineages concurrently (they fit trivially on one GPU) and cross-play them every 200 iterations. Any convention that is a private handshake shows up immediately as a cross-lineage score drop.
3. **Opponent noise injection.** Frozen opponents play with `ε = 0.05` uniform-random action mixing and temperature `T ∈ [0.8, 1.3]`. Conventions that require an exactly-matching partner do not survive noisy partners.
4. **Held-out lineages.** Lineage C never enters A's or B's pool; it is used only for evaluation. Its score is the honest generalization metric.
5. **Exploitability probe** (§9.2): freeze the agent in `P−1` seats and train a fresh best-responder in the remaining seat for a fixed budget. Report the best-response gap. This is the only metric that actually detects brittleness; the others are proxies.

---

## 5. Network & Feature Representation (revised)

### 5.1 Disjoint policy heads (fix 3.1)

```python
self.bid_head  = nn.Linear(H, 11)
self.play_head = nn.Linear(H, 52)
```

- During bidding, only `bid_head` is evaluated and masked with the 11-bit legal-bid mask.
- During play, only `play_head` is evaluated and masked with the 52-bit legal-play mask.
- Action ids are stored together with a phase flag; a play action's id is the card index directly, a bid action's id is the bid value.
- **It is structurally impossible to emit a bid during play**, because the play-phase logits tensor has no bid entries. The v1 flat-63 head required a runtime invariant to guarantee this; we replace an invariant with a type.

The `-1e9` sentinel from v1 is replaced by `-1e30` in fp32 / `-inf` handling in bf16 — `-1e9` is *not* small enough after the softmax when a row has few legal actions and large logits, and in bf16 `-1e9` rounds catastrophically. We also `assert legal.any(dim=-1).all()`.

### 5.2 Feature layout registry (fix 4.1)

A single Rust const table is the authority; Python imports it and asserts against it.

```rust
pub struct Block { pub name: &'static str, pub offset: usize, pub size: usize, pub suit_rule: SuitRule }

pub enum SuitRule { Agnostic, CardIndexed { stride: usize }, SuitIndexed { stride: usize } }

pub const LAYOUT: &[Block] = &[
  Block{ name:"my_hand",        offset:   0, size: 52,  suit_rule: SuitRule::CardIndexed{stride:1} },
  Block{ name:"played_by_rel",  offset:  52, size: 312, suit_rule: SuitRule::CardIndexed{stride:1} }, // 6*52
  Block{ name:"trick_cards",    offset: 364, size: 312, suit_rule: SuitRule::CardIndexed{stride:1} }, // pos*52
  Block{ name:"led_suit",       offset: 676, size:   5, suit_rule: SuitRule::SuitIndexed{stride:1} }, // 4 suits + none
  Block{ name:"my_voids",       offset: 681, size:   4, suit_rule: SuitRule::SuitIndexed{stride:1} },
  Block{ name:"other_voids",    offset: 685, size:  24, suit_rule: SuitRule::SuitIndexed{stride:1} }, // 6*4
  Block{ name:"unseen_by_suit", offset: 709, size:  56, suit_rule: SuitRule::SuitIndexed{stride:14} },// per suit count 0..13
  Block{ name:"bids_rel",       offset: 765, size:  78, suit_rule: SuitRule::Agnostic },              // 6*(11+1 unknown)
  Block{ name:"tricks_rel",     offset: 843, size:  66, suit_rule: SuitRule::Agnostic },
  Block{ name:"need_rel",       offset: 909, size: 126, suit_rule: SuitRule::Agnostic },              // 6*(bid-tricks, -10..10)
  Block{ name:"role_flags",     offset:1035, size:  30, suit_rule: SuitRule::Agnostic },
  Block{ name:"round_ctx",      offset:1065, size:  40, suit_rule: SuitRule::Agnostic },
  Block{ name:"match_ctx",      offset:1105, size: 120, suit_rule: SuitRule::Agnostic },              // §8
];
pub const N_FEATURES: usize = 1225;

const _: () = assert_layout_disjoint_and_complete(LAYOUT, N_FEATURES); // compile-time
```

Guarantees:
- Compile-time check that blocks are contiguous, non-overlapping, and sum to `N_FEATURES`.
- `debug_assert!(idx >= block.offset && idx < block.offset + block.size)` on every push in debug builds.
- Python does `from plump_rs import LAYOUT; assert net.fc0.num_embeddings == LAYOUT.n_features`.
- A test enumerates 10⁵ random states and asserts every emitted index is claimed by exactly one block.

### 5.3 Dead stock and unseen cards (fix 4.2)

Because `52 − P·C` cards are never dealt, an "I have not seen this card" signal is ambiguous between *opponent holds it* and *it is dead*. We disambiguate explicitly:

- `unseen_by_suit`: for each suit, a one-hot of the count of cards of that suit not in my hand and not yet played (0..13). This is the quantity that actually matters for play decisions.
- `round_ctx` includes one-hots for `P`, `C`, `undealt_count = 52 − P·C`, and `tricks_remaining`.
- **Belief head classes:** `n_class = 8` = `{rel_seat 0..5, PLAYED, UNDEALT}`. Impossible classes are masked to `−inf` before the softmax using `PublicKnowledge` (e.g. a card in the actor's own hand has all classes masked except `rel_seat 0`; a played card is deterministically `PLAYED`). Masking removes trivially-learnable classes and forces the head to spend capacity on the genuinely uncertain cards.
- **Belief loss** is computed **only over genuinely uncertain cards** (those with ≥2 legal classes), otherwise the loss is dominated by trivial predictions and the reported number is meaningless.
- Belief targets come from the engine's ground-truth deal, expressed in actor-relative seat indices, and recomputed at every step (so a played card's target flips to `PLAYED`).

### 5.4 Suit permutation, done structurally (fix 3.3)

We never inspect a raw index and guess whether it is a card. We build, once at startup, a permutation table from the layout registry:

```rust
pub fn build_perm_table() -> Vec<i16> {   // 24 * N_FEATURES
    let mut t = vec![0i16; 24 * N_FEATURES];
    for (pi, perm) in all_permutations_of_4().enumerate() {
        for b in LAYOUT {
            for k in 0..b.size {
                let mapped = match b.suit_rule {
                    SuitRule::Agnostic => k,
                    SuitRule::CardIndexed{stride} => {
                        let (outer, c) = (k / (52*stride), k % 52);
                        outer*52*stride + (perm[c/13] as usize)*13 + c%13
                    }
                    SuitRule::SuitIndexed{stride} => {
                        let (outer, s) = (k / (4*stride), (k / stride) % 4);
                        let inner = k % stride;
                        outer*4*stride + (perm[s] as usize)*stride + inner
                    }
                };
                t[pi*N_FEATURES + b.offset + k] = (b.offset + mapped) as i16;
            }
        }
    }
    t
}
```

Applying an augmentation is then `idx[i] = table[pi*N_FEATURES + idx[i]]` — a single lookup, provably identity on agnostic blocks, and impossible to get wrong per-block because the rule lives next to the block definition.

Action relabeling: `play` action `c → perm[c/13]*13 + c%13`; `bid` action unchanged; belief target class `rel_seat` unchanged, but the *card axis* of the belief target is permuted.

**Note:** with per-deal suit relabeling in the engine (§4.5.1) the S₄ augmentation is largely redundant for training-data diversity, but the table remains essential for (a) test-time symmetrization (average the policy over all 24 permutations for a free strength boost in evaluation/search) and (b) the parity property test:

```
encode(permute_state(s, σ)) == permute_indices(encode(s), σ)     ∀ s, σ
```

This one test catches essentially every symmetry bug.

### 5.5 Network

```python
class PlumpNet(nn.Module):
    def __init__(self, n_feat=1225, H=768, n_atoms=25):
        self.embed = nn.EmbeddingBag(n_feat + 1, H, mode='sum', include_last_offset=True,
                                     padding_idx=n_feat)   # index n_feat is a no-op pad
        self.b0 = nn.Parameter(torch.zeros(H))
        self.body = nn.Sequential(
            nn.LayerNorm(H), nn.ReLU(), nn.Linear(H, H),
            nn.LayerNorm(H), nn.ReLU(), nn.Linear(H, H))
        self.bid_head   = nn.Linear(H, 11)
        self.play_head  = nn.Linear(H, 52)
        self.value_head = nn.Linear(H, n_atoms)
        self.trick_head = nn.Linear(H, 11)
        self.belief_head= nn.Linear(H, 52*8)
```

- Fixed-width `[B, K_MAX]` index tensor with `padding_idx` for unused slots → `EmbeddingBag` with `include_last_offset` and a *constant* offsets tensor. **Static shapes, no `lengths` variability**, which is what makes CUDA graphs possible. (v1's variable-length `lengths` would have forced dynamic shapes every step.)
- LayerNorm before each Linear stabilizes training with the wide dynamic range of summed sparse embeddings; v1's raw residual-without-norm on a sum-of-embeddings input is a known blow-up mode.
- ~1.2 M params in the embedding + 1.2 M in the body ≈ 2.5 M total.

---

## 6. Search: PIMC done correctly

### 6.1 Legal world sampling (fix 2.2, 2.4)

**Never** sample the 52 cards independently from the belief marginals. Instead:

**Step 1 — support.** Take `PublicKnowledge.possible[p]` and `remaining_hand_size[p]` plus the dead-stock slot with capacity `undealt_count − (already-known dead cards)`. Unknown cards `U = FULL_DECK \ (my_hand ∪ all_played)`. This is an exact bipartite assignment problem: `|U|` cards into `P+1` bins with fixed capacities and per-card allowed-bin sets.

**Step 2 — feasible initialization.** Sequential most-constrained-first assignment:

```
sort U by |allowed_bins(card)| ascending
for card in U:
    w[bin] = belief_prob(card, bin) * (capacity_left[bin] > 0) * allowed(card, bin)
    if all zero -> restart with Hall-violation repair (rare; bounded retries)
    bin ~ Categorical(w); capacity_left[bin] -= 1
```

Hall's-condition feasibility is guaranteed by the constraint structure (voids only ever remove *suits*, and capacities sum exactly to `|U|`), but we keep a bounded-retry + deterministic-repair fallback and a hard `assert_world_valid`.

**Step 3 — belief refinement via swap MCMC.** The sequential sampler is biased (order-dependent) relative to the belief distribution. Fix it with `n_swap ≈ 200` Metropolis swap proposals:

```
pick cards c1 ∈ bin_a, c2 ∈ bin_b (a ≠ b), both swaps legal
α = min(1, [q(c1,b) q(c2,a)] / [q(c1,a) q(c2,b)])
accept with prob α
```

Swaps preserve capacities and legality exactly, so **every intermediate state is a valid world**. This is the standard fix for constrained sampling from marginals and it costs microseconds.

**Step 4 — bid consistency filter (optional).** Reject/reweight worlds whose implied bids are wildly inconsistent with the observed bids under the current policy (`w ∝ Π_p π_bid(b_p | hand_p)`). This is importance weighting toward worlds that *explain the auction* and is the single biggest quality improvement over uniform sampling.

**Bootstrapping order:** v1 was right that uniform sampling over `U` is a fine first implementation — but it must still be the *constrained* uniform (respecting voids and dead-stock cardinality), i.e. steps 1–2 with `belief_prob ≡ 1`. Unconstrained sampling is never used.

### 6.2 Bidding without strategy fusion (fix 2.1)

Double-dummy trick counts are systematically optimistic because the solver sees hidden cards. Feeding them into a bid decision produces exactly the over-bidding pathology described. Three corrections:

**(a) Bidding uses policy rollouts, not double-dummy.** For each sampled world `w`, play the round out with the *current neural policy for all seats* (which is non-clairvoyant by construction: each simulated player sees only its own information set). Record the actor's trick count `t_w`. The distribution `{t_w}` over `W ≈ 200` worlds is an unbiased estimate of the trick distribution *under the policy that will actually be executed*. Then:

```
bid* = argmax_{b legal} Σ_w score(b, t_w) / W
```

This is a decision-theoretically correct bid and contains no fusion, because no simulated agent ever conditions on hidden information.

**(b) Double-dummy is used only as a cheap feature, after calibration.** DD is ~100× faster than a policy rollout, so we keep it — but we fit a calibration model `P(t_actual | t_DD, C, P, seat_position)` from logged self-play data (a small 4-D histogram, smoothed). Bids then use `Σ_t P(t | t_DD) score(b, t)`. The calibration absorbs the fusion bias empirically rather than pretending it does not exist.

**(c) Bid distribution comes from the net, not from the search, during training.** During training rollouts we do not search at all (throughput). Search is applied in the Expert-Iteration phase (§6.4) and at deployment.

### 6.3 Play-phase search

At the root, PIMC already avoids fusion (one action is chosen for all worlds). Fusion remains *below* the root in a pure DD solver. Mitigations:

- **Depth-limited determinized search with a neural leaf value.** Expand the current trick fully (all legal plays for the actor), then evaluate each resulting position with the network's factored value `Σ_t p(t|s)·score(bid, t)` rather than solving to the end. This eliminates deep fusion entirely, is far cheaper, and inherits the net's imperfect-information play.
- **Full DD only for endgames** (`tricks_remaining ≤ 3`), where fusion is negligible and exact play is worth a lot.
- **Action scoring** = mean over worlds of the leaf value, with a per-world common-random-numbers trick (the same world set is used for every candidate action) to cancel variance.

### 6.4 Search as a policy-improvement operator, not an oracle (fix 2.3)

PIMC cannot bluff, signal, or hide information; blending it directly with a trained policy at deployment can *reduce* strength by breaking the net's own (imperfect-information-correct) behaviour. We therefore:

1. **Expert Iteration.** Periodically (every ~1000 PPO iterations) run search on a sample of ~50 k states, producing improved action distributions `π_search`. Add a distillation loss `KL(π_search ‖ π_θ)` to the PPO objective. The improvement is absorbed *into an imperfect-information policy*, which then continues to be validated in real self-play — so any search recommendation that is bad because of fusion or non-locality gets punished by the RL objective and is not retained.
2. **Bounded blending at deployment.** `π_final ∝ π_net^α · π_search^(1−α)` with `α` tuned on held-out duplicate deals, **and with a hard rule that blending is disabled if it fails to beat pure `π_net` by ≥ 2 σ**. We ship whatever wins, and we measure it rather than assuming search helps.
3. **Roadmap:** replace PIMC with IS-MCTS over public belief states (ReBeL-style) if the ExIt gains plateau. Explicitly out of scope for the 3-week plan.

---

## 7. Systems & Performance

### 7.1 Iteration budget

Per iteration, `B = 16384`, `P = 5`, `C = 10` → `D = 55` steps.

| Stage | Work | Estimate |
|---|---|---|
| Deal + reset | Rust, rayon | 3 ms |
| Encode ×55 | 900 k states, ~60 features each, 16 threads | 250–400 ms |
| H2D copy ×55 | 4 MB each, pinned, async | overlapped |
| Forward ×55 ×2 nets | 2.5 M params, bf16, CUDA graph | 180 ms |
| Engine step ×55 | Rust | 20 ms |
| PPO update (2 epochs × 8 minibatches) | fwd+bwd on 900 k samples | 500 ms |
| **Total** | | **~0.9–1.1 s** |

With encoding double-buffered against the GPU, the effective iteration is `max(encode, forward) + update ≈ 0.9 s`. 10 k iterations ≈ **2.5 h**, matching the target.

### 7.2 Pipelining detail

```
thread A (Rust):  encode(t+1) into buffer[(t+1)%2]
thread M (Python): H2D(buffer[t%2]) -> forward -> sample -> step(actions)
```

Synchronization is one `crossbeam` channel round-trip per step. The engine's `step()` must complete before `encode(t+1)` begins, so the true overlap is `encode(t+1) ‖ (forward(t) + sample(t))` — still enough to hide most of the encode cost.

### 7.3 Trajectory storage (fix 5.3)

All buffers are **pre-allocated once** on GPU with static shapes:

| Tensor | Shape | dtype | Bytes |
|---|---|---|---|
| `obs_idx` | `[D, B, K_MAX]` | int16 | 55·16384·128·2 = **231 MB** |
| `action` | `[D, B]` | uint8 | 0.9 MB |
| `old_logp` | `[D, B]` | float16 | 1.8 MB |
| `value` | `[D, B]` | float16 | 1.8 MB |
| `legal_bits` | `[D, B]` | uint64 (play) | 7.2 MB |
| `legal_bid` | `[D, B]` | uint16 | 1.8 MB |
| `actor_seat` | `[D, B]` | uint8 | 0.9 MB |
| `is_learner` | `[D, B]` | bool | 0.9 MB |
| `belief_tgt` | `[D, B, 52]` | uint8 | 47 MB |
| **Total** | | | **~293 MB** |

Notes:
- Legal masks are stored **as bitmasks** (8 bytes), not as 52 booleans (52 bytes), and unpacked on the fly inside the minibatch loop. 6.5× saving.
- Belief targets are the dominant secondary cost; if VRAM is tight, subsample the belief loss to 1-in-4 transitions and store only those.
- **Minibatching is by index permutation** (`perm = torch.randperm(D*B, device='cuda')`, then `flat_obs[perm[mb]]`). No Python lists, no `torch.cat`, no host round-trip, no fragmentation.
- Buffers are reused across iterations; the allocator sees a constant footprint.

---

## 8. Round-Level vs. Match-Level (fix 6)

Optimizing per-round score is provably suboptimal in a ladder: when trailing on the last round, a low-probability high-value bid can be correct even at negative EV in points; when leading, minimizing variance is correct.

**Stage A (rounds 1–10 k iterations): round-score objective, match-conditioned inputs.**
The `match_ctx` feature block (§5.2) is present from the very first iteration and encodes:
- own cumulative score bucket, and each opponent's score *relative to mine* (bucketed, actor-relative order);
- rounds remaining in the ladder, one-hot;
- the remaining round-size schedule (e.g. "descending 10→1"), summarized;
- position in the standings (rank one-hot).

During Stage A these features are populated from a *simulated* random match context (sample a plausible standings vector for each deal) so the network learns to read them, and the reward remains the round score. The net therefore learns `V_round(s | context)` but is context-conditional and *ready* to change behaviour.

**Stage B (fine-tune, ~2 k iterations): match utility.**
The engine gains a `Match` driver that plays a full ladder of rounds. We define terminal utility `U` from final placement (e.g. `U = [1.0, 0.4, 0.1, 0.0, ...]` by rank, or the actual ladder payout), and train a small **match-value network** `V_match(standings, rounds_left)` — a 3-layer MLP over ~50 inputs, trained by regression on simulated ladders.

Round-level reward is then replaced by the potential-difference reward

```
r_round = V_match(standings_after, rounds_left − 1) − V_match(standings_before, rounds_left)
```

This is a **potential-based shaping** of the true match objective: it is dense (one signal per round), it is on the same scale everywhere, and summing it over a match telescopes to `U − V_match(initial)`, so the optimal policy is unchanged from optimizing `U` directly. Crucially, it lets us keep the round as the PPO episode — no long-horizon credit assignment, no 10× longer trajectories — while optimizing the correct objective.

`V_match` and the policy are trained alternately: freeze the policy, resimulate ladders, refit `V_match`, unfreeze. Two or three alternations suffice.

**Validation:** the Stage-B agent should visibly change behaviour — measurably higher bid variance when trailing in the final rounds, lower when leading. We assert this in an evaluation test; if the behaviour does not appear, the match features are not being read and something is broken.

---

## 9. Evaluation

### 9.1 Duplicate deals

Pre-generate 20 000 deals per `(P, C)`. For each deal, play all `P` seat rotations with the candidate in each seat against a fixed reference lineup. Report the mean paired score difference and its standard error. Paired-by-deal + all-rotations removes essentially all card luck; residual variance comes only from stochastic policies (which we can also fix by seeding).

### 9.2 Metrics

- **Points per round vs. reference** (paired, with CI).
- **Plump rate** and **signed bid error histogram** — an agent that misses by +1 too often has a play problem; by −1 too often has a bidding problem.
- **Bid calibration curve:** `P(tricks = b | bid = b)` and reliability diagrams from the trick head. A well-calibrated trick head is the strongest single predictor of playing strength here.
- **Belief head accuracy on uncertain cards only** (§5.3), reported as top-1 and NLL.
- **Best-response gap (exploitability proxy):** freeze the agent in `P−1` seats, train a fresh net in the last seat for 1 GPU-hour, report the gap in points/round. Track this over training — a rising gap means the agent is drifting into conventions.
- **Cross-lineage score** against the held-out lineage.
- **Match win rate** (Stage B only) over full ladders.

### 9.3 Correctness harness

- **Brute-force ground truth** for `P=3, C=2`: enumerate the full game tree with a perfect-recall best response; compare the agent's decisions and the value head against exact values. This is the only place we get absolute rather than relative numbers.
- **Human CLI** for rule-adherence sanity checks.

---

## 10. Testing Strategy (expanded)

**Rust unit / property tests**
- `legal_bids`: no underflow; only the dealer is constrained; mask never empty; forbidden bid out of range handled.
- `legal_plays`: follow-suit obeyed; void → free choice.
- `trick_winner`: highest card of led suit; no trump.
- Scoring table matches the configured variant, for all `(bid, tricks)`.
- **Round invariant:** every game consumes exactly `P + P·C` decisions; hands empty at the end; `Σ tricks_won = C`.
- **Encoder parity:** `encode(permute(s, σ)) == permute_indices(encode(s), σ)` for 10⁵ random `(s, σ)`.
- **Encoder relabeling invariance:** rotating absolute seat labels leaves the actor-relative encoding unchanged.
- **Layout:** every emitted index falls in exactly one block; `max index < N_FEATURES`; `len ≤ K_MAX`.
- **World sampler:** 10⁶ sampled worlds all pass `assert_world_valid` (capacities, voids, no duplicate cards, played cards not re-dealt, dead-stock size exact).
- **Swap-MCMC:** on a tiny instance, the empirical world distribution matches the exact constrained distribution (χ² test).

**Python integration tests**
- Rust and Torch legal masks agree bit-for-bit.
- Rewards returned by the engine equal the scoring function applied to `(bids, tricks)`.
- PPO buffer: every stored transition's `seat` matches the engine's `actor`; every seat owns exactly `1 + C` transitions per round; returns are constant within a seat's transitions.
- Value-head consistency: `|V − Σ p(t)·score(b,t)| < ε` post-bid on a trained net.
- Differential test: Rust encoder vs. Triton encoder (if Plan B activated).

---

## 11. Revised Milestones

| Phase | Days | Content | Exit criterion |
|---|---|---|---|
| 0 | 0 | Scaffolding, card types, scoring config | `cargo test` green |
| 1 | 1–2 | Single-game engine + `PublicKnowledge` + legal masks | Property tests green; brute-force `P=3,C=2` tree enumerable |
| 2 | 3–4 | Layout registry, encoder, permutation table, SoA batch | Parity + invariance tests green; encode ≥ 2 M states/s on 16 cores |
| 3 | 5 | PyO3 bridge, pinned buffers, double buffering | Full rollout of 16 k games at ≥ 40 steps/s |
| 4 | 6–8 | Network, per-seat PPO, distributional value, aux heads | Beats random by > 8 pts/round; beats heuristic on `P=4,C=5` |
| 5 | 9–11 | All configs, per-deal suit relabeling, opponent pool (K=2), duplicate-deal baselining | Trick-head calibration ECE < 0.03; best-response gap measured |
| 6 | 12–13 | Multi-lineage population, cross-play, exploitability probe | Cross-lineage score within 1 pt of self-play score |
| 7 | 14–16 | Constrained world sampler, depth-limited PIMC, DD endgame, bid-by-policy-rollout | Search beats net by ≥ 2 σ on duplicate deals, or is disabled |
| 8 | 17–18 | Expert Iteration distillation | Distilled net ≥ search-blended net at 1/100 the inference cost |
| 9 | 19–21 | Match simulator, `V_match`, Stage-B fine-tune, final eval | Measurable standings-dependent bid variance; match win rate ↑ |

---

## 12. Residual Risks (honestly stated)

| Risk | Status |
|---|---|
| PIMC provides no gain over a well-trained net | Plausible. Mitigated by the "must beat net by 2 σ or be disabled" rule; the ExIt path also degrades gracefully to a no-op. |
| Best-response gap stays large | Expected — PPO self-play is not an equilibrium solver. If the gap is unacceptable we would need CFR/ReBeL, which is out of scope. We report the number rather than claim Nash. |
| CPU encoding remains the bottleneck at `P=6, C=10` | Plan B (GPU encoder) is designed and testable; budget 2 days if triggered. |
| Match-level fine-tuning destabilizes round-level play | Alternating optimization with a frozen `V_match` and a small LR; fall back to Stage-A weights if match win rate regresses. |
| "Super-human" claim | Removed. We claim *strong*, measured against duplicate-deal baselines, heuristics, held-out lineages, and a best-response probe. Absolute optimality is only verified on the `P=3, C=2` toy. |