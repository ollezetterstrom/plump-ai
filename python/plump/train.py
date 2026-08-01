# PPO trainer skeleton (§4).
#
# Phase 4+ material. This module fixes the structure and the loss math from
# §4.1–§4.3; the rollout loop is wired once the Rust engine bridge lands.
#
# Key contracts enforced here when implemented:
#   * per-seat trajectory attribution (§4.1): returns are per-seat scalars,
#     no cross-seat GAE;
#   * distributional value CE + trick-count factorization (§4.2);
#   * disjoint bid/play heads selected by phase (§4.3), `-1e30` masking;
#   * approx-KL early stopping and an entropy floor (§4.3).

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from . import config
from .net import PlumpNet

EPS = 1e-8


@dataclass
class PolicyLosses:
    policy: torch.Tensor
    value_ce: torch.Tensor
    value_consistency: torch.Tensor
    trick_ce: torch.Tensor
    belief_ce: torch.Tensor
    entropy: torch.Tensor
    total: torch.Tensor
    approx_kl: Optional[torch.Tensor]


def _select_logits(net_outputs: dict, is_bid: torch.Tensor) -> torch.Tensor:
    """Merge the disjoint bid/play heads into a [B, 52] action logits tensor.

    Bid ids live in 0..=10 and play ids are card indices 0..51. Bid rows are
    gathered from `bid` into the leading 11 slots; play rows pass through
    unchanged. Cross-phase actions are structurally impossible (§5.1).
    """
    bid = net_outputs["bid"]  # [B, 11]
    play = net_outputs["play"]  # [B, 52]
    logits = play.clone()
    bid_b = is_bid.nonzero(as_tuple=True)[0]
    if bid_b.numel():
        logits[bid_b, :11] = bid[bid_b]
    return logits


def _logp_ratio(logits: torch.Tensor, action: torch.Tensor, legal: torch.Tensor,
                old_logp: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~legal, -1e30)
    logp = torch.log_softmax(masked, -1).gather(-1, action.unsqueeze(-1)).squeeze(-1)
    return (logp - old_logp).exp()


def compute_policy_loss(
    net: PlumpNet,
    net_outputs: dict,
    is_bid: torch.Tensor,
    action: torch.Tensor,
    legal: torch.Tensor,
    old_logp: torch.Tensor,
    ret: torch.Tensor,
    atom_idx: torch.Tensor,
    trick_tgt: torch.Tensor,
    belief_tgt: torch.Tensor,
    post_bid: torch.Tensor,
    cfg: config.PpoConfig,
    entropy_coef: float,
) -> PolicyLosses:
    """PPO objective from §4.3 with all auxiliary losses gradient-scaled."""
    logits = _select_logits(net_outputs, is_bid)
    ratio = _logp_ratio(logits, action, legal, old_logp)
    adv = ret - net.value(net_outputs["value"])
    adv = (adv - adv.mean()) / (adv.std() + EPS)

    policy_loss = -torch.min(
        ratio * adv,
        ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv,
    ).mean()

    value_ce = F.cross_entropy(net_outputs["value"], atom_idx)
    trick_ce = F.cross_entropy(net_outputs["tricks"], trick_tgt)

    # Belief CE computed only over genuinely uncertain cards (≥2 legal
    # classes) — the loss is masked to those positions (§5.3).
    belief_ce = _belief_ce(net, net_outputs["belief"], belief_tgt)

    # Post-bid factored value consistency: ||V - sum_t p(t)*score(bid,t)||².
    value_consistency = torch.tensor(0.0, device=adv.device)
    if post_bid.any():
        pb = post_bid
        p_tricks = torch.softmax(net_outputs["tricks"][pb], -1)  # [n, 11]
        # V_factored = sum_t p(t) * score_table[bid][t]; bid is the action id.
        bid = action[pb].to(torch.long)
        v_factored = (p_tricks * net.score_table[bid]).sum(-1)
        value_consistency = F.mse_loss(
            net.value(net_outputs["value"][pb]), v_factored)

    ent = net.entropy(logits, legal).mean()
    ent = ent.clamp(min=cfg.entropy_floor)

    total = (
        policy_loss
        + cfg.value_coef * value_ce
        + cfg.consistency_coef * value_consistency
        + cfg.trick_coef * trick_ce
        + cfg.belief_coef * belief_ce
        - entropy_coef * ent
    )

    with torch.no_grad():
        # approx-KL needs the new logp of the sampled action, which is
        # available during `update()`; computed there for early stopping.
        approx_kl = None

    return PolicyLosses(policy=policy_loss, value_ce=value_ce,
                        value_consistency=value_consistency, trick_ce=trick_ce,
                        belief_ce=belief_ce, entropy=ent, total=total,
                        approx_kl=approx_kl)


def _belief_ce(net: PlumpNet, belief_logits: torch.Tensor,
               belief_tgt: torch.Tensor) -> torch.Tensor:
    # Phase 5: mask to uncertain cards via the engine's PublicKnowledge.
    # Skeleton returns the plain mean; the mask lands with the engine bridge.
    b, _ = belief_logits.shape
    logits = belief_logits.view(b, 52, net.n_belief_classes)
    return F.cross_entropy(logits.permute(0, 2, 1), belief_tgt)


class PPOTrainer:
    """Stages the fixed-shape buffers from §7.3 and the update loop from §4.

    Not runnable until the engine bridge provides `plump.env` rollouts.
    """

    def __init__(self, net: PlumpNet, env_cfg: config.EnvConfig,
                 ppo_cfg: config.PpoConfig, device: str):
        self.net = net.to(device)
        self.env_cfg = env_cfg
        self.ppo_cfg = ppo_cfg
        self.device = device
        self.opt = torch.optim.Adam(net.parameters(), lr=ppo_cfg.lr)
        self.rollout_len = (1 + env_cfg.n_cards) * env_cfg.n_players
        self._alloc_buffers()

    def _alloc_buffers(self):
        # §7.3: pre-allocated once, fixed shapes, reused across iterations.
        D, B = self.rollout_len, self.env_cfg.batch
        d = self.device
        self.obs_idx = torch.zeros(D, B, config.K_MAX, dtype=torch.int16, device=d)
        self.action = torch.zeros(D, B, dtype=torch.uint8, device=d)
        self.old_logp = torch.zeros(D, B, dtype=torch.float16, device=d)
        self.value = torch.zeros(D, B, dtype=torch.float16, device=d)
        self.legal = torch.zeros(D, B, 52, dtype=torch.bool, device=d)
        self.is_bid = torch.zeros(D, B, dtype=torch.bool, device=d)
        self.seat = torch.zeros(D, B, dtype=torch.uint8, device=d)
        self.is_learner = torch.zeros(D, B, dtype=torch.bool, device=d)
        self.ret = torch.zeros(D, B, dtype=torch.float32, device=d)

    def rollout(self):
        # Phase 4: one iteration of deal -> step x D -> per-seat MC returns
        # (§4.1), opponent mixing K=2 (§4.4), suit relabeling (§4.5.1).
        raise NotImplementedError("wired when the engine bridge lands")

    def update(self):
        # §4.3: 2 epochs x 8 minibatches, index-permutation shuffling
        # (§7.3), approx-KL early stop at target_kl.
        raise NotImplementedError("wired when the engine bridge lands")

    def train(self, iterations: int):
        for _ in range(iterations):
            self.rollout()
            self.update()
