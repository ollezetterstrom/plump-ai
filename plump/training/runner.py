# plump/training/runner.py — game runner, composes env+encode+models. No training logic.
from __future__ import annotations

import random
import torch
from ..env.engine import PlumpEnv
from ..encode.legacy import encode_state_278, encode_state_294, mask_actions
from ..env.cards import INDEX_TO_CARD
from ..config.settings import terminal_reward
from .rewards import shaped_reward


def pick_action(q_values: torch.Tensor, legal_mask: torch.Tensor, epsilon: float) -> int:
    masked = q_values.clone()
    masked[~legal_mask] = -1e9
    if random.random() < epsilon:
        idxs = legal_mask.nonzero(as_tuple=True)[0]
        return idxs[random.randint(0, len(idxs) - 1)].item()
    return masked.argmax().item()


def run_one_game(env: PlumpEnv, bid_model, play_model, champ_bid, champ_play, epsilon, device, is_eval=False):
    """Decoupled runner — house rules inside."""
    env.new_round(round_cards=random.randint(1, 10))

    bid_transitions: list[dict] = []
    play_transitions: list[dict] = []

    TRAIN_RANDOM_CHANCE = 0.07  # from config, kept local to avoid circular import

    # bidding
    for turn in range(4):
        is_ai = (turn == 0) if is_eval else True
        is_random = False if is_eval else (random.random() < TRAIN_RANDOM_CHANCE)
        if is_eval and not is_ai:
            state = encode_state_278(env, turn)
            legal = mask_actions(env, turn, "bid")
            legal_t = torch.tensor(legal, dtype=torch.bool, device=device)
            if champ_bid is not None:
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32, device=device)
                    bid = pick_action(champ_bid(state_t), legal_t, 0.0)
            else:
                bid = legal_t.nonzero(as_tuple=True)[0][random.randrange(legal_t.sum().item())].item()
        elif is_random:
            legal = mask_actions(env, turn, "bid")
            legal_t = torch.tensor(legal, dtype=torch.bool)
            bid = legal_t.nonzero(as_tuple=True)[0][random.randrange(legal_t.sum().item())].item()
        else:
            state = encode_state_278(env, turn)
            legal = mask_actions(env, turn, "bid")
            legal_t = torch.tensor(legal, dtype=torch.bool, device=device)
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=device)
                bid = pick_action(bid_model(state_t), legal_t, epsilon)
            if not is_eval:
                bid_transitions.append({"player": turn, "state": state, "action": bid, "legal": legal})
        env.bids[turn] = bid

    # house: highest earliest leads
    try:
        max_bid = max(env.bids)
        current_player = next(i for i, b in enumerate(env.bids) if b == max_bid)
    except ValueError:
        current_player = 0

    for _trick in range(env.round_cards):
        trick_last_idx: dict[int, int] = {}
        for _ in range(4):
            is_ai = (current_player == 0) if is_eval else True
            is_random = False if is_eval else (random.random() < TRAIN_RANDOM_CHANCE)
            if is_eval and not is_ai:
                state = encode_state_278(env, current_player)
                legal = mask_actions(env, current_player, "play")
                legal_t = torch.tensor(legal, dtype=torch.bool, device=device)
                if champ_play is not None:
                    with torch.no_grad():
                        state_t = torch.tensor(state, dtype=torch.float32, device=device)
                        action = pick_action(champ_play(state_t), legal_t, 0.0)
                else:
                    action = legal_t.nonzero(as_tuple=True)[0][random.randrange(legal_t.sum().item())].item()
                env.play_card(current_player, INDEX_TO_CARD[action])
            elif is_random:
                legal = mask_actions(env, current_player, "play")
                legal_t = torch.tensor(legal, dtype=torch.bool)
                action = legal_t.nonzero(as_tuple=True)[0][random.randrange(legal_t.sum().item())].item()
                env.play_card(current_player, INDEX_TO_CARD[action])
            else:
                state = encode_state_294(env, current_player)
                legal = mask_actions(env, current_player, "play")
                legal_t = torch.tensor(legal, dtype=torch.bool, device=device)
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32, device=device)
                    action = pick_action(play_model(state_t), legal_t, epsilon)
                env.play_card(current_player, INDEX_TO_CARD[action])
                if not is_eval:
                    play_transitions.append(
                        {
                            "player": current_player,
                            "state": state,
                            "action": action,
                            "legal": legal,
                            "shaped_r": 0.0,
                            "post_state": encode_state_294(env, current_player),
                        }
                    )
                    trick_last_idx[current_player] = len(play_transitions) - 1
            current_player = (current_player + 1) % 4
        winner, _ = env.resolve_trick()
        if not is_eval:
            for p, tidx in trick_last_idx.items():
                t = play_transitions[tidx]
                t["shaped_r"] = shaped_reward(p == winner, env.tricks_won[p], env.bids[p])
        current_player = winner

    if is_eval:
        p0_score = terminal_reward(env.tricks_won[0], env.bids[0])
        return [], [], p0_score

    BID_STATE_SIZE = len(bid_transitions[0]["state"]) if bid_transitions else 278
    zero_bid_state = [0.0] * BID_STATE_SIZE
    zero_bid_mask = [False] * 11
    formatted_bid = []
    for t in bid_transitions:
        p = t["player"]
        reward = terminal_reward(env.tricks_won[p], env.bids[p])
        formatted_bid.append((t["state"], t["action"], reward, zero_bid_state, True, zero_bid_mask))
    formatted_play = []
    for i, t in enumerate(play_transitions):
        p = t["player"]
        next_t = None
        for j in range(i + 1, len(play_transitions)):
            if play_transitions[j]["player"] == p:
                next_t = play_transitions[j]
                break
        is_done = next_t is None
        if is_done:
            term_r = terminal_reward(env.tricks_won[p], env.bids[p])
            step_reward = term_r + t["shaped_r"]
            true_next_state = t["post_state"]
            next_mask = [False] * 52
        else:
            step_reward = t["shaped_r"]
            true_next_state = next_t["state"]
            next_mask = next_t["legal"]
        formatted_play.append((t["state"], t["action"], step_reward, true_next_state, is_done, next_mask))
    p0_score = int(terminal_reward(env.tricks_won[0], env.bids[0]))
    return formatted_bid, formatted_play, p0_score
