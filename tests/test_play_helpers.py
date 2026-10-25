import io

import numpy as np
import pytest
from rich.console import Console

import play
from plump_engine import PyFastEnv


class FakeConsole:
    def __init__(self, inputs):
        self.inputs = list(inputs)
        self.lines = []

    def input(self, prompt=""):
        self.lines.append(prompt)
        if not self.inputs:
            raise EOFError
        return self.inputs.pop(0)

    def print(self, *args, **kwargs):
        if any(hasattr(arg, "__rich_console__") for arg in args):
            buffer = io.StringIO()
            renderer = Console(
                file=buffer, width=120, force_terminal=False, color_system=None
            )
            renderer.print(*args, **kwargs)
            self.lines.append(buffer.getvalue())
        else:
            self.lines.append(" ".join(str(arg) for arg in args))


def settings():
    return play.PlaySettings(
        autoplay=False, use_search=True, worlds=4, model_available=True
    )


def test_parse_card():
    # Spec suit order: 0 = hearts, 1 = spades, 2 = diamonds, 3 = clubs.
    assert play.parse_card("HA") == 12
    assert play.parse_card("SA") == 25
    assert play.parse_card("sA") == 25
    assert play.parse_card("HK") == 11
    assert play.parse_card("D10") == 34
    assert play.parse_card("C7") == 44
    assert play.parse_card("♠A") == 25
    assert play.parse_card("A♠") == 25
    assert play.parse_card("10D") == 34
    assert play.parse_card("ZZ") is None
    assert play.parse_card("S") is None
    assert play.parse_card("S1") is None


def test_mask_to_bool():
    mask = (1 << 0) | (1 << 10)
    result = play.mask_to_bool(mask, 11)
    assert result.dtype == np.bool_
    assert result.sum() == 2
    assert result[0] and result[10] and not result[1]


def test_human_bid_rejects_out_of_range_then_passes():
    env = PyFastEnv(seed=1)
    env.new_round(10, 0)
    console = FakeConsole(["11", "pass"])
    assert play.human_bid(console, env, settings()) == 0


def test_human_bid_rejects_dealer_balancing_bid():
    env = PyFastEnv(seed=1)
    env.new_round(4, 1)
    for _ in range(3):
        assert int(env.current_player) != 0
        env.step_bid(0)
    assert int(env.current_player) == 0
    assert env.forbidden_bid() == 4
    console = FakeConsole(["4", "0"])
    assert play.human_bid(console, env, settings()) == 0
    assert any("forbidden" in line for line in console.lines)


def test_human_card_rejects_invalid_then_plays_legal_lead():
    env = PyFastEnv(seed=2)
    env.new_round(2, 0)
    for _ in range(4):
        env.step_bid(0)
    assert env.phase == "Playing"
    assert int(env.current_player) == 0
    hand = list(env.state()["hands"][0])
    legal = play.mask_to_bool(int(env.legal_cards_mask(0)), 52)
    legal_card = next(card for card in hand if legal[card])
    console = FakeConsole(["ZZ", play.card_code(legal_card)])
    assert play.human_card(console, env, settings()) == legal_card


def test_pannplump_ui_hides_own_card_until_play():
    env = PyFastEnv(seed=3)
    env.new_round(1, 0)
    own = int(env.state()["hands"][0][0])
    console = FakeConsole([])
    play.render_state(console, env, [0] * 4, [0] * 4, 1, settings(), None)
    bidding_text = "\n".join(console.lines)
    assert "??" in bidding_text
    assert "on your forehead!" in bidding_text
    assert play.card_plain(own) not in bidding_text

    for _ in range(4):
        env.step_bid(0)
    assert env.phase == "Playing"
    console = FakeConsole([])
    play.render_state(console, env, [0] * 4, [0] * 4, 1, settings(), None)
    playing_text = "\n".join(console.lines)
    assert play.card_plain(own) in playing_text
    assert "revealed" in playing_text


def test_round_status_labels():
    assert "PASSED" in play.round_status(0, 0)
    assert "MADE" in play.round_status(2, 2)
    assert "PLUMP" in play.round_status(2, 1)
    assert "PLUMP" in play.round_status(0, 1)


def test_round_result_uses_scoreboard_headers():
    env = PyFastEnv(seed=4)
    env.new_round(1, 0)
    for _ in range(4):
        env.step_bid(0)
    while env.phase == "Playing":
        player = int(env.current_player)
        mask = int(env.legal_cards_mask(player))
        card = (mask & -mask).bit_length() - 1
        env.step_play(card)
    assert env.phase == "Terminal"
    console = FakeConsole([])
    play.show_round_result(console, env, 1, [5, 5, 5, 5], [0] * 4, [0] * 4)
    rendered = "\n".join(console.lines)
    for header in (
        "Tricks Won",
        "Status",
        "Round Points",
        "Total Plumps",
        "Total Score",
    ):
        assert header in rendered


def test_search_on_without_model_stays_disabled():
    config = play.PlaySettings(
        autoplay=False, use_search=False, worlds=4, model_available=False
    )
    console = FakeConsole([])
    assert play.handle_command("search on", config, console)
    assert config.use_search is False
    assert any("heuristic" in line for line in console.lines)

    config = settings()
    assert play.handle_command("search on", config, console)
    assert config.use_search is True


def test_human_card_explains_follow_suit():
    env = None
    illegal_card = None
    legal_card = None
    for seed in range(100):
        candidate = PyFastEnv(seed=seed)
        candidate.new_round(2, 1)
        for _ in range(4):
            candidate.step_bid(0)
        for _ in range(3):
            player = int(candidate.current_player)
            mask = int(candidate.legal_cards_mask(player))
            card = (mask & -mask).bit_length() - 1
            candidate.step_play(card)
        if candidate.phase != "Playing" or int(candidate.current_player) != 0:
            continue
        hand = list(candidate.state()["hands"][0])
        legal = play.mask_to_bool(int(candidate.legal_cards_mask(0)), 52)
        bad = [card for card in hand if not legal[card]]
        if bad:
            env = candidate
            illegal_card = bad[0]
            legal_card = next(card for card in hand if legal[card])
            break
    assert env is not None
    console = FakeConsole([play.card_code(illegal_card), play.card_code(legal_card)])
    assert play.human_card(console, env, settings()) == legal_card
    assert any("illegal" in line.lower() for line in console.lines)
