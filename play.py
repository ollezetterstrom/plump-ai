"""Play Swedish Plump against the trained transformer (or a heuristic)."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import plump_engine
from plump.agents import heuristic_bid, heuristic_play, masked_argmax
from plump.checkpoint import load_model as load_transformer
from plump.config import BEST_CHECKPOINT, LATEST_CHECKPOINT, ROUND_SCHEDULE


def _unicode_supported() -> bool:
    encoding = getattr(sys.stdout, "encoding", None)
    if not encoding:
        return False
    try:
        "♥♠♦♣→".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


UNICODE = _unicode_supported()
SUITS = ("♥", "♠", "♦", "♣") if UNICODE else ("H", "S", "D", "C")
ARROW = "→" if UNICODE else "->"
DASH = "—" if UNICODE else "-"
SUIT_CODES = "HSDC"
RANK_LABELS = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}
RANK_PARSE = {label: rank for rank, label in RANK_LABELS.items()}
RANK_PARSE["T"] = 10
SUIT_PARSE = {
    "H": 0,
    "S": 1,
    "D": 2,
    "C": 3,
    "♥": 0,
    "♠": 1,
    "♦": 2,
    "♣": 3,
}
SEAT_NAMES = ("Seat 0", "Seat 1", "Seat 2", "Seat 3")


class QuitGame(Exception):
    pass


@dataclass
class PlaySettings:
    autoplay: bool
    use_search: bool
    worlds: int
    model_available: bool


@dataclass
class Decision:
    method: str
    elapsed_ms: float = 0.0
    extra: str = ""


def card_code(card: int) -> str:
    return f"{SUIT_CODES[card // 13]}{RANK_LABELS[card % 13 + 2]}"


def card_markup(card: int) -> str:
    suit = card // 13
    color = "bold red" if suit in (0, 2) else "bold cyan"
    return f"[{color}]{SUITS[suit]}{RANK_LABELS[card % 13 + 2]}[/{color}]"


def card_plain(card: int) -> str:
    return f"{SUITS[card // 13]}{RANK_LABELS[card % 13 + 2]}"


def parse_card(text: str) -> Optional[int]:
    raw = text.strip().upper().replace("10", "T")
    if len(raw) < 2:
        return None
    if raw[0] in SUIT_PARSE:
        suit_ch, rank_ch = raw[0], raw[1:]
    elif raw[-1] in SUIT_PARSE:
        suit_ch, rank_ch = raw[-1], raw[:-1]
    else:
        return None
    suit = SUIT_PARSE.get(suit_ch)
    rank = RANK_PARSE.get(rank_ch)
    if suit is None or rank is None:
        return None
    return suit * 13 + (rank - 2)


def mask_to_bool(mask: int, size: int) -> np.ndarray:
    return np.array([(int(mask) >> i) & 1 for i in range(size)], dtype=np.bool_)


def normalized_prior(values: np.ndarray, legal: np.ndarray) -> np.ndarray:
    masked = np.where(legal, values, -np.inf)
    finite = masked[np.isfinite(masked)]
    if finite.size == 0:
        return np.zeros_like(masked, dtype=np.float32)
    low = float(finite.min())
    high = float(finite.max())
    if high - low < 1e-6:
        return np.where(legal, 0.5, 0.0).astype(np.float32)
    return np.where(legal, (masked - low) / (high - low), 0.0).astype(
        np.float32
    )


def bid_text(bid: int) -> str:
    if bid < 0:
        return DASH
    if bid == 0:
        return "PASS"
    return f"BID {bid}"


def decision_suffix(decision: Optional[Decision]) -> str:
    if decision is None:
        return ""
    text = f"[dim]({decision.method}"
    if decision.elapsed_ms > 0.0:
        text += f", {decision.elapsed_ms:.1f} ms"
    if decision.extra:
        text += f", {decision.extra}"
    return text + ")[/dim]"


def print_help(console: Console) -> None:
    console.print(
        Panel(
            "[bold]Commands[/]\n"
            "  [cyan]search on[/]   enable PIMC search for the AI\n"
            "  [cyan]search off[/]  use the transformer Q-heads directly\n"
            "  [cyan]worlds N[/]    set the number of PIMC worlds (default 32)\n"
            "  [cyan]help[/]        show this help\n"
            "  [cyan]quit[/]        leave the table\n\n"
            "[bold]Bids[/]\n"
            "  Enter 0 or 'pass' to pass, otherwise a number up to the round size.\n"
            "  The dealer's balancing bid is forbidden.\n\n"
            "[bold]Cards[/]\n"
            "  Enter suit then rank: [cyan]HA[/], [cyan]SK[/], [cyan]D10[/], [cyan]C7[/]\n"
            + (
                "  Unicode suits also work: [cyan]♥A[/], [cyan]♠K[/], [cyan]♦10[/], [cyan]♣7[/]"
                if UNICODE
                else "  Suits: H=hearts, S=spades, D=diamonds, C=clubs"
            )
            + "\n\n[bold]Pannplump[/]\n"
            "  On a 1-card round your own card stays on your forehead, hidden\n"
            "  until every bid is in. You can see the other three foreheads.",
            title="Plump help",
            border_style="cyan",
        )
    )


def handle_command(
    raw: str, settings: PlaySettings, console: Console
) -> bool:
    command = raw.strip().lower()
    if command in ("quit", "exit", "q"):
        raise QuitGame
    if command == "help":
        print_help(console)
        return True
    if command == "search on":
        if not settings.model_available:
            console.print(
                "[yellow]No checkpoint loaded; PIMC search needs the model's "
                "beliefs. AI stays heuristic.[/yellow]"
            )
            return True
        settings.use_search = True
        console.print("[green]AI search enabled.[/green]")
        return True
    if command == "search off":
        settings.use_search = False
        if settings.model_available:
            console.print("[yellow]AI search disabled; using model heads.[/yellow]")
        else:
            console.print(
                "[yellow]No checkpoint loaded; AI stays heuristic.[/yellow]"
            )
        return True
    if command.startswith("worlds "):
        try:
            worlds = int(command.split(None, 1)[1])
        except ValueError:
            console.print("[red]Usage: worlds N[/red]")
            return True
        if worlds < 1:
            console.print("[red]Worlds must be at least 1.[/red]")
            return True
        settings.worlds = worlds
        console.print(f"[green]PIMC worlds: {worlds}[/green]")
        return True
    if command == "search":
        state = "on" if settings.use_search else "off"
        worlds = f" ({settings.worlds} worlds)" if settings.use_search else ""
        console.print(f"AI search is {state}{worlds}.")
        return True
    return False


def encode_tensors(env, player: int, device: torch.device):
    tokens, feats, _ = env.encode(player)
    token_tensor = torch.from_numpy(tokens).unsqueeze(0).long().to(device)
    feat_tensor = torch.from_numpy(feats).unsqueeze(0).float().to(device)
    return token_tensor, feat_tensor


def model_q(model, env, player: int, phase: str, device: torch.device) -> np.ndarray:
    tokens, feats = encode_tensors(env, player, device)
    with torch.inference_mode():
        q = model(tokens, feats, phase=phase)[0].float().cpu().numpy()
    return q


def choose_bid(env, model, settings: PlaySettings, device) -> Tuple[int, Decision]:
    player = int(env.current_player)
    legal = mask_to_bool(int(env.legal_bids_mask(player)), 11)
    start = time.perf_counter()
    if model is None:
        bid = heuristic_bid(legal, int(env.round_cards))
        decision = Decision("heuristic")
    elif settings.use_search and int(env.round_cards) == 1:
        q = model_q(model, env, player, "bid", device)
        ev = np.asarray(env.search_pannplump_bid_values(), dtype=np.float32)
        combined = 0.5 * normalized_prior(q, legal) + 0.5 * normalized_prior(
            ev, legal
        )
        bid = int(np.argmax(combined))
        decision = Decision(
            "pannplump + bid head", extra=f"q={q[bid]:.2f}, ev={ev[bid]:.2f}"
        )
    else:
        q = model_q(model, env, player, "bid", device)
        bid = int(masked_argmax(q[None, :], legal[None, :])[0])
        decision = Decision("bid head", extra=f"q={q[bid]:.2f}")
    decision.elapsed_ms = (time.perf_counter() - start) * 1000.0
    return bid, decision


def choose_card(env, model, settings: PlaySettings, device) -> Tuple[int, Decision]:
    player = int(env.current_player)
    legal = mask_to_bool(int(env.legal_cards_mask(player)), 52)
    start = time.perf_counter()
    if int(legal.sum()) == 1:
        card = int(np.flatnonzero(legal)[0])
        decision = Decision("only legal card")
        decision.elapsed_ms = (time.perf_counter() - start) * 1000.0
        return card, decision
    if model is None:
        bid = int(env.bids[player])
        tricks = int(env.tricks_won[player])
        card = heuristic_play(legal, bid, tricks)
        decision = Decision("heuristic")
    elif settings.use_search:
        tokens, feats = encode_tensors(env, player, device)
        with torch.inference_mode():
            q_out, belief_logits, _ = model.forward_with_aux(
                tokens, feats, phase="play"
            )
            q = q_out[0].float().cpu().numpy()
            belief = torch.sigmoid(belief_logits[0]).float().cpu().numpy()
        belief = np.ascontiguousarray(belief, dtype=np.float32)
        ev = np.asarray(
            env.search_card_values(settings.worlds, belief), dtype=np.float32
        )
        state = env.state()
        tricks_left = int(env.round_cards) - int(state["current_trick"])
        if tricks_left <= 5:
            q_weight, ev_weight = 0.25, 0.75
            mode = "endgame"
        else:
            q_weight, ev_weight = 0.50, 0.50
            mode = "midgame"
        combined = q_weight * normalized_prior(
            q, legal
        ) + ev_weight * normalized_prior(ev, legal)
        card = int(np.argmax(np.where(legal, combined, -np.inf)))
        decision = Decision(
            f"PIMC {settings.worlds}w + play head",
            extra=f"q={q[card]:.2f} ev={ev[card]:.2f} {mode}",
        )
    else:
        q = model_q(model, env, player, "play", device)
        card = int(masked_argmax(q[None, :], legal[None, :])[0])
        decision = Decision("play head", extra=f"q={q[card]:.2f}")
    decision.elapsed_ms = (time.perf_counter() - start) * 1000.0
    return card, decision


def append_plain_card(text: Text, card: int) -> None:
    suit = card // 13
    style = "bold red" if suit in (0, 2) else "bold cyan"
    text.append(f"{SUITS[suit]}{RANK_LABELS[card % 13 + 2]}", style=style)


def render_hand(console: Console, hand) -> None:
    text = Text()
    for suit in range(4):
        cards = sorted(
            [c for c in hand if c // 13 == suit], key=lambda c: c % 13, reverse=True
        )
        if not cards:
            continue
        style = "bold red" if suit in (0, 2) else "bold cyan"
        text.append(f"{SUITS[suit]}  ", style=style)
        for card in cards:
            text.append(f"{RANK_LABELS[card % 13 + 2]}  ", style=style)
        text.append("\n")
    if not hand:
        text.append("(empty)", style="dim")
    console.print(
        Panel(text, title=f"Your hand {DASH} Seat 0", border_style="cyan")
    )


def render_forehead(console: Console, state, reveal: bool) -> None:
    text = Text()
    own = state["hands"][0]
    if reveal and own:
        text.append("Your card: ", style="bold")
        append_plain_card(text, int(own[0]))
        text.append("  ")
        text.append("(revealed!)", style="bold magenta")
        text.append("\n")
    elif reveal:
        text.append("Your card: (already played)\n", style="dim")
    else:
        text.append("Your card: ", style="bold")
        text.append("??", style="bold magenta")
        text.append("  (on your forehead!)\n", style="dim")
    text.append("Opponents' foreheads:\n", style="bold")
    for player in (1, 2, 3):
        hand = state["hands"][player]
        text.append(f"  {SEAT_NAMES[player]}: ")
        if hand:
            for card in hand:
                append_plain_card(text, int(card))
                text.append(" ")
        else:
            text.append(DASH, style="dim")
        text.append("\n")
    console.print(
        Panel(
            text, title=f"Pannplump {DASH} forehead round", border_style="magenta"
        )
    )


def illegal_reason(env, card: int, hand) -> str:
    state = env.state()
    if card not in hand:
        return "that card is not in your hand"
    if state["led_suit"] == 255:
        return "you are leading, so any card is legal"
    led = int(state["led_suit"])
    if card // 13 == led:
        return "that card is legal"
    if any(c // 13 == led for c in hand):
        return f"you must follow {SUITS[led]}"
    return "you are void in the led suit"


def render_state(
    console: Console,
    env,
    totals,
    plumps,
    round_no: int,
    settings: PlaySettings,
    last_trick: Optional[str],
) -> None:
    state = env.state()
    leader = int(state["leader"])
    leader_bid = int(state["bids"][leader])
    dealer = (int(state["start_bidder"]) + 3) % 4
    contract = ""
    if state["phase"] == "Playing":
        contract = f"  contract: {SEAT_NAMES[leader]} ({bid_text(leader_bid)})"
    console.print(
        f"[bold]Round {round_no}/{len(ROUND_SCHEDULE)}[/]  "
        f"{state['round_cards']} cards  dealer: {SEAT_NAMES[dealer]}  "
        f"phase: {state['phase']}{contract}"
    )

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("Player")
    table.add_column("Bid", justify="right")
    table.add_column("Tricks Won", justify="right")
    table.add_column("Total Plumps", justify="right")
    table.add_column("Total Score", justify="right")
    for player in range(4):
        active = (
            player == int(state["current_player"])
            and state["phase"] != "Terminal"
        )
        name = Text(SEAT_NAMES[player], style="bold cyan" if active else "")
        bid = int(state["bids"][player])
        tricks = int(state["tricks_won"][player])
        table.add_row(
            name,
            bid_text(bid),
            str(tricks),
            str(plumps[player]),
            str(totals[player]),
        )
    console.print(table)

    if int(state["round_cards"]) == 1:
        render_forehead(console, state, reveal=state["phase"] == "Playing")
    else:
        render_hand(console, state["hands"][0])

    if state["table"]:
        trick = "   ".join(
            f"{SEAT_NAMES[int(player)]} {card_markup(int(card))}"
            for player, card in state["table"]
        )
    else:
        trick = f"[dim]{DASH}[/dim]"
    console.print(f"[bold]Current trick:[/] {trick}")
    if last_trick:
        console.print(f"[bold]Last trick:[/] {last_trick}")
    if state["phase"] == "Playing" and state["led_suit"] != 255:
        suit = int(state["led_suit"])
        console.print(f"[bold]Led suit:[/] {SUITS[suit]}")

    search_state = "on" if settings.use_search else "off"
    worlds = f" ({settings.worlds} worlds)" if settings.use_search else ""
    console.print(
        f"[dim]AI search: {search_state}{worlds} {DASH} "
        "commands: search on/off, worlds N, help, quit[/dim]"
    )


def human_bid(console: Console, env, settings: PlaySettings) -> int:
    state = env.state()
    legal = mask_to_bool(int(env.legal_bids_mask(0)), 11)
    options = ", ".join(str(bid) for bid in range(11) if legal[bid])
    console.print(f"[bold]Legal bids:[/] {options}   [dim]0 = pass[/dim]")
    forbidden = env.forbidden_bid()
    while True:
        try:
            raw = console.input("[bold cyan]Your bid[/] (0=pass): ")
        except EOFError:
            raise QuitGame
        if handle_command(raw, settings, console):
            continue
        text = raw.strip().lower()
        if text in ("", "p", "pass"):
            bid = 0
        elif text.isdigit():
            bid = int(text)
        else:
            console.print("[red]Enter a bid number, 'pass', or a command.[/red]")
            continue
        if bid < 0 or bid > int(state["round_cards"]):
            console.print(
                f"[red]Bid must be between 0 and {state['round_cards']}.[/red]"
            )
            continue
        if not legal[bid]:
            if forbidden == bid:
                console.print(
                    f"[red]Bid {bid} is forbidden for the dealer.[/red]"
                )
            else:
                console.print(f"[red]Bid {bid} is not legal.[/red]")
            continue
        return bid


def human_card(console: Console, env, settings: PlaySettings) -> int:
    state = env.state()
    hand = state["hands"][0]
    legal = mask_to_bool(int(env.legal_cards_mask(0)), 52)
    legal_cards = [c for c in range(52) if legal[c]]
    console.print(
        "[bold]Legal cards:[/] "
        + "  ".join(card_markup(card) for card in legal_cards)
        + "   [dim](e.g. HA, S10)[/dim]"
    )
    illegal = [c for c in hand if not legal[c]]
    if illegal:
        reason = illegal_reason(env, illegal[0], hand)
        console.print(
            "[dim]Illegal: "
            + "  ".join(card_code(card) for card in illegal)
            + f" ({reason})[/dim]"
        )
    while True:
        try:
            raw = console.input("[bold cyan]Your card[/] (e.g. HA): ")
        except EOFError:
            raise QuitGame
        if handle_command(raw, settings, console):
            continue
        card = parse_card(raw)
        if card is None:
            console.print(
                "[red]Use suit+rank, e.g. HA, SK, D10, C7.[/red]"
            )
            continue
        if card not in hand:
            console.print(f"[red]{card_code(card)} is not in your hand.[/red]")
            continue
        if not legal[card]:
            console.print(
                f"[red]{card_markup(card)} is illegal: "
                f"{illegal_reason(env, card, hand)}.[/red]"
            )
            continue
        return card


def round_summary_line(env, round_no: int, round_pts) -> str:
    state = env.state()
    bids = " ".join(
        f"{SEAT_NAMES[p]}:{bid_text(int(state['bids'][p]))}" for p in range(4)
    )
    tricks = " ".join(
        f"{SEAT_NAMES[p]}:{state['tricks_won'][p]}" for p in range(4)
    )
    points = " ".join(
        f"{SEAT_NAMES[p]}:{round_pts[p]:+d}" for p in range(4)
    )
    leader = int(env.get_leader())
    return (
        f"[bold]Round {round_no:02d}[/] {state['round_cards']}c | "
        f"{bids} | tricks {tricks} | pts {points} | "
        f"lead {SEAT_NAMES[leader]}({bid_text(int(state['bids'][leader]))})"
    )


def round_status(bid: int, tricks: int) -> str:
    if bid < 0:
        return f"[dim]{DASH}[/dim]"
    if tricks == bid:
        return "[green]MADE[/green]" if bid > 0 else "[green]PASSED[/green]"
    return "[red]PLUMP[/red]"


def show_round_result(
    console: Console, env, round_no: int, round_pts, totals, plumps
) -> None:
    state = env.state()
    table = Table(title=f"Round {round_no} result", box=box.ROUNDED)
    table.add_column("Player")
    table.add_column("Bid", justify="right")
    table.add_column("Tricks Won", justify="right")
    table.add_column("Status")
    table.add_column("Round Points", justify="right")
    table.add_column("Total Plumps", justify="right")
    table.add_column("Total Score", justify="right")
    for player in range(4):
        bid = int(state["bids"][player])
        tricks = int(state["tricks_won"][player])
        table.add_row(
            SEAT_NAMES[player],
            bid_text(bid),
            str(tricks),
            round_status(bid, tricks),
            f"{round_pts[player]:+d}",
            str(plumps[player]),
            str(totals[player]),
        )
    console.print(table)


def show_final(console: Console, totals, plumps) -> None:
    best = max(totals)
    winners = [SEAT_NAMES[p] for p in range(4) if totals[p] == best]
    table = Table(title="Final scoreboard", box=box.DOUBLE)
    table.add_column("Player")
    table.add_column("Total Plumps", justify="right")
    table.add_column("Total Score", justify="right")
    for player in range(4):
        style = "bold green" if totals[player] == best else ""
        table.add_row(
            Text(SEAT_NAMES[player], style=style),
            str(plumps[player]),
            str(totals[player]),
        )
    console.print(table)
    console.print(
        Panel.fit(
            f"[bold green]{', '.join(winners)} wins with {best} points![/bold green]",
            border_style="green",
        )
    )


def play(
    console: Console,
    settings: PlaySettings,
    model,
    device,
    seed: Optional[int],
    max_rounds: int,
) -> None:
    env = plump_engine.PyFastEnv(seed)
    totals = [0, 0, 0, 0]
    plumps = [0, 0, 0, 0]
    total_rounds = min(max_rounds, len(ROUND_SCHEDULE))

    for round_index in range(total_rounds):
        round_cards = ROUND_SCHEDULE[round_index]
        start_bidder = round_index % 4
        dealer = (start_bidder + 3) % 4
        env.new_round(round_cards, start_bidder)
        last_trick: Optional[str] = None

        if not settings.autoplay:
            console.rule(
                f"[bold]Round {round_index + 1}/{total_rounds}[/] {DASH} "
                f"{round_cards} cards {DASH} dealer: {SEAT_NAMES[dealer]} {DASH} "
                f"first bidder: {SEAT_NAMES[start_bidder]}"
            )
        else:
            console.print(
                f"[dim]Round {round_index + 1:02d}/{total_rounds} {DASH} "
                f"{round_cards} cards {DASH} dealer {SEAT_NAMES[dealer]}[/dim]"
            )

        while env.phase == "Bidding":
            player = int(env.current_player)
            if int(env.bid_offset(player)) == 3 and not settings.autoplay:
                forbidden = env.forbidden_bid()
                if forbidden is not None:
                    console.print(
                        f"[yellow]Gå går inte: {SEAT_NAMES[player]} cannot bid "
                        f"{forbidden} (bids would sum to {round_cards}).[/yellow]"
                    )
            if player == 0 and not settings.autoplay:
                render_state(
                    console, env, totals, plumps, round_index + 1, settings, last_trick
                )
                bid = human_bid(console, env, settings)
                decision = None
            else:
                bid, decision = choose_bid(env, model, settings, device)
            env.step_bid(bid)
            if not settings.autoplay:
                console.print(
                    f"{SEAT_NAMES[player]} bids {bid_text(bid)} "
                    f"{decision_suffix(decision)}"
                )

        leader = int(env.get_leader())
        if not settings.autoplay:
            console.print(
                f"[bold]Highest bid {int(env.bids[leader])} by "
                f"{SEAT_NAMES[leader]} (earliest in bidding order) "
                "leads Trick 1.[/bold]"
            )

        if round_cards == 1:
            state = env.state()
            own = state["hands"][0]
            if own:
                console.print(
                    f"[bold magenta]Forehead reveal![/] {SEAT_NAMES[0]} was "
                    f"holding {card_markup(int(own[0]))}."
                )

        trick_number = 0
        while env.phase == "Playing":
            state = env.state()
            table_before = list(state["table"])
            trick_number = int(state["current_trick"]) + 1
            player = int(env.current_player)
            if player == 0 and not settings.autoplay:
                render_state(
                    console, env, totals, plumps, round_index + 1, settings, last_trick
                )
                card = human_card(console, env, settings)
                decision = None
            else:
                card, decision = choose_card(env, model, settings, device)

            winner = env.step_play(card)
            if winner is not None:
                trick = "  ".join(
                    f"{SEAT_NAMES[int(p)]} {card_markup(int(c))}"
                    for p, c in table_before + [(player, card)]
                )
                last_trick = f"{trick}  {ARROW}  {SEAT_NAMES[winner]}"
            if not settings.autoplay:
                console.print(
                    f"{SEAT_NAMES[player]} plays {card_markup(card)} "
                    f"{decision_suffix(decision)}"
                )
                if winner is not None:
                    console.print(
                        f"  [green]{SEAT_NAMES[winner]} wins trick {trick_number}[/green]"
                    )

        round_pts = [int(env.display_points_for(p)) for p in range(4)]
        for player in range(4):
            bid = int(env.bids[player])
            tricks = int(env.tricks_won[player])
            if bid >= 0 and tricks != bid:
                plumps[player] += 1
            totals[player] += round_pts[player]

        if settings.autoplay:
            console.print(round_summary_line(env, round_index + 1, round_pts))
        else:
            show_round_result(
                console, env, round_index + 1, round_pts, totals, plumps
            )

    show_final(console, totals, plumps)


def load_ai_model(args, console: Console, device: torch.device):
    if args.no_model:
        console.print("[yellow]Model disabled; AI uses the heuristic fallback.[/yellow]")
        return None
    if args.model is not None:
        path = Path(args.model)
    elif BEST_CHECKPOINT.exists():
        path = BEST_CHECKPOINT
    elif LATEST_CHECKPOINT.exists():
        path = LATEST_CHECKPOINT
    else:
        console.print(
            "[yellow]No checkpoint found (plump_transformer_best.pt / "
            "plump_transformer_latest.pt). AI uses the heuristic fallback.[/yellow]"
        )
        return None
    if not path.exists():
        console.print(f"[yellow]Checkpoint {path} does not exist; using heuristic.[/yellow]")
        return None
    try:
        model, _ = load_transformer(path, device)
    except Exception as exc:
        console.print(f"[red]Failed to load {path}: {exc}[/red]")
        console.print("[yellow]AI uses the heuristic fallback.[/yellow]")
        return None
    console.print(
        f"[green]Loaded transformer:[/] {path.name} on {device}",
        highlight=False,
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Swedish Plump")
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--search", type=int, default=32)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=len(ROUND_SCHEDULE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = load_ai_model(args, console, device)
    settings = PlaySettings(
        autoplay=args.autoplay,
        use_search=args.search > 0 and model is not None,
        worlds=args.search if args.search > 0 else 32,
        model_available=model is not None,
    )

    if args.autoplay:
        search_desc = (
            f"on ({settings.worlds} worlds)"
            if settings.use_search
            else "off (policy head)"
        )
        console.print(
            Panel.fit(
                f"[bold]Autoplay[/] {DASH} 4 AI players {DASH} "
                f"search {search_desc} {DASH} model "
                f"{'loaded' if model is not None else 'heuristic'}",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel.fit(
                "[bold]Swedish Plump[/]\n"
                "20 rounds: 10 cards down to 1, then back up to 10.\n"
                "Bid exactly your tricks to score. Miss and it is a plump.\n"
                "You are Seat 0. Type 'help' for commands.",
                title="Welcome",
                border_style="green",
            )
        )

    try:
        play(
            console,
            settings,
            model,
            device,
            args.seed,
            max(1, min(args.max_rounds, len(ROUND_SCHEDULE))),
        )
    except QuitGame:
        console.print("[dim]Left the table.[/dim]")
    except KeyboardInterrupt:
        console.print("\n[dim]Left the table.[/dim]")


if __name__ == "__main__":
    main()
