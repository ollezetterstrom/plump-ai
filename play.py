import os
import torch
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import IntPrompt
from rich import box

from plump_env import (
    PlumpEnv,
    encode_state_278,
    encode_state_294,
    mask_actions,
    INDEX_TO_CARD,
    CARD_INDEX,
)
from train import DuelingQNetwork

console = Console()

SUIT_SYMBOL = {"H": "♥", "D": "♦", "S": "♠", "C": "♣"}
SUIT_COLOR = {"H": "red bold", "D": "red bold", "S": "blue bold", "C": "blue bold"}
PLAYER_NAMES = ["You", "AI 1", "AI 2", "AI 3"]
PLAYER_COLORS = ["green", "cyan", "magenta", "yellow"]

ENCODERS = {
    278: encode_state_278,
    294: encode_state_294,
}


def card_rich(card):
    suit, rank = card
    sym = SUIT_SYMBOL.get(suit, suit)
    color = SUIT_COLOR.get(suit, "white")
    
    if rank == 14: rank_str = "A"
    elif rank == 13: rank_str = "K"
    elif rank == 12: rank_str = "Q"
    elif rank == 11: rank_str = "J"
    else: rank_str = str(rank)
    
    return f"[{color}]{rank_str}{sym}[/]"


def load_model(base_name, output_size, device):
    """Try checkpoint paths in priority order; auto-detect input dim from weights."""
    paths = [
        f"{base_name}_interrupted_v2.pt",
        f"{base_name}_interrupted.pt",
        f"{base_name}_latest_v2.pt",
        f"{base_name}_latest.pt",
        f"{base_name}_best_v2.pt",
        f"{base_name}_best.pt",
        f"{base_name}_champion.pt",
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location=device, weights_only=True)
            except TypeError:
                # Fallback for older PyTorch versions that don't support weights_only
                state_dict = torch.load(path, map_location=device)
                
            input_size = state_dict["features.0.weight"].shape[1]
            model = DuelingQNetwork(input_size, output_size)
            model.load_state_dict(state_dict)
            console.print(
                f"  [green]✓[/] Loaded [dim]{path}[/]  (input_dim={input_size})"
            )
            return model, input_size

    raise FileNotFoundError(f"Could not find weights for {base_name}")


def get_encoder(input_size):
    if input_size in ENCODERS:
        return ENCODERS[input_size]
    available = ", ".join(str(k) for k in ENCODERS)
    raise ValueError(
        f"No encoder for input_size={input_size}.  "
        f"Available encoders produce: [{available}].  "
        f"Add an encode_state_{input_size} function to plump_env.py and import it."
    )


def ai_action(model, encoder, env, player, phase):
    state = encoder(env, player)
    legal = mask_actions(env, player, phase)
    legal_t = torch.from_numpy(np.array(legal, dtype=np.bool_))
    state_t = torch.from_numpy(np.array(state, dtype=np.float32))
    with torch.no_grad():
        q = model(state_t)
    q[~legal_t] = -1e9
    action = q.argmax().item()
    if phase == "bid":
        return action
    return INDEX_TO_CARD[action]


def play_against_ai():
    device = torch.device("cpu")

    with console.status("[bold cyan]Loading AI models…[/]"):
        bid_model, bid_input = load_model("plump_bid_model", 11, device)
        play_model, play_input = load_model("plump_play_model", 52, device)

    bid_encoder = get_encoder(bid_input)
    play_encoder = get_encoder(play_input)

    bid_model.eval()
    play_model.eval()

    env = PlumpEnv()
    total_scores = [0] * 4
    rounds_to_play = list(range(10, 0, -1))

    console.print()
    console.print(
        Panel(
            Text.from_markup(
                "🃏 [bold bright_white]P L U M P[/]\n"
                "[dim]Human vs 3 AI Champions — Full Game[/]"
            ),
            box=box.HEAVY,
            border_style="bright_blue",
            padding=(1, 6),
        )
    )

    for round_idx, round_cards in enumerate(rounds_to_play):
        env.new_round(round_cards=round_cards)

        console.print()
        console.rule(
            f"[bold yellow]Round {round_idx + 1}[/] · {round_cards} card{'s' if round_cards != 1 else ''}",
            style="dim yellow",
        )

        # ── Hand ────────────────────────────────────────────────────
        hand_sorted = sorted(env.hands[0], key=lambda c: (c[0], c[1]))
        hand_str = "  ".join(card_rich(c) for c in hand_sorted)
        console.print(f"\n[b]Your Hand:[/]  {hand_str}")

        # ── Bidding ─────────────────────────────────────────────────
        console.print(f"\n[bold]📋 Bidding Phase[/]")

        for turn in range(4):
            if turn == 0:
                while True:
                    bid = IntPrompt.ask(
                        f"  [{PLAYER_COLORS[0]}]Your bid[/] [dim](0–{round_cards})[/]",
                        console=console,
                    )
                    legal_mask = mask_actions(env, 0, "bid")
                    if 0 <= bid <= round_cards and legal_mask[bid]:
                        break
                    console.print("  [red]✗ Illegal bid. Try again.[/]")
            else:
                bid = ai_action(bid_model, bid_encoder, env, turn, "bid")
                console.print(
                    f"  [{PLAYER_COLORS[turn]}]{PLAYER_NAMES[turn]}[/] bids [bold]{bid}[/]"
                )
            env.bids[turn] = bid

        # Bids summary
        bid_tbl = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 3))
        for p in range(4):
            bid_tbl.add_column(
                PLAYER_NAMES[p], style=PLAYER_COLORS[p], justify="center"
            )
        bid_tbl.add_row(*[str(env.bids[p]) for p in range(4)])
        console.print(bid_tbl)

        # ── Playing ─────────────────────────────────────────────────
        # House rule: earliest highest bidder leads
        max_bid = max(env.bids)
        current_player = next(i for i, b in enumerate(env.bids) if b == max_bid)
        console.print(f"  [dim]Highest bid {max_bid} by {PLAYER_NAMES[current_player]} leads[/]")

        for trick in range(round_cards):
            console.print(f"\n  [bold dim]── Trick {trick + 1}/{round_cards} ──[/]")

            for _ in range(4):
                if current_player == 0:
                    if env.table:
                        tbl_str = " ".join(card_rich(c) for _, c in env.table)
                        console.print(f"    [dim]Table:[/] {tbl_str}")
                    if env.led_suit:
                        led_sym = SUIT_SYMBOL.get(env.led_suit, env.led_suit)
                        led_color = SUIT_COLOR.get(env.led_suit, "white")
                        console.print(f"    [dim]Led suit:[/] [{led_color}]{led_sym}[/]")

                    legal_mask = mask_actions(env, 0, "play")
                    my_hand = env.hands[0]

                    ct = Table(box=None, show_header=False, padding=(0, 1))
                    ct.add_column(justify="right", style="dim")
                    ct.add_column()
                    ct.add_column(justify="center", width=6)

                    for idx, card in enumerate(my_hand):
                        is_legal = legal_mask[CARD_INDEX[card]]
                        if is_legal:
                            ct.add_row(str(idx), card_rich(card), "[green]✔[/]")
                        else:
                            ct.add_row(str(idx), f"[dim]{card_rich(card)}[/]", "[red]✗[/]")

                    console.print(ct)

                    while True:
                        choice = IntPrompt.ask(
                            "    [bold]Play[/]", console=console
                        )
                        if 0 <= choice < len(my_hand):
                            card = my_hand[choice]
                            if legal_mask[CARD_INDEX[card]]:
                                break
                        console.print("    [red]✗ Illegal choice — follow suit![/]")

                    console.print(f"    [bold]→[/] You played {card_rich(card)}")
                    env.play_card(0, card)
                else:
                    card = ai_action(play_model, play_encoder, env, current_player, "play")
                    console.print(
                        f"    [bold]→[/] [{PLAYER_COLORS[current_player]}]"
                        f"{PLAYER_NAMES[current_player]}[/] plays {card_rich(card)}"
                    )
                    env.play_card(current_player, card)

                current_player = (current_player + 1) % 4

            winner, _ = env.resolve_trick()
            console.print(
                f"    🏆 [{PLAYER_COLORS[winner]}]{PLAYER_NAMES[winner]}[/] "
                f"takes the trick"
            )
            current_player = winner

        # ── Round Scoring ───────────────────────────────────────────
        rt = Table(
            title=f"Round {round_idx + 1} Results",
            box=box.ROUNDED,
            show_lines=True,
            title_style="bold",
        )
        rt.add_column("Player", style="bold")
        rt.add_column("Bid", justify="center")
        rt.add_column("Won", justify="center")
        rt.add_column("Result", justify="center")
        rt.add_column("Points", justify="center")
        rt.add_column("Total", justify="center", style="bold")

        for p in range(4):
            passed = env.tricks_won[p] == env.bids[p]
            if passed:
                pts = (5 if env.bids[p]==0 else 10 + env.tricks_won[p])
                total_scores[p] += pts
                result, pts_s = "[green]PASSED[/]", f"[green]+{pts}[/]"
            else:
                result, pts_s = "[red]PLUMPED[/]", "[red]+0[/]"

            rt.add_row(
                f"[{PLAYER_COLORS[p]}]{PLAYER_NAMES[p]}[/]",
                str(env.bids[p]),
                str(env.tricks_won[p]),
                result,
                pts_s,
                str(total_scores[p]),
            )

        console.print()
        console.print(rt)

    # ── Final Results ───────────────────────────────────────────────
    max_score = max(total_scores)
    rankings = sorted(range(4), key=lambda p: total_scores[p], reverse=True)

    ft = Table(
        box=box.HEAVY,
        show_lines=True,
        title="🏆  Final Standings  🏆",
        title_style="bold yellow",
    )
    ft.add_column("Rank", justify="center", style="bold")
    ft.add_column("Player", style="bold")
    ft.add_column("Score", justify="center", style="bold")

    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    for rank, p in enumerate(rankings, 1):
        is_champ = total_scores[p] == max_score
        medal = medals.get(rank, "  ")
        name_style = "bold yellow" if is_champ else PLAYER_COLORS[p]
        score_style = "bold yellow" if is_champ else None
        ft.add_row(
            f"{medal} {rank}",
            f"[{name_style}]{PLAYER_NAMES[p]}[/]",
            f"[{score_style}]{total_scores[p]}[/]" if score_style else str(total_scores[p]),
        )

    console.print()
    console.print(Panel(ft, border_style="bright_yellow", padding=(1, 4)))

    winner_idx = total_scores.index(max_score)
    if winner_idx == 0:
        console.print("\n[bold green]🎉 You won the game![/]\n")
    else:
        console.print(f"\n[bold red]💀 {PLAYER_NAMES[winner_idx]} wins the game![/]\n")


if __name__ == "__main__":
    play_against_ai()