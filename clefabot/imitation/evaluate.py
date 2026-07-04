"""Evaluate a trained net head-to-head (spec §7 acceptance / M4).

`BCPlayer` drives battles with the shared PolicyValueNet. Evaluation pits it
against a RandomPlayer (the §7 baseline) and the SimpleHeuristicsPlayer teacher,
team held constant on both sides (as the eval gate will, §9). The M4 acceptance
bar: beat the random baseline clearly.

Doubles order execution: the policy's per-slot choice is turned into a *legal*
order using poke-env's target enumeration + join_orders; if the specific pick is
illegal it falls back to a random legal move for that decision, so play is always
legal. The fallback rate is reported for transparency.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import Player, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.player.battle_order import DoubleBattleOrder, SingleBattleOrder

from ..env.actions import (
    MOVE_ACTIONS,
    decode_action,
    index_to_move_target,
    legal_action_mask,
)
from ..env.features import encode_battle
from ..env.players import REG_MB_FORMAT
from ..team.parser import parse_team
from .net import PolicyValueNet


class BCPlayer(Player):
    """Plays using the shared PolicyValueNet (greedy over legal actions)."""

    def __init__(self, *args, net: PolicyValueNet, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net
        self.net.eval()
        self.decisions = 0
        self.fallbacks = 0

    def _slot_orders(self, battle, slot: int, action_index: int):
        """Return a list of legal SingleBattleOrders for the policy's choice."""
        dec = decode_action(int(action_index))
        active = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
        # No early return on active is None: on force-switch turns the fainted
        # slot has no active mon, yet it is exactly the slot that must switch.
        if dec["kind"] == "switch":
            switches = battle.available_switches[slot] if slot < len(battle.available_switches) else []
            if dec["bench_slot"] < len(switches):
                # Chosen switch first, remaining as alternatives: join_orders
                # rejects both slots switching to the same mon (double-KO
                # turns), and the alternatives let it resolve that collision
                # instead of falling back to a random order.
                chosen = switches[dec["bench_slot"]]
                rest = [SingleBattleOrder(s) for s in switches if s is not chosen]
                return [SingleBattleOrder(chosen), *rest]
            return []
        if dec["kind"] == "move":
            if active is None:
                return []
            moves = battle.available_moves[slot] if slot < len(battle.available_moves) else []
            active_moves = list(active.moves.values()) if active.moves else []
            if dec["move_slot"] >= len(active_moves):
                return []
            move = active_moves[dec["move_slot"]]
            if move not in moves and not any(m.id == move.id for m in moves):
                return []
            targets = battle.get_possible_showdown_targets(move, active)
            if not targets:
                return [SingleBattleOrder(move)]
            desired = index_to_move_target(dec["target"])
            target = desired if desired in targets else targets[0]
            return [SingleBattleOrder(move, move_target=target)]
        return []

    def choose_move(self, battle):
        try:
            obs = torch.from_numpy(encode_battle(battle)).float()
            mask = torch.from_numpy(legal_action_mask(battle)).bool()
            action = self.net.act(obs, mask, greedy=True)  # (n_slots,)
            first = self._slot_orders(battle, 0, action[0])
            second = self._slot_orders(battle, 1, action[1])
            self.decisions += 1
            if first or second:
                combos = DoubleBattleOrder.join_orders(first or [], second or [])
                if combos:
                    return combos[0]
        except Exception:
            pass
        self.fallbacks += 1
        return self.choose_random_move(battle)


def load_net(checkpoint: str) -> PolicyValueNet:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net = PolicyValueNet(obs_size=ckpt["obs_size"])
    net.load_state_dict(ckpt["state_dict"])
    return net


async def evaluate(checkpoint: str, team_text: str, n: int = 100) -> dict:
    team = parse_team(team_text).to_showdown_team()
    net = load_net(checkpoint)
    common = dict(
        battle_format=REG_MB_FORMAT,
        server_configuration=LocalhostServerConfiguration,
        team=team,
        max_concurrent_battles=10,
    )
    results = {}
    for label, make_opp in (
        ("vs_random", lambda: RandomPlayer(
            account_configuration=AccountConfiguration("EvalRand", None), **common)),
        ("vs_heuristic", lambda: SimpleHeuristicsPlayer(
            account_configuration=AccountConfiguration("EvalHeur", None), **common)),
    ):
        bc = BCPlayer(
            account_configuration=AccountConfiguration(f"BC_{label[:6]}", None),
            net=net, **common)
        opp = make_opp()
        await bc.battle_against(opp, n_battles=n)
        wr = bc.n_won_battles / max(1, bc.n_finished_battles)
        results[label] = {
            "win_rate": wr, "games": bc.n_finished_battles,
            "fallback_rate": bc.fallbacks / max(1, bc.decisions),
        }
        print(f"{label}: win_rate={wr:.3f} over {bc.n_finished_battles} "
              f"(fallback={results[label]['fallback_rate']:.2f})")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint (M4)")
    ap.add_argument("--checkpoint", default="checkpoints/bc_baseline.pt")
    ap.add_argument("--team", default="config/current_team.txt")
    ap.add_argument("--games", type=int, default=100)
    args = ap.parse_args()
    team_text = Path(args.team).read_text()
    res = asyncio.run(evaluate(args.checkpoint, team_text, n=args.games))
    if res["vs_random"]["win_rate"] > 0.5:
        print("PASS: beats random baseline (spec §7 acceptance).")
    else:
        print("BELOW BAR: does not beat random baseline yet.")


if __name__ == "__main__":
    main()
