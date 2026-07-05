"""Evaluate a trained net head-to-head (spec §7 acceptance / M4).

`BCPlayer` drives battles with the shared PolicyValueNet. Evaluation pits it
against a RandomPlayer (the §7 baseline) and the SimpleHeuristicsPlayer teacher,
team held constant on both sides (as the eval gate will, §9). The M4 acceptance
bar: beat the random baseline clearly.

Order execution goes through poke-env's ``action_to_order`` — the same single
source of truth as the mask and the training labels. ``strict=False`` degrades
an illegal pick to a default order instead of crashing; the masked argmax makes
that path rare, and the rate is reported for transparency.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import Player, RandomPlayer, SimpleHeuristicsPlayer

from ..env.actions import action_to_order, legal_action_mask
from ..env.features import encode_battle
from ..env.players import REG_MB_FORMAT
from ..team.parser import parse_team
from .net import PolicyValueNet


class BCPlayer(Player):
    """Plays using the shared PolicyValueNet (greedy over legal actions)."""

    def __init__(self, *args, net: PolicyValueNet, auto_mega: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.net = net
        self.net.eval()
        # Upgrade plain moves to mega when legal. Heuristic, not gospel: No
        # Guard also lets opponents never miss, and megaing drops Lightning
        # Rod's electric immunity — PPO learns the real timing at M5.
        self.auto_mega = auto_mega
        self.decisions = 0
        self.fallbacks = 0

    def choose_move(self, battle):
        try:
            obs = torch.from_numpy(encode_battle(battle)).float()
            mask = torch.from_numpy(legal_action_mask(battle)).bool()
            action = self.net.act(obs, mask, greedy=True).numpy().astype(np.int64)

            if self.auto_mega:
                # Upgrade a plain move to its mega variant when legal (+20
                # shifts into the mega block); one mega per battle.
                for s in range(len(action)):
                    if 7 <= action[s] <= 26 and mask[s, action[s] + 20]:
                        action[s] += 20
                        break

            # Resolve identical-switch collisions (double-KO turns): keep slot
            # 0's pick, give slot 1 its next-best differing legal action.
            if action[0] == action[1] and 1 <= action[0] <= 6:
                logits, _ = self.net(obs.unsqueeze(0))
                slot1 = logits.squeeze(0)[1].masked_fill(~mask[1], float("-inf"))
                slot1[action[0]] = float("-inf")
                action[1] = int(slot1.argmax())

            self.decisions += 1
            return action_to_order(action, battle, strict=False)
        except Exception:
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
            "fallback_rate": bc.fallbacks / max(1, bc.decisions + bc.fallbacks),
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
