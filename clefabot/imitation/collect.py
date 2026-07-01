"""Collect (state, action) pairs for behavior cloning (spec §7).

The teacher is poke-env's SimpleHeuristicsPlayer — a scripted policy that plays
measurably better than random. We record, at every decision, the fixed-shape
observation, the teacher's chosen per-slot action index, and the legality mask.

This is the local, network-independent data path. The replay-log ingestion path
(learning from real human replays via poke-env's protocol parser) lands in
`replay_ingest.py` and slots in once the replay corpus is reachable (blocked by
the current environment network policy — see README / PLAN §0).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import SimpleHeuristicsPlayer

from ..env.actions import (
    N_ACTIVE_SLOTS,
    PASS_ACTION,
    PER_SLOT_ACTIONS,
    encode_order_slot,
    legal_action_mask,
)
from ..env.features import encode_battle
from ..env.players import REG_MB_FORMAT
from ..team.parser import parse_team


class RecordingTeacher(SimpleHeuristicsPlayer):
    """SimpleHeuristicsPlayer that records (obs, action, mask) each decision."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs: list[np.ndarray] = []
        self.actions: list[list[int]] = []
        self.masks: list[np.ndarray] = []

    def choose_move(self, battle):
        order = super().choose_move(battle)
        first = getattr(order, "first_order", None)
        second = getattr(order, "second_order", None)
        if first is not None or second is not None:
            obs = encode_battle(battle)
            mask = legal_action_mask(battle)
            actives = battle.active_pokemon
            switches = battle.available_switches
            slot_actions = []
            for s, sub in enumerate((first, second)):
                if sub is None:
                    slot_actions.append(PASS_ACTION)
                    continue
                active = actives[s] if s < len(actives) else None
                sw = switches[s] if s < len(switches) else []
                slot_actions.append(encode_order_slot(sub, active, sw))
            self.obs.append(obs)
            self.actions.append(slot_actions)
            self.masks.append(mask)
        return order


async def collect(team_text: str, n_battles: int, out_path: str) -> int:
    team = parse_team(team_text).to_showdown_team()
    common = dict(
        battle_format=REG_MB_FORMAT,
        server_configuration=LocalhostServerConfiguration,
        team=team,
        max_concurrent_battles=10,
    )
    teacher = RecordingTeacher(
        account_configuration=AccountConfiguration("Teacher", None), **common)
    sparring = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration("TeacherOpp", None), **common)

    await teacher.battle_against(sparring, n_battles=n_battles)

    obs = np.asarray(teacher.obs, dtype=np.float32)
    actions = np.asarray(teacher.actions, dtype=np.int64)
    masks = np.asarray(teacher.masks, dtype=bool)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, obs=obs, actions=actions, masks=masks)
    print(f"collected {len(obs)} decisions over {n_battles} games -> {out_path}")
    return len(obs)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Collect BC data from a teacher")
    ap.add_argument("--team", default="config/current_team.txt")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--out", default="data/bc_dataset.npz")
    args = ap.parse_args()
    team_text = Path(args.team).read_text()
    asyncio.run(collect(team_text, args.games, args.out))


if __name__ == "__main__":
    main()
