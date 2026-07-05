"""Collect (state, action) pairs for behavior cloning (spec §7).

The teacher is poke-env's SimpleHeuristicsPlayer wrapped with one domain rule:
it has no Mega Evolution logic of its own (audited: zero mega handling), so
``MegaTeacher`` sets the mega flag on the stone-holder's move at the first legal
opportunity. For this team that is near-universally correct (Mega Raichu Y is a
strict upgrade); PPO refines the timing later.

Labels use poke-env's own ``order_to_action`` (single source of truth with the
action mask — validated live at 100% label-in-mask), so the dataset speaks the
same 107-per-slot encoding as the mask, the net, and order execution.

The replay-log ingestion path (learning from real human replays) lives in
`replay_ingest.py` and slots in once the replay corpus is reachable (blocked by
the current environment network policy — see README / PLAN §0).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import SimpleHeuristicsPlayer

from ..env.actions import legal_action_mask, order_to_action
from ..env.features import encode_battle
from ..env.players import REG_MB_FORMAT
from ..team.parser import parse_team


class MegaTeacher(SimpleHeuristicsPlayer):
    """SimpleHeuristicsPlayer that mega evolves at the first legal chance."""

    def choose_move(self, battle):
        order = super().choose_move(battle)
        can_mega = getattr(battle, "can_mega_evolve", None)
        if can_mega and hasattr(order, "first_order"):
            from poke_env.battle import Move

            for slot, sub in enumerate((order.first_order, order.second_order)):
                if (
                    sub is not None
                    and slot < len(can_mega)
                    and can_mega[slot]
                    and isinstance(getattr(sub, "order", None), Move)
                ):
                    sub.mega = True
                    break  # only one mega per battle
        return order


class RecordingTeacher(MegaTeacher):
    """MegaTeacher that records (obs, action, mask) at every real decision."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []

    def choose_move(self, battle):
        order = super().choose_move(battle)
        if hasattr(order, "first_order"):
            try:
                action = np.asarray(
                    order_to_action(order, battle, strict=False), dtype=np.int64)
                # Negative actions are default/forfeit sentinels, not real
                # decisions — recording them would poison the BC targets.
                if (action >= 0).all():
                    self.obs.append(encode_battle(battle))
                    self.actions.append(action)
                    self.masks.append(legal_action_mask(battle))
            except Exception:
                pass  # skip unlabelable decisions rather than poison the set
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
    sparring = MegaTeacher(
        account_configuration=AccountConfiguration("TeacherOpp", None), **common)

    await teacher.battle_against(sparring, n_battles=n_battles)

    obs = np.asarray(teacher.obs, dtype=np.float32)
    actions = np.stack(teacher.actions).astype(np.int64)
    masks = np.stack(teacher.masks)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, obs=obs, actions=actions, masks=masks)
    n_mega = int(((actions >= 27) & (actions <= 46)).sum())
    print(f"collected {len(obs)} decisions over {n_battles} games "
          f"({n_mega} mega labels) -> {out_path}")
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
