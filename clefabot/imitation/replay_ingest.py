"""Replay-log ingestion path for imitation learning (spec §7).

Spec §7 calls for learning from real human replays by feeding saved replay log
lines through poke-env's *own* protocol parser — the same parser it uses on a
live socket, pointed at static logs. This module implements that state
reconstruction: given a Showdown replay log, it rebuilds the DoubleBattle turn by
turn and emits the fixed-shape observation at each turn boundary.

Status: the external human-replay corpus for gen9championsvgc2026regmb is not
reachable from the current environment (network policy blocks
replay.pokemonshowdown.com — see README / PLAN §0), so this path is exercised
against locally generated logs for now. Producing *labeled* (state, action)
pairs from third-party replays additionally requires reconstructing each
player's legal choice set (the private request), which replays don't include;
until the corpus is available the local teacher path (collect.py) supplies the
BC labels. This module gives the state-reconstruction half so the corpus slots
in with minimal extra work.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from poke_env.battle import DoubleBattle

from ..env.features import encode_battle

_LOGGER = logging.getLogger("clefabot.replay_ingest")


def iter_states_from_log(log_text: str, *, gen: int = 9,
                         battle_tag: str = "replay",
                         username: str = "p1") -> Iterator[np.ndarray]:
    """Yield the fixed-shape observation at each turn boundary of a replay log.

    Lines are fed through DoubleBattle.parse_message exactly as poke-env does for
    live battles. Malformed / unsupported lines are skipped rather than fatal, so
    partial logs still yield the states they can.
    """
    battle = DoubleBattle(battle_tag, username, _LOGGER, gen=gen)
    last_turn = -1
    for raw in log_text.splitlines():
        line = raw.rstrip("\n")
        if not line.startswith("|"):
            continue
        split = line.split("|")
        try:
            battle.parse_message(split)
        except Exception:
            # Some message types touch state we didn't fully initialize from a
            # bare log; skip them rather than abort the whole replay.
            continue
        if battle.turn != last_turn:
            last_turn = battle.turn
            try:
                yield encode_battle(battle)
            except Exception:
                continue


def states_from_log(log_text: str, **kwargs) -> list[np.ndarray]:
    return list(iter_states_from_log(log_text, **kwargs))
