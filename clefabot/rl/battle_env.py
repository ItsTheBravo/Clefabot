"""Clefabot's doubles RL environment (spec §8 / plan M5).

Subclasses poke-env's ``DoublesEnv`` (PettingZoo two-agent ParallelEnv), which
already provides the async battle plumbing, the native 107-per-slot action
encoding, and the legality mask. We supply:

- ``embed_battle``: the fixed-shape, team-agnostic M3 observation (1992-d)
- ``calc_reward``: terminal win/loss, plus small step-shaping from the HP/faint
  differential. Shaping weights are deliberately small relative to the terminal
  reward (plan §7 tunable) so the value head stays approximately interpretable
  as a win estimate for the analytics layer (§11).

Reward shaping uses battle-state deltas (HP fractions, faints) rather than the
damage calculator per turn — the calculator stays the shared utility for
state features and analytics (§5), while the shaping signal here needs to
reflect what actually happened, which the battle state gives directly.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from gymnasium.spaces import Box

from poke_env.battle import AbstractBattle
from poke_env.environment import DoublesEnv

from ..env.features import OBS_SIZE, encode_battle

# Reward weights (plan §7 tunables).
W_WIN = 1.0          # terminal +/-
W_HP = 0.3           # total swing across a game if HP goes 100% -> 0%
W_FAINT = 0.05       # per faint differential


class ClefabotDoublesEnv(DoublesEnv):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.observation_spaces = {
            agent: Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
            for agent in self.possible_agents
        }
        # Per-battle reward bookkeeping (battle_tag -> last potential).
        self._last_potential: dict[str, float] = {}

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return encode_battle(battle)

    def _potential(self, battle: AbstractBattle) -> float:
        """Dense progress signal: our HP/faints vs theirs, in [-1, 1]-ish."""
        my_hp = sum(m.current_hp_fraction or 0.0 for m in battle.team.values())
        opp_hp = sum(m.current_hp_fraction or 0.0 for m in battle.opponent_team.values())
        my_faints = sum(1 for m in battle.team.values() if m.fainted)
        opp_faints = sum(1 for m in battle.opponent_team.values() if m.fainted)
        # Normalize by brought-team size (4 in VGC).
        n = 4.0
        return (W_HP * (my_hp - opp_hp) / n) + (W_FAINT * (opp_faints - my_faints))

    def calc_reward(self, battle: AbstractBattle) -> float:
        """Difference-of-potential shaping + terminal win/loss."""
        pot = self._potential(battle)
        last = self._last_potential.get(battle.battle_tag, 0.0)
        self._last_potential[battle.battle_tag] = pot
        reward = pot - last
        if battle.finished:
            self._last_potential.pop(battle.battle_tag, None)
            if battle.won:
                reward += W_WIN
            elif battle.lost:
                reward -= W_WIN
        return reward
