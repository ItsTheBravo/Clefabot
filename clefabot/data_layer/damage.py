"""Shared damage-calculation utility (spec §5, plan M2).

The whole pipeline routes damage math through here so the model never re-derives
it from raw stats, and reward shaping (§5) and analytics (§11) use one source.

Backed by poke-env's built-in `calculate_damage`, which computes from the live
battle's Pokémon (post-Mega stats/typing included). Per its own docstring it
ignores some edge cases, so treat results as close estimates, not the exact
in-game roll — adequate for reward shaping and analytics. Coverage of specific
Reg M-B items should be spot-checked as they come up (M2 open item).
"""

from __future__ import annotations

from typing import Optional

from poke_env.battle import DoubleBattle, Move
from poke_env.calc import calculate_damage


def damage_range(
    attacker_identifier: str,
    defender_identifier: str,
    move: Move,
    battle: DoubleBattle,
    is_critical: bool = False,
) -> Optional[tuple[int, int]]:
    """Return (min, max) damage for a move, or None if it can't be computed."""
    try:
        return calculate_damage(
            attacker_identifier, defender_identifier, move, battle, is_critical
        )
    except (AssertionError, KeyError, ValueError, AttributeError):
        # Unknown mon/move state (e.g. opponent stats not yet revealed) — the
        # caller decides how to treat a missing estimate rather than crashing.
        return None


def expected_damage(*args, **kwargs) -> Optional[float]:
    """Midpoint of the damage range, convenient for reward shaping."""
    rng = damage_range(*args, **kwargs)
    if rng is None:
        return None
    lo, hi = rng
    return (lo + hi) / 2.0
