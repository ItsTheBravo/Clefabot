"""Fixed-shape, team-agnostic observation encoding (spec §4, plan §1.3 / M3).

This is the warm-start contract. The observation vector's length is a pure
function of the constants below — never of *which* Pokémon or team is in play —
so a team edit keeps the network's input layer identical and the learned weights
transfer (that is what makes an edit a re-train, not a re-architecture).

Every Pokémon is described by generic features (base stats, computed stats,
typing, status, boosts, item presence, and its move slots), padded to a fixed
number of team slots. Absent slots are zero-filled. Two entirely different legal
teams therefore produce identically shaped observations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from poke_env.battle import (
    DoubleBattle,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonType,
    Status,
    Weather,
)

# --- fixed dimensions ---------------------------------------------------
MAX_TEAM = 6          # slots per side
MOVES_PER_MON = 4     # move slots per Pokémon
TYPES = list(PokemonType)          # fixed ordering (len 20)
STATUSES = list(Status)            # fixed ordering (len 7)
WEATHERS = list(Weather)           # len 9
FIELDS = list(Field)               # len 15
BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")

N_TYPES = len(TYPES)
N_STATUSES = len(STATUSES)

# Per-move feature block: [present, bp/200, accuracy, phys, spec, status,
# priority/5, pp_fraction] + type one-hot.
MOVE_FEATS = 8 + N_TYPES

# Per-Pokémon block.
_MON_SCALARS = 5                     # present, active, fainted, hp_frac, level/100
_MON_STATS = 2 * len(STAT_KEYS)      # base + computed
_MON_MISC = N_TYPES + N_STATUSES + len(BOOST_KEYS) + 1  # types, status, boosts, has_item
MON_FEATS = _MON_SCALARS + _MON_STATS + _MON_MISC + MOVES_PER_MON * MOVE_FEATS

# Global field block.
FIELD_FEATS = len(WEATHERS) + len(FIELDS)

# Total observation size — a pure function of the constants above.
OBS_SIZE = 2 * MAX_TEAM * MON_FEATS + FIELD_FEATS

_TYPE_IDX = {t: i for i, t in enumerate(TYPES)}
_STATUS_IDX = {s: i for i, s in enumerate(STATUSES)}


def _one_hot(idx: Optional[int], n: int) -> list[float]:
    v = [0.0] * n
    if idx is not None and 0 <= idx < n:
        v[idx] = 1.0
    return v


def encode_move(move: Optional[Move]) -> list[float]:
    """Fixed-length feature block for one move slot (zeros if empty)."""
    if move is None:
        return [0.0] * MOVE_FEATS
    acc = move.accuracy if isinstance(move.accuracy, (int, float)) else 1.0
    pp_frac = (move.current_pp / move.max_pp) if getattr(move, "max_pp", 0) else 1.0
    feats = [
        1.0,
        min((move.base_power or 0) / 200.0, 1.0),
        float(acc),
        1.0 if move.category == MoveCategory.PHYSICAL else 0.0,
        1.0 if move.category == MoveCategory.SPECIAL else 0.0,
        1.0 if move.category == MoveCategory.STATUS else 0.0,
        (move.priority or 0) / 5.0,
        float(pp_frac),
    ]
    feats += _one_hot(_TYPE_IDX.get(move.type), N_TYPES)
    return feats


def encode_pokemon(mon: Optional[Pokemon], *, is_active: bool = False) -> list[float]:
    """Fixed-length feature block for one team slot (zeros if empty)."""
    if mon is None:
        return [0.0] * MON_FEATS

    hp_frac = mon.current_hp_fraction if mon.current_hp_fraction is not None else 0.0
    feats = [
        1.0,
        1.0 if is_active else 0.0,
        1.0 if mon.fainted else 0.0,
        float(hp_frac),
        (mon.level or 50) / 100.0,
    ]

    # Base stats (always known from the dex).
    feats += [(mon.base_stats.get(k, 0) or 0) / 255.0 for k in STAT_KEYS]
    # Computed stats (known for our own mons; may be absent for opponents).
    stats = mon.stats or {}
    feats += [((stats.get(k) or 0) / 255.0) for k in STAT_KEYS]

    # Typing (1 or 2 types).
    type_vec = [0.0] * N_TYPES
    for t in mon.types:
        if t is not None and t in _TYPE_IDX:
            type_vec[_TYPE_IDX[t]] = 1.0
    feats += type_vec

    # Status + boosts + item presence.
    feats += _one_hot(_STATUS_IDX.get(mon.status), N_STATUSES) if mon.status else [0.0] * N_STATUSES
    boosts = mon.boosts or {}
    feats += [(boosts.get(k, 0) or 0) / 6.0 for k in BOOST_KEYS]
    feats.append(1.0 if mon.item else 0.0)

    # Move slots (padded to MOVES_PER_MON).
    moves = list(mon.moves.values()) if mon.moves else []
    for i in range(MOVES_PER_MON):
        feats += encode_move(moves[i] if i < len(moves) else None)

    return feats


def _encode_side(team: dict, active: list) -> list[float]:
    """Encode one side's team (dict of mons) padded to MAX_TEAM slots."""
    active_ids = {id(m) for m in active if m is not None}
    mons = list(team.values()) if team else []
    feats: list[float] = []
    for i in range(MAX_TEAM):
        mon = mons[i] if i < len(mons) else None
        feats += encode_pokemon(mon, is_active=(mon is not None and id(mon) in active_ids))
    return feats


def encode_battle(battle: DoubleBattle) -> np.ndarray:
    """Encode a full DoubleBattle state into a fixed-length float32 vector.

    The returned length always equals ``OBS_SIZE`` regardless of the teams
    involved (the M3 invariant).
    """
    feats: list[float] = []
    feats += _encode_side(battle.team, battle.active_pokemon)
    feats += _encode_side(battle.opponent_team, battle.opponent_active_pokemon)

    weather_keys = set(battle.weather.keys()) if battle.weather else set()
    feats += [1.0 if w in weather_keys else 0.0 for w in WEATHERS]
    field_keys = set(battle.fields.keys()) if battle.fields else set()
    feats += [1.0 if f in field_keys else 0.0 for f in FIELDS]

    arr = np.asarray(feats, dtype=np.float32)
    assert arr.shape[0] == OBS_SIZE, f"obs {arr.shape[0]} != OBS_SIZE {OBS_SIZE}"
    return arr
