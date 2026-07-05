"""Fixed-shape, team-agnostic action space for doubles (plan §1.3 / M3).

This is a thin adapter over poke-env's native doubles action encoding
(``DoublesEnv``), adopted in place of an earlier hand-rolled 21-action scheme
because the native one additionally covers Mega Evolution / Tera orders (the
hand-rolled scheme deferred gimmicks — untenable for a team built around Mega
Raichu Y) and ships with maintained, battle-tested ``action_to_order`` /
``order_to_action`` / ``get_action_mask`` implementations (verified live:
100% teacher-label-in-mask over sampled games, zero conversion errors).

Per-slot encoding (poke-env's, documented in DoublesEnv.action_to_order;
each move block is 5 consecutive targets in order -2, -1, 0, 1, 2):

    0                : pass
    1..6             : switch to team slot 1..6
    7..26            : move 1..4 x target
    27..46           : move 1..4 x target, mega evolve
    47..66           : move 1..4 x target, z-move   (dead in gen 9, masked out)
    67..86           : move 1..4 x target, dynamax  (dead in gen 9, masked out)
    87..106          : move 1..4 x target, terastallize
    (-2 default / -1 forfeit exist in the scheme but are never emitted here)

The action space stays a pure function of constants — team-agnostic, which is
what keeps a team edit a re-train rather than a re-architecture.
"""

from __future__ import annotations

import numpy as np

from poke_env.battle import DoubleBattle
from poke_env.environment import DoublesEnv

GEN = 9
N_ACTIVE_SLOTS = 2
PER_SLOT_ACTIONS = DoublesEnv.get_action_space_size(GEN)  # 107 for gen 9

# Layout constants for decode/analytics.
PASS_ACTION = 0
SWITCH_BASE = 1          # 1..6
MOVE_BASE = 7            # 7..: 4 moves x 5 targets per gimmick block
TARGETS = (-2, -1, 0, 1, 2)
N_MOVES = 4
N_TARGETS = len(TARGETS)
GIMMICKS = ("none", "mega", "zmove", "dynamax", "tera")

# Re-exported order conversions (single source of truth: poke-env).
action_to_order = DoublesEnv.action_to_order
order_to_action = DoublesEnv.order_to_action


def action_space_shape() -> tuple[int, int]:
    """MultiDiscrete shape: (n_active_slots, per_slot_actions)."""
    return (N_ACTIVE_SLOTS, PER_SLOT_ACTIONS)


def legal_action_mask(battle: DoubleBattle) -> np.ndarray:
    """(N_ACTIVE_SLOTS, PER_SLOT_ACTIONS) boolean legality mask."""
    flat = np.array(DoublesEnv.get_action_mask(battle), dtype=bool)
    return flat.reshape(N_ACTIVE_SLOTS, PER_SLOT_ACTIONS)


def decode_action(action_index: int) -> dict:
    """Decode one per-slot action index into a structured description."""
    a = int(action_index)
    if a == PASS_ACTION:
        return {"kind": "pass"}
    if a < 0:
        return {"kind": "default" if a == -2 else "forfeit"}
    if SWITCH_BASE <= a < MOVE_BASE:
        return {"kind": "switch", "team_slot": a - SWITCH_BASE}
    rel = a - MOVE_BASE
    block, within = divmod(rel, N_MOVES * N_TARGETS)
    move_slot, target_idx = divmod(within, N_TARGETS)
    return {
        "kind": "move",
        "move_slot": move_slot,
        "target": TARGETS[target_idx],
        "gimmick": GIMMICKS[block],
    }
