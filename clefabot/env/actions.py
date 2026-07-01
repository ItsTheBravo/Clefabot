"""Fixed-shape, team-agnostic action space for doubles (plan §1.3 / M3).

Like the observation encoding, the action space is sized by constants, not by the
current team, so the policy head keeps the same width across team edits.

Encoding (per active slot):
    0 .. N_MOVES*N_TARGETS-1   : use move m at target t
    then N_SWITCH              : switch to bench slot s
    then 1                     : pass / default (e.g. slot is fainted)

A doubles action is a pair (one entry per active slot) -> MultiDiscrete of two
identical per-slot spaces. Gimmick use (Mega/Tera) is exposed as a separate
binary flag rather than doubling the discrete space; wiring the flag through to
order execution lands with the Gym env in M5. The legality mask below is what
PPO consumes to avoid illegal actions.
"""

from __future__ import annotations

import numpy as np

from poke_env.battle import DoubleBattle

N_MOVES = 4          # move slots
N_TARGETS = 4        # generic targets: opp_0, opp_1, ally, self
N_SWITCH = 4         # bench switch slots (padded; VGC uses <=2)

MOVE_ACTIONS = N_MOVES * N_TARGETS         # 16
PASS_ACTION = MOVE_ACTIONS + N_SWITCH      # index of the pass action
PER_SLOT_ACTIONS = MOVE_ACTIONS + N_SWITCH + 1   # 21
N_ACTIVE_SLOTS = 2                          # doubles

# Target index -> semantic label (documentation / decode aid).
TARGET_LABELS = ("opp_0", "opp_1", "ally", "self")


def action_space_shape() -> tuple[int, int]:
    """MultiDiscrete shape: (n_active_slots, per_slot_actions)."""
    return (N_ACTIVE_SLOTS, PER_SLOT_ACTIONS)


def _slot_mask(battle: DoubleBattle, slot: int) -> np.ndarray:
    """Boolean legality mask (length PER_SLOT_ACTIONS) for one active slot."""
    mask = np.zeros(PER_SLOT_ACTIONS, dtype=bool)

    active = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
    if active is None or active.fainted:
        mask[PASS_ACTION] = True
        return mask

    # Legal moves for this slot.
    moves = battle.available_moves[slot] if slot < len(battle.available_moves) else []
    active_moves = list(active.moves.values()) if active.moves else []
    for move in moves:
        try:
            m_idx = active_moves.index(move)
        except ValueError:
            m_idx = next((i for i, mv in enumerate(active_moves) if mv.id == move.id), None)
        if m_idx is None or m_idx >= N_MOVES:
            continue
        for t in range(N_TARGETS):
            mask[m_idx * N_TARGETS + t] = True

    # Legal switches for this slot.
    switches = battle.available_switches[slot] if slot < len(battle.available_switches) else []
    for s_idx in range(min(len(switches), N_SWITCH)):
        mask[MOVE_ACTIONS + s_idx] = True

    if not mask.any():
        mask[PASS_ACTION] = True
    return mask


def legal_action_mask(battle: DoubleBattle) -> np.ndarray:
    """Return a (N_ACTIVE_SLOTS, PER_SLOT_ACTIONS) boolean legality mask."""
    return np.stack([_slot_mask(battle, s) for s in range(N_ACTIVE_SLOTS)])


def decode_action(action_index: int) -> dict:
    """Decode a single per-slot action index into a structured description.

    Order execution against poke-env's BattleOrder happens in the Gym env (M5);
    this decoder is the shared source of truth for what each index means and is
    used by the tests to pin the contract.
    """
    if action_index >= PASS_ACTION:
        return {"kind": "pass"}
    if action_index >= MOVE_ACTIONS:
        return {"kind": "switch", "bench_slot": action_index - MOVE_ACTIONS}
    move_slot, target = divmod(action_index, N_TARGETS)
    return {"kind": "move", "move_slot": move_slot,
            "target": target, "target_label": TARGET_LABELS[target]}
