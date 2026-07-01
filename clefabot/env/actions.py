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


# --- target <-> index mapping (doubles) --------------------------------
# poke-env doubles targets: +1/+2 = opponent slots, -1/-2 = own side,
# 0 = no explicit target (spread/self/status). We keep a single consistent
# mapping so labels collected from a teacher and orders reconstructed for
# execution agree. Exact target legality is validated against the move at
# execution time (M5); this mapping just pins the encoding.
def move_target_to_index(move_target: int) -> int:
    if move_target == 1:
        return 0  # opp_0
    if move_target == 2:
        return 1  # opp_1
    if move_target < 0:
        return 2  # ally
    return 3      # self / spread / no explicit target


def index_to_move_target(target_index: int) -> int:
    return {0: 1, 1: 2, 2: -1, 3: 0}.get(target_index, 0)


def encode_order_slot(order, active, switches) -> int:
    """Map one poke-env SingleBattleOrder to a per-slot action index.

    ``active`` is the Pokemon in this slot; ``switches`` is the slot's list of
    available switch targets. Returns PASS_ACTION for pass/default orders.
    """
    from poke_env.battle import Move, Pokemon

    inner = getattr(order, "order", None)
    if isinstance(inner, Move):
        move_list = list(active.moves.values()) if active and active.moves else []
        m_idx = next((i for i, mv in enumerate(move_list) if mv.id == inner.id), None)
        if m_idx is None or m_idx >= N_MOVES:
            return PASS_ACTION
        t_idx = move_target_to_index(getattr(order, "move_target", 0))
        return m_idx * N_TARGETS + t_idx
    if isinstance(inner, Pokemon):
        s_idx = next((i for i, mon in enumerate(switches)
                      if mon.species == inner.species), None)
        if s_idx is None or s_idx >= N_SWITCH:
            return PASS_ACTION
        return MOVE_ACTIONS + s_idx
    return PASS_ACTION


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
