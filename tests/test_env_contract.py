"""M3 contract tests: the observation and action shapes must be team-agnostic.

The whole warm-start premise (spec §4) is that swapping the team does not change
the network's input/output widths. These tests pin that invariant using
Pokémon/Move objects built offline (no server needed).
"""

import numpy as np
from poke_env.battle import Move, Pokemon

from clefabot.env import actions, features


def _mon(species: str) -> Pokemon:
    return Pokemon(gen=9, species=species)


def test_move_block_fixed_length():
    assert len(features.encode_move(None)) == features.MOVE_FEATS
    assert len(features.encode_move(Move("zapcannon", gen=9))) == features.MOVE_FEATS
    assert len(features.encode_move(Move("protect", gen=9))) == features.MOVE_FEATS


def test_pokemon_block_fixed_length_across_species():
    # Wildly different species must all yield the same-length block.
    for species in ["raichumegay", "clefable", "kingambit", "meowscarada",
                    "ditto", "fluttermane", "miraidon"]:
        block = features.encode_pokemon(_mon(species))
        assert len(block) == features.MON_FEATS, species
    # Empty slot too.
    assert len(features.encode_pokemon(None)) == features.MON_FEATS


def test_obs_size_is_pure_constant():
    # OBS_SIZE derives only from constants, not from any team.
    expected = 2 * features.MAX_TEAM * features.MON_FEATS + features.FIELD_FEATS
    assert features.OBS_SIZE == expected


def test_action_space_shape_fixed():
    assert actions.action_space_shape() == (
        actions.N_ACTIVE_SLOTS, actions.PER_SLOT_ACTIONS)
    assert actions.PER_SLOT_ACTIONS == actions.N_MOVES * actions.N_TARGETS + \
        actions.N_SWITCH + 1


def test_action_decode_roundtrip():
    assert actions.decode_action(0) == {
        "kind": "move", "move_slot": 0, "target": 0, "target_label": "opp_0"}
    assert actions.decode_action(actions.N_TARGETS)["move_slot"] == 1
    sw = actions.decode_action(actions.MOVE_ACTIONS)
    assert sw == {"kind": "switch", "bench_slot": 0}
    assert actions.decode_action(actions.PASS_ACTION) == {"kind": "pass"}


def test_two_different_teams_same_obs_length():
    """Encoding two disjoint teams' worth of mons yields identical lengths."""
    team_a = ["raichumegay", "clefable", "mamoswine", "ceruledge",
              "kingambit", "meowscarada"]
    team_b = ["miraidon", "fluttermane", "chiyu", "ironhands",
              "gholdengo", "ursaluna"]
    enc_a = np.concatenate([features.encode_pokemon(_mon(s)) for s in team_a])
    enc_b = np.concatenate([features.encode_pokemon(_mon(s)) for s in team_b])
    assert enc_a.shape == enc_b.shape == (6 * features.MON_FEATS,)


class _FakeBattle:
    """Minimal DoubleBattle stand-in for mask logic tests."""

    def __init__(self, force_switch, actives, moves, switches):
        self.force_switch = force_switch
        self.active_pokemon = actives
        self.available_moves = moves
        self.available_switches = switches


def test_force_switch_mask_inverts_slots():
    """Regression: the forced slot must switch; the other slot must pass.

    (Bug found in audit: a fainted active previously produced a pass-only mask
    on exactly the slot that was required to switch.)
    """
    bench = [_mon("clefable"), _mon("kingambit")]
    battle = _FakeBattle(
        force_switch=[True, False],
        actives=[None, _mon("mamoswine")],
        moves=[[], []],
        switches=[bench, bench],
    )
    mask = actions.legal_action_mask(battle)
    # Slot 0 (forced): exactly the two switch actions, nothing else.
    assert mask[0, actions.MOVE_ACTIONS:actions.MOVE_ACTIONS + 2].all()
    assert not mask[0, :actions.MOVE_ACTIONS].any()
    assert not mask[0, actions.PASS_ACTION]
    # Slot 1 (not forced): pass only.
    assert mask[1, actions.PASS_ACTION]
    assert mask[1].sum() == 1


def test_force_switch_no_bench_falls_back_to_pass():
    battle = _FakeBattle(
        force_switch=[True, False],
        actives=[None, _mon("mamoswine")],
        moves=[[], []],
        switches=[[], []],
    )
    mask = actions.legal_action_mask(battle)
    assert mask[0, actions.PASS_ACTION] and mask[0].sum() == 1
