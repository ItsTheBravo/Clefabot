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
    # poke-env native gen-9 doubles encoding: 107 actions per active slot.
    assert actions.action_space_shape() == (2, 107)
    assert actions.PER_SLOT_ACTIONS == 107


def test_action_decode_layout():
    assert actions.decode_action(0) == {"kind": "pass"}
    assert actions.decode_action(1) == {"kind": "switch", "team_slot": 0}
    assert actions.decode_action(6) == {"kind": "switch", "team_slot": 5}
    d = actions.decode_action(7)
    assert d == {"kind": "move", "move_slot": 0, "target": -2, "gimmick": "none"}
    # 27 opens the mega block: move 1, first target, mega evolve.
    d = actions.decode_action(27)
    assert d["kind"] == "move" and d["move_slot"] == 0 and d["gimmick"] == "mega"
    # 87 opens the tera block.
    assert actions.decode_action(87)["gimmick"] == "tera"
    # Last index decodes cleanly.
    d = actions.decode_action(actions.PER_SLOT_ACTIONS - 1)
    assert d == {"kind": "move", "move_slot": 3, "target": 2, "gimmick": "tera"}


def test_two_different_teams_same_obs_length():
    """Encoding two disjoint teams' worth of mons yields identical lengths."""
    team_a = ["raichumegay", "clefable", "mamoswine", "ceruledge",
              "kingambit", "meowscarada"]
    team_b = ["miraidon", "fluttermane", "chiyu", "ironhands",
              "gholdengo", "ursaluna"]
    enc_a = np.concatenate([features.encode_pokemon(_mon(s)) for s in team_a])
    enc_b = np.concatenate([features.encode_pokemon(_mon(s)) for s in team_b])
    assert enc_a.shape == enc_b.shape == (6 * features.MON_FEATS,)
