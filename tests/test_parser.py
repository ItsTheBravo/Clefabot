"""Tests for the team parser, focused on the Mega-resolution contract (plan §1.1)."""

from clefabot.team.parser import parse_team

SAMPLE = """Raichu-Mega-Y @ Raichunite Y
Ability: Lightning Rod
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Grass Knot
- Zap Cannon
- Focus Blast
- Protect

Clefable @ Sitrus Berry
Ability: Cute Charm
Level: 50
EVs: 32 HP / 17 Def / 12 SpD / 5 Spe
Bold Nature
- Moonblast
- Follow Me
- Helping Hand
- Protect
"""


def test_parses_all_slots():
    team = parse_team(SAMPLE)
    assert len(team.pokemon) == 2
    assert team.raw_text == SAMPLE  # raw paste retained for traceability


def test_mega_normalized_to_base_species():
    team = parse_team(SAMPLE)
    raichu = team.pokemon[0]
    assert raichu.species == "Raichu-Mega-Y"
    assert raichu.base_species == "Raichu"
    assert raichu.is_mega is True
    assert raichu.mega_forme == "Mega-Y"
    # Pre-mega ability preserved as pasted (becomes No Guard on evolve in-battle).
    assert raichu.ability == "Lightning Rod"
    assert raichu.item == "Raichunite Y"


def test_showdown_string_uses_base_species():
    team = parse_team(SAMPLE)
    out = team.to_showdown_team()
    assert out.startswith("Raichu @ Raichunite Y")
    assert "Raichu-Mega-Y" not in out


def test_parses_evs_and_moves():
    team = parse_team(SAMPLE)
    clef = team.pokemon[1]
    assert clef.evs == {"hp": 32, "def": 17, "spd": 12, "spe": 5}
    assert clef.moves == ["Moonblast", "Follow Me", "Helping Hand", "Protect"]
    assert clef.nature == "Bold"
