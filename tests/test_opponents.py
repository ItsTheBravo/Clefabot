"""Tests for the opponent-pool machinery's offline parts (M6)."""

from clefabot.opponents.archetype import is_near_duplicate, jaccard, species_set, tag_team
from clefabot.opponents.packed import unpack_team
from clefabot.team.parser import parse_team

# Two mons in Showdown packed format (ids, as they appear in |showteam| lines).
PACKED = (
    "Raichu||raichunitey|lightningrod|grassknot,zapcannon,focusblast,protect"
    "|Timid|2,,,32,,32||||50|,,,,,Electric]"
    "Kingambit||focussash|defiant|kowtowcleave,suckerpunch,ironhead,lowkick"
    "|Adamant|2,32,,,,32||||50|"
)


def test_unpack_team_fields():
    mons = unpack_team(PACKED)
    assert len(mons) == 2
    raichu = mons[0]
    assert raichu.species == "Raichu"
    assert raichu.item == "raichunitey"
    assert raichu.moves == ["grassknot", "zapcannon", "focusblast", "protect"]
    assert raichu.nature == "Timid"
    assert raichu.evs == {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32}
    assert raichu.ivs["atk"] == 31  # empty IVs default to 31
    assert raichu.level == 50
    assert raichu.tera_type == "Electric"


def test_unpacked_text_reparses_with_team_parser():
    mons = unpack_team(PACKED)
    text = "\n\n".join(m.to_showdown_text() for m in mons)
    team = parse_team(text)  # round-trips through the §4 parser
    assert len(team.pokemon) == 2
    assert team.pokemon[1].item == "focussash"


def test_archetype_tagging():
    mons = unpack_team(PACKED)
    # No trick room / weather; 1 offense item -> balance-ish, never crashes.
    assert tag_team(mons) in {"balance", "hyper_offense", "bulky_balance",
                              "redirection_support"}
    mons[0].moves[0] = "trickroom"
    assert tag_team(mons) == "trick_room"


def test_jaccard_and_near_duplicates():
    a = unpack_team(PACKED)
    b = unpack_team(PACKED)
    assert jaccard(species_set(a), species_set(b)) == 1.0
    assert is_near_duplicate(a, b)
    b[0].species = "Miraidon"
    b[1].species = "Fluttermane"
    assert not is_near_duplicate(a, b)
