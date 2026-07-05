"""Rule-based archetype tagging + species-overlap diversity (spec §6).

Deliberately rough: the tags exist to power "win rate by archetype" analytics
(§11), not to be a perfect taxonomy. Rules operate on move/item/species ids as
they appear in packed teams.
"""

from __future__ import annotations

from .packed import PackedMon

TRICK_ROOM_MOVES = {"trickroom"}
REDIRECTION_MOVES = {"followme", "ragepowder"}
WEATHER_ABILITIES = {"drought", "drizzle", "sandstream", "snowwarning",
                     "orichalcumpulse", "desolateland", "primordialsea"}
OFFENSE_ITEMS = {"choiceband", "choicespecs", "choicescarf", "lifeorb",
                 "focussash", "boosterenergy"}
BULK_ITEMS = {"leftovers", "sitrusberry", "assaultvest", "rockyhelmet",
              "covertcloak"}


def tag_team(mons: list[PackedMon]) -> str:
    """Return a single rough archetype label for a team."""
    moves = {m for mon in mons for m in mon.moves}
    items = {mon.item for mon in mons}
    abilities = {mon.ability for mon in mons}

    if moves & TRICK_ROOM_MOVES:
        return "trick_room"
    if abilities & WEATHER_ABILITIES:
        return "weather"
    n_off = len(items & OFFENSE_ITEMS)
    n_bulk = len(items & BULK_ITEMS)
    if moves & REDIRECTION_MOVES and n_bulk >= 1:
        return "redirection_support"
    if n_off >= 3:
        return "hyper_offense"
    if n_bulk >= 3:
        return "bulky_balance"
    return "balance"


def species_set(mons: list[PackedMon]) -> frozenset[str]:
    return frozenset(m.species.lower().replace(" ", "").replace("-", "")
                     for m in mons)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def is_near_duplicate(a: list[PackedMon], b: list[PackedMon],
                      threshold: float = 0.67) -> bool:
    """True when two teams share most of their species (4+/6 by default)."""
    return jaccard(species_set(a), species_set(b)) >= threshold
