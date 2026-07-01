"""Parse a Showdown team export into a generic, team-agnostic representation.

Design note (spec §4): the representation is deliberately *generic* — a list of
slot specs with typed fields — never keyed on "Pokemon #3" or on this specific
team. That is what lets a team edit be a re-train rather than a re-architecture.

Mega handling (plan §1.1): the current team pastes ``Raichu-Mega-Y @ Raichunite
Y`` with the pre-Mega ability (Lightning Rod). Showdown's team format wants the
*base* species + the stone; the Mega form (stats/typing/ability -> No Guard) is
applied in-battle. So we normalize mega formes to their base species for the
outgoing team string while recording the Mega details in the parsed spec.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Stat keys in Showdown's canonical order.
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
_STAT_ALIASES = {
    "hp": "hp",
    "atk": "atk",
    "def": "def",
    "spa": "spa",
    "spd": "spd",
    "spe": "spe",
}


@dataclass
class PokemonSpec:
    """One parsed team slot in generic form."""

    species: str  # as pasted, e.g. "Raichu-Mega-Y"
    base_species: str  # normalized base, e.g. "Raichu" (used for team string)
    item: Optional[str]
    ability: str  # pre-mega ability as pasted (e.g. "Lightning Rod")
    level: int
    nature: Optional[str]
    evs: dict[str, int]
    ivs: dict[str, int]
    moves: list[str]
    nickname: Optional[str] = None
    gender: Optional[str] = None
    tera_type: Optional[str] = None
    is_mega: bool = False
    mega_forme: Optional[str] = None  # e.g. "Mega-Y"

    def to_showdown_set(self) -> str:
        """Render back to a Showdown export block using the base species.

        Mega formes are emitted as their base species so the sim's own
        mega-from-stone logic applies at battle time.
        """
        head = self.base_species
        if self.nickname:
            head = f"{self.nickname} ({self.base_species})"
        if self.item:
            head = f"{head} @ {self.item}"
        lines = [head, f"Ability: {self.ability}"]
        if self.level != 50:
            lines.append(f"Level: {self.level}")
        if self.tera_type:
            lines.append(f"Tera Type: {self.tera_type}")
        ev_str = self._stat_line(self.evs, default=0)
        if ev_str:
            lines.append(f"EVs: {ev_str}")
        if self.nature:
            lines.append(f"{self.nature} Nature")
        iv_str = self._stat_line(self.ivs, default=31)
        if iv_str:
            lines.append(f"IVs: {iv_str}")
        lines.extend(f"- {m}" for m in self.moves)
        return "\n".join(lines)

    @staticmethod
    def _stat_line(stats: dict[str, int], default: int) -> str:
        parts = [
            f"{stats[k]} {k.upper()}"
            for k in STAT_KEYS
            if k in stats and stats[k] != default
        ]
        return " / ".join(parts)


@dataclass
class TeamSpec:
    """A whole team plus the raw paste, for traceability (spec §4)."""

    pokemon: list[PokemonSpec]
    raw_text: str = ""
    extras: dict = field(default_factory=dict)

    def to_showdown_team(self) -> str:
        """Full team export string, safe to hand to poke-env's Player(team=...)."""
        return "\n\n".join(p.to_showdown_set() for p in self.pokemon)


# Mega/primal formes are identified by a "-Mega" / "-Primal" suffix on the paste.
_MEGA_SUFFIX = re.compile(r"-(Mega(?:-[XY])?|Primal)$", re.IGNORECASE)


def _split_mega(species: str) -> tuple[str, bool, Optional[str]]:
    """Return (base_species, is_mega, mega_forme) for a pasted species name."""
    m = _MEGA_SUFFIX.search(species)
    if not m:
        return species, False, None
    base = species[: m.start()]
    forme = m.group(1)
    return base, True, forme


def _parse_stat_block(value: str) -> dict[str, int]:
    """Parse 'EVs: 2 HP / 32 SpA / 32 Spe' style RHS into {stat: int}."""
    out: dict[str, int] = {}
    for chunk in value.split("/"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split()
        if len(parts) != 2:
            continue
        amount, stat = parts
        stat_key = _STAT_ALIASES.get(stat.lower())
        if stat_key is None:
            continue
        try:
            out[stat_key] = int(amount)
        except ValueError:
            continue
    return out


def parse_pokemon(block: str) -> PokemonSpec:
    """Parse a single Showdown set block into a PokemonSpec."""
    lines = [ln.rstrip() for ln in block.strip().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty Pokemon block")

    # Header line: "Nick (Species) (Gender) @ Item" with many optional parts.
    header = lines[0]
    item = None
    if "@" in header:
        header, item = (s.strip() for s in header.rsplit("@", 1))

    gender = None
    gender_m = re.search(r"\((M|F)\)\s*$", header)
    if gender_m:
        gender = gender_m.group(1)
        header = header[: gender_m.start()].strip()

    nickname = None
    paren_m = re.search(r"^(.*?)\s*\(([^)]+)\)\s*$", header)
    if paren_m:
        nickname = paren_m.group(1).strip()
        species = paren_m.group(2).strip()
    else:
        species = header.strip()

    base_species, is_mega, mega_forme = _split_mega(species)

    spec = PokemonSpec(
        species=species,
        base_species=base_species,
        item=item or None,
        ability="",
        level=50,
        nature=None,
        evs={},
        ivs={},
        moves=[],
        nickname=nickname,
        gender=gender,
        is_mega=is_mega,
        mega_forme=mega_forme,
    )

    for ln in lines[1:]:
        low = ln.lower()
        if low.startswith("ability:"):
            spec.ability = ln.split(":", 1)[1].strip()
        elif low.startswith("level:"):
            spec.level = int(ln.split(":", 1)[1].strip())
        elif low.startswith("tera type:"):
            spec.tera_type = ln.split(":", 1)[1].strip()
        elif low.startswith("evs:"):
            spec.evs = _parse_stat_block(ln.split(":", 1)[1])
        elif low.startswith("ivs:"):
            spec.ivs = _parse_stat_block(ln.split(":", 1)[1])
        elif low.endswith("nature"):
            spec.nature = ln.rsplit(" ", 1)[0].strip()
        elif ln.startswith("-"):
            move = ln[1:].strip()
            if move:
                spec.moves.append(move)

    if not spec.ability:
        raise ValueError(f"no ability parsed for {species!r}")
    return spec


def parse_team(text: str) -> TeamSpec:
    """Parse a full Showdown team export (blocks separated by blank lines)."""
    # Normalize newlines and split on blank-line boundaries.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    blocks = re.split(r"\n\s*\n", normalized)
    pokemon = [parse_pokemon(b) for b in blocks if b.strip()]
    if not pokemon:
        raise ValueError("no Pokemon parsed from team text")
    return TeamSpec(pokemon=pokemon, raw_text=text)
