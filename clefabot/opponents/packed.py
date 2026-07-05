"""Unpack Showdown's packed team format (spec §6).

Reg M-B games run with Open Team Sheets, so every replay log carries both
players' full teams as ``|showteam|pN|<packed>`` lines — no separate team-dump
source needed. This module converts a packed team into (a) a plain dict
structure for tagging/diversity analysis and (b) a Showdown-importable text
block (ids are fine: Showdown's importer and poke-env's teambuilder normalize
names to ids anyway).

Packed format (sim/TEAMS.md): mons joined by ']', fields by '|':
  NICKNAME|SPECIES|ITEM|ABILITY|MOVE1,MOVE2,..|NATURE|EV1,..,EV6|GENDER|
  IV1,..,IV6|SHINY|LEVEL|HAPPINESS,POKEBALL,HPTYPE,GMAX,DMAXLEVEL,TERATYPE
SPECIES is empty when identical to NICKNAME; empty EVS mean 0, empty IVS 31.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STAT_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")


@dataclass
class PackedMon:
    species: str
    item: str
    ability: str
    moves: list[str]
    nature: str
    evs: dict[str, int]
    ivs: dict[str, int]
    level: int = 50
    gender: str = ""
    tera_type: str = ""
    nickname: str = ""
    extras: dict = field(default_factory=dict)

    def to_showdown_text(self) -> str:
        head = self.species
        if self.item:
            head += f" @ {self.item}"
        lines = [head, f"Ability: {self.ability}"]
        if self.level and self.level != 100:
            lines.append(f"Level: {self.level}")
        if self.tera_type:
            lines.append(f"Tera Type: {self.tera_type}")
        ev_parts = [f"{v} {k.upper()}" for k, v in self.evs.items() if v]
        if ev_parts:
            lines.append("EVs: " + " / ".join(ev_parts))
        if self.nature:
            lines.append(f"{self.nature.capitalize()} Nature")
        iv_parts = [f"{v} {k.upper()}" for k, v in self.ivs.items() if v != 31]
        if iv_parts:
            lines.append("IVs: " + " / ".join(iv_parts))
        lines += [f"- {m}" for m in self.moves]
        return "\n".join(lines)


def _stats(csv: str, default: int) -> dict[str, int]:
    if not csv:
        return {k: default for k in STAT_ORDER}
    vals = csv.split(",")
    return {
        k: int(vals[i]) if i < len(vals) and vals[i] else default
        for i, k in enumerate(STAT_ORDER)
    }


def unpack_mon(chunk: str) -> PackedMon:
    f = chunk.split("|")
    if len(f) < 11:
        f += [""] * (11 - len(f))
    nickname, species = f[0], f[1] or f[0]
    misc = f[11].split(",") if len(f) > 11 and f[11] else []
    tera = misc[5] if len(misc) > 5 else ""
    return PackedMon(
        species=species,
        item=f[2],
        ability=f[3],
        moves=[m for m in f[4].split(",") if m],
        nature=f[5],
        evs=_stats(f[6], 0),
        gender=f[7],
        ivs=_stats(f[8], 31),
        level=int(f[10]) if f[10] else 100,
        tera_type=tera,
        nickname=nickname if f[1] else "",
    )


def unpack_team(packed: str) -> list[PackedMon]:
    return [unpack_mon(c) for c in packed.strip().split("]") if c.strip()]


def team_to_showdown_text(mons: list[PackedMon]) -> str:
    return "\n\n".join(m.to_showdown_text() for m in mons)
