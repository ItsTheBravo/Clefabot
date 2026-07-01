"""Write games, turns, and team versions into the shared store (spec §11).

Everything that plays a game — M1 scripted play now, PPO self-play later — logs
through this single object so the training loop and the analytics layer share one
pipeline (spec §8).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ..team.parser import TeamSpec
from . import schema


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def team_version_id(team: TeamSpec) -> str:
    """Deterministic id from the parsed team, so identical teams collapse and
    any edit produces a new version (spec §4/§10)."""
    payload = json.dumps(
        [asdict(p) for p in team.pokemon], sort_keys=True, separators=(",", ":")
    )
    return "tv_" + hashlib.sha1(payload.encode()).hexdigest()[:12]


class GameLogger:
    """Thin writer over the SQLite schema."""

    def __init__(self, db_path: str | Path):
        self.conn = schema.init_db(db_path)

    def close(self) -> None:
        self.conn.close()

    # --- team versions -------------------------------------------------
    def register_team(self, team: TeamSpec, label: str | None = None,
                      parent_checkpoint: str | None = None) -> str:
        vid = team_version_id(team)
        parsed = json.dumps([asdict(p) for p in team.pokemon])
        self.conn.execute(
            "INSERT INTO team_versions"
            "(version_id, created_at, label, raw_text, parsed_json, parent_checkpoint) "
            "VALUES(?,?,?,?,?,?) ON CONFLICT(version_id) DO NOTHING",
            (vid, _now(), label, team.raw_text, parsed, parent_checkpoint),
        )
        self.conn.commit()
        return vid

    # --- games ---------------------------------------------------------
    def log_game(self, *, team_version: str, result: str, turn_count: int,
                 source: str, checkpoint_id: str | None = None,
                 opponent_archetype: str | None = None,
                 game_format: str | None = None,
                 turns: list[dict] | None = None) -> str:
        game_id = "g_" + uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO games"
            "(game_id, timestamp, team_version, checkpoint_id, opponent_archetype,"
            " result, turn_count, source, format) VALUES(?,?,?,?,?,?,?,?,?)",
            (game_id, _now(), team_version, checkpoint_id, opponent_archetype,
             result, turn_count, source, game_format),
        )
        if turns:
            self.conn.executemany(
                "INSERT INTO turns"
                "(game_id, turn_number, win_probability, move_probs_json,"
                " active_self, active_opponent, action_taken, damage_dealt,"
                " damage_received) VALUES(?,?,?,?,?,?,?,?,?)",
                [
                    (
                        game_id, t.get("turn_number"), t.get("win_probability"),
                        json.dumps(t["move_probs"]) if t.get("move_probs") else None,
                        t.get("active_self"), t.get("active_opponent"),
                        t.get("action_taken"), t.get("damage_dealt"),
                        t.get("damage_received"),
                    )
                    for t in turns
                ],
            )
        self.conn.commit()
        return game_id

    # --- export --------------------------------------------------------
    def export_csv(self, out_dir: str | Path) -> list[Path]:
        """Dump every table to CSV (spec §11 secondary output)."""
        import csv

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        tables = [r[0] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        )]
        for table in tables:
            rows = self.conn.execute(f"SELECT * FROM {table}").fetchall()
            path = out / f"{table}.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                if rows:
                    writer.writerow(rows[0].keys())
                    writer.writerows([tuple(r) for r in rows])
                else:
                    cols = [c[1] for c in self.conn.execute(
                        f"PRAGMA table_info({table})")]
                    writer.writerow(cols)
            written.append(path)
        return written
