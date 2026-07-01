"""SQLite schema for the shared game/analytics data pipeline (spec §11).

One store, one schema — local self-play, opponent-pool, and (future) live games
all write here. WAL mode is enabled so concurrent battle-loggers (plan §1.4)
don't lock each other out.

The ``source`` column already carries the ``live`` value even though Phase 1
only produces local games, so Phase 2 live data slots in with no migration.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- One row per distinct team version / edit (spec §4, §10).
CREATE TABLE IF NOT EXISTS team_versions (
    version_id        TEXT PRIMARY KEY,   -- content hash of the parsed team
    created_at        TEXT NOT NULL,
    label             TEXT,
    raw_text          TEXT NOT NULL,      -- exact paste, for traceability
    parsed_json       TEXT NOT NULL,      -- generic representation
    parent_checkpoint TEXT                -- warm-start parent (spec §10)
);

-- One row per checkpoint produced by training (spec §8, §9).
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    team_version  TEXT REFERENCES team_versions(version_id),
    parent_id     TEXT,                   -- lineage / warm-start parent
    path          TEXT,
    is_champion   INTEGER NOT NULL DEFAULT 0
);

-- One row per game played anywhere in the pipeline (spec §11).
CREATE TABLE IF NOT EXISTS games (
    game_id            TEXT PRIMARY KEY,
    timestamp          TEXT NOT NULL,
    team_version       TEXT REFERENCES team_versions(version_id),
    checkpoint_id      TEXT,
    opponent_archetype TEXT,
    result             TEXT NOT NULL,      -- win / loss / tie
    turn_count         INTEGER,
    source             TEXT NOT NULL       -- local_selfplay | local_opponent_pool | live
        CHECK (source IN ('local_selfplay', 'local_opponent_pool', 'live', 'local_scripted')),
    format             TEXT
);

-- Per-turn detail (spec §11). win_probability / move_probs populate once the
-- PPO value+policy heads exist (M5); for M1 the structural fields fill in.
CREATE TABLE IF NOT EXISTS turns (
    game_id         TEXT NOT NULL REFERENCES games(game_id),
    turn_number     INTEGER NOT NULL,
    win_probability REAL,                 -- from PPO value head (M5+)
    move_probs_json TEXT,                 -- policy head distribution (M5+)
    active_self     TEXT,
    active_opponent TEXT,
    action_taken    TEXT,
    damage_dealt    REAL,
    damage_received REAL,
    PRIMARY KEY (game_id, turn_number)
);

-- Aggregated per-Pokemon stats, per team version (spec §11).
CREATE TABLE IF NOT EXISTS pokemon_stats (
    team_version   TEXT NOT NULL,
    species        TEXT NOT NULL,
    games_brought  INTEGER NOT NULL DEFAULT 0,
    wins_when_brought INTEGER NOT NULL DEFAULT 0,
    kos            INTEGER NOT NULL DEFAULT 0,
    faints         INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (team_version, species)
);

-- Aggregated per-move stats, per team version (spec §11).
CREATE TABLE IF NOT EXISTS move_stats (
    team_version TEXT NOT NULL,
    move         TEXT NOT NULL,
    uses         INTEGER NOT NULL DEFAULT 0,
    wins         INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (team_version, move)
);

-- Evaluation-gate results (spec §9).
CREATE TABLE IF NOT EXISTS evaluations (
    eval_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT NOT NULL,
    champion_id   TEXT,
    challenger_id TEXT,
    n_games       INTEGER NOT NULL,
    win_rate      REAL NOT NULL,
    threshold     REAL NOT NULL,
    promoted      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_games_team ON games(team_version);
CREATE INDEX IF NOT EXISTS idx_games_source ON games(source);
CREATE INDEX IF NOT EXISTS idx_turns_game ON turns(game_id);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating dirs as needed) a WAL-mode connection with FKs on."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Create the schema if absent and return an open connection."""
    conn = connect(db_path)
    conn.executescript(DDL)
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
    return conn
