# Clefabot

A local training pipeline that learns to pilot one fixed VGC team (Pokémon
Champions, **Regulation M-B**, doubles) via imitation-warm-started self-play,
with a shared data/analytics layer and a Discord digest on top.

See [`vgc-agent-phase1-requirements.md`](vgc-agent-phase1-requirements.md) for
the full Phase 1 spec and [`PLAN.md`](PLAN.md) for the implementation plan and
build order.

## Status

| Milestone | State |
|---|---|
| **M1 — Environment + one logged game** | ✅ done |
| M2 — Data layer + damage-calc utility | schema in place; damage calc pending |
| M3 — Fixed-shape env contract | next |
| M4 — Imitation baseline | pending |
| M5 — PPO self-play loop | pending |
| M6 — Opponent pool | pending |
| M7 — Evaluation gate | pending |
| M8 — Retrain trigger | pending |
| M9 — Analytics + "analyze this turn" | pending |
| M10 — Discord digest | pending |

M1 proves the full spine: a real doubles game plays on a local Reg M-B server
and is logged end to end into SQLite (games + per-turn snapshots + team version),
exportable to CSV. Mega Evolution resolves correctly in-battle (the turn log
shows `raichu` → `raichumegay`).

## Setup

Requires Node.js and Python 3.10+.

```bash
# 1. Python deps
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Local Showdown server (fetched from GitHub — the npm package is too stale
#    to include Champions/Reg M; see PLAN.md §0)
scripts/setup_server.sh
```

## Run M1

```bash
# terminal 1: start the local server
scripts/start_server.sh

# terminal 2: play and log one game with the current team
.venv/bin/python -m clefabot.cli.play \
    --team config/current_team.txt --db data/clefabot.sqlite --games 1
```

The current team lives in [`config/current_team.txt`](config/current_team.txt)
as a standard Showdown export (edit it to retrain later — that path lands in M8).

## Tests

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests/ -q
```

## Layout

```
clefabot/team/       # Showdown-paste parser -> generic representation (Mega-aware)
clefabot/data_layer/ # SQLite schema (WAL) + writer + CSV export
clefabot/env/        # poke-env players (Reg M-B doubles)
clefabot/cli/        # entry points (play; train/retrain/report land later)
scripts/             # server fetch/build/start
config/              # current team paste (webhook secrets stay out of git)
```

## Format

Confirmed against the local server's format list: `[Gen 9 Champions] VGC 2026
Reg M-B` (id `gen9championsvgc2026regmb`) — standard VGC doubles, Flat Rules +
VGC Timer + Open Team Sheets.
