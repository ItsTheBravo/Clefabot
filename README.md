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
| **M2 — Data layer + damage-calc utility** | ✅ done |
| **M3 — Fixed-shape env contract** | ✅ done |
| **M4 — Imitation baseline** | ✅ done |
| M5 — PPO self-play loop | next |
| M6 — Opponent pool | pending |
| M7 — Evaluation gate | pending |
| M8 — Retrain trigger | pending |
| M9 — Analytics + "analyze this turn" | pending |
| M10 — Discord digest | pending |

M1 proves the full spine: a real doubles game plays on a local Reg M-B server
and is logged end to end into SQLite (games + per-turn snapshots + team version),
exportable to CSV. Mega Evolution resolves correctly in-battle (the turn log
shows `raichu` → `raichumegay`).

M2 wires the shared damage utility (`clefabot/data_layer/damage.py`) over
poke-env's built-in calculator — one source of damage math for reward shaping
and analytics.

M3 locks the warm-start contract: a fixed-size, team-agnostic observation
(`OBS_SIZE = 1992`) and action space, verified team-invariant both offline and
on live battles. This is what makes a team edit a re-train rather than a
re-architecture. The action space is poke-env's native gen-9 doubles encoding
(`MultiDiscrete([107, 107])` + legality mask) — adopted over an initial
hand-rolled 21-action scheme because it covers Mega Evolution/Tera orders and
shares one source of truth (`action_to_order` / `order_to_action` /
`get_action_mask`) between labels, mask, and execution.

M4 trains the imitation baseline: the shared `PolicyValueNet` (used verbatim by
PPO at M5, so warm-start is a literal weight copy) is behavior-cloned from
poke-env's `SimpleHeuristicsPlayer`. The cloned net **wins 100% vs a random baseline**
(spec §7 acceptance: beat random), with clean doubles order execution (~0% fallback
incl. force-switch turns) and Mega Evolution firing in play. The value/policy heads also back the "analyze this turn" readout
(plan §1.2). `replay_ingest.py` provides the §7 replay-log→state path for real
human replays once that corpus is reachable.

### Imitation pipeline (M4)

```bash
# collect (state, action) pairs from the teacher, behavior-clone, evaluate
.venv/bin/python -m clefabot.imitation.collect   --games 150 --out data/bc_dataset.npz
.venv/bin/python -m clefabot.imitation.train_bc  --dataset data/bc_dataset.npz --out checkpoints/bc_baseline.pt
.venv/bin/python -m clefabot.imitation.evaluate  --checkpoint checkpoints/bc_baseline.pt --games 40
```

### Known data-access limitation

`replay.pokemonshowdown.com` and pokepaste hosts are blocked by this
environment's network policy, so the real human-replay corpus (M4 quality) and
scraped opponent teams (M6) can't be fetched here yet. The pipeline is built and
tested against local data; widening the policy or dropping in a data export
unblocks the external corpus without code changes.

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
