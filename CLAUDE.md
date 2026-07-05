# Clefabot — Claude Code context

RL agent that pilots one fixed VGC team (Pokémon Champions, Regulation M-B,
doubles) via behavior-cloning warm start + PPO self-play, with a SQLite
analytics layer and (upcoming) Discord digest. Spec:
`vgc-agent-phase1-requirements.md`. Plan + verified facts + deviations:
`PLAN.md`. Milestone status table: `README.md`.

## Commands

```bash
scripts/bootstrap.sh                              # one-time full setup
scripts/ensure_server.sh                          # start local Showdown (idempotent)
.venv/bin/python -m pytest tests/ -q              # test suite
.venv/bin/python -m clefabot.cli.play             # one logged game (smoke test)
.venv/bin/python -m clefabot.imitation.collect --games 150   # BC teacher data
.venv/bin/python -m clefabot.imitation.train_bc              # BC baseline
.venv/bin/python -m clefabot.imitation.evaluate --games 40   # must beat random
.venv/bin/python -m clefabot.rl.selfplay --games 200         # PPO self-play
.venv/bin/python -m clefabot.eval.gate --champion X --challenger Y  # promotion gate
.venv/bin/python -m clefabot.opponents.scrape --pages 10     # opponent pool (network)
```

Always use `.venv/bin/python` (never bare `python3`) — torch and poke-env live
in the venv.

## Architecture (one line per package)

- `clefabot/team/` — Showdown paste → generic representation; resolves
  `Raichu-Mega-Y` → base `Raichu` + stone (sim megas in-battle).
- `clefabot/env/` — `features.py`: fixed 1992-dim team-agnostic observation;
  `actions.py`: thin adapter over poke-env's native `MultiDiscrete([107,107])`
  doubles encoding (mask/order conversions re-exported from `DoublesEnv`).
- `clefabot/imitation/` — `net.py`: shared `PolicyValueNet` (BC and PPO use the
  same module, so warm start = `load_state_dict`); collect/train/evaluate.
- `clefabot/rl/` — `ppo.py`: compact PPO-clip (deliberate deviation from the
  spec's stable-baselines3 — see PLAN.md §3); `selfplay.py`: the training loop
  (rollouts via `battle_against`, NOT PokeEnv's step loop — it stalls).
- `clefabot/eval/gate.py` — champion-vs-challenger, promote iff wr > 0.55, logs
  to `evaluations`.
- `clefabot/opponents/` — replay scraper (network), packed-team unpacker,
  archetype tags, Jaccard dedupe. Output: `opponents_pool/` (committed).
- `clefabot/data_layer/` — WAL SQLite schema v2 + writer + CSV export + damage
  util. EVERY game must log through `GameLogger` (one pipeline, spec §8).

## Hard-won facts (do not re-derive)

- Format id: `gen9championsvgc2026regmb` on a current GitHub build of
  pokemon-showdown. The npm package is STALE (no Champions) — always fetch the
  server via `scripts/setup_server.sh`.
- Reg M-B has Open Team Sheets → replay logs contain both full teams
  (`|showteam|` lines); that's the whole opponent-pool data source.
- The obs/action shapes are team-agnostic constants. Never size anything to
  the current team — that's what makes a team edit a re-train (spec §4).
- Force-switch turns invert per-slot legality (the fainted slot must act);
  poke-env's `get_action_mask`/`action_to_order` handle this — don't hand-roll.
- `order_to_action` can return -2/-1 sentinels (default/forfeit); filter
  negatives before training on labels.
- A doubles battle turn = multiple decisions; `turns` is keyed on
  (game_id, decision_idx), not turn_number.
- SimpleHeuristicsPlayer has zero mega logic — `MegaTeacher` adds it for
  demonstrations. Auto-mega-ASAP is NOT strictly correct (No Guard lets
  opponents never miss; losing Lightning Rod matters) — mega timing is PPO's
  job.
- Value-head "win probability" is not yet calibrated (small training volume);
  don't present it as gospel.
- poke-env teampreview default is random 4-of-6 — team selection is NOT
  learned yet (open item).

## Conventions

- Commits: descriptive message; NO "Co-Authored-By: Claude" or any AI
  trailer/model id (user preference, firm).
- Do not commit: `data/`, `checkpoints/`, `*.pt`, `*.npz`, `.venv/`, `server/`
  (all gitignored, regenerable). DO commit `opponents_pool/`.
- Tunables (checkpoint interval, eval size, promotion threshold, reward
  weights) are named constants with defaults — keep them visible, not buried.
- Discord webhook URL (M10) comes from env var / local config, never the repo.

## Current state (see README table)

M1–M4 + M7 done and validated; M5 core loop validated (long soak pending);
M6 scraper ready, needs a networked machine. Next: run scraper locally, wire
opponent pool into selfplay, M8 retrain trigger, M9 analytics + "analyze this
turn" CLI, M10 Discord digest. First gate run already promoted the PPO
checkpoint over the BC baseline (58% over 100 games).
