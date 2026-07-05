# Clefabot — Phase 1 Implementation Plan

Working plan for the VGC Team Agent described in `vgc-agent-phase1-requirements.md`.
This document locks in the technical decisions we agreed on, corrects a few
assumptions in the original requirements after verification, and defines the
build order. Nothing here changes the *goals* of the spec — it makes them
concrete and buildable.

---

## 0. Verified facts (was uncertain in the requirements)

These were flagged as "verify, don't assume" in §14. Resolved:

- **Format is real and live.** Pokémon Champions *Regulation M-B* is the official
  in-game Ranked / VGC format (17 Jun – 2 Sep 2026, incl. Worlds). Mega Evolution
  is legal.
- **It is standard VGC doubles.** 4-of-6 team preview, 2v2 on the field. No new
  battle mechanics vs. normal VGC — poke-env / SB3 doubles handling applies
  unchanged.
- **Showdown supports it.** Playable format id is `gen9championsvgc2026regmb`
  (M-A is `...regma`). Replays and usage data exist publicly.
- **Mega Raichu Y is a real M-B Mega**, ability **No Guard** post-Mega (base ability
  Lightning Rod). No Guard makes Zap Cannon / Focus Blast always hit — that is the
  team's core engine, and the damage/feature layer must model the ability swap on
  Mega.

Still to confirm *at build time* (cheap checks, not blockers):
- Exact local Showdown server version ↔ poke-env version compatibility.
- Damage-calc library coverage for M-B Megas and new items (needed for reward
  shaping + analytics).
- Replay-search endpoint availability/volume for `gen9championsvgc2026regmb`.

---

## 1. Corrections & refinements vs. the original requirements

1. **Team paste parsing must resolve base ↔ Mega.** The current team lists
   `Raichu-Mega-Y @ Raichunite Y / Ability: Lightning Rod`. Showdown expects the
   **base** species + pre-Mega ability, with the Mega form (stats, typing, ability
   → No Guard) applied when it evolves in-battle. The §4 parser handles this
   generically (stone → Mega mapping), not as a special case.

2. **The "probability" feature is the agent's own policy + value, exposed.**
   Original spec framed it as after-the-fact win-probability charts. What's
   actually wanted: given a board state, the agent reports **(a) its win-chance
   estimate** and **(b) a ranked % breakdown of each candidate move per Pokémon**
   for that turn. This is precisely the PPO network's native output — policy head =
   move %s, value head = win estimate. So we **expose the model's own numbers** via
   a small "analyze this turn" entry point (usable live-ish and on replayed games),
   rather than training a separate estimator. Simpler and more faithful.

3. **Warm-start requires a fixed-shape, team-agnostic encoding.** Observation and
   action spaces are sized to *generic slots* (Pokémon slot 1–6, move slot 1–4 +
   target, switch targets), never to this specific team's species/moves. This is
   what makes a team edit a re-train (copy weights) instead of a re-architecture.
   Hard requirement, baked in from the first env commit.

4. **CPU-only practicality:** run many battles concurrently (poke-env supports
   concurrent battles against one local server); open SQLite in **WAL mode** so
   parallel game-logging writers don't lock each other. Both are cheap settings
   that prevent slow training / write-crashes later.

---

## 2. Cost

Zero software cost. Every dependency is free/open-source, and everything runs
locally on CPU — no cloud, GPU, subscription, or per-game fee. The only real
cost is **wall-clock training time** on your own machine (hours→days in the
background for a strong agent). Optional future spend: renting a cloud CPU box to
train faster — never required for Phase 1.

---

## 3. Tech stack (pinned intent)

| Concern | Choice |
|---|---|
| Battle server | local `pokemon-showdown start --no-security` (Node) |
| Protocol client / RL env | `poke-env` (Gymnasium-compatible) |
| RL algorithm | PPO-clip implemented directly on the shared net (~150 lines; documented deviation from the spec's `stable-baselines3` — SB3 cannot express the two-agent battle env + MultiDiscrete masking + external warm start without extensive custom-policy surgery) |
| NN framework | `torch` (CPU; MPS opportunistic, never required) |
| Imitation baseline | behavior cloning, **same net architecture as the PPO policy** |
| Damage calc | ruleset-compatible calc lib (verify M-B coverage) — shared utility |
| Storage | `sqlite3` (WAL), CSV export per table |
| Charts | `matplotlib` (static image artifacts) |
| Discord | `requests` → one-way webhook |

---

## 4. Project layout (proposed)

```
clefabot/
  env/            # Showdown launch helpers, poke-env Player + Gym wrapper
  team/           # paste parser, generic feature representation, versioning
  data_layer/     # SQLite schema, writers (WAL), CSV export, damage-calc util
  imitation/      # replay ingestion (poke-env parser) -> (state,action) -> BC
  rl/             # PPO self-play loop, checkpointing, warm-start loader
  eval/           # evaluation gate + promotion logic
  opponents/      # replay/pokepaste scraping, archetype tagging, diversity check
  analytics/      # facts generator, chart generation, "analyze this turn" tool
  discord/        # webhook digest builder
  cli/            # entry points: play, train, retrain(team paste), report
  config/         # webhook URL etc. via env var / local config (never committed)
  tests/
```

Single shared data pipeline: local self-play, opponent-pool, and (future) live
all write through `data_layer` with the same schema.

---

## 5. Build order

Sequenced so each step produces something runnable and testable. Maps to the
§13 acceptance milestones.

**M1 — Environment + one logged game (thin vertical slice).**
Launch local Showdown, parse the current team (incl. Mega resolution) into the
generic representation, play one full game agent-vs-scripted, and write it end to
end into the SQLite schema. Proves the spine works before layering ML on it.
*Accept:* a complete game plays and appears fully in `games`/`turns`.

**M2 — Data layer + damage-calc utility.** Finalize schema (see §6), WAL mode,
CSV export, and the shared damage calculator wrapper (verify M-B coverage;
document gaps publicly, don't guess). Everything downstream logs through this.
*Accept:* tables populate from M1 games; CSV export works; damage calc returns
sane numbers for a known M-B interaction.

**M3 — Fixed-shape env contract.** Lock the generic observation/action encoding
(team-agnostic slots) and action masking for doubles. This is the warm-start
foundation — do it before any training so IL and PPO share identical I/O shapes.
*Accept:* obs/action shapes are invariant when the team is swapped for a
different legal M-B team.

**M4 — Imitation baseline.** Ingest replays via poke-env's own protocol parser →
(state, action) pairs → behavior-clone a net matching the PPO policy net's
input/output shape. *Accept:* beats random/scripted baseline head-to-head.

**M5 — PPO self-play loop.** PPO initialized from the M4 checkpoint (real warm
start = load weights). Games drawn from self-play (vs current/past checkpoints) +
opponent pool. Concurrent battles, checkpoint every N games (start N=500,
tunable), all logged through the data layer. *Accept:* sustained run without
crashing, checkpoints save/reload.

**M6 — Opponent pool.** Scrape 50–100 M-B teams (replay search + pokepastes),
parse into the §4 representation, rule-based archetype tags, Jaccard diversity
check. *Accept:* ≥50 distinct tagged teams; near-duplicates flagged.

**M7 — Evaluation gate + promotion.** New checkpoint plays a fixed set (e.g. 100
games) vs current champion, team constant both sides. Promote iff win rate >
threshold (start 55%, tunable, logged). Every eval logged (win rate, N,
promoted?). *Accept:* two checkpoints auto-pitted, decision logged.

**M8 — Retrain trigger (team edit).** CLI takes a new team paste, creates a new
team version linked to the current champion as warm-start parent, fine-tunes
(M5), runs the gate (M7). Small and large edits use the identical path.
*Accept:* an edit produces a new version, warm-starts, runs the gate.

**M9 — Analytics + "analyze this turn".** Facts generator (streaks, closest
game, top Pokémon, biggest upset/blunder — deterministic, no model). Charts
(learning curve, win-rate-by-archetype, per-Pokémon usage, notable-game curve).
The turn-analysis tool exposing policy %s + value estimate (refinement #2).
*Accept:* learning curve + win-rate-by-archetype render from real data; turn tool
returns move %s and win estimate for a given board state.

**M10 — Discord digest.** End-of-session webhook post: session summary +
2–3 facts + one chart image. Webhook URL from env var / local config.
*Accept:* real session summary + facts + chart post to a test webhook.

---

## 6. Data schema (from §11, unchanged intent)

- `team_versions` — version_id, timestamp, raw_paste, parsed_repr, parent_checkpoint
- `games` — game_id, ts, team_version, opponent_archetype, result, turn_count,
  checkpoint_id, source (`local_selfplay` / `local_opponent_pool` / `live`)
- `turns` — game_id, turn_number, win_prob (value head), move_probs (policy head,
  supports refinement #2), active mons both sides, action, dmg dealt/received
- `pokemon_stats` — per mon per team_version: brought, win-rate-when-brought, KOs, faints
- `move_stats` — per move: usage counts + outcomes, per team_version
- `evaluations` — champion_id, challenger_id, n_games, win_rate, threshold, promoted

`source` and the `live` value exist now so Phase 2 live data slots in with no
schema change.

---

## 7. Open decisions parked as tunables (rule exists, number is adjustable)

- Checkpoint interval (default 500 games)
- Eval set size (default 100 games) and promotion threshold (default >55%)
- Reward shaping weight (damage-calc-based) vs terminal win/loss
- Session size N that triggers a Discord digest

---

## 8. Non-goals (unchanged, Phase 1)

No live ladder / real account, no screen/OCR analysis, no Reg M-B legality
validation, no persistent dashboard app, no interactive Discord bot, no
multi-team support.
