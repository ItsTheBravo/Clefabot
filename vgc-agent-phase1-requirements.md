# VGC Team Agent — Phase 1 Requirements

## 1. Objective

Build a local training pipeline that learns to pilot one specific, fixed VGC team (Pokémon Champions, Regulation M-B, doubles) through iterative self-play, starting from an imitation-learned baseline. The system must support re-training from a warm start whenever the team is edited, without needing a full rebuild.

This is also a data analysis project, not just a training pipeline. Every game played — local training, self-play, or (in later phases) live ladder — must be logged into a structured, queryable data layer that produces visualizations, notable facts, and a periodic Discord digest of session results. The data schema should be forward-compatible with live ladder games even though Phase 1 only produces local ones (see §11).

Output of Phase 1 is a working local loop: play games → learn → evaluate → promote if better → repeat — plus the analytics and Discord reporting built on top of that same game log. Live deployment and screen/video analysis are still later phases.

## 2. Non-Goals (Phase 1)

- No connection to live Pokémon Showdown ladder or a real bot account
- No Pokémon Champions screen/video analysis or OCR
- No team legality validation against the Reg M-B ban list (explicitly deferred)
- No persistent interactive dashboard app (e.g. a Streamlit/web UI) — visualizations are generated artifacts (static charts + exportable tables), not a live app. If an interactive dashboard is wanted later, Power BI against the exported tables is the natural route, not a custom-built one.
- No interactive Discord bot (slash commands, queries) — v1 is a one-way webhook digest, not a bot with a gateway connection
- No support for multiple simultaneous teams — single fixed team per training lineage

## 3. Environment & Dependencies

- Node.js — to run a local Pokémon Showdown server (`pokemon-showdown start --no-security`) for unlimited, fast, non-ladder games
- Python 3.10+
  - `poke-env` — Showdown protocol client, Gymnasium-compatible RL environment
  - `stable-baselines3` — PPO implementation
  - `torch` — required for the imitation baseline (see §7 — architecture must be warm-start-compatible with the PPO policy network)
  - `sqlite3` (stdlib) — structured game/turn/analytics storage
  - `matplotlib` or `plotly` — chart generation
  - `requests` — Discord webhook delivery
  - A damage calculator library compatible with the current ruleset (Mega Evolution, Reg M-B items/movepool) — verify coverage before relying on it for reward shaping (see §14)
- Target hardware: CPU-only. No GPU-dependent code paths. If building on Apple Silicon, PyTorch's MPS backend can be used opportunistically but must not be required.

## 4. Team Input & Representation

**Input:** Standard Showdown export text block (species / item / ability / level / EVs / nature / moves), as pasted by the user.

**Requirement:** Parse into a *generic* feature representation — base stats, typing, movepool, ability, item, computed stats from EVs/nature/level — never hardcoded per-slot ("Pokémon #3"). This applies identically to the user's own team and to opponent teams. This is the design requirement that makes a future team edit a re-train rather than a re-architecture.

**Versioning:** Each distinct team (or edit to it) gets a version ID and timestamp, and is linked to the checkpoint lineage it produces (see §10). Store the raw pasted text alongside the parsed representation for traceability.

## 5. Battle Environment

- Local Showdown server, dev mode, no ladder/rating involvement
- `poke-env` `Player` classes for both sides
- Format: the current Reg M-B doubles format as it actually exists on the Showdown server being used — **do not hardcode a guessed format string**; confirm it against the server's live format list at setup time (see §14)
- Damage calculator wired in as a shared utility, used both for reward shaping/state features and for the analytics layer — don't let the model re-derive damage math from raw stats

## 6. Opponent Pool

- Source real Reg M-B teams from Showdown's replay search API for the confirmed format, supplemented by public team dumps (Pokepaste links, Smogon RMT-equivalent threads) if available
- Target starting size: 50–100 distinct teams, growing over time
- Include a basic diversity check (e.g. Jaccard similarity on species composition) so the pool isn't dominated by near-duplicate netdecks
- Store as parseable team files using the same representation as §4, tagged with an archetype label (e.g. Trick Room, hyper offense, bulky balance) — even a rough rule-based tag is enough to power the "win rate by archetype" analytics in §11

## 7. Imitation Learning Baseline

- Parse historical replay logs (same target format) into (state, action) pairs by feeding saved replay log lines through poke-env's own protocol message parser — same parser it uses for live battles, just pointed at static logs instead of a socket
- **Architecture requirement:** the baseline model must be a neural network with an architecture compatible with the PPO policy network used in §8 — this is what makes "warm start" literally mean initializing PPO's weights from this model, not just a vague head start. Use behavior cloning (supervised training toward matching the PPO policy net's input/output shape), not a tree-based model — trees can't donate weights to a neural net.
- Output: a checkpoint that plays measurably better than a random or simple scripted baseline before any RL begins

## 8. Self-Play RL Loop

- Gymnasium-wrapped poke-env environment, PPO via stable-baselines3
- Policy/value network initialized from the §7 imitation checkpoint, not random init
- Training games drawn from: self-play (current model vs. current or recent-past checkpoints) and opponent-pool play (current model vs. the real teams from §6)
- Regular checkpointing (by game count or update count — pick a concrete interval, e.g. every 500 games)
- Every game played here must be logged through the same schema as §11, not a separate ad hoc log — the training loop and the analytics layer share one data pipeline

## 9. Evaluation Gate

- Every new checkpoint plays a fixed evaluation set (e.g. 100 games) against the current "champion" checkpoint, team held constant on both sides
- Promotion rule: new checkpoint becomes champion only if it wins clearly more than it loses against the previous champion (define and log an explicit threshold, e.g. >55% win rate over the eval set — exact number is tunable, but there must be a rule, not a manual judgment call)
- Log every evaluation result (win rate, game count, promoted true/false) for later reporting

## 10. Team Versioning & Retrain Trigger

- Script/CLI entry point: accept a new team paste (full team or a described edit)
- Create a new team version record, linked to the current champion checkpoint as its warm-start parent
- Kick off a fine-tune run (§8 loop) from that parent, against the same opponent pool
- Run through the same evaluation gate (§9) before the new team version's model is marked current
- Small edits (single move/item swap) and large edits (swapped Pokémon) both go through this same path — expect the latter to need more training games to reconverge, but the pipeline doesn't change

## 11. Data & Analytics Layer

**Storage:** SQLite as the primary store — queryable locally, and importable into Power BI directly (native ODBC/connector support) if an interactive dashboard is ever wanted on top of this. CSV export of each table as a secondary output.

**Schema (minimum):**
- `games` — game_id, timestamp, team_version, opponent_archetype, result, turn_count, checkpoint_id, source (`local_selfplay` / `local_opponent_pool` / `live` — only the first two populate in Phase 1, but the field exists now so live data slots in later without a schema change)
- `turns` — game_id, turn_number, win_probability (from the PPO value head, if exposed), active Pokémon both sides, action taken, damage dealt/received
- `pokemon_stats` — per-Pokémon, per-team-version: games brought, win rate when brought, KOs, times fainted
- `move_stats` — per-move usage counts and outcomes, per team version

**Visualizations (generated artifacts, not a live app):**
- Win rate over training iterations (learning curve)
- Win rate by opponent archetype
- Per-Pokémon contribution/usage chart
- Win-probability-over-time chart for notable individual games (closest win, biggest upset)

Generate these as static images (matplotlib/plotly) on a schedule tied to training sessions, not on every single game.

**Facts generator:** deterministic, rule-based extraction from the tables above — no separate model needed for this. Minimum set: longest win/loss streak, closest game (smallest margin or largest win-probability swing that still won), most-used/highest-win-rate Pokémon of the session, biggest upset (won against an opponent archetype it has a low historical win rate against), biggest blunder (a turn where win probability dropped sharply in a game that was ultimately lost).

## 12. Discord Summary

- **Delivery mechanism:** Discord webhook (a plain HTTP POST to a webhook URL) — not a full bot with a gateway connection. This is a one-way digest, which is all v1 needs.
- **Trigger:** end of each completed training/battling session (a batch run of N games), rather than a fixed daily cron — this is more robust to CPU-only training that runs in bursts rather than continuously. If training does end up running continuously, a daily cutoff can be layered on later without changing the underlying digest logic.
- **Content of each post:**
  - Session summary: games played, wins/losses, win rate, checkpoint promoted (yes/no)
  - 2–3 facts pulled from §11's facts generator
  - One attached chart image (e.g. the session's win-probability curve or the updated learning curve) generated by the visualization module
- **Config:** webhook URL supplied via environment variable or local config file — never hardcoded, since it's effectively a secret

## 13. Milestones & Acceptance Criteria

1. **Environment stood up** — the team can play a full local game start-to-finish against a scripted opponent, fully logged
2. **Imitation baseline trained** — beats a random/scripted baseline in head-to-head evaluation
3. **Self-play loop running** — PPO trains for a sustained run without crashing, checkpoints save correctly
4. **Evaluation gate working** — two arbitrary checkpoints can be pitted against each other automatically and a promotion decision is logged
5. **Retrain trigger working** — a team edit produces a new version, warm-starts correctly from the prior champion, and runs through the same gate
6. **Data layer populated** — games/turns/pokemon_stats/move_stats tables are correctly filled from real training runs, and can be exported to CSV
7. **Visualizations generated** — at least the learning curve and win-rate-by-archetype charts can be produced from real data
8. **Discord digest working** — a real session's summary, facts, and chart post successfully to a test webhook end-to-end

## 14. Open Items to Resolve During Build (do not assume — verify)

- Exact current Reg M-B doubles format identifier on the Showdown server in use — pull from the server's live format list, don't hardcode
- `poke-env` version compatibility with the Showdown server version being deployed locally
- Damage calculator library coverage for Mega Evolution and any Reg M-B-specific items/mechanics — confirm before relying on it for reward shaping; note publicly, don't guess
- Availability and format of Reg M-B replay data via Showdown's replay search endpoint for the confirmed format string above
- Whether stable-baselines3's PPO exposes the value head cleanly enough for per-turn win-probability logging (§11) — if not, note the fallback (e.g. a separate small value estimator) rather than silently dropping that feature

---

## Appendix A: Current Team (Reg M-B)

```
Raichu-Mega-Y @ Raichunite Y
Ability: Lightning Rod
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Grass Knot
- Zap Cannon
- Focus Blast
- Protect

Clefable @ Sitrus Berry
Ability: Cute Charm
Level: 50
EVs: 32 HP / 17 Def / 12 SpD / 5 Spe
Bold Nature
- Moonblast
- Follow Me
- Helping Hand
- Protect

Mamoswine @ Life Orb
Ability: Oblivious
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Jolly Nature
- High Horsepower
- Ice Shard
- Icicle Crash
- Protect

Ceruledge @ Colbur Berry
Ability: Flash Fire
Level: 50
EVs: 31 HP / 25 Atk / 10 Def
Adamant Nature
- Bitter Blade
- Bulk Up
- Shadow Sneak
- Protect

Kingambit @ Focus Sash
Ability: Defiant
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Iron Head
- Low Kick

Meowscarada @ Black Glasses
Ability: Protean
Level: 50
EVs: 32 Atk / 2 SpA / 32 Spe
Adamant Nature
- Protect
- Knock Off
- Flower Trick
- Thunder Punch
```
