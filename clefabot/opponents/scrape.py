"""Scrape real Reg M-B opponent teams + replay logs (spec §6, M4 corpus).

NETWORK REQUIRED: talks to replay.pokemonshowdown.com, which is blocked in the
cloud dev container — run this on a normal machine (see README "Run it
locally"). Everything downstream is offline once the data is on disk.

Two outputs per run:
- ``opponents_pool/`` (committed): deduped, diversity-checked opponent teams as
  Showdown text files + ``manifest.json`` with archetype tags (§6). Teams come
  from Open Team Sheets (``|showteam|`` lines), so every replay yields both
  players' full teams.
- ``data/replays/`` (gitignored, regenerable): raw replay logs for the
  imitation corpus (§7).

Politeness: sequential requests with a small delay; resumable (skips replay
ids already downloaded).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import requests

from .archetype import is_near_duplicate, species_set, tag_team
from .packed import PackedMon, team_to_showdown_text, unpack_team

BASE = "https://replay.pokemonshowdown.com"
FORMAT_ID = "gen9championsvgc2026regmb"
SHOWTEAM_RE = re.compile(r"^\|showteam\|p[12]\|(.+)$", re.MULTILINE)


def search_replays(pages: int, session: requests.Session) -> list[dict]:
    out: list[dict] = []
    for page in range(1, pages + 1):
        r = session.get(f"{BASE}/search.json",
                        params={"format": FORMAT_ID, "page": page}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        time.sleep(0.5)
    return out


def fetch_replay_log(replay_id: str, session: requests.Session) -> str:
    r = session.get(f"{BASE}/{replay_id}.json", timeout=30)
    r.raise_for_status()
    return r.json().get("log", "")


def team_hash(mons: list[PackedMon]) -> str:
    payload = json.dumps(
        sorted((m.species, m.item, m.ability, sorted(m.moves)) for m in mons))
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def run(pages: int, pool_dir: str, replay_dir: str,
        max_teams: int | None = None) -> dict:
    pool = Path(pool_dir)
    pool.mkdir(parents=True, exist_ok=True)
    replays = Path(replay_dir)
    replays.mkdir(parents=True, exist_ok=True)
    manifest_path = pool / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    known_teams: dict[str, list[PackedMon]] = {}
    for th, meta in manifest.items():
        text = (pool / meta["file"]).read_text()
        # species-only reload is enough for the diversity check
        known_teams[th] = [PackedMon(s, "", "", [], "", {}, {})
                           for s in meta["species"]]

    session = requests.Session()
    session.headers["User-Agent"] = "clefabot-opponent-pool/0.1 (local research)"

    found = search_replays(pages, session)
    print(f"replay search: {len(found)} results over <= {pages} pages")

    new_teams, dupes, near_dupes = 0, 0, 0
    for entry in found:
        rid = entry["id"]
        log_path = replays / f"{rid}.log"
        if log_path.exists():
            log = log_path.read_text()
        else:
            try:
                log = fetch_replay_log(rid, session)
            except requests.RequestException as e:
                print(f"  skip {rid}: {e}")
                continue
            log_path.write_text(log)
            time.sleep(0.5)

        for packed in SHOWTEAM_RE.findall(log):
            mons = unpack_team(packed)
            if len(mons) < 6:
                continue
            th = team_hash(mons)
            if th in manifest:
                dupes += 1
                continue
            if any(is_near_duplicate(mons, other)
                   for other in known_teams.values()):
                near_dupes += 1
                continue
            fname = f"team_{th}.txt"
            (pool / fname).write_text(team_to_showdown_text(mons))
            manifest[th] = {
                "file": fname,
                "archetype": tag_team(mons),
                "species": sorted(species_set(mons)),
                "source_replay": rid,
            }
            known_teams[th] = mons
            new_teams += 1
            if max_teams and len(manifest) >= max_teams:
                break
        if max_teams and len(manifest) >= max_teams:
            break

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    by_arch: dict[str, int] = {}
    for meta in manifest.values():
        by_arch[meta["archetype"]] = by_arch.get(meta["archetype"], 0) + 1
    stats = {"pool_size": len(manifest), "new": new_teams,
             "exact_dupes_skipped": dupes, "near_dupes_skipped": near_dupes,
             "by_archetype": by_arch}
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scrape Reg M-B opponent teams + replays (network required)")
    ap.add_argument("--pages", type=int, default=10,
                    help="replay-search pages to walk (~50 replays/page)")
    ap.add_argument("--pool-dir", default="opponents_pool")
    ap.add_argument("--replay-dir", default="data/replays")
    ap.add_argument("--max-teams", type=int, default=None)
    args = ap.parse_args()
    run(args.pages, args.pool_dir, args.replay_dir, args.max_teams)


if __name__ == "__main__":
    main()
