#!/usr/bin/env bash
# Fetch and build a local Pokemon Showdown server that includes the Champions
# Reg M-B format.
#
# NOTE: the npm-published `pokemon-showdown` package is stale (only up to Reg
# G/H — no Champions/Reg M). We pull the current source from GitHub instead
# (verified to contain "[Gen 9 Champions] VGC 2026 Reg M-B", id
# gen9championsvgc2026regmb). See PLAN.md §0.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d server ]; then
  echo "server/ already exists — remove it to re-fetch."
  exit 0
fi

echo "Downloading pokemon-showdown source..."
curl -sSL -o /tmp/ps.tar.gz \
  "https://codeload.github.com/smogon/pokemon-showdown/tar.gz/refs/heads/master"
tar xzf /tmp/ps.tar.gz
mv pokemon-showdown-master server
rm -f /tmp/ps.tar.gz

echo "Installing server dependencies..."
( cd server && npm install --no-audit --no-fund )

echo "Building..."
( cd server && node build )

echo "Done. Start it with: scripts/start_server.sh"
