#!/usr/bin/env bash
# One-shot local setup: Python venv + deps, Showdown server fetch + build.
# Requirements: Python 3.10+, Node.js 18+, git/curl. macOS or Linux (Windows:
# use WSL).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== checking prerequisites =="
command -v node >/dev/null || { echo "Node.js not found — install from https://nodejs.org"; exit 1; }
PY=python3
$PY -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
  || { echo "Python 3.10+ required"; exit 1; }
echo "node $(node --version), $($PY --version)"

echo "== python venv + deps =="
[ -d .venv ] || $PY -m venv .venv
.venv/bin/pip install -q --upgrade pip
# CPU-only torch wheels are much smaller; fall back to the default PyPI wheel
# (works on CPU, just bigger) if the pytorch index is unreachable.
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu \
  || .venv/bin/pip install -q torch
.venv/bin/pip install -q -r requirements.txt pytest

echo "== showdown server =="
scripts/setup_server.sh

echo "== sanity =="
.venv/bin/python -m pytest tests/ -q

cat <<'EOF'

Done. To run:

  scripts/ensure_server.sh                  # start the battle server (idempotent)
  .venv/bin/python -m clefabot.cli.play     # play one logged game (smoke test)

Rebuild the training state (regenerable artifacts are not in git):

  .venv/bin/python -m clefabot.imitation.collect   --games 150
  .venv/bin/python -m clefabot.imitation.train_bc
  .venv/bin/python -m clefabot.rl.selfplay --games 200

EOF
