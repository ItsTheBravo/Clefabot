#!/usr/bin/env bash
# Start the local Showdown server in dev mode (no ladder/auth), for local,
# non-ladder training games only (spec §5).
set -euo pipefail
cd "$(dirname "$0")/../server"
exec node pokemon-showdown start --no-security
