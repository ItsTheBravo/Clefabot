#!/usr/bin/env bash
# Idempotent: start the local Showdown server if it isn't already listening.
# Fully detached (setsid + nohup) so it survives the launching shell's exit.
set -euo pipefail
cd "$(dirname "$0")/.."

if node -e "require('net').connect(8000,'127.0.0.1',()=>process.exit(0)).on('error',()=>process.exit(1))" 2>/dev/null; then
  echo "server already up"
  exit 0
fi

LOG="${PS_SERVER_LOG:-/tmp/ps-server.log}"
setsid nohup node server/pokemon-showdown start --no-security >"$LOG" 2>&1 < /dev/null &
for i in $(seq 1 30); do
  sleep 1
  if node -e "require('net').connect(8000,'127.0.0.1',()=>process.exit(0)).on('error',()=>process.exit(1))" 2>/dev/null; then
    echo "server up (log: $LOG)"
    exit 0
  fi
done
echo "server failed to start; last log lines:" >&2
tail -5 "$LOG" >&2
exit 1
