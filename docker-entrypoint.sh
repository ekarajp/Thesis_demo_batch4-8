#!/usr/bin/env bash
# Container entrypoint: prepare portable runtime -> fail-closed preflight -> exec CMD.
# Default CMD launches the dashboard on 0.0.0.0:8765 (see Dockerfile).
# Override to run a one-off CLI command, e.g.:
#   docker run --rm rc-fragility-poc python -m fragility_poc.cli --help
set -euo pipefail
cd /app

echo "[entrypoint] Preparing portable runtime..."
python /app/prepare_portable_runtime.py --root /app

echo "[entrypoint] Running fail-closed scientific preflight..."
python /app/preflight_portable.py --root /app

echo "[entrypoint] Exec: $*"
exec "$@"
