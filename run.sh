#!/usr/bin/env bash
# Linux launcher — equivalent of START_UI.bat + SETUP_ENV.bat.
#
# The bundled .bat launchers and the offline wheelhouse/ are Windows-only
# (every binary wheel is *-win_amd64.whl, and openseespywin has no Linux build).
# On Linux we install the locked dependencies from PyPI online, swapping the
# Windows OpenSeesPy backend (openseespywin) for the Linux one (openseespylinux,
# pulled automatically by openseespy==3.5.1.3), then run preflight + dashboard.
#
# Requires: 64-bit CPython 3.10 on PATH as `python3.10`
#   Debian/Ubuntu: sudo apt install python3.10 python3.10-venv
set -euo pipefail

# --- cd to this script's directory (like `cd /d "%~dp0"`) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# ===========================================================================
# [1/4] Locate 64-bit CPython 3.10 (preflight_portable.py enforces this exactly)
# ===========================================================================
PYBIN=""
for cand in python3.10 python3.10.exe; do
  if command -v "$cand" >/dev/null 2>&1; then PYBIN="$cand"; break; fi
done
if [ -z "$PYBIN" ]; then
  echo "ERROR: python3.10 not found on PATH." >&2
  echo "Install 64-bit CPython 3.10 first (e.g. 'sudo apt install python3.10 python3.10-venv')." >&2
  exit 1
fi
if ! "$PYBIN" -c "import struct,sys; assert sys.version_info[:2]==(3,10) and struct.calcsize('P')*8==64" >/dev/null 2>&1; then
  echo "ERROR: $PYBIN is not 64-bit CPython 3.10." >&2
  exit 1
fi
echo "[1/4] Using interpreter: $("$PYBIN" -c 'import sys;print(sys.executable)')"

VENV_PY="$SCRIPT_DIR/.venv/bin/python"

# ===========================================================================
# [2/4] Create venv if missing (equivalent of SETUP_ENV.bat)
# ===========================================================================
if [ ! -x "$VENV_PY" ]; then
  echo "[2/4] Creating project-local environment..."
  "$PYBIN" -m venv "$SCRIPT_DIR/.venv"
  if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: venv creation failed (install python3.10-venv)." >&2
    exit 1
  fi
  "$VENV_PY" -m pip install --upgrade pip >/dev/null
else
  echo "[2/4] Existing project-local environment found."
fi

# ===========================================================================
# [3/4] Install locked dependencies (online — wheelhouse is Windows-only)
# ===========================================================================
if ! "$VENV_PY" -c "import fragility_poc, numpy, scipy, sklearn, openseespy.opensees" >/dev/null 2>&1; then
  echo "[3/4] Installing locked dependencies from PyPI..."
  "$VENV_PY" -m pip install setuptools==75.8.2 wheel==0.45.1
  # Drop the Windows-only backend pin; openseespy==3.5.1.3 resolves openseespylinux.
  REQS_FILTERED="$(mktemp)"
  grep -v -iE '^openseespywin' "$SCRIPT_DIR/requirements-lock.txt" > "$REQS_FILTERED"
  "$VENV_PY" -m pip install -r "$REQS_FILTERED"
  rm -f "$REQS_FILTERED"

  echo "[3/4] Verifying RC Fragility imports..."
  "$VENV_PY" -c "import fragility_poc, numpy, scipy, sklearn, openseespy.opensees; print('RC Fragility environment import: PASS')"
else
  echo "[3/4] Dependencies already installed."
fi

# ===========================================================================
# Prepare portable paths + verify ground-motion files
# ===========================================================================
echo "Preparing portable paths and verifying ground-motion files..."
"$VENV_PY" "$SCRIPT_DIR/prepare_portable_runtime.py" --root "$SCRIPT_DIR"

# ===========================================================================
# Fail-closed scientific preflight
# ===========================================================================
echo "Running fail-closed scientific preflight..."
"$VENV_PY" "$SCRIPT_DIR/preflight_portable.py" --root "$SCRIPT_DIR"

# ===========================================================================
# [4/4] Open dashboard (overridable via env: DASH_HOST / DASH_PORT)
# ===========================================================================
DASH_HOST="${DASH_HOST:-127.0.0.1}"
DASH_PORT="${DASH_PORT:-8765}"
URL="http://$DASH_HOST:$DASH_PORT"
echo "[4/4] Opening dashboard at $URL"
( command -v xdg-open >/dev/null 2>&1 && xdg-open "$URL" >/dev/null 2>&1 & ) || true

"$VENV_PY" "$SCRIPT_DIR/server_dashboard.py" --host "$DASH_HOST" --port "$DASH_PORT"
