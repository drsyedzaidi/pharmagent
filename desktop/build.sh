#!/usr/bin/env bash
# Build the PharmAgent desktop app (macOS: desktop/dist/PharmAgent.app).
#
#   desktop/build.sh            # frontend build + PyInstaller bundle
#   desktop/build.sh --smoke    # ...then launch headless and hit /api/health
#
# Requires: backend/.venv with requirements.txt + requirements-desktop.txt,
# node for the frontend. Unsigned: first launch needs right-click > Open (or
# `xattr -dr com.apple.quarantine PharmAgent.app`) until the bundle is signed
# and notarized.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/backend/.venv/bin/python"

echo "== frontend build"
npm --prefix "$ROOT/frontend" run build

echo "== pyinstaller"
"$PY" -m PyInstaller "$ROOT/desktop/PharmAgent.spec" --noconfirm \
  --distpath "$ROOT/desktop/dist" --workpath "$ROOT/desktop/build"

APP="$ROOT/desktop/dist/PharmAgent.app"
BIN="$APP/Contents/MacOS/PharmAgent"
[ -x "$BIN" ] || BIN="$ROOT/desktop/dist/PharmAgent/PharmAgent"
echo "== built: $APP"
du -sh "$APP" 2>/dev/null || true

if [ "${1:-}" = "--smoke" ]; then
  echo "== smoke: headless launch"
  PORT=${PHARMAGENT_DESKTOP_PORT:-8765}
  TMPDATA="$(mktemp -d)"
  PHARMAGENT_DESKTOP_NO_WINDOW=1 PHARMAGENT_DESKTOP_PORT=$PORT \
    PHARMAGENT_DESKTOP_DATA_DIR="$TMPDATA" "$BIN" &
  PID=$!
  trap 'kill $PID 2>/dev/null || true' EXIT
  for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then break; fi
    sleep 1
  done
  curl -fsS "http://127.0.0.1:$PORT/api/health"; echo
  curl -fsS "http://127.0.0.1:$PORT/" | grep -q '<div id="root">' && echo "frontend served OK"
  kill $PID; wait $PID 2>/dev/null || true
  echo "== smoke OK (data dir $TMPDATA)"
fi
