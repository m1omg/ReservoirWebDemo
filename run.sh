#!/bin/sh
# Serve the Reservoir Computing Playground locally and open it in your browser.
# Works on any Linux distribution (and macOS) that has Python 3.
#
#   ./run.sh          # serves on port 8000
#   ./run.sh 9000     # pick another port
set -e
cd "$(dirname "$0")"
PORT="${1:-8000}"
# 127.0.0.1 counts as a secure origin, which browsers require for the microphone.
URL="http://127.0.0.1:$PORT/"

PY=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info[0] < 3)' 2>/dev/null; then
    PY="$candidate"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3 is needed for the little local web server." >&2
  echo "Install it with your package manager (apt install python3 / pacman -S python / dnf install python3)," >&2
  echo "or serve this folder with any other static web server." >&2
  exit 1
fi

echo "Reservoir Computing Playground: $URL"
echo "Press Ctrl+C to stop."
( sleep 1; xdg-open "$URL" >/dev/null 2>&1 || open "$URL" >/dev/null 2>&1 || true ) &
exec "$PY" -m http.server "$PORT" --bind 127.0.0.1
