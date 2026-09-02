#!/bin/bash
# GA Clinic - offline macOS launcher (double-click). Keep this window open while using the app.
cd "$(dirname "$0")" || exit 1
DIR="$(pwd)"
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
export OCT_BM_DL=1
export OCT_CLINIC_DATA="$DIR/user_data"
mkdir -p "$OCT_CLINIC_DATA"
PY="$DIR/runtime/python/bin/python3"
LIBS="$DIR/libs"

# A second double-click reopens the existing localhost instance.
if curl -fsS "http://127.0.0.1:8021/api/health" 2>/dev/null | grep -q '"app":"ga-clinic"'; then
  open "http://127.0.0.1:8021/"
  exit 0
fi

if [ ! -x "$PY" ]; then
  echo "GA Clinic - unpacking Python (first run only)..."
  tar -xzf "$DIR/runtime/python.tar.gz" -C "$DIR/runtime" || { echo "Unpack failed."; read -r _; exit 1; }
fi
if [ ! -d "$LIBS" ]; then
  echo "GA Clinic - first-time setup (a couple of minutes, no internet needed)..."
  TMP_LIBS="$DIR/.libs-install"
  rm -rf "$TMP_LIBS"
  "$PY" -m pip install --no-index --no-deps --find-links "$DIR/wheels" --target "$TMP_LIBS" -r "$DIR/requirements.txt" \
    || { rm -rf "$TMP_LIBS"; echo "Setup failed (see the messages above)."; read -r _; exit 1; }
  mv "$TMP_LIBS" "$LIBS"
fi
export PYTHONPATH="$LIBS:$DIR/app"
echo "GA Clinic starting - your browser opens when ready."
( for i in $(seq 1 150); do
    curl -s -o /dev/null "http://127.0.0.1:8021/api/health" && { open "http://127.0.0.1:8021/"; break; }
    sleep 0.8
  done ) &
exec "$PY" -m uvicorn clinic.api.app:app --host 127.0.0.1 --port 8021
