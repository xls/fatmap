#!/bin/sh
# fatmap bootstrap (Linux / macOS / MSYS). See bootstrap.py --help.
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
exec "$PY" bootstrap.py "$@"
