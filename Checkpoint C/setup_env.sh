#!/usr/bin/env bash
# Create (or update) the project virtual environment in .venv and install
# requirements.txt into it.
#
# Usage:
#   ./setup_env.sh              # uses python3 on PATH
#   PYTHON=python3.12 ./setup_env.sh
#   ./setup_env.sh --fresh      # delete and recreate .venv

set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
VENV_DIR=".venv"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: $PYTHON not found; set PYTHON=/path/to/python3.11+" >&2
    exit 1
fi

if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    echo "error: Python 3.11+ required (pandas 3.x); found $("$PYTHON" --version)" >&2
    exit 1
fi

if [[ "${1:-}" == "--fresh" && -d "$VENV_DIR" ]]; then
    echo "Removing existing $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating $VENV_DIR with $("$PYTHON" --version)"
    "$PYTHON" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip --quiet
"$VENV_DIR/bin/python" -m pip install -r requirements.txt

echo
echo "Environment ready. Activate it with:"
echo "  source \"$(pwd)/$VENV_DIR/bin/activate\""
echo "or run the pipeline directly with ./run.sh"
