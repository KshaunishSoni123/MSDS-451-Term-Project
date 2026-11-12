#!/usr/bin/env bash
# Run Checkpoint B and Checkpoint C end to end inside one virtual environment.
# Both checkpoints have the same requirements.txt, so Checkpoint B's .venv
# serves both. All arguments are passed through to run_all.py.
#
# Usage:
#   ./run_all.sh                    # full refit, every stage
#   ./run_all.sh --help             # stages and options
#
# The ingest stage needs an SEC contact identity when SEC data is not cached:
#   export EDGAR_IDENTITY="Your Name your.email@example.com"

set -euo pipefail

cd "$(dirname "$0")"
VENV_PYTHON="Checkpoint B/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "No virtual environment found; running Checkpoint B/setup_env.sh first"
    "Checkpoint B/setup_env.sh"
fi

exec "$VENV_PYTHON" run_all.py "$@"
