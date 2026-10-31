#!/usr/bin/env bash
# Run the portfolio optimizer pipeline inside the project virtual environment.
# All arguments are passed through to portfolio_optimizer.py.
#
# Usage:
#   ./run.sh                          # full pipeline
#   ./run.sh --skip ingest            # reuse downloaded data
#   ./run.sh --only simulate --paths 2000
#
# The ingest stage needs an SEC contact identity:
#   export EDGAR_IDENTITY="Your Name your.email@example.com"

set -euo pipefail

cd "$(dirname "$0")"
VENV_PYTHON=".venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "No virtual environment found; running ./setup_env.sh first"
    ./setup_env.sh
fi

exec "$VENV_PYTHON" portfolio_optimizer.py "$@"
