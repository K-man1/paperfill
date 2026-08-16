#!/usr/bin/env bash
# Start PaperFill in production (gunicorn). Used by Nest and any plain VM.
#
#   ./deploy/run.sh
#
# Reads .env (PORT, AI_API_KEY). PORT comes from `nest get-port`.
set -euo pipefail
# This script lives in deploy/; everything below assumes the repo root.
cd "$(dirname "$0")/.."

# Activate the venv if present. The Nest container uses .venv; a plain VM
# setup may use venv. Prefer whichever exists.
if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [ -d venv ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

exec gunicorn -c deploy/gunicorn.conf.py paperfill.app:app
