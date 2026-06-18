#!/usr/bin/env bash
# Start PaperFill in production (gunicorn). Used by Nest and any plain VM.
#
#   ./run.sh
#
# Reads .env (PORT, HCAI_API_KEY). PORT comes from `nest get-port`.
set -euo pipefail
cd "$(dirname "$0")"

# Activate the venv if present. The Nest container uses .venv; a plain VM
# setup may use venv. Prefer whichever exists.
if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [ -d venv ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

exec gunicorn -c gunicorn.conf.py app:app
