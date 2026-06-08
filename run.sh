#!/usr/bin/env bash
# Start PaperFill in production (gunicorn). Used by Nest and any plain VM.
#
#   ./run.sh
#
# Reads .env (PORT, HCAI_API_KEY, ONEDM_*). PORT comes from `nest get-port`.
set -euo pipefail
cd "$(dirname "$0")"

# Activate the venv if it exists (Nest setup creates ./venv).
if [ -d venv ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

exec gunicorn -c gunicorn.conf.py app:app
