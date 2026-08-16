"""
Production gunicorn config for PaperFill.

Reads PORT from the environment so it drops straight into Hack Club Nest:
run `nest get-port`, put the number in .env as PORT, and gunicorn binds to it.
Defaults to 8080 if PORT is unset.
"""

import os
import sys
from pathlib import Path

# The app lives in src/paperfill/. Put src/ and the repo root on sys.path here,
# in the config file gunicorn imports before the app, so `paperfill.app:app`
# resolves without a `pip install -e .` step. That keeps the Nest deploy a
# plain `git pull` + `systemctl restart` with nothing else to remember.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# gunicorn's cwd decides where uploads/ and outputs/ land, so pin it to the
# repo root rather than wherever the service happened to be started from.
chdir = str(_REPO_ROOT)

# Bind. On Nest, Caddy proxies to this port on the same box.
bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"

# Two workers is plenty for a single-user-ish tool. Jobs are mirrored to disk
# (see save_job/load_job in app.py) so a fill can land on a different worker
# than the upload did.
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))

# /api/fill can make several LLM calls and then rasterise + stamp handwriting
# for every slot on the page, all inside the one request. Keep the worker
# timeout well above a realistic worst-case fill so a slow multi-page job
# doesn't get the worker killed mid-render.
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "180"))

# Recycle workers periodically to bound any slow memory growth from PyMuPDF.
max_requests = 200
max_requests_jitter = 40

# Log to stdout/stderr so `nest logs` / journald captures everything.
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("GUNICORN_LOGLEVEL", "info")
