"""
Production gunicorn config for PaperFill.

Reads PORT from the environment so it drops straight into Hack Club Nest:
run `nest get-port`, put the number in .env as PORT, and gunicorn binds to it.
Defaults to 8080 if PORT is unset.
"""

import os

# Bind. On Nest, Caddy proxies to this port on the same box.
bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"

# Two workers is plenty for a single-user-ish tool. Jobs are mirrored to disk
# (see save_job/load_job in app.py) so a fill can land on a different worker
# than the upload did.
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))

# /api/fill can call the LLM and then the One-DM handwriting service, whose
# own client timeout is 150s. Keep the worker timeout safely above that so a
# slow handwriting render doesn't get the worker killed mid-request.
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "180"))

# Recycle workers periodically to bound any slow memory growth from PyMuPDF.
max_requests = 200
max_requests_jitter = 40

# Log to stdout/stderr so `nest logs` / journald captures everything.
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("GUNICORN_LOGLEVEL", "info")
