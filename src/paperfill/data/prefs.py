"""
Per-account fill preferences, set on the /settings page: whether to stamp the
Goodnotes watermark, whether to auto-fill "Name"/"Date" header blanks (and
what name to use), and the default way a freshly-detected graph answer is
drawn ("points", "curve", or "none" — see GRAPH_MODES in index.html).

File-backed JSON keyed by the same user key as the credit ledger (_user_key()
in app.py — a Google sub or "email:<addr>"), same pattern as the ads-enabled
/ auto-Pro settings, just keyed per account instead of global.

Two gunicorn workers share the file, so it is locked the same way usage.json
is: an exclusive lock around save()'s read-modify-write (without it the second
writer drops the first one's row) and a shared lock around the read (without
it a fill mid-save reads a truncated file, and falling back to DEFAULTS there
stamps the watermark onto a sheet for someone who turned it off).
"""

from __future__ import annotations

import json

try:                      # POSIX only; on anything else we run lock-free.
    import fcntl
except ImportError:       # pragma: no cover - Windows dev boxes
    fcntl = None

from paperfill.paths import REPO_ROOT

PREFS_PATH = REPO_ROOT / "user_prefs.json"

PLOT_MODES = ("points", "curve", "none")

DEFAULTS = {
    "watermark": True,
    "fill_name_date": False,
    "name": "",
    "graph_plot": "points",
}


def _read() -> dict:
    """Every account's stored preferences. Empty when nobody has saved any;
    a JSONDecodeError here is real corruption, not a race, and is left to
    raise rather than being quietly answered with DEFAULTS."""
    try:
        fh = open(PREFS_PATH, encoding="utf-8")
    except FileNotFoundError:
        return {}
    with fh:
        if fcntl is not None:
            fcntl.flock(fh, fcntl.LOCK_SH)
        try:
            raw = fh.read().strip()
        finally:
            if fcntl is not None:
                fcntl.flock(fh, fcntl.LOCK_UN)
    data = json.loads(raw) if raw else {}
    return data if isinstance(data, dict) else {}


def get(user_key: str) -> dict:
    """This account's preferences, defaults filled in for anything missing or
    unset. Safe to call for an account that's never saved any."""
    out = dict(DEFAULTS)
    if not user_key:
        return out
    stored = _read().get(user_key) or {}
    out.update({k: stored[k] for k in DEFAULTS if k in stored})
    if out["graph_plot"] not in PLOT_MODES:
        out["graph_plot"] = "points"
    return out


def save(user_key: str, raw: dict) -> dict:
    """Validate and persist this account's preferences. Returns the cleaned
    values actually stored."""
    if not user_key:
        raise ValueError("no user key")
    clean = {
        "watermark": bool(raw.get("watermark", True)),
        "fill_name_date": bool(raw.get("fill_name_date", False)),
        "name": str(raw.get("name") or "")[:200].strip(),
        "graph_plot": raw.get("graph_plot")
                     if raw.get("graph_plot") in PLOT_MODES else "points",
    }
    with open(PREFS_PATH, "a+", encoding="utf-8") as fh:
        if fcntl is not None:
            fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            body = fh.read().strip()
            data = json.loads(body) if body else {}
            if not isinstance(data, dict):
                data = {}
            data[user_key] = clean
            fh.seek(0)
            fh.truncate()
            json.dump(data, fh, indent=2)
        finally:
            if fcntl is not None:
                fcntl.flock(fh, fcntl.LOCK_UN)
    return clean
