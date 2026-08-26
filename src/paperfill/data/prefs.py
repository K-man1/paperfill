"""
Per-account fill preferences, set on the /settings page: whether to stamp the
Goodnotes watermark, whether to auto-fill "Name"/"Date" header blanks (and
what name to use), and the default way a freshly-detected graph answer is
drawn ("points", "curve", or "none" — see GRAPH_MODES in index.html).

File-backed JSON keyed by the same user key as the credit ledger (_user_key()
in app.py — a Google sub or "email:<addr>"), same pattern as the ads-enabled
/ auto-Pro settings, just keyed per account instead of global.
"""

from __future__ import annotations

import json

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
    try:
        return json.loads(PREFS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write(data: dict) -> None:
    PREFS_PATH.write_text(json.dumps(data, indent=2))


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
    data = _read()
    data[user_key] = clean
    _write(data)
    return clean
