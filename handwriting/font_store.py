"""
On-disk store for user-built handwriting fonts.

One ``.otf`` per font under ``handwriting/fonts/<id>.otf`` plus a small JSON
index of human labels.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

FONTS_DIR = Path(__file__).resolve().parent / "fonts"
_INDEX = FONTS_DIR / "index.json"


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return s[:32] or "hand"


def _read_index() -> dict:
    try:
        return json.loads(_INDEX.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_index(idx: dict) -> None:
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    _INDEX.write_text(json.dumps(idx, indent=2))


def font_path(font_id: str) -> Path | None:
    if not font_id:
        return None
    p = FONTS_DIR / f"{_slug(font_id)}.otf"
    return p if p.exists() else None


def save_font(name: str, otf_bytes: bytes) -> str:
    """Persist an OTF under a slug derived from ``name``; return its id.
    A colliding id is suffixed so existing fonts aren't clobbered."""
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    base = _slug(name)
    font_id = base
    n = 2
    while (FONTS_DIR / f"{font_id}.otf").exists():
        font_id = f"{base}-{n}"
        n += 1
    (FONTS_DIR / f"{font_id}.otf").write_bytes(otf_bytes)
    idx = _read_index()
    idx[font_id] = {"label": (name or font_id).strip()[:40], "created": time.time()}
    _write_index(idx)
    return font_id


def list_fonts() -> list[dict]:
    """[{id, label}] for every stored font (newest first)."""
    idx = _read_index()
    out = []
    for p in FONTS_DIR.glob("*.otf"):
        fid = p.stem
        meta = idx.get(fid, {})
        out.append({"id": fid, "label": meta.get("label", fid),
                    "created": meta.get("created", 0)})
    out.sort(key=lambda f: f["created"], reverse=True)
    return [{"id": f["id"], "label": f["label"]} for f in out]


def has_fonts() -> bool:
    try:
        return any(FONTS_DIR.glob("*.otf"))
    except OSError:
        return False
