"""
On-disk store for user-built handwriting fonts.

Each user has exactly ONE handwriting "font", which may be made of up to 3
*variants* (one per filled template copy they upload) so repeated letters can
look different. The font id is derived from the user's session subject, so a
user can only ever have one font and can't reach anyone else's.

Layout under ``handwriting/fonts/``:
  <id>.otf       primary variant
  <id>.v2.otf    second variant (optional)
  <id>.v3.otf    third variant (optional)
plus a small JSON index of metadata (owner, created, variant count).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

try:                      # POSIX only; on anything else we run lock-free.
    import fcntl
except ImportError:       # pragma: no cover - Windows dev boxes
    fcntl = None

from paperfill.paths import REPO_ROOT

# Built by users at RUNTIME, so this must not be anchored on __file__ the way
# the shipped template files below are: the package moved into src/ but these
# fonts did not, and repointing them would orphan every font already built on
# the server. Physical location is unchanged from before the src/ layout.
FONTS_DIR = REPO_ROOT / "handwriting" / "fonts"
_INDEX = FONTS_DIR / "index.json"

LABEL = "Your handwriting"   # fixed — users don't name their font anymore
MAX_VARIANTS = 3

# Per-user handwriting appearance knobs, tuned on /handwriting/settings.
# letter_spacing / font_size / word_spacing are percentages (100 = the scanned
# handwriting's natural size/spacing, unchanged). pen_thickness is a REAL,
# calibrated stroke width in millimetres on the printed page — the renderer
# measures the font's actual stroke and dilates/erodes it to hit this target,
# so e.g. 0.3 always means a ~0.3mm line, regardless of what pen was used to
# fill the template or how large a given answer is written. 0.4mm (a typical
# fine ballpoint) is the default; a bold fountain-pen nib runs 0.8-1.2mm.
DEFAULT_SETTINGS = {
    "letter_spacing": 100.0,
    "font_size": 100.0,
    "word_spacing": 100.0,
    "pen_thickness": 0.4,
}
SETTING_RANGES = {
    "letter_spacing": (2.0, 300.0),
    "font_size": (2.0, 300.0),
    "word_spacing": (2.0, 300.0),
    "pen_thickness": (0.1, 2.0),
}


def coerce_settings(raw: dict | None) -> dict:
    """Return a full settings dict with every value validated and clamped to its
    allowed range, filling anything missing or malformed with the default."""
    out = dict(DEFAULT_SETTINGS)
    raw = raw or {}
    for key, (lo, hi) in SETTING_RANGES.items():
        if raw.get(key) is None:
            continue
        try:
            out[key] = max(lo, min(hi, float(raw[key])))
        except (TypeError, ValueError):
            pass
    return out


def user_font_id(sub: str) -> str:
    """Deterministic, unguessable font id for a user (their session subject).
    Same user → same id, so rebuilding replaces their font and one user can
    never address another's."""
    h = hashlib.sha256((sub or "").encode("utf-8")).hexdigest()[:16]
    return f"u{h}"


def _read_index() -> dict:
    try:
        return json.loads(_INDEX.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _mutate(fn):
    """Run ``fn(idx)`` against the index and persist whatever it leaves behind,
    returning fn's result (same shape as ``data.usage._mutate``).

    Two gunicorn workers share this file, so the whole read-modify-write runs
    under an exclusive flock or one of them loses its update. The lock lives on
    a sibling rather than on the index itself, because the new index is swapped
    in with os.replace — that leaves a lock held on the index's old inode
    guarding nothing. Replacing rather than truncating in place also means a
    concurrent _read_index never sees a half-written file, which it would
    otherwise take for an empty index and then overwrite everyone else's entry
    with.

    Unlike the usage counter this does NOT fail open: an index we can't write
    is a font the user thinks they saved and didn't."""
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_INDEX.with_suffix(".lock"), "a+", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            idx = _read_index()
            result = fn(idx)
            tmp = _INDEX.with_suffix(".tmp")
            tmp.write_text(json.dumps(idx, indent=2))
            os.replace(tmp, _INDEX)
            return result
        finally:
            if fcntl is not None:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _variant_paths(font_id: str) -> list[Path]:
    """All on-disk variant OTFs for a font id, primary first."""
    if not font_id:
        return []
    out: list[Path] = []
    primary = FONTS_DIR / f"{font_id}.otf"
    if primary.exists():
        out.append(primary)
    for i in range(2, MAX_VARIANTS + 1):
        p = FONTS_DIR / f"{font_id}.v{i}.otf"
        if p.exists():
            out.append(p)
    return out


def font_path(font_id: str) -> Path | None:
    """Path to the primary variant if it exists, else None. Kept for callers
    that just need to know a font exists (e.g. style validation, samples)."""
    if not font_id:
        return None
    p = FONTS_DIR / f"{font_id}.otf"
    return p if p.exists() else None


def font_variant_paths(font_id: str) -> list[str]:
    """String paths of every variant for rendering (the renderer picks one per
    word). Empty list if the font doesn't exist."""
    return [str(p) for p in _variant_paths(font_id)]


def save_user_font(sub: str, otf_variants: list[bytes]) -> str:
    """Persist a user's handwriting as 1–3 OTF variants, replacing any previous
    font they had. Returns the font id."""
    if not otf_variants:
        raise ValueError("no font variants to save")
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    font_id = user_font_id(sub)

    # Clear any previous variants so a rebuild fully replaces the old font.
    for old in _variant_paths(font_id):
        old.unlink()

    variants = [b for b in otf_variants if b][:MAX_VARIANTS]
    for i, b in enumerate(variants):
        name = f"{font_id}.otf" if i == 0 else f"{font_id}.v{i + 1}.otf"
        (FONTS_DIR / name).write_bytes(b)

    def apply(idx: dict) -> None:
        # Rebuilding replaces the glyphs but keeps the appearance settings the
        # user tuned — a fresh scan shouldn't silently reset their
        # spacing/size choices.
        prior = (idx.get(font_id) or {}).get("settings")
        idx[font_id] = {"label": LABEL, "owner": sub, "created": time.time(),
                        "variants": len(variants),
                        "settings": coerce_settings(prior)}

    _mutate(apply)
    return font_id


def get_settings(font_id: str) -> dict:
    """The font's tuned appearance settings, defaults filled in for anything
    missing. Safe to call for a font that has never been tuned."""
    entry = _read_index().get(font_id) or {}
    return coerce_settings(entry.get("settings"))


def save_settings(font_id: str, settings: dict) -> dict:
    """Persist validated appearance settings for an existing font. Returns the
    clamped settings actually stored. Raises KeyError if the font doesn't
    exist."""
    if font_path(font_id) is None:
        raise KeyError(font_id)
    clean = coerce_settings(settings)

    def apply(idx: dict) -> None:
        # The OTFs on disk are what "having a font" means everywhere else, and
        # the index can fall behind them (a lost write, a font restored from
        # backup). Rebuild the entry from disk rather than refusing to store
        # settings for a font the user can plainly see.
        entry = idx.get(font_id)
        if not isinstance(entry, dict):
            entry = {"label": LABEL, "created": time.time(),
                     "variants": len(_variant_paths(font_id))}
            idx[font_id] = entry
        entry["settings"] = clean

    _mutate(apply)
    return clean


def user_font(sub: str) -> dict | None:
    """The user's font as {id, label, variants}, or None if they have none."""
    font_id = user_font_id(sub)
    paths = _variant_paths(font_id)
    if not paths:
        return None
    return {"id": font_id, "label": LABEL, "variants": len(paths)}


def list_fonts_for(sub: str) -> list[dict]:
    """The user's font(s) as a list (0 or 1 entry) — keeps the API shape the
    front-end expects while enforcing one-font-per-user."""
    f = user_font(sub)
    return [{"id": f["id"], "label": f["label"]}] if f else []
