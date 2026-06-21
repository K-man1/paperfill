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
import time
from pathlib import Path

FONTS_DIR = Path(__file__).resolve().parent / "fonts"
_INDEX = FONTS_DIR / "index.json"

LABEL = "Your handwriting"   # fixed — users don't name their font anymore
MAX_VARIANTS = 3


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


def _write_index(idx: dict) -> None:
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    _INDEX.write_text(json.dumps(idx, indent=2))


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

    idx = _read_index()
    idx[font_id] = {"label": LABEL, "owner": sub, "created": time.time(),
                    "variants": len(variants)}
    _write_index(idx)
    return font_id


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
