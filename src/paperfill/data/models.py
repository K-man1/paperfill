"""
Which model each AI call runs on, editable from the admin dashboard.

Every model id used to be pinned to an environment variable, so trying a new
one meant editing .env and restarting both gunicorn workers. An override saved
here lands in ai_models.json and wins over the env var; clear it and the slot
falls back to the env var, then to the slot it inherits from. Read fresh on
each call, so a change takes effect on the next fill in every worker without a
restart.

The slots are separate because they are different jobs. Detection needs
bounding-box grounding and gets a strict JSON schema; the text fill needs
neither. Which provider serves a call is not one of those jobs: the primary
proxy and the OpenRouter fallback share one model-id namespace, so a call that
fails over is retried on the same model rather than a separate one.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import NamedTuple

from paperfill.paths import REPO_ROOT

_PATH = REPO_ROOT / "ai_models.json"

# Only used when a slot has neither an override, an env var, nor anything to
# inherit from — i.e. a deployment that configured nothing at all.
DEFAULT_MODEL = "openai/gpt-5.5"


class ModelSlot(NamedTuple):
    env: str
    label: str
    inherits: str | None
    needs_vision: bool
    note: str


# Order is the order the dashboard renders them in.
SLOTS: dict[str, ModelSlot] = {
    "vision": ModelSlot(
        "VISION_MODEL", "Vision — Free tier", None, True,
        "Answers the worksheet from the page image. Reading and reasoning, no "
        "coordinates. The default every other vision slot inherits."),
    "vision_pro": ModelSlot(
        "VISION_MODEL_PRO", "Vision — Pro", "vision", True,
        "What Pro accounts get for the same work."),
    "ocr": ModelSlot(
        "OCR_MODEL", "Scanned-page OCR", "vision", True,
        "Fires automatically on image-only pages. The ONLY slot that emits "
        "coordinates itself: normalized [x0,y0,x1,y1], top-left origin. A model "
        "that answers well but grounds badly puts answers in the wrong place, "
        "and nothing catches it."),
    "detect": ModelSlot(
        "MULTIMODAL_MODEL", "Blank detection", "vision", True,
        "Names each blank by the printed text beside it, transcribed verbatim; "
        "code resolves that back to a box. Fails on paraphrase, not on bad "
        "coordinates."),
    "regions": ModelSlot(
        "REGION_MODEL", "Region selection", "vision", True,
        "Picks region ids off a numbered overlay code already drew. Multiple "
        "choice over boxes that exist, so it cannot invent a location."),
    "text_fill": ModelSlot(
        "AI_MODEL", "Text fill — Free tier", None, False,
        "Answers from the transcribed text alone, when the vision fill is off."),
    "text_fill_pro": ModelSlot(
        "AI_MODEL_PRO", "Text fill — Pro", "text_fill", False,
        "What Pro accounts get for the same work."),
}


def _read() -> dict:
    try:
        data = json.loads(_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def get(slot: str) -> str:
    """The model id this slot resolves to right now: saved override, else the
    env var, else whatever the slot inherits from."""
    spec = SLOTS[slot]
    saved = str(_read().get(slot) or "").strip()
    if saved:
        return saved
    env = os.environ.get(spec.env, "").strip()
    if env:
        return env
    return get(spec.inherits) if spec.inherits else DEFAULT_MODEL


def overrides() -> dict:
    """Only the slots an admin has actually pinned, for rendering the form."""
    saved = _read()
    return {k: str(saved.get(k) or "").strip() for k in SLOTS}


def resolved() -> dict:
    return {k: get(k) for k in SLOTS}


def save(raw: dict) -> dict:
    """Persist model overrides. An empty value clears the slot back to its env
    var. Ids with whitespace are rejected rather than stored — a typo here
    would fail every call the slot serves."""
    clean = {}
    for slot in SLOTS:
        val = str(raw.get(slot) or "").strip()
        if not val:
            continue
        if len(val) > 120 or any(c.isspace() for c in val):
            raise ValueError(f"{SLOTS[slot].label}: {val!r} is not a model id")
        clean[slot] = val
    _PATH.write_text(json.dumps(clean, indent=2))
    return clean


# ---- Provider catalog ----------------------------------------------------
# The primary provider publishes what it can serve, with modalities and list
# prices attached. Fetching it means the dashboard offers a picker of models
# that actually exist instead of a text box where one typo breaks every fill.
# Cached per worker: the catalog moves on the order of days, and a dashboard
# load should not wait on a second HTTP round-trip to a third party.

_CATALOG_TTL = 900
_catalog_cache: tuple[float, list[dict]] | None = None
_catalog_lock = threading.Lock()


def catalog() -> list[dict]:
    """Models the primary provider can serve: {id, vision, price_in, price_out}.
    Empty when the provider is unreachable — the picker degrades to free text
    rather than blocking the page."""
    global _catalog_cache
    now = time.monotonic()
    if _catalog_cache and now - _catalog_cache[0] < _CATALOG_TTL:
        return _catalog_cache[1]
    with _catalog_lock:
        if _catalog_cache and now - _catalog_cache[0] < _CATALOG_TTL:
            return _catalog_cache[1]
        try:
            entries = _fetch_catalog()
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            print(f"[models] could not fetch provider catalog: {e}")
            # Cache the failure too, so an outage doesn't add its timeout to
            # every dashboard load until it clears.
            entries = []
        _catalog_cache = (now, entries)
        return entries


def _fetch_catalog() -> list[dict]:
    base = (os.environ.get("AI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL", "https://ai.hackclub.com/proxy/v1"))
    key = (os.environ.get("AI_API_KEY") or os.environ.get("HCAI_API_KEY")
           or os.environ.get("OPENAI_API_KEY") or "")
    req = urllib.request.Request(
        base.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {key}"} if key else {},
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        payload = json.loads(resp.read().decode())

    def price(entry, field):
        try:                       # provider quotes $/token; we bill $/1M
            return round(float(entry.get("pricing", {}).get(field, 0)) * 1e6, 6)
        except (TypeError, ValueError):
            return 0.0

    seen, out = set(), []
    for m in payload.get("data") or []:
        mid = m.get("id") or ""
        if not mid or mid in seen:
            continue
        seen.add(mid)
        modalities = (m.get("architecture") or {}).get("input_modalities") or []
        out.append({
            "id": mid,
            "vision": "image" in modalities,
            "price_in": price(m, "prompt"),
            "price_out": price(m, "completion"),
        })
    return sorted(out, key=lambda e: e["id"])
