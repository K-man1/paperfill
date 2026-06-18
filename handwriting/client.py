"""
Nest-side client for the One-DM Modal service.

Keeps the GPU service at arm's length: one batched HTTP call per worksheet,
returns {overlay_id: transparent_png_bytes}. If the service is unconfigured
or errors, callers fall back to normal text rendering.
"""

import base64
import os

import requests

MODAL_URL = os.environ.get("ONEDM_MODAL_URL", "").strip()
AUTH_TOKEN = os.environ.get("ONEDM_AUTH_TOKEN", "").strip()
TIMEOUT_S = float(os.environ.get("ONEDM_TIMEOUT", "150"))


def handwriting_enabled() -> bool:
    """True if handwriting output is available at all: either the One-DM Modal
    service is configured, or at least one user-built font exists locally."""
    if MODAL_URL:
        return True
    try:
        from .font_store import has_fonts
        return has_fonts()
    except Exception:
        return False


def generate_handwriting(style_b64: str, items: dict[str, str]) -> dict[str, bytes]:
    """
    items: {overlay_id: answer_text}. Returns {overlay_id: png_bytes}.
    One request for the whole worksheet (Modal loads the model once, loops
    on the warm GPU). Empty answers are dropped before sending.
    """
    items = {k: v for k, v in items.items() if str(v).strip()}
    if not (MODAL_URL and style_b64 and items):
        return {}

    resp = requests.post(
        MODAL_URL,
        json={"auth_token": AUTH_TOKEN, "style_b64": style_b64, "items": items},
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()
    return {k: base64.b64decode(v) for k, v in data.items()}
