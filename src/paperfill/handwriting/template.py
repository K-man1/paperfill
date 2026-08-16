"""
Printed handwriting template + its calibrated geometry.

The template itself is ``template.pdf`` (a multi-page raster the user prints and
fills in). Its machine-readable geometry — per-page marker centres and per-cell
drawing rectangles in canonical page pixels — is calibrated once into
``template_geometry.json`` (see ``calibrate.py``). The font builder reads that
geometry; nothing here re-derives it at runtime.

``data/Calligraphr-Template.pdf`` is the upstream Calligraphr sheet that
``template.pdf`` was originally derived from. No code loads it — it's kept as
provenance so the template can be regenerated or re-cut later. Deleting it
costs nothing at runtime but loses that trail.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE_PDF = HERE / "data" / "template.pdf"
GEOMETRY_JSON = HERE / "data" / "template_geometry.json"


@lru_cache(maxsize=1)
def geometry() -> dict:
    return json.loads(GEOMETRY_JSON.read_text())


def pages() -> list[dict]:
    """Per-page geometry: {index, width, height, markers[4], cells[{glyph, draw}]}."""
    return geometry()["pages"]


def descenders() -> set[str]:
    """Glyphs whose tails dip below the baseline (negative font units)."""
    return set(geometry().get("descenders", "gjpqy"))


def glyphs() -> list[str]:
    """Every glyph the template captures, across all pages, in layout order."""
    return [c["glyph"] for p in pages() for c in p["cells"]]


def template_pdf_bytes() -> bytes:
    return TEMPLATE_PDF.read_bytes()


def page_count() -> int:
    return len(pages())
