"""
One-time calibration of the printed template (handwriting/template.pdf) into a
geometry spec the font builder reads: per-page marker centres and per-cell
drawing rectangles, in canonical page pixels (the PDF rendered at DPI).

The template is a flat raster, so we measure it instead of reading coordinates
from the PDF: detect the four corner markers, detect the regular cell grid from
the fully-populated first page, and reuse that grid on every page (the layout is
identical across pages — only which cells hold a glyph differs).

Re-run after changing the template:  python -m handwriting.calibrate
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import fitz

from .font_build import _find_markers

HERE = Path(__file__).resolve().parent
TEMPLATE_PDF = HERE / "template.pdf"
GEOMETRY_JSON = HERE / "template_geometry.json"

DPI = 150
LABEL_FRAC = 0.16          # top of cell reserved for its printed label
INSET = 6                  # px inset from cell borders for the drawing area
DESCENDERS = "gjpqy"

# Row-major glyph layout of the printed template, one string per grid row, per
# page. Empty/QR cells are simply omitted (rows are left-aligned). Edit this to
# match the template if its glyph set changes.
GLYPH_MAP = {
    # Page 0: the QR occupies the last two columns of rows 0-1, so those rows
    # hold only 6 glyphs; the remaining rows are full width (8).
    0: ["!\"'()+", ",-.123", "456789:;", "<>?ABCDE",
        "FGHIJKLM", "NOPQRSTU", "VWXYZabc", "defghijk"],
    # Page 1: the larger QR occupies the last three columns of rows 0-1, so
    # those rows hold only 5 glyphs.
    1: ["lmnop", "qrstu", "vwxyz±×á", "éíñó÷ú→√", "∛∞≈≠≤≥"],
}


def _render(page) -> np.ndarray:
    pix = page.get_pixmap(dpi=DPI)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    code = cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR
    return cv2.cvtColor(arr, code)


def _peaks(proj: np.ndarray, frac: float) -> list[int]:
    thr = proj.max() * frac
    out, i, n = [], 0, len(proj)
    while i < n:
        if proj[i] >= thr:
            j = i
            while j < n and proj[j] >= thr:
                j += 1
            out.append((i + j) // 2)
            i = j
        else:
            i += 1
    return out


def _detect_grid(gray: np.ndarray) -> tuple[list[int], list[int]]:
    """Return (column x-boundaries, row y-boundaries) for the regular grid."""
    bw = (gray < 235).astype(np.uint8) * 255
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))
    xs = _peaks(vert.sum(axis=0), 0.4)
    ys = _peaks(horiz.sum(axis=1), 0.4)
    # Each row boundary has a faint label-separator line ~0.16 cell below it;
    # keep only the true row tops (lines preceded by a full cell of space).
    rowtops = [ys[0]] + [y for p, y in zip(ys, ys[1:]) if y - p > 100]
    return xs, rowtops


def calibrate() -> dict:
    doc = fitz.open(str(TEMPLATE_PDF))
    g0 = cv2.cvtColor(_render(doc[0]), cv2.COLOR_BGR2GRAY)
    xs, rowtops = _detect_grid(g0)
    rh = (rowtops[-1] - rowtops[0]) / (len(rowtops) - 1)

    pages = []
    for pi in range(len(doc)):
        if pi not in GLYPH_MAP:
            continue
        img = _render(doc[pi])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        markers = _find_markers(gray)
        if markers is None:
            raise RuntimeError(f"page {pi}: could not find 4 corner markers")
        cells = []
        for r, rowstr in enumerate(GLYPH_MAP[pi]):
            yt, yb = rowtops[r], rowtops[r + 1]
            for c, ch in enumerate(rowstr):
                x0, x1 = xs[c], xs[c + 1]
                draw = [x0 + INSET, yt + rh * LABEL_FRAC, x1 - INSET, yb - INSET]
                cells.append({"glyph": ch, "draw": [round(v, 1) for v in draw]})
        pages.append({"index": pi, "width": gray.shape[1],
                      "height": gray.shape[0],
                      "markers": [[round(x, 1), round(y, 1)] for x, y in markers],
                      "cells": cells})
    doc.close()
    return {"dpi": DPI, "descenders": DESCENDERS, "pages": pages}


if __name__ == "__main__":
    geo = calibrate()
    GEOMETRY_JSON.write_text(json.dumps(geo, indent=1, ensure_ascii=False))
    n = sum(len(p["cells"]) for p in geo["pages"])
    print(f"wrote {GEOMETRY_JSON} — {len(geo['pages'])} pages, {n} glyph cells")
