"""
Build a real ``.otf`` from a photo / scan of a filled handwriting template.

Pipeline (no FontForge, no GPU):

  1. find the four concentric-square registration markers,
  2. warp the page back to canonical pixel space (template.py) so it works on
     phone photos, not just clean scans,
  3. crop each glyph's cell (cells are mapped by the known template layout —
     we never OCR the labels),
  4. threshold to an ink mask that keeps dark ink but drops the light ruled
     guides, denoise while keeping the dots on i / j,
  5. trace each mask with potrace (`-b svg --flat`) and build a CFF glyph,
     applying potrace's group transform composed with our font-unit transform,
  6. assemble an OTF with fontTools, synthesising space / period / comma.

CLI:  python -m handwriting.font_build <template image> <out.otf>
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from fontTools.agl import UV2AGL
from fontTools.fontBuilder import FontBuilder
from fontTools.misc.transform import Transform
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.pens.transformPen import TransformPen
from fontTools.svgLib.path import parse_path

from . import template as T

# Font design space.
UPM = 1000
ASCENT = 950
DESCENT = -350
SIDE_BEARING = 60          # font units of left/right bearing per glyph
MIN_COMPONENT_PX = 18      # connected components smaller than this are noise
INK_THRESHOLD = 120        # dark <= this is ink; lighter (ruled guides) dropped
DESCENDER_BASELINE = 0.65  # baseline sits this far down a descender's ink box


# ---- Marker detection + rectification ------------------------------------

def _is_square(cnt) -> bool:
    peri = cv2.arcLength(cnt, True)
    if peri < 40:
        return False
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return False
    _, _, w, h = cv2.boundingRect(approx)
    if w < 12 or h < 12:
        return False
    ar = w / float(h)
    return 0.6 < ar < 1.6


def _nested_square_depth(contours, hier, i: int) -> int:
    """Deepest chain of nested square contours below contour ``i``."""
    best = 0
    child = hier[i][2]
    while child != -1:
        if _is_square(contours[child]):
            best = max(best, 1 + _nested_square_depth(contours, hier, child))
        child = hier[child][0]
    return best


def _find_markers(gray: np.ndarray) -> list[tuple[float, float]] | None:
    """Return four marker centres ordered TL, TR, BR, BL, or None."""
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hier = cv2.findContours(th, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return None
    hier = hier[0]
    cands = []
    for i, cnt in enumerate(contours):
        # A marker's outer square encloses a white square enclosing a black
        # square: at least two levels of nested squares below it.
        if _is_square(cnt) and _nested_square_depth(contours, hier, i) >= 2:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cands.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
    if len(cands) < 4:
        return None

    h, w = gray.shape[:2]
    corners = [(0, 0), (w, 0), (w, h), (0, h)]  # TL, TR, BR, BL
    chosen: list[tuple[float, float]] = []
    used: set[int] = set()
    for corner in corners:
        best_i, best_d = -1, None
        for i, (cx, cy) in enumerate(cands):
            if i in used:
                continue
            d = (cx - corner[0]) ** 2 + (cy - corner[1]) ** 2
            if best_d is None or d < best_d:
                best_d, best_i = d, i
        if best_i < 0:
            return None
        used.add(best_i)
        chosen.append(cands[best_i])
    return chosen


def rectify(img: np.ndarray) -> np.ndarray:
    """Warp a photo/scan to canonical template pixel space using the four
    markers. Falls back to a plain resize if the markers can't be found
    (works for an already-square scan)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    markers = _find_markers(gray)
    if markers is None:
        return cv2.resize(img, (T.CANON_W, T.CANON_H),
                          interpolation=cv2.INTER_AREA)
    src = np.array(markers, dtype=np.float32)
    dst = np.array(T.marker_centers(), dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, H, (T.CANON_W, T.CANON_H),
                               flags=cv2.INTER_LINEAR,
                               borderValue=(255, 255, 255))


# ---- Ink extraction -------------------------------------------------------

def _ink_mask(cell_gray: np.ndarray) -> np.ndarray:
    """Binary ink mask (255 = ink) for one drawing cell. Drops the light ruled
    guides, denoises, but keeps small marks like the dots on i / j."""
    _, mask = cv2.threshold(cell_gray, INK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= MIN_COMPONENT_PX:
            keep[labels == lbl] = 255
    return keep


def _trace_svg(mask: np.ndarray):
    """Run potrace on an ink mask. Returns (path_d, G_transform) or None."""
    # potrace traces dark regions of the input, so feed ink as black on white.
    bmp = 255 - mask
    with tempfile.TemporaryDirectory() as d:
        in_path = os.path.join(d, "cell.bmp")
        out_path = os.path.join(d, "cell.svg")
        cv2.imwrite(in_path, bmp)
        subprocess.run(["potrace", in_path, "-b", "svg", "--flat",
                        "-o", out_path], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        svg = open(out_path, encoding="utf-8").read()

    pm = re.search(r"<path[^>]*\bd=\"([^\"]+)\"", svg)
    if not pm:
        return None
    d_attr = pm.group(1)

    # potrace wraps the path in <g transform="translate(tx,ty) scale(sx,sy)">
    # with sy NEGATIVE and scale ~0.1 (10x internal units). This MUST be
    # applied or glyphs come out ~10x oversized.
    G = Transform()
    gm = re.search(r"<g\b[^>]*transform=\"([^\"]+)\"", svg)
    if gm:
        tr = gm.group(1)
        tm = re.search(r"translate\(([-\d.]+)[ ,]+([-\d.]+)\)", tr)
        sm = re.search(r"scale\(([-\d.]+)(?:[ ,]+([-\d.]+))?\)", tr)
        if tm:
            G = G.translate(float(tm.group(1)), float(tm.group(2)))
        if sm:
            sx = float(sm.group(1))
            sy = float(sm.group(2)) if sm.group(2) is not None else sx
            G = G.scale(sx, sy)
    return d_attr, G


# ---- Glyph + font assembly ------------------------------------------------

def _glyph_name(ch: str) -> str:
    return UV2AGL.get(ord(ch)) or f"uni{ord(ch):04X}"


def _build_charstring(mask: np.ndarray, ch: str, draw_h: int):
    """Trace a cell mask into a CFF charstring. Returns (charstring, advance)
    or None if the cell is empty."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None

    traced = _trace_svg(mask)
    if traced is None:
        return None
    d_attr, G = traced

    left = int(xs.min())
    top = int(ys.min())
    bottom = int(ys.max())
    glyph_h = max(bottom - top, 1)

    # All glyphs share one scale (1000 units == one drawing-cell height) so the
    # relative sizes the user drew are preserved.
    U = UPM / float(draw_h)

    # Baseline = ink bottom, so the letter sits on the line. For descenders,
    # lift the baseline into the glyph so the tail dips below it.
    if ch in T.DESCENDERS:
        baseline_py = top + DESCENDER_BASELINE * glyph_h
    else:
        baseline_py = bottom

    # font_units = M . (cell pixels). Flip y (image y-down -> font y-up).
    M = Transform(U, 0, 0, -U, SIDE_BEARING - left * U, baseline_py * U)

    pen = T2CharStringPen(0, None)
    # G first (potrace coords -> top-origin cell pixels), then M (-> font units).
    parse_path(d_attr, TransformPen(TransformPen(pen, M), G))

    width = int(round((int(xs.max()) - left) * U + 2 * SIDE_BEARING))
    return pen.getCharString(), width


def _synth_charstring(ch: str):
    """Fallback outlines for space / period / comma when not drawn."""
    pen = T2CharStringPen(0, None)
    if ch == " ":
        return pen.getCharString(), 300
    if ch in ".,":
        r = 45
        cx, cy = 90, 40
        pen.moveTo((cx - r, cy))
        pen.curveTo((cx - r, cy + r), (cx + r, cy + r), (cx + r, cy))
        pen.curveTo((cx + r, cy - r), (cx - r, cy - r), (cx - r, cy))
        if ch == ",":
            pen.closePath()
            pen.moveTo((cx - r, cy - r))            # little tail
            pen.lineTo((cx, cy - r - 90))
            pen.lineTo((cx + r, cy - r))
        pen.closePath()
        return pen.getCharString(), 220
    return None


def build_font(image_path: str, out_path: str, variants: int = 1,
               family: str = "Paperfill Hand") -> str:
    """Build an OTF from a filled-template image. Returns ``out_path``."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not read image (unsupported or corrupt file)")
    canon = rectify(img)
    gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)

    # Collect the masks per glyph (keep the best-inked variant).
    draw_h = None
    best: dict[str, np.ndarray] = {}
    best_area: dict[str, int] = {}
    for glyph, _variant, rect in T.cells(variants):
        dx0, dy0, dx1, dy1 = (int(round(v)) for v in T.drawing_rect(rect))
        draw_h = dy1 - dy0
        cell = gray[dy0:dy1, dx0:dx1]
        mask = _ink_mask(cell)
        area = int((mask > 0).sum())
        if area > best_area.get(glyph, 0):
            best[glyph] = mask
            best_area[glyph] = area

    charstrings: dict[str, object] = {}
    advances: dict[str, int] = {}

    notdef = T2CharStringPen(600, None)
    charstrings[".notdef"] = notdef.getCharString()
    advances[".notdef"] = 600

    cmap: dict[int, str] = {}
    for glyph in T.GLYPHS:
        mask = best.get(glyph)
        result = None
        if mask is not None and best_area.get(glyph, 0) >= MIN_COMPONENT_PX:
            result = _build_charstring(mask, glyph, draw_h)
        if result is None:
            result = _synth_charstring(glyph)   # period / comma fallback
        if result is None:
            continue
        name = _glyph_name(glyph)
        charstrings[name] = result[0]
        advances[name] = result[1]
        cmap[ord(glyph)] = name

    # Always provide a space.
    if "space" not in charstrings:
        cs, w = _synth_charstring(" ")
        charstrings["space"] = cs
        advances["space"] = w
        cmap[ord(" ")] = "space"

    glyph_order = [".notdef"] + [n for n in charstrings if n != ".notdef"]

    fb = FontBuilder(UPM, isTTF=False)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupCFF(family, {"FullName": family}, charstrings, {})
    metrics = {n: (advances[n], 0) for n in glyph_order}
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ASCENT, descent=DESCENT)
    fb.setupNameTable({"familyName": family, "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=ASCENT, sTypoDescender=DESCENT,
                usWinAscent=ASCENT, usWinDescent=-DESCENT)
    fb.setupPost()
    fb.font.save(out_path)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python -m handwriting.font_build <template.png> <out.otf> "
              "[variants]", file=sys.stderr)
        sys.exit(2)
    var = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    path = build_font(sys.argv[1], sys.argv[2], variants=var)
    print(f"wrote {path}")
