"""
Tests for the in-app handwriting-font feature.

The build pipeline needs the `potrace` binary and a TrueType font to synthesise
a filled template; both are skipped gracefully if unavailable so the suite
stays portable.
"""

import glob
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont

from handwriting import font_store, template as T

_HAS_POTRACE = shutil.which("potrace") is not None


def _system_font():
    for pat in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/**/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None


requires_build = pytest.mark.skipif(
    not (_HAS_POTRACE and _system_font()),
    reason="needs potrace + a system TrueType font",
)


def _make_filled(path, font_path, variants=1, warp=False):
    """Synthesise a 'filled' template by stamping a real font into each cell."""
    img = Image.new("RGB", (T.CANON_W, T.CANON_H), "white")
    d = ImageDraw.Draw(img)
    for cx, cy in T.marker_centers():
        s = T.MARKER_OUTER
        for frac, fill in ((1.0, "black"), (0.58, "white"), (0.26, "black")):
            h = s * frac / 2
            d.rectangle([cx - h, cy - h, cx + h, cy + h], fill=fill)
    for glyph, _v, rect in T.cells(variants):
        dx0, dy0, dx1, dy1 = T.drawing_rect(rect)
        baseline = dy0 + (dy1 - dy0) * 0.86
        f = ImageFont.truetype(font_path, int((dy1 - dy0) * 0.62))
        d.text((dx0 + (dx1 - dx0) * 0.30, baseline), glyph, font=f,
               fill="black", anchor="ls")
    if warp:
        import cv2
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w = arr.shape[:2]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[18, 30], [w - 40, 8], [w - 12, h - 25], [35, h - 12]])
        M = cv2.getPerspectiveTransform(src, dst)
        cv2.imwrite(str(path), cv2.warpPerspective(arr, M, (w, h),
                    borderValue=(255, 255, 255)))
    else:
        img.save(path)


@pytest.fixture
def filled(tmp_path):
    fp = _system_font()
    out = tmp_path / "filled.png"
    _make_filled(out, fp)
    return str(out)


# ---- template -------------------------------------------------------------

def test_template_layout_covers_required_glyphs():
    glyphs = {g for g, _v, _r in T.cells(1)}
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,":
        assert ch in glyphs
    assert len(T.marker_centers()) == 4


def test_blank_png_renders():
    png = T.render_blank_png(1)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert Image.open(__import__("io").BytesIO(png)).size == (T.CANON_W, T.CANON_H)


# ---- font_build -----------------------------------------------------------

@requires_build
def test_build_emits_required_glyphs(filled, tmp_path):
    from handwriting.font_build import build_font
    otf = build_font(filled, str(tmp_path / "f.otf"))
    f = TTFont(otf)
    cmap = f.getBestCmap()
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,":
        assert ord(ch) in cmap, f"missing {ch!r}"


@requires_build
def test_glyph_bounds_sane(filled, tmp_path):
    from handwriting.font_build import build_font
    f = TTFont(build_font(filled, str(tmp_path / "f.otf")))
    cmap, gs = f.getBestCmap(), f.getGlyphSet()

    def bounds(ch):
        p = BoundsPen(gs)
        gs[cmap[ord(ch)]].draw(p)
        return p.bounds

    # Caps sit on the baseline (bottom near 0) and rise well above it.
    for ch in "AH":
        x0, y0, x1, y1 = bounds(ch)
        assert -25 < y0 < 25, f"{ch} bottom {y0} not on baseline"
        assert y1 > 250, f"{ch} too short"
    # Descenders dip below the baseline (negative).
    for ch in "gpy":
        assert bounds(ch)[1] < -40, f"{ch} has no descender"


@requires_build
def test_build_survives_perspective(tmp_path):
    """Marker detection + homography recover the grid from a warped 'photo'."""
    from handwriting.font_build import build_font
    warped = tmp_path / "warp.png"
    _make_filled(warped, _system_font(), warp=True)
    f = TTFont(build_font(str(warped), str(tmp_path / "f.otf")))
    cmap = f.getBestCmap()
    assert all(ord(c) in cmap for c in "ABCXYZagpz")


# ---- font_render ----------------------------------------------------------

@requires_build
def test_render_text_png(filled, tmp_path):
    from handwriting.font_build import build_font
    from handwriting.font_render import render_text_png
    otf = build_font(filled, str(tmp_path / "f.otf"))
    png = render_text_png("Hello world.", otf)
    im = Image.open(__import__("io").BytesIO(png))
    assert im.mode == "RGBA"
    assert np.asarray(im)[..., 3].max() > 0          # has opaque ink
    assert render_text_png("", otf) == b""           # empty -> empty
    assert render_text_png("   ", otf) == b""


# ---- font_store + gating --------------------------------------------------

def test_font_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(font_store, "FONTS_DIR", tmp_path / "fonts")
    monkeypatch.setattr(font_store, "_INDEX", tmp_path / "fonts" / "index.json")

    assert not font_store.has_fonts()

    fid = font_store.save_font("My Hand", b"not-a-real-otf-but-bytes")
    assert font_store.font_path(fid) is not None
    assert {"id": fid, "label": "My Hand"} in font_store.list_fonts()
    assert font_store.has_fonts()

    # Colliding names don't clobber.
    fid2 = font_store.save_font("My Hand", b"second")
    assert fid2 != fid
