"""
Tests for the in-app handwriting-font feature (calibrated multi-page template).

The build pipeline needs the `potrace` binary and a TrueType font to synthesise
a filled template; both are skipped gracefully if unavailable so the suite
stays portable.
"""

import glob
import shutil

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
    not (_HAS_POTRACE and _system_font() and T.TEMPLATE_PDF.exists()),
    reason="needs potrace + a system TrueType font + template.pdf",
)


def _filled_pages(font_path):
    """Stamp each glyph into its calibrated cell over the real template pages,
    returning a list of BGR images (one per template page)."""
    import cv2
    import fitz
    pages = []
    doc = fitz.open(str(T.TEMPLATE_PDF))
    for gp in T.pages():
        pix = doc[gp["index"]].get_pixmap(dpi=T.geometry()["dpi"])
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        code = cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR
        pil = Image.fromarray(cv2.cvtColor(cv2.cvtColor(arr, code), cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for cell in gp["cells"]:
            x0, y0, x1, y1 = cell["draw"]
            f = ImageFont.truetype(font_path, int((y1 - y0) * 0.62))
            d.text(((x0 + x1) / 2, y1 - (y1 - y0) * 0.12), cell["glyph"],
                   font=f, fill=(10, 10, 10), anchor="ms")
        pages.append(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
    doc.close()
    return pages


def _warp(img):
    import cv2
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[16, 26], [w - 34, 7], [w - 10, h - 22], [30, h - 11]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))


@pytest.fixture
def filled(tmp_path):
    import cv2
    paths = []
    for i, page in enumerate(_filled_pages(_system_font())):
        p = tmp_path / f"page{i}.png"
        cv2.imwrite(str(p), page)
        paths.append(str(p))
    return paths


# ---- geometry -------------------------------------------------------------

def test_geometry_loads_and_has_glyphs():
    assert T.page_count() >= 1
    gl = T.glyphs()
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
        assert ch in gl
    for p in T.pages():
        assert len(p["markers"]) == 4
        assert p["cells"]


# ---- font_build -----------------------------------------------------------

@requires_build
def test_build_emits_required_glyphs(filled, tmp_path):
    from handwriting.font_build import build_font
    f = TTFont(build_font(filled, str(tmp_path / "f.otf")))
    cmap = f.getBestCmap()
    for ch in T.glyphs():
        assert ord(ch) in cmap, f"missing {ch!r}"
    assert ord(" ") in cmap                       # space synthesised
    for ch in "áéíñóú":                            # accents carried through
        assert ord(ch) in cmap


@requires_build
def test_glyph_bounds_sane(filled, tmp_path):
    from handwriting.font_build import build_font
    f = TTFont(build_font(filled, str(tmp_path / "f.otf")))
    cmap, gs = f.getBestCmap(), f.getGlyphSet()

    def bounds(ch):
        p = BoundsPen(gs)
        gs[cmap[ord(ch)]].draw(p)
        return p.bounds

    for ch in "AH":                                # caps sit on the baseline
        x0, y0, x1, y1 = bounds(ch)
        assert -25 < y0 < 25 and y1 > 250
    for ch in "gpy":                               # descenders go negative
        assert bounds(ch)[1] < -40
    for ch in T.glyphs():                          # nothing wildly oversized
        x0, y0, x1, y1 = bounds(ch)
        assert (y1 - y0) < 1050, f"{ch!r} too tall: {(y0, y1)}"


@requires_build
def test_build_survives_perspective(tmp_path):
    """Marker detection + homography recover the grid from warped 'photos'."""
    import cv2
    from handwriting.font_build import build_font
    paths = []
    for i, page in enumerate(_filled_pages(_system_font())):
        p = tmp_path / f"w{i}.png"
        cv2.imwrite(str(p), _warp(page))
        paths.append(str(p))
    f = TTFont(build_font(paths, str(tmp_path / "f.otf")))
    cmap = f.getBestCmap()
    assert all(ord(c) in cmap for c in "ABCXYZabcxyz123")


# ---- font_render ----------------------------------------------------------

@requires_build
def test_render_text_png(filled, tmp_path):
    import io
    from handwriting.font_build import build_font
    from handwriting.font_render import render_text_png
    otf = build_font(filled, str(tmp_path / "f.otf"))
    png = render_text_png("Hola energía.", otf)
    im = Image.open(io.BytesIO(png))
    assert im.mode == "RGBA"
    assert np.asarray(im)[..., 3].max() > 0          # has opaque ink
    assert render_text_png("", otf) == b""
    assert render_text_png("   ", otf) == b""


# ---- font_store -----------------------------------------------------------

def test_font_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(font_store, "FONTS_DIR", tmp_path / "fonts")
    monkeypatch.setattr(font_store, "_INDEX", tmp_path / "fonts" / "index.json")

    assert not font_store.has_fonts()
    fid = font_store.save_font("My Hand", b"not-a-real-otf-but-bytes")
    assert font_store.font_path(fid) is not None
    assert {"id": fid, "label": "My Hand"} in font_store.list_fonts()
    assert font_store.has_fonts()
    assert font_store.save_font("My Hand", b"second") != fid   # no clobber
