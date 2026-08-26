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

from paperfill.handwriting import font_store, template as T

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
    from paperfill.handwriting.font_build import build_font
    f = TTFont(build_font(filled, str(tmp_path / "f.otf")))
    cmap = f.getBestCmap()
    for ch in T.glyphs():
        assert ord(ch) in cmap, f"missing {ch!r}"
    assert ord(" ") in cmap                       # space synthesised
    for ch in "áéíñóú":                            # accents carried through
        assert ord(ch) in cmap


@requires_build
def test_glyph_bounds_sane(filled, tmp_path):
    from paperfill.handwriting.font_build import build_font
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


def _glyph_bounds(otf_path) -> dict:
    """Every glyph's outline bounds, keyed by character."""
    f = TTFont(otf_path)
    cmap, gs = f.getBestCmap(), f.getGlyphSet()
    out = {}
    for ch in T.glyphs():
        name = cmap.get(ord(ch))
        if name is None:
            continue
        pen = BoundsPen(gs)
        gs[name].draw(pen)
        if pen.bounds:
            out[ch] = pen.bounds
    return out


@requires_build
def test_build_survives_perspective(filled, tmp_path):
    """Marker detection + homography recover the grid from warped 'photos'.

    Asserting the glyphs merely EXIST proves nothing: when _find_markers fails,
    rectify falls back to resizing the whole photo, and a resized cell still
    holds enough ink to trace something for every character. What the fallback
    can't do is put the ink back where it belongs — it shifts and crops the
    glyphs. So compare the warped build's outlines against the flat scan's:
    with the homography the two agree to a fraction of an em, without it they
    drift by tens of units and the worst glyph is cropped to pieces."""
    import cv2
    from paperfill.handwriting.font_build import build_font
    paths = []
    for i, page in enumerate(_filled_pages(_system_font())):
        p = tmp_path / f"w{i}.png"
        cv2.imwrite(str(p), _warp(page))
        paths.append(str(p))

    flat = _glyph_bounds(build_font(filled, str(tmp_path / "flat.otf")))
    warped = _glyph_bounds(build_font(paths, str(tmp_path / "warped.otf")))
    assert set(flat) == set(warped)
    assert set("ABCXYZabcxyz123") <= set(flat)

    drift = {ch: max(abs(a - b) for a, b in zip(flat[ch], warped[ch]))
             for ch in flat}
    worst, off = max(drift.items(), key=lambda kv: kv[1])
    # Font units, UPM = 1000. The homography leaves a wobble of a couple of
    # units from the resampling; the resize fallback averages ~100 units off
    # with its worst glyph out by ~800, so either bound separates the two by a
    # factor of four or better.
    assert sum(drift.values()) / len(drift) < 8, "outlines drifted across the board"
    assert off < 60, f"{worst!r} is off by {off:.0f} units"


# ---- font_render ----------------------------------------------------------

@requires_build
def test_render_text_png(filled, tmp_path):
    import io
    from paperfill.handwriting.font_build import build_font
    from paperfill.handwriting.font_render import render_text_png
    otf = build_font(filled, str(tmp_path / "f.otf"))
    png = render_text_png("Hola energía.", otf)
    im = Image.open(io.BytesIO(png))
    assert im.mode == "RGBA"
    assert np.asarray(im)[..., 3].max() > 0          # has opaque ink
    assert render_text_png("", otf) == b""
    assert render_text_png("   ", otf) == b""


def _stamped_ink(pdf_path, slot_bbox):
    """The alpha of the handwriting image stamped into ``slot_bbox``, plus the
    page points one of its pixels covers — enough to read a stroke width back
    off the page in millimetres."""
    import io
    import fitz
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    slot = fitz.Rect(slot_bbox)
    # The watermark is stamped too, so pick the image that lands in the slot.
    stamped = [(img, r) for img in page.get_images(full=True)
               for r in page.get_image_rects(img[0]) if r.intersects(slot)]
    assert len(stamped) == 1, f"{len(stamped)} images in the answer slot"
    img, rect = stamped[0]
    # insert_image splits an RGBA PNG into an RGB image plus a soft mask, and
    # the ink is entirely in that mask (the colour plane is flat black).
    smask = doc.extract_image(img[1])["image"]
    alpha = np.array(Image.open(io.BytesIO(smask)).convert("L"))
    doc.close()
    return alpha, rect.width / alpha.shape[1]


@requires_build
def test_pen_thickness_lands_on_the_page_as_a_real_stroke_width(filled, tmp_path):
    """The mm setting has to survive the whole path: build the font, render the
    answer, stamp it, then measure the ink that actually reached the page.

    render_overlays_pdf catches any stamping failure and quietly falls back to
    typeset text, so a bug here doesn't raise — the worksheet just comes out
    typed. And a calibration that erodes the ink away, or ignores the setting,
    still stamps a perfectly valid image. So check the ink itself."""
    import fitz
    from paperfill.handwriting.font_build import build_font
    from paperfill.handwriting.font_render import (MM_PER_PT, estimate_stroke_px,
                                                   render_text_png)
    from paperfill.ai.render import render_overlays_pdf

    src = tmp_path / "src.pdf"
    doc = fitz.open()
    doc.new_page(width=300, height=200)
    doc.save(str(src))
    doc.close()

    otf = build_font(filled, str(tmp_path / "f.otf"))
    png = render_text_png("Handwriting sample", otf, max_width_px=1400)
    bbox = [20, 40, 280, 120]

    def stamp(mm):
        out = tmp_path / f"out{mm}.pdf"
        render_overlays_pdf(str(src), [{"id": "ov0", "page": 0, "bbox": bbox,
                                        "text": "UNIQUEANSWER"}], str(out),
                            images={"ov0": png}, pen_thickness_mm=mm)
        with fitz.open(str(out)) as stamped:
            assert "UNIQUEANSWER" not in stamped[0].get_text(), \
                "fell back to typeset text"
        alpha, pt_per_px = _stamped_ink(out, bbox)
        assert (alpha > 60).any(), "the calibrated answer has no ink left"
        return estimate_stroke_px(alpha) * pt_per_px * MM_PER_PT

    fine = stamp(0.4)
    # estimate_stroke_px reads a dilated glyph a few percent thick (corners and
    # joins cost perimeter), and the dilation radius is a whole number of
    # render pixels — ~0.04mm each here — so allow a few pixels of slack. It
    # still excludes the font's own uncalibrated stroke, 0.25mm on this build.
    assert abs(fine - 0.4) < 0.12, f"0.4mm pen measured {fine:.3f}mm"
    # The top of the slider has to keep moving: MAX_PEN_RADIUS_PX capping too
    # low would flatten a bold nib back onto the fine one.
    assert stamp(1.2) > 1.5 * fine


# ---- font_store -----------------------------------------------------------

def test_font_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(font_store, "FONTS_DIR", tmp_path / "fonts")
    monkeypatch.setattr(font_store, "_INDEX", tmp_path / "fonts" / "index.json")

    sub = "user-123"
    assert font_store.user_font(sub) is None
    assert font_store.list_fonts_for(sub) == []

    fid = font_store.save_user_font(sub, [b"otf-one", b"otf-two"])
    assert fid == font_store.user_font_id(sub)            # id derives from sub
    assert font_store.font_path(fid) is not None
    assert font_store.font_variant_paths(fid) == [
        str(tmp_path / "fonts" / f"{fid}.otf"),
        str(tmp_path / "fonts" / f"{fid}.v2.otf"),
    ]
    assert font_store.user_font(sub) == {
        "id": fid, "label": font_store.LABEL, "variants": 2,
    }
    assert font_store.list_fonts_for(sub) == [{"id": fid, "label": font_store.LABEL}]

    # Rebuilding replaces in place: same id, stale variants are cleared.
    assert font_store.save_user_font(sub, [b"only-one"]) == fid
    assert font_store.user_font(sub)["variants"] == 1
    assert font_store.font_variant_paths(fid) == [str(tmp_path / "fonts" / f"{fid}.otf")]

    # A different user gets a different, isolated id.
    assert font_store.save_user_font("user-456", [b"x"]) != fid
