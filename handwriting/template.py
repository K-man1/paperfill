"""
Canonical handwriting-template geometry, shared by the printable blank
template and the font builder.

The template is a single page with four concentric-square registration
markers in the corners and a grid of labelled cells, one (or a few) per
glyph. ``font_build`` finds the four markers in a phone photo / scan, warps
the page back to this canonical pixel space, and then reads each glyph from
its fixed cell rectangle below. Because both the generator and the builder
import these constants, the cell map is guaranteed to line up — we never OCR
the printed labels.

Coordinates are in canonical pixels (A4 at 150 dpi).
"""

from __future__ import annotations

import io

# ---- Canonical page -------------------------------------------------------

CANON_W = 1240
CANON_H = 1754

# Concentric-square markers. Centre points sit a fixed inset from each corner;
# ``font_build`` maps the four detected centres onto these.
MARKER_OUTER = 64          # outer black square, px
MARKER_INSET = 74          # marker centre distance from each page edge

def marker_centers() -> list[tuple[float, float]]:
    """Canonical marker centres, ordered TL, TR, BR, BL."""
    m = MARKER_INSET
    return [
        (m, m),                       # top-left
        (CANON_W - m, m),             # top-right
        (CANON_W - m, CANON_H - m),   # bottom-right
        (m, CANON_H - m),             # bottom-left
    ]

# ---- Glyph grid -----------------------------------------------------------

# Row-major order of the glyphs a user draws. Space is synthesised, not drawn.
# Builder synthesises period/comma too if their cells come back empty.
GLYPHS = (
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abcdefghijklmnopqrstuvwxyz")
    + list("0123456789")
    + list(".,'\"!?-:;()")
)

# Glyphs whose tails dip below the baseline; the builder lifts the baseline
# into the glyph so the descender renders as negative font units.
DESCENDERS = set("gjpqy")

COLS = 8
GRID_X0 = 64
GRID_X1 = CANON_W - 64
GRID_Y0 = 172
GRID_Y1 = CANON_H - 108

# Each cell reserves a thin strip at the top for its printed label; the rest is
# the drawing area. The builder insets the drawing area a little more to stay
# clear of the printed cell border and label.
LABEL_STRIP_H = 22
DRAW_INSET = 6


def _layout(variants: int) -> tuple[int, int, float, float]:
    """Return (n_cells, rows, cell_w, cell_h) for ``variants`` rows per glyph."""
    n_cells = len(GLYPHS) * variants
    rows = (n_cells + COLS - 1) // COLS
    cell_w = (GRID_X1 - GRID_X0) / COLS
    cell_h = (GRID_Y1 - GRID_Y0) / rows
    return n_cells, rows, cell_w, cell_h


def cells(variants: int = 1):
    """Yield ``(glyph, variant_index, (x0, y0, x1, y1))`` for every cell, in
    row-major order. With ``variants`` > 1 each glyph gets that many adjacent
    cells so the user can draw a few takes; the builder keeps the best-inked
    one."""
    n_cells, rows, cell_w, cell_h = _layout(variants)
    seq = [(g, v) for g in GLYPHS for v in range(variants)]
    for idx, (glyph, variant) in enumerate(seq):
        r, c = divmod(idx, COLS)
        x0 = GRID_X0 + c * cell_w
        y0 = GRID_Y0 + r * cell_h
        yield glyph, variant, (x0, y0, x0 + cell_w, y0 + cell_h)


def drawing_rect(cell_rect) -> tuple[float, float, float, float]:
    """The ink region of a cell: below the label strip, inset from borders."""
    x0, y0, x1, y1 = cell_rect
    return (x0 + DRAW_INSET, y0 + LABEL_STRIP_H,
            x1 - DRAW_INSET, y1 - DRAW_INSET)


# ---- Blank template rendering --------------------------------------------

def render_blank_png(variants: int = 1) -> bytes:
    """Render the printable blank template as PNG bytes."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (CANON_W, CANON_H), "white")
    d = ImageDraw.Draw(img)

    def font(size, bold=False):
        names = (["DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold
                 else ["DejaVuSans.ttf", "Arial.ttf"])
        for n in names:
            try:
                return ImageFont.truetype(n, size)
            except OSError:
                continue
        return ImageFont.load_default()

    GUIDE = (190, 205, 225)     # faint blue ruled lines
    BORDER = (215, 215, 215)    # faint cell borders
    LABEL = (140, 140, 140)

    # Markers (concentric black / white / black squares).
    for cx, cy in marker_centers():
        s = MARKER_OUTER
        for frac, fill in ((1.0, "black"), (0.58, "white"), (0.26, "black")):
            h = s * frac / 2
            d.rectangle([cx - h, cy - h, cx + h, cy + h], fill=fill)

    # Branding (replaces Calligraphr's logo + QR code).
    d.text((CANON_W / 2, 92), "Paperfill", font=font(46, bold=True),
           fill="black", anchor="mm")
    d.text((CANON_W / 2, 132), "Handwriting Template", font=font(20),
           fill=(110, 110, 110), anchor="mm")

    # Cells.
    for glyph, _variant, rect in cells(variants):
        x0, y0, x1, y1 = rect
        d.rectangle([x0, y0, x1, y1], outline=BORDER, width=1)
        d.text((x0 + 6, y0 + 4), glyph, font=font(14), fill=LABEL, anchor="lm")
        dx0, dy0, dx1, dy1 = drawing_rect(rect)
        # Three ruled guides: cap line, x-height, baseline.
        for t in (0.10, 0.46, 0.86):
            gy = dy0 + (dy1 - dy0) * t
            d.line([dx0, gy, dx1, gy], fill=GUIDE, width=1)

    d.text((CANON_W / 2, CANON_H - 64),
           "Print, draw one glyph per cell on the lines, then photograph or "
           "scan the whole page — keep all four corner markers in frame.",
           font=font(16), fill=(120, 120, 120), anchor="mm")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_blank_pdf(variants: int = 1) -> bytes:
    """Wrap the blank PNG in a single A4 PDF page (nice for printing)."""
    import fitz

    png = render_blank_png(variants)
    doc = fitz.open()
    page = doc.new_page(width=595.28, height=841.89)  # A4 in points
    page.insert_image(page.rect, stream=png)
    out = doc.tobytes()
    doc.close()
    return out


if __name__ == "__main__":  # python -m handwriting.template out.png [variants]
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "template.png"
    variants = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    data = render_blank_pdf(variants) if out.lower().endswith(".pdf") \
        else render_blank_png(variants)
    with open(out, "wb") as f:
        f.write(data)
    print(f"wrote {out} ({len(data)} bytes, variants={variants})")
