"""
Render text in a user's handwriting font to a transparent PNG, in the exact
format ``render._insert_handwriting_image`` expects: dark ink on alpha, paper
knocked out to transparent.

We layout the words with Pillow (HarfBuzz/raqm when available, so ``+calt`` /
``+liga`` fire). The ink sits on a flat, even baseline — no warping, rotation,
or per-word jitter — so the script stays clean and legible. When a wrap width
is supplied the text flows onto multiple lines at a constant size instead of
being squeezed onto one long line; each line occupies a fixed-height band so
the PDF stamper can scale every answer to the same line height.
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

RENDER_PX = 110            # glyph size; the PDF stamper scales it down
PAD = 6                    # horizontal padding around a line
SPACE_FRAC = 0.32          # word gap as a fraction of the em

# Each rendered line lives in a band of this fixed pixel height with its
# baseline at BASELINE_FRAC down the band. Because the band height is constant,
# a multi-line image is exactly ``nlines * LINE_BAND_PX`` tall, which lets the
# stamper recover the line count and scale every answer to one line height.
LINE_BAND_PX = 150
BASELINE_FRAC = 0.74


def _layout_engine():
    try:
        return ImageFont.Layout.RAQM
    except AttributeError:
        return ImageFont.Layout.BASIC


_FEATURES = ["+calt", "+liga"]


def _render_word(word: str, font: ImageFont.FreeTypeFont,
                 feats) -> tuple[Image.Image, int]:
    """Render one word to a tight grayscale (black-on-white) image on a flat
    baseline. Returns (image, baseline_y_within_image)."""
    ascent, descent = font.getmetrics()
    probe = Image.new("L", (8, 8), 255)
    bbox = ImageDraw.Draw(probe).textbbox((0, 0), word, font=font,
                                          anchor="la", features=feats)
    w = max(bbox[2] - bbox[0], 1)
    img = Image.new("L", (w, ascent + descent), 255)
    # anchor "la" draws the text top at y=0, so the baseline sits at `ascent`.
    ImageDraw.Draw(img).text((-bbox[0], 0), word, font=font,
                             fill=0, anchor="la", features=feats)
    return img, ascent


def _load_font(path: str) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, RENDER_PX, layout_engine=_layout_engine())
    except OSError:
        return ImageFont.truetype(path, RENDER_PX)


def _compose_line(words: list[tuple[Image.Image, int]], space: int) -> Image.Image:
    """Paste a line of (word_image, baseline) onto a fixed-height band with a
    common, flat baseline."""
    baseline_y = int(LINE_BAND_PX * BASELINE_FRAC)
    width = sum(im.width for im, _ in words) + space * (len(words) - 1) + 2 * PAD
    band = Image.new("L", (max(width, 1), LINE_BAND_PX), 255)
    x = PAD
    for im, baseline in words:
        y = baseline_y - baseline
        # Clamp so a tall font can't spill out of the band (rare).
        y = max(0, min(y, LINE_BAND_PX - im.height))
        region = band.crop((x, y, x + im.width, y + im.height))
        band.paste(ImageChops.darker(region, im), (x, y))
        x += im.width + space
    return band


def render_text_png(text: str, otf_path, seed: int | None = None,
                    max_width_px: float | None = None) -> bytes:
    """Render ``text`` to transparent-PNG bytes (dark ink on alpha). Empty /
    whitespace text yields b''.

    ``otf_path`` is a font path, or a list of variant paths (one per filled
    template copy the user uploaded). With multiple variants a font is chosen
    per word so repeated words/letters across the page don't look stamped.

    If ``max_width_px`` is given, words are wrapped onto multiple fixed-height
    line bands so each line stays within that pixel width; otherwise everything
    is laid out on a single line (used for the handwriting-sample preview)."""
    text = (text or "").strip()
    if not text:
        return b""
    rng = random.Random(seed if seed is not None else text)

    paths = [otf_path] if isinstance(otf_path, (str, bytes)) else list(otf_path)
    paths = [str(p) for p in paths if p]
    if not paths:
        return b""
    fonts = [_load_font(p) for p in paths]

    # raqm may be unavailable in the Pillow build; fall back to no features.
    feats = _FEATURES
    try:
        ImageDraw.Draw(Image.new("L", (4, 4))).textbbox(
            (0, 0), "x", font=fonts[0], features=feats)
    except Exception:
        feats = None

    space = int(RENDER_PX * SPACE_FRAC)
    # Render each word (variant chosen per word for natural variation).
    rendered = [_render_word(w, rng.choice(fonts), feats) for w in text.split()]
    if not rendered:
        return b""

    # Greedy word-wrap into lines that fit max_width_px (render-space px).
    lines: list[list[tuple[Image.Image, int]]] = []
    cur: list[tuple[Image.Image, int]] = []
    cur_w = 0
    for im, baseline in rendered:
        add = im.width if not cur else cur_w + space + im.width
        if max_width_px and cur and add > max_width_px:
            lines.append(cur)
            cur, cur_w = [(im, baseline)], im.width
        else:
            cur.append((im, baseline))
            cur_w = add
    if cur:
        lines.append(cur)

    line_imgs = [_compose_line(line, space) for line in lines]
    total_w = max(im.width for im in line_imgs)
    canvas = Image.new("L", (total_w, LINE_BAND_PX * len(line_imgs)), 255)
    for i, im in enumerate(line_imgs):
        canvas.paste(im, (0, i * LINE_BAND_PX))

    # Knock the white paper out to alpha; keep dark ink. Crop horizontally only
    # — the full band height per line is the stamper's scaling contract.
    gray = np.asarray(canvas, dtype=np.uint8)
    alpha = 255 - gray
    cols = np.where(alpha.max(axis=0) > 8)[0]
    if cols.size:
        alpha = alpha[:, cols[0]:cols[-1] + 1]
    h, w = alpha.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)   # black ink (RGB stays 0)
    rgba[..., 3] = alpha
    out = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(out, format="PNG")
    return out.getvalue()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python -m handwriting.font_render <otf> <text> [out.png]",
              file=sys.stderr)
        sys.exit(2)
    png = render_text_png(sys.argv[2], sys.argv[1])
    out = sys.argv[3] if len(sys.argv) > 3 else "rendered.png"
    with open(out, "wb") as f:
        f.write(png)
    print(f"wrote {out} ({len(png)} bytes)")
