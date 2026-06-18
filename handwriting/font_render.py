"""
Render text in a user's handwriting font to a transparent PNG, in the exact
format ``render._insert_handwriting_image`` expects: dark ink on alpha, paper
knocked out to transparent.

We layout the words with Pillow (HarfBuzz/raqm when available, so ``+calt`` /
``+liga`` fire), then add a GENTLE per-word elastic warp plus a touch of
rotation and baseline jitter so repeated letters don't look stamped. The warp
is deliberately small — a strong warp mangles the letterforms.
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, map_coordinates

RENDER_PX = 128            # tall render; the PDF stamper scales it down
PAD = 24                   # padding around each word (room for warp/rotation)
SPACE_FRAC = 0.32          # word gap as a fraction of the em
WARP_DISP = 1.3            # elastic displacement, ~px per em-hundred (gentle)
WARP_SIGMA = 14            # smoothness of the elastic field
MAX_ROTATE = 2.2           # degrees of per-word rotation
MAX_BASELINE_JITTER = 4    # px of per-word vertical jitter


def _layout_engine():
    try:
        return ImageFont.Layout.RAQM
    except AttributeError:
        return ImageFont.Layout.BASIC


_FEATURES = ["+calt", "+liga"]


def _elastic(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Gentle elastic warp of a grayscale (white-bg) word image."""
    h, w = arr.shape
    seed = rng.randrange(2**31)
    state = np.random.RandomState(seed)
    disp = WARP_DISP * RENDER_PX / 100.0
    dx = gaussian_filter(state.rand(h, w) * 2 - 1, WARP_SIGMA)
    dy = gaussian_filter(state.rand(h, w) * 2 - 1, WARP_SIGMA)
    # Normalise the smoothed field to a known peak displacement.
    for d in (dx, dy):
        m = np.abs(d).max() or 1.0
        d *= disp / m
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = [(yy + dy).ravel(), (xx + dx).ravel()]
    out = map_coordinates(arr, coords, order=1, mode="constant", cval=255.0)
    return out.reshape(h, w)


def _render_word(word: str, font: ImageFont.FreeTypeFont, feats,
                 rng: random.Random) -> tuple[Image.Image, int]:
    """Render one word to a grayscale (black-on-white) image with a gentle
    warp + rotation. Returns (image, baseline_y_within_image)."""
    ascent, descent = font.getmetrics()
    probe = Image.new("L", (8, 8), 255)
    bbox = ImageDraw.Draw(probe).textbbox((0, 0), word, font=font,
                                          anchor="la", features=feats)
    w = max(bbox[2] - bbox[0], 1)
    img = Image.new("L", (w + 2 * PAD, ascent + descent + 2 * PAD), 255)
    ImageDraw.Draw(img).text((PAD - bbox[0], PAD), word, font=font,
                             fill=0, anchor="la", features=feats)

    img = Image.fromarray(_elastic(np.asarray(img, dtype=np.float64), rng)
                          .clip(0, 255).astype(np.uint8))
    angle = rng.uniform(-MAX_ROTATE, MAX_ROTATE)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=255)

    baseline = PAD + ascent + (img.height - (ascent + descent + 2 * PAD)) // 2
    return img, baseline


def render_text_png(text: str, otf_path: str, seed: int | None = None) -> bytes:
    """Render ``text`` in the font at ``otf_path`` to transparent-PNG bytes
    (dark ink on alpha). Empty / whitespace text yields b''."""
    text = (text or "").strip()
    if not text:
        return b""
    rng = random.Random(seed if seed is not None else text)
    try:
        font = ImageFont.truetype(otf_path, RENDER_PX,
                                  layout_engine=_layout_engine())
    except OSError:
        font = ImageFont.truetype(otf_path, RENDER_PX)

    # raqm may be unavailable in the Pillow build; fall back to no features.
    feats = _FEATURES
    try:
        ImageDraw.Draw(Image.new("L", (4, 4))).textbbox(
            (0, 0), "x", font=font, features=feats)
    except Exception:
        feats = None

    space = int(RENDER_PX * SPACE_FRAC)
    words = [(_render_word(w, font, feats, rng)) for w in text.split()]
    if not words:
        return b""

    above = max(b for _, b in words) + MAX_BASELINE_JITTER + 2
    below = max(im.height - b for im, b in words) + MAX_BASELINE_JITTER + 2
    total_w = sum(im.width for im, _ in words) + space * (len(words) - 1) + 2 * PAD
    canvas = Image.new("L", (total_w, above + below), 255)

    x = PAD
    for im, baseline in words:
        jitter = rng.randint(-MAX_BASELINE_JITTER, MAX_BASELINE_JITTER)
        y = above - baseline + jitter
        region = canvas.crop((x, y, x + im.width, y + im.height))
        canvas.paste(ImageChops.darker(region, im), (x, y))
        x += im.width + space

    # Knock the white paper out to alpha; keep dark ink.
    gray = np.asarray(canvas, dtype=np.uint8)
    alpha = 255 - gray
    cols = np.where(alpha.max(axis=0) > 8)[0]
    rows = np.where(alpha.max(axis=1) > 8)[0]
    if cols.size and rows.size:
        x0, x1 = cols[0], cols[-1] + 1
        y0, y1 = rows[0], rows[-1] + 1
        alpha = alpha[y0:y1, x0:x1]
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
