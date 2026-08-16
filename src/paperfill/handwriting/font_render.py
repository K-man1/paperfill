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
import math
import random

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

RENDER_PX = 110            # base glyph size; the PDF stamper scales it down
PAD = 6                    # horizontal padding around a line
SPACE_FRAC = 0.32          # word gap as a fraction of the em

# Each rendered line lives in a band of this fixed pixel height with its
# baseline at BASELINE_FRAC down the band. Because the band height is constant,
# a multi-line image is exactly ``nlines * LINE_BAND_PX`` tall, which lets the
# stamper recover the line count and scale every answer to one line height.
# The font-size setting scales the glyphs AND this band together, so the
# baseline stays at the same fraction of a (now taller/shorter) band.
LINE_BAND_PX = 150
BASELINE_FRAC = 0.74

# --- appearance settings (see handwriting.font_store) ----------------------
# How the four user-tunable knobs map into render-space pixels:
#   letter_spacing % -> extra tracking added between glyphs, as a fraction of
#                       the em per 100% of deviation from natural (100%).
#   word_spacing  %  -> straight multiplier on the natural word gap.
#   font_size     %  -> scales the glyph em and the line band together.
#   pen_thickness mm -> the ACTUAL stroke width the ink is calibrated to on the
#                       printed page. We measure the font's own stroke width and
#                       dilate/erode the ink to hit the target (see
#                       calibrate_stroke). The real page scale is only known at
#                       stamp time, so the fill path calibrates there; the
#                       preview approximates it at the open-response scale below.
TRACK_FRAC = 0.28
_DEFAULTS = {"letter_spacing": 100.0, "font_size": 100.0,
             "word_spacing": 100.0, "pen_thickness": 0.4}

# Physical-width calibration. A PDF point is 1/72", so:
MM_PER_PT = 25.4 / 72.0
# The reference writing size used to turn a target millimetre into a render-px
# stroke width for the PREVIEW (which isn't stamped, so it has no page scale of
# its own). It mirrors render.HW_EM_REGION — the open-response answer em — so
# the preview reads like a typical filled-in answer. The actual fill uses the
# per-slot page scale at stamp time instead, so it's exact there.
REF_EM_PT = 12.0
_RENDER_PX_PER_MM = RENDER_PX / (REF_EM_PT * MM_PER_PT)

# Pen-thickness calibration is clamped to this radius so an extreme mm value
# can't dilate the ink into a solid blob or erode it to nothing. It must be
# generous enough to actually reach the top of the slider's advertised range
# (2.0mm) at the OPEN-RESPONSE page scale, the more demanding of the two slot
# kinds: hitting 2.0mm there needs ~52 render-px of stroke width, so from even
# a very fine natural scan (a few px) the dilation radius runs into the 20s.
# Undershooting this cap silently flattens the top of the slider — turning it
# up stops doing anything — so err generous rather than tight. The PNG is
# cropped with this much horizontal margin (CROP_MARGIN_PX) left around the
# ink so a caller recalibrating the stroke LATER — the fill path does this at
# stamp time, once the real page scale is known — has room to grow into
# without clipping against the crop edge.
MAX_PEN_RADIUS_PX = 26
CROP_MARGIN_PX = MAX_PEN_RADIUS_PX + 6


def _layout_engine():
    try:
        return ImageFont.Layout.RAQM
    except AttributeError:
        return ImageFont.Layout.BASIC


_FEATURES = ["+calt", "+liga"]


def _render_word(word: str, font: ImageFont.FreeTypeFont,
                 feats, track_px: float = 0.0) -> tuple[Image.Image, int]:
    """Render one word to a tight grayscale (black-on-white) image on a flat
    baseline. Returns (image, baseline_y_within_image).

    With ``track_px == 0`` the whole word is drawn in one call so kerning and
    contextual/ligature features fire — the natural, default look. When letter
    spacing is dialled off natural we instead lay each glyph down at its own
    advance plus ``track_px`` of extra tracking; features don't apply per glyph,
    but that's fine for the print-style template fonts this pipeline builds."""
    ascent, descent = font.getmetrics()
    if not track_px:
        probe = Image.new("L", (8, 8), 255)
        bbox = ImageDraw.Draw(probe).textbbox((0, 0), word, font=font,
                                              anchor="la", features=feats)
        w = max(bbox[2] - bbox[0], 1)
        img = Image.new("L", (w, ascent + descent), 255)
        # anchor "la" draws the text top at y=0, so the baseline sits at `ascent`.
        ImageDraw.Draw(img).text((-bbox[0], 0), word, font=font,
                                 fill=0, anchor="la", features=feats)
        return img, ascent

    advances = [font.getlength(ch) for ch in word]
    total = sum(advances) + track_px * max(len(word) - 1, 0)
    w = max(int(math.ceil(total)) + 4, 1)
    img = Image.new("L", (w, ascent + descent), 255)
    draw = ImageDraw.Draw(img)
    x = 2.0 if track_px < 0 else 0.0   # tiny left inset so negative tracking can't clip
    for ch, adv in zip(word, advances):
        # anchor "ls" = left edge, baseline — so each glyph keeps its own metrics.
        draw.text((x, ascent), ch, font=font, fill=0, anchor="ls")
        x += adv + track_px
    return img, ascent


def _load_font(path: str, px: int = RENDER_PX) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, px, layout_engine=_layout_engine())
    except OSError:
        return ImageFont.truetype(path, px)


def _compose_line(words: list[tuple[Image.Image, int]], space: int,
                  band_px: int, pad: int) -> Image.Image:
    """Paste a line of (word_image, baseline) onto a fixed-height band with a
    common, flat baseline. ``band_px`` scales with the font-size setting."""
    baseline_y = int(band_px * BASELINE_FRAC)
    width = sum(im.width for im, _ in words) + space * (len(words) - 1) + 2 * pad
    band = Image.new("L", (max(width, 1), band_px), 255)
    x = pad
    for im, baseline in words:
        y = baseline_y - baseline
        # Clamp so a tall font can't spill out of the band (rare).
        y = max(0, min(y, band_px - im.height))
        region = band.crop((x, y, x + im.width, y + im.height))
        band.paste(ImageChops.darker(region, im), (x, y))
        x += im.width + space
    return band


def estimate_stroke_px(alpha: np.ndarray) -> float:
    """Estimate the ink's stroke width, in pixels, from an alpha (ink) channel.

    For a long thin stroke the area is ``width * length`` and the perimeter is
    ``~2 * length`` (the two long sides), so ``2 * area / perimeter`` recovers
    the width. We approximate the perimeter as the count of ink pixels that
    touch a non-ink 4-neighbour. Cheap, and good enough to calibrate against."""
    mask = alpha > 60
    area = int(mask.sum())
    if area == 0:
        return 0.0
    bg = ~mask
    edge = mask & (np.roll(bg, 1, 0) | np.roll(bg, -1, 0)
                   | np.roll(bg, 1, 1) | np.roll(bg, -1, 1))
    perim = int(edge.sum())
    if perim == 0:                       # a solid blob; treat as very thick
        return float(min(alpha.shape))
    return 2.0 * area / perim


def calibrate_stroke(alpha: np.ndarray, target_px: float) -> np.ndarray:
    """Dilate or erode the ink so its stroke width lands near ``target_px``.

    A morphological dilation grows the stroke by ~``radius`` px on each side, so
    to move the width from ``w`` to ``target`` we use ``radius = (target-w)/2``.
    The radius is clamped so an extreme setting can't blow the ink out to a blob
    or erase it entirely — beyond the clamp it saturates rather than destroys.

    Uses cv2's dilate/erode rather than Pillow's MaxFilter/MinFilter: they
    produce identical output (both are a flat rectangular structuring element),
    but Pillow's rank filter is a naive O(kernel^2)-per-pixel scan — ~400ms at
    the largest radius this needs — while cv2's is a proper separable
    implementation, ~3ms at the same radius. This runs per answer per fill, so
    that difference is the whole render staying fast instead of a multi-second
    worksheet."""
    if target_px <= 0:
        return alpha
    w = estimate_stroke_px(alpha)
    if w <= 0:
        return alpha
    radius = int(round((target_px - w) / 2.0))
    radius = max(-MAX_PEN_RADIUS_PX, min(MAX_PEN_RADIUS_PX, radius))
    if radius == 0:
        return alpha
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (2 * abs(radius) + 1, 2 * abs(radius) + 1))
    op = cv2.dilate if radius > 0 else cv2.erode
    return op(alpha, kernel)


def render_text_png(text: str, otf_path, seed: int | None = None,
                    max_width_px: float | None = None,
                    settings: dict | None = None,
                    apply_pen: bool = False) -> bytes:
    """Render ``text`` to transparent-PNG bytes (dark ink on alpha). Empty /
    whitespace text yields b''.

    ``otf_path`` is a font path, or a list of variant paths (one per filled
    template copy the user uploaded). With multiple variants a font is chosen
    per word so repeated words/letters across the page don't look stamped.

    ``settings`` carries the user's tuned appearance knobs (letter spacing, font
    size, word spacing, pen thickness); omitted/None means the neutral defaults.

    ``apply_pen`` bakes the pen-thickness calibration into the ink at the
    open-response reference scale. The FILL path leaves it False and calibrates
    at stamp time, where the exact per-slot page scale is known; the PREVIEW
    passes True so the on-screen sample shows the chosen stroke weight.

    If ``max_width_px`` is given, words are wrapped onto multiple fixed-height
    line bands so each line stays within that pixel width; otherwise everything
    is laid out on a single line (used for the handwriting-sample preview)."""
    text = (text or "").strip()
    if not text:
        return b""
    rng = random.Random(seed if seed is not None else text)

    s = {**_DEFAULTS, **(settings or {})}
    fs = s["font_size"] / 100.0                 # glyph + band scale
    px = max(int(round(RENDER_PX * fs)), 4)     # font size the glyphs render at
    band_px = max(int(round(LINE_BAND_PX * fs)), 8)
    # The visual pad scales with font size, but pen-thickness calibration needs
    # a FIXED (not fs-scaled) blank margin around the line to dilate into —
    # the target stroke width in render-px is independent of fs (the stamp
    # scales the whole image by a constant regardless of font size), so the
    # headroom it needs doesn't shrink just because the glyphs did. Reserve it
    # up front so the later crop (below) has real canvas to keep, not just
    # clamp back down to.
    pad = max(int(round(PAD * fs)), 1) + CROP_MARGIN_PX
    space = max(int(round(px * SPACE_FRAC * s["word_spacing"] / 100.0)), 0)
    track_px = (s["letter_spacing"] / 100.0 - 1.0) * px * TRACK_FRAC

    paths = [otf_path] if isinstance(otf_path, (str, bytes)) else list(otf_path)
    paths = [str(p) for p in paths if p]
    if not paths:
        return b""
    fonts = [_load_font(p, px) for p in paths]

    # raqm may be unavailable in the Pillow build; fall back to no features.
    feats = _FEATURES
    try:
        ImageDraw.Draw(Image.new("L", (4, 4))).textbbox(
            (0, 0), "x", font=fonts[0], features=feats)
    except Exception:
        feats = None

    # Render each word (variant chosen per word for natural variation).
    rendered = [_render_word(w, rng.choice(fonts), feats, track_px)
                for w in text.split()]
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

    line_imgs = [_compose_line(line, space, band_px, pad) for line in lines]
    total_w = max(im.width for im in line_imgs)
    canvas = Image.new("L", (total_w, band_px * len(line_imgs)), 255)
    for i, im in enumerate(line_imgs):
        canvas.paste(im, (0, i * band_px))

    # Knock the white paper out to alpha; keep dark ink. Crop horizontally only
    # — the full band height per line is the stamper's scaling contract. Leave
    # CROP_MARGIN_PX of transparent slack on each side so a later stamp-time
    # pen-thickness recalibration (see render._insert_handwriting_image) has
    # room to dilate the stroke without clipping against the crop edge.
    gray = np.asarray(canvas, dtype=np.uint8)
    alpha = 255 - gray
    if apply_pen:
        alpha = calibrate_stroke(alpha, s["pen_thickness"] * _RENDER_PX_PER_MM)
    cols = np.where(alpha.max(axis=0) > 8)[0]
    if cols.size:
        lo = max(0, cols[0] - CROP_MARGIN_PX)
        hi = min(alpha.shape[1], cols[-1] + 1 + CROP_MARGIN_PX)
        alpha = alpha[:, lo:hi]
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
