"""
Renderer: takes the structure JSON + answers dict and produces a filled
PDF. Coordinates are never seen by the LLM — they come from the
preprocessor. Text auto-fits to the slot bbox.
"""

import os
import re

import fitz

from paperfill.handwriting.font_render import RENDER_PX, BASELINE_FRAC
from paperfill.paths import REPO_ROOT


# Tunables
HANDWRITING_FONT = "helv"  # built-in PDF font
MIN_FONT_SIZE = 6

# "Made with Goodnotes" badge stamped on the bottom-left of every rendered page.
WATERMARK_PATH = str(REPO_ROOT / "assets" / "goodnotes_watermark.png")
WATERMARK_WIDTH = 117                                                                                            # px wide on the page (aspect ratio preserved)
WATERMARK_MARGIN = 20   # px from the left and bottom edges
# Handwriting is rendered by the font pipeline as fixed-height line bands
# (LINE_BAND_PX tall) whose glyphs are drawn at RENDER_PX em. We scale every
# band so the *em* lands at a target px on the page — sizing by the actual
# writing height, not the band's ascent/descent whitespace, so short answers
# read at a comfortable size. Inline blanks (a matching letter, a verb form)
# are written large to match the printed text and sit *on* the underscore;
# open-response answers are smaller so a multi-line answer fits its region
# without crowding the next question.
HW_EM_INLINE = 16       # px em for single-line inline-blank answers
HW_EM_REGION = 12       # px em for wrapped open-response answers
HW_THIN_H = 24          # bbox heights below this are single-line slots (inline blanks)
# The slot's y1 is the printed text-cell bottom, which sits ~a descender below
# the underscore the student writes on. Raise the handwriting baseline by a
# fraction of the em so words rest *on* the line instead of the line striking
# through them.
HW_BASELINE_RAISE = 0.30

_HW_SCALE_INLINE = HW_EM_INLINE / RENDER_PX
_HW_SCALE_REGION = HW_EM_REGION / RENDER_PX


def hw_wrap_width(bbox) -> float | None:
    """Render-space pixel width to wrap a handwriting answer to so it flows onto
    multiple lines at the open-response em height instead of being squeezed onto
    one line. Returns None for thin inline-blank slots, which stay on a single
    line."""
    box_w = bbox[2] - bbox[0]
    box_h = bbox[3] - bbox[1]
    if box_h < HW_THIN_H:
        return None
    # A band scaled by _HW_SCALE_REGION should stay within box_w, so the wrap
    # width in render-space px is box_w divided by that scale.
    return box_w / _HW_SCALE_REGION

_OV_DEFAULTS = {
    "mode": "region",
    "font": "sans",
    "size": 11,
    "bold": False,
    "italic": False,
    "underline": False,
}

# Multiple-choice answer mark: a hand-drawn-looking oval around the chosen
# option's label. Blue like a pen, a couple of points of padding so the ring
# clears the letter, and a slightly heavy stroke so it reads as deliberate.
_CIRCLE_COLOR = (0.10, 0.20, 0.75)
_CIRCLE_WIDTH = 1.6
_CIRCLE_PAD = 2.5


def wrap_text_to_width(text: str, width: float, font: str, size: float) -> list[str]:
    """Greedy word-wrap to fit a given width."""
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if fitz.get_text_length(trial, fontname=font, fontsize=size) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def insert_text_in_region(page, region, text: str,
                          size: float = 10, line_gap: float = 3) -> None:
    """Place a multi-line answer inside an open-response region with word wrap."""
    text = text.strip()
    if not text:
        return
    width = region[2] - region[0]
    available_height = region[3] - region[1]
    current_size = size
    while current_size >= MIN_FONT_SIZE:
        lines = wrap_text_to_width(text, width, HANDWRITING_FONT, current_size)
        line_height = current_size + line_gap
        if line_height * len(lines) <= available_height:
            break
        current_size -= 0.5
    else:
        lines = wrap_text_to_width(text, width, HANDWRITING_FONT, MIN_FONT_SIZE)
        line_height = MIN_FONT_SIZE + line_gap

    y = region[1] + current_size  # first baseline
    for line in lines:
        if y > region[3]:
            break
        page.insert_text((region[0], y), line, fontname=HANDWRITING_FONT,
                         fontsize=current_size, color=(0, 0, 0))
        y += line_height


def _hex_to_rgb(hex_color: str) -> tuple:
    h = (hex_color or "").lstrip("#")
    if len(h) != 6:
        return (0.1, 0.1, 0.1)
    try:
        return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)
    except ValueError:
        return (0.1, 0.1, 0.1)


def _html_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace("\n", "<br>"))


def _overlay_to_html(ov: dict) -> str | None:
    text = (ov.get("text") or "").strip()
    if not text:
        return None
    font = ov.get("font", "sans")
    family = {"serif": "serif", "mono": "monospace"}.get(font, "sans-serif")
    size = float(ov.get("size", 11))
    weight = "700" if ov.get("bold") else "400"
    style_italic = "italic" if ov.get("italic") else "normal"
    decoration = "underline" if ov.get("underline") else "none"
    css = (
        f"font-family: {family}; "
        f"font-size: {size}pt; "
        f"font-weight: {weight}; "
        f"font-style: {style_italic}; "
        f"text-decoration: {decoration}; "
        f"color: #000000; "
        f"line-height: 1.15; "
        f"margin: 0; padding: 0;"
    )
    return f'<p style="{css}">{_html_escape(text)}</p>'


def _insert_handwriting_image(page, bbox, png_bytes: bytes,
                              pen_thickness_mm: float | None = None) -> None:
    """Stamp a transparent handwriting PNG into the slot at a consistent writing
    size. The PNG is a stack of fixed-height (LINE_BAND_PX) line bands drawn at
    RENDER_PX em; scaling by _HW_SCALE lands the em at HW_EM_PX, so every answer
    — a single matching letter or a full sentence — reads at the same size.
    Single-line inline blanks place their baseline *on* the underscore; multi-
    line region answers top-anchor so they sit next to the question. The PNG
    already has the paper knocked out to alpha, so it overlays cleanly.

    If ``pen_thickness_mm`` is given, the ink's stroke width is recalibrated
    here — not when the PNG was first rendered — because the true page scale
    (including any shrink-to-fit for an oversized answer) is only known at this
    point. That's what makes the millimetre setting land as an actual physical
    stroke width on the page regardless of how big or small the answer got
    written."""
    import io
    import numpy as np
    from PIL import Image
    from paperfill.handwriting.font_render import MM_PER_PT, calibrate_stroke

    x0, y0, x1, y1 = bbox
    box_w, box_h = x1 - x0, y1 - y0
    img = Image.open(io.BytesIO(png_bytes))
    img_w, img_h = img.size
    if not (img_w and img_h):
        return

    if box_h < HW_THIN_H:
        # Inline blank: single line, written large. Only shrink if the answer
        # would overrun the blank's width. Anchor the glyph baseline onto the
        # underscore (~y1) so writing sits on the line with descenders below it.
        scale = _HW_SCALE_INLINE
        if img_w * scale > box_w:
            scale = box_w / img_w
        draw_w, draw_h = img_w * scale, img_h * scale
        # Underscore sits ~a descender above the slot bottom; raise onto it.
        # A single-line image is exactly one band tall, so its baseline sits at
        # BASELINE_FRAC of img_h — derive it from the image so a font-size-scaled
        # band (taller/shorter than LINE_BAND_PX) still lands on the underscore.
        baseline_y = y1 - 1 - HW_BASELINE_RAISE * HW_EM_INLINE
        top = baseline_y - BASELINE_FRAC * img_h * scale
        rect = fitz.Rect(x0 + 1, top, x0 + 1 + draw_w, top + draw_h)
    else:
        # Open-response region: wrapped lines, top-anchored next to the question.
        # If the wrapped block is taller than the region, scale it down to fit.
        scale = _HW_SCALE_REGION
        if img_h * scale > box_h:
            scale = box_h / img_h
        draw_w, draw_h = img_w * scale, img_h * scale
        rect = fitz.Rect(x0 + 1, y0, x0 + 1 + draw_w, y0 + draw_h)

    if pen_thickness_mm:
        # scale maps this PNG's own render-px to page-pt, so the render-px
        # width that will land at the target mm is the physical target
        # (converted to pt) divided by that same scale.
        target_render_px = (pen_thickness_mm / MM_PER_PT) / scale
        # np.array (not asarray): Pillow hands out its pixel buffer read-only,
        # so asarray gives a view that raises on the alpha assignment below —
        # which the caller swallows, silently typesetting the answer instead.
        rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
        rgba[..., 3] = calibrate_stroke(rgba[..., 3], target_render_px)
        buf = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
        png_bytes = buf.getvalue()

    page.insert_image(rect, stream=png_bytes, keep_proportion=True, overlay=True)


def _stamp_watermark(doc) -> None:
    """Stamp the 'Made with Goodnotes' badge on the bottom-left of every page.
    The PNG is transparent, so it overlays cleanly. Sized to WATERMARK_WIDTH
    with aspect ratio preserved and a small margin from the page edges."""
    if not os.path.exists(WATERMARK_PATH):
        return
    import io
    from PIL import Image

    with open(WATERMARK_PATH, "rb") as f:
        png_bytes = f.read()
    img_w, img_h = Image.open(io.BytesIO(png_bytes)).size
    if not (img_w and img_h):
        return
    scale = WATERMARK_WIDTH / img_w
    draw_w, draw_h = img_w * scale, img_h * scale

    for page in doc:
        ph = page.rect.height
        x0 = WATERMARK_MARGIN
        y1 = ph - WATERMARK_MARGIN
        rect = fitz.Rect(x0, y1 - draw_h, x0 + draw_w, y1)
        page.insert_image(rect, stream=png_bytes, keep_proportion=True, overlay=True)


def render_overlays_pdf(pdf_path: str, overlays: list[dict], out_path: str,
                        images: dict[str, bytes] | None = None,
                        pen_thickness_mm: float | None = None) -> None:
    """
    Render the flat overlay list onto a copy of the PDF. Each overlay carries
    its own formatting (font, size, bold/italic/underline) which is applied
    via PyMuPDF's HTML/Story renderer.

    If `images` maps an overlay id -> PNG bytes (rendered handwriting), that
    overlay is stamped as an image instead of typeset text. `pen_thickness_mm`,
    if given, recalibrates every stamped answer's ink to that physical stroke
    width (see `_insert_handwriting_image`).
    """
    images = images or {}
    doc = fitz.open(pdf_path)
    for ov in overlays:
        page_idx = ov.get("page", 0)
        if page_idx < 0 or page_idx >= len(doc):
            continue
        page = doc[page_idx]

        if ov.get("kind") == "circle":
            x0, y0, x1, y1 = ov["bbox"]
            rect = fitz.Rect(x0 - _CIRCLE_PAD, y0 - _CIRCLE_PAD,
                             x1 + _CIRCLE_PAD, y1 + _CIRCLE_PAD)
            try:
                page.draw_oval(rect, color=_CIRCLE_COLOR, width=_CIRCLE_WIDTH)
            except Exception:
                pass
            continue

        if ov.get("kind") == "points":
            plot = ov.get("plot", "points")
            pts = [(float(x), float(y)) for x, y in (ov.get("points") or [])]
            if plot == "none" or not pts:
                continue
            try:
                shape = page.new_shape()
                if plot == "curve":
                    shape.draw_polyline([fitz.Point(x, y)
                                         for x, y in curve_through(pts)])
                    shape.finish(color=_POINT_COLOR, width=_CURVE_WIDTH,
                                 closePath=False, lineCap=1, lineJoin=1)
                else:
                    for x, y in pts:
                        shape.draw_circle(fitz.Point(x, y), _POINT_RADIUS)
                    shape.finish(color=_POINT_COLOR, fill=_POINT_COLOR,
                                 width=0.4)
                shape.commit()
            except Exception:
                pass
            continue

        if ov.get("kind") == "ink":
            points = ov.get("points") or []
            if len(points) >= 2:
                try:
                    shape = page.new_shape()
                    shape.draw_polyline([fitz.Point(x, y) for x, y in points])
                    shape.finish(color=_hex_to_rgb(ov.get("color")),
                                width=float(ov.get("width", 2.0)),
                                closePath=False, lineCap=1, lineJoin=1)
                    shape.commit()
                except Exception:
                    pass
            continue

        png = images.get(ov.get("id"))
        if png:
            try:
                _insert_handwriting_image(page, ov["bbox"], png, pen_thickness_mm)
                continue
            except Exception:
                pass  # fall through to text rendering on any image failure

        html = _overlay_to_html(ov)
        if not html:
            continue
        rect = fitz.Rect(*ov["bbox"])
        try:
            page.insert_htmlbox(rect, html)
        except Exception:
            # htmlbox failed (bad rect, unsupported font) — fall back to plain text
            insert_text_in_region(page, ov["bbox"], ov.get("text", ""))
    _stamp_watermark(doc)
    # garbage=4 + deflate strip orphaned objects and compress streams; without
    # them PyMuPDF leaves the source PDF's bloat in place (a study guide ballooned
    # to ~178MB). With them the same file lands around 10MB.
    doc.save(out_path, garbage=4, deflate=True, deflate_images=True,
             deflate_fonts=True, clean=True)
    doc.close()


_POINT_RADIUS = 2.0
_CURVE_WIDTH = 1.4
_POINT_COLOR = (0.11, 0.36, 0.86)

# Minus arrives as ASCII, as the real minus sign, or as an en dash depending on
# whether the model was writing "maths" or "text".
_NUMBER = r"[-−–]?\d+(?:\.\d+)?"
_PAIR_RE = re.compile(rf"[(\[]\s*({_NUMBER})\s*[,;]\s*({_NUMBER})\s*[)\]]")
_NUMBER_RE = re.compile(_NUMBER)


def _number(text: str) -> float:
    return float(text.replace("−", "-").replace("–", "-"))


def curve_through(points: list[tuple[float, float]],
                  samples: int = 16) -> list[tuple[float, float]]:
    """Densify a handful of plotted points into a smooth curve.

    A model gives 7 to 15 points, and joining those straight looks like a
    polygon rather than a graph. Catmull-Rom is the right fit here because it
    passes exactly THROUGH every control point — the points are the answer, so
    a curve that merely approximates them would be plotting something the model
    didn't say. The end segments duplicate the outer points so the curve starts
    and stops at the data instead of overshooting.
    """
    if len(points) < 3:
        return list(points)
    ordered = sorted(points)
    padded = [ordered[0]] + ordered + [ordered[-1]]
    out: list[tuple[float, float]] = [ordered[0]]
    for i in range(len(padded) - 3):
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = padded[i:i + 4]
        for step in range(1, samples + 1):
            t = step / samples
            t2, t3 = t * t, t * t * t
            out.append((
                0.5 * ((2 * x1) + (-x0 + x2) * t
                       + (2 * x0 - 5 * x1 + 4 * x2 - x3) * t2
                       + (-x0 + 3 * x1 - 3 * x2 + x3) * t3),
                0.5 * ((2 * y1) + (-y0 + y2) * t
                       + (2 * y0 - 5 * y1 + 4 * y2 - y3) * t2
                       + (-y0 + 3 * y1 - 3 * y2 + y3) * t3),
            ))
    return out


def parse_points(answer: str) -> list[tuple[float, float]]:
    """Pull (x, y) pairs out of a model's graph answer.

    The answer normally arrives as prose around a list of coordinates
    ("(-2, 5), (0, 1), (2, 5)"), so the pairs are matched rather than the string
    parsed. Square brackets count too.

    JSON mode is the awkward case: a model that answers with a nested list gets
    flattened to a bare "x, y, x, y" run with every bracket stripped before this
    ever sees it, so an even run of loose numbers is paired up rather than
    thrown away. A graph answer has no other sensible reading.
    """
    pairs = _PAIR_RE.findall(answer or "")
    if pairs:
        return [(_number(x), _number(y)) for x, y in pairs]

    loose = _NUMBER_RE.findall(answer or "")
    if len(loose) >= 4 and len(loose) % 2 == 0:
        values = [_number(n) for n in loose]
        return list(zip(values[::2], values[1::2]))
    return []


def plot_point(graph: dict, x: float, y: float) -> tuple[float, float] | None:
    """Map a graph coordinate onto the page, or None if it falls off the plot.

    The origin comes from the detected axis lines rather than from interpolating
    the range, because a hand-made grid is rarely centred to the pixel and the
    axes are the one thing we located exactly.
    """
    span_x = graph["x_max"] - graph["x_min"]
    span_y = graph["y_max"] - graph["y_min"]
    if span_x <= 0 or span_y <= 0:
        return None
    px = graph["origin_x"] + x * (graph["right"] - graph["left"]) / span_x
    py = graph["origin_y"] - y * (graph["bottom"] - graph["top"]) / span_y
    if not (graph["left"] - 1 <= px <= graph["right"] + 1):
        return None
    if not (graph["top"] - 1 <= py <= graph["bottom"] + 1):
        return None
    return px, py


def _match_option(answer: str, options: list[dict]) -> dict | None:
    """Find the option the model chose. It usually returns just the label ("C",
    "III"), but tolerate "C.", "c)", "C) 2(3m-5n)" or "III only" — compare the
    answer's leading letter/numeral token against each option's label. Falls back
    to the first character for a lettered list when the token doesn't match whole
    (e.g. answer "Cfoo")."""
    ans = (answer or "").strip().upper()
    if not ans or not options:
        return None
    by_label = {str(o["label"]).upper(): o for o in options}
    m = re.match(r"[A-Z]+", ans)
    tok = m.group(0) if m else ""
    if tok in by_label:
        return by_label[tok]
    if tok[:1] in by_label:
        return by_label[tok[:1]]
    return None


def build_overlays_from_structure(structure: dict, answers: dict,
                                  default_plot: str = "points") -> list[dict]:
    """
    Turn the preprocessor's structured units + LLM answers into a flat list
    of editable overlays. Inline blanks get a small region just above the
    underscore; open-response answers use their detected answer_region;
    multiple-choice answers become a "circle" overlay on the chosen option.

    `default_plot` is the user's saved default for a freshly-built graph
    overlay ("points", "curve", or "none" — see GRAPH_MODES in index.html);
    it's still just the starting value; the editor can flip an individual
    graph to a different mode afterward via its own `ov.plot`.
    """
    overlays: list[dict] = []
    nid = 0
    for u in structure["units"]:
        page = u["page"]
        if u["type"] == "inline_blanks":
            for slot in u["slots"]:
                x0, y0, x1, y1 = slot["bbox"]
                overlays.append({
                    **_OV_DEFAULTS,
                    "id": f"ov{nid}", "page": page,
                    "bbox": [x0, y1 - 13, x1, y1 + 1],
                    "text": answers.get(slot["slot_id"], ""),
                })
                nid += 1
        elif u["type"] == "open_response":
            overlays.append({
                **_OV_DEFAULTS,
                "id": f"ov{nid}", "page": page,
                "bbox": list(u["answer_region"]),
                "text": answers.get(u["unit_id"], ""),
            })
            nid += 1
        elif u["type"] == "graph":
            graph = u.get("graph") or {}
            answer = answers.get(u["unit_id"], "")
            points = parse_points(answer)
            plotted = [p for p in (plot_point(graph, x, y) for x, y in points)
                       if p is not None]
            # An unplottable graph answer draws nothing at all, which looks
            # identical to no answer, so say which of the two it was.
            if answer and not plotted:
                print(f"[render] graph {u['unit_id']}: {len(points)} point(s) "
                      f"parsed, none on the grid, from {answer[:120]!r}")
            if plotted:
                overlays.append({
                    **_OV_DEFAULTS,
                    "id": f"ov{nid}", "page": page,
                    "kind": "points",
                    "bbox": list(u["bbox"]),
                    "points": plotted,
                    # How the reader wants it drawn; switched in the editor.
                    "plot": default_plot,
                    "text": "",
                })
                nid += 1
        elif u["type"] == "multiple_choice":
            match = _match_option(answers.get(u["unit_id"], ""),
                                  u.get("options") or [])
            if match is not None:
                overlays.append({
                    **_OV_DEFAULTS,
                    "id": f"ov{nid}", "page": page,
                    "kind": "circle",
                    "bbox": list(match["bbox"]),
                    "text": "",
                })
                nid += 1
        elif u["type"] == "table":
            for row in u["table_cells"]:
                for cell in row:
                    if cell is None:
                        continue
                    for slot in cell["slots"]:
                        x0, y0, x1, y1 = slot["bbox"]
                        overlays.append({
                            **_OV_DEFAULTS,
                            "id": f"ov{nid}", "page": page,
                            "bbox": [x0, y1 - 13, x1, y1 + 1],
                            "text": answers.get(slot["slot_id"], ""),
                        })
                        nid += 1
    return overlays
