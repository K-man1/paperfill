"""
Renderer: takes the structure JSON + answers dict and produces a filled
PDF. Coordinates are never seen by the LLM — they come from the
preprocessor. Text auto-fits to the slot bbox.
"""

import os

import fitz

from handwriting.font_render import LINE_BAND_PX, RENDER_PX, BASELINE_FRAC


# Tunables
HANDWRITING_FONT = "helv"  # built-in PDF font
MIN_FONT_SIZE = 6

# "Made with Goodnotes" badge stamped on the bottom-left of every rendered page.
WATERMARK_PATH = os.path.join(os.path.dirname(__file__), "assets", "goodnotes_watermark.png")
WATERMARK_WIDTH = 110   # px wide on the page (aspect ratio preserved)
WATERMARK_MARGIN = 18   # px from the left and bottom edges

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

_HW_SCALE_INLINE = HW_EM_INLINE / RENDER_PX
_HW_SCALE_REGION = HW_EM_REGION / RENDER_PX
_HW_BASELINE_OFF = BASELINE_FRAC * LINE_BAND_PX  # baseline depth within a band (render px)


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


def _insert_handwriting_image(page, bbox, png_bytes: bytes) -> None:
    """Stamp a transparent handwriting PNG into the slot at a consistent writing
    size. The PNG is a stack of fixed-height (LINE_BAND_PX) line bands drawn at
    RENDER_PX em; scaling by _HW_SCALE lands the em at HW_EM_PX, so every answer
    — a single matching letter or a full sentence — reads at the same size.
    Single-line inline blanks place their baseline *on* the underscore; multi-
    line region answers top-anchor so they sit next to the question. The PNG
    already has the paper knocked out to alpha, so it overlays cleanly."""
    import io
    from PIL import Image

    x0, y0, x1, y1 = bbox
    box_w, box_h = x1 - x0, y1 - y0
    img_w, img_h = Image.open(io.BytesIO(png_bytes)).size
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
        baseline_y = y1 - 1                      # the underscore line
        top = baseline_y - _HW_BASELINE_OFF * scale
        rect = fitz.Rect(x0 + 1, top, x0 + 1 + draw_w, top + draw_h)
    else:
        # Open-response region: wrapped lines, top-anchored next to the question.
        # If the wrapped block is taller than the region, scale it down to fit.
        scale = _HW_SCALE_REGION
        if img_h * scale > box_h:
            scale = box_h / img_h
        draw_w, draw_h = img_w * scale, img_h * scale
        rect = fitz.Rect(x0 + 1, y0, x0 + 1 + draw_w, y0 + draw_h)

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
                        images: dict[str, bytes] | None = None) -> None:
    """
    Render the flat overlay list onto a copy of the PDF. Each overlay carries
    its own formatting (font, size, bold/italic/underline) which is applied
    via PyMuPDF's HTML/Story renderer.

    If `images` maps an overlay id -> PNG bytes (rendered handwriting), that
    overlay is stamped as an image instead of typeset text.
    """
    images = images or {}
    doc = fitz.open(pdf_path)
    for ov in overlays:
        page_idx = ov.get("page", 0)
        if page_idx < 0 or page_idx >= len(doc):
            continue
        page = doc[page_idx]

        png = images.get(ov.get("id"))
        if png:
            try:
                _insert_handwriting_image(page, ov["bbox"], png)
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


def build_overlays_from_structure(structure: dict, answers: dict) -> list[dict]:
    """
    Turn the preprocessor's structured units + LLM answers into a flat list
    of editable overlays. Inline blanks get a small region just above the
    underscore; open-response answers use their detected answer_region.
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
