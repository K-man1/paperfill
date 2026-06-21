"""
Renderer: takes the structure JSON + answers dict and produces a filled
PDF. Coordinates are never seen by the LLM — they come from the
preprocessor. Text auto-fits to the slot bbox.
"""

import os

import fitz


# Tunables
HANDWRITING_FONT = "helv"  # built-in PDF font
MIN_FONT_SIZE = 6

# "Made with Goodnotes" badge stamped on the bottom-left of every rendered page.
WATERMARK_PATH = os.path.join(os.path.dirname(__file__), "assets", "goodnotes_watermark.png")
WATERMARK_WIDTH = 110   # px wide on the page (aspect ratio preserved)
WATERMARK_MARGIN = 18   # px from the left and bottom edges

# One-DM handwriting is generated at 64px tall; squeezing it into a ~14px slot
# turns legible cursive into an illegible scratch. Render it noticeably taller
# than typeset text (real handwriting is bigger than print and overshoots the
# line), bottom-anchored a hair below the underscore so descenders dip under it.
HW_TARGET_HEIGHT = 22   # px tall to aim for at the default font size, width allowing
HW_BASE_SIZE = 11       # overlay font size HW_TARGET_HEIGHT is calibrated to
HW_DESCENDER_DROP = 3   # px the image bottom sits below the underscore line

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


def _insert_handwriting_image(page, bbox, png_bytes: bytes,
                              size: float = HW_BASE_SIZE) -> None:
    """Stamp a transparent handwriting PNG into the slot. Rendered at a real
    handwriting height (HW_TARGET_HEIGHT, scaled by the overlay's font `size`
    so the size picker drives handwriting too) rather than crammed into the
    thin text slot, with the baseline sitting on the underscore and descenders
    dipping just below it. Left-aligned; the PNG already has the paper
    background knocked out to alpha, so it overlays cleanly."""
    import io
    from PIL import Image

    x0, y0, x1, y1 = bbox
    box_w = x1 - x0
    img_w, img_h = Image.open(io.BytesIO(png_bytes)).size
    if not (img_w and img_h):
        return

    # Aim for HW_TARGET_HEIGHT scaled to the chosen font size, but if the word
    # would then run past the blank's width, fall back to fitting the width
    # (same behaviour typeset text has when a long answer must shrink). Aspect
    # ratio is always preserved.
    try:
        target_h = HW_TARGET_HEIGHT * (float(size) / HW_BASE_SIZE)
    except (TypeError, ValueError):
        target_h = HW_TARGET_HEIGHT
    scale = max(target_h, 1.0) / img_h
    if img_w * scale > box_w:
        scale = box_w / img_w
    draw_w, draw_h = img_w * scale, img_h * scale

    bottom = y1 + HW_DESCENDER_DROP            # let g/y/p tails dip under the line
    rect = fitz.Rect(x0 + 1, bottom - draw_h, x0 + 1 + draw_w, bottom)
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

    If `images` maps an overlay id -> PNG bytes (handwriting from One-DM), that
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
                _insert_handwriting_image(page, ov["bbox"], png,
                                          ov.get("size", _OV_DEFAULTS["size"]))
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
    doc.save(out_path)
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
