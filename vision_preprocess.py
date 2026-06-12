"""
Vision-based detection for scanned (image-only) PDF pages.

The text-layer pipeline in preprocess.py needs real characters and
underscore runs. Scanned worksheets have neither — each page is a single
raster image. This module renders such a page, asks a vision model to
locate every fill-in prompt and its blank line(s), and converts the
result into the same Unit/Slot structure the renderer consumes.

Only inline_blanks and open_response units are produced here; tables on
a scan are emitted as their individual blank slots.
"""

import base64
import os
import re

import fitz

from json_utils import extract_json_object
from preprocess import Slot, Unit


VISION_MODEL = os.environ.get("VISION_MODEL", "openai/gpt-5.5")
VISION_DPI = int(os.environ.get("VISION_DPI", "200"))

_BLANK_TOKEN = "___"

_SYSTEM = (
    "You analyze a scanned worksheet page image and locate every place a "
    "student is expected to write an answer. Return ONLY a JSON object — no "
    "prose, no markdown fences.\n"
    "\n"
    "Shape:\n"
    '{ "items": [ {\n'
    '    "kind": "inline" | "open",\n'
    '    "prompt": "the full sentence or question, with each fill-in blank '
    f'written as the literal token {_BLANK_TOKEN} in reading order",\n'
    '    "blanks": [ [x0,y0,x1,y1], ... ]\n'
    "} ] }\n"
    "\n"
    "Rules:\n"
    "- Coordinates are normalized floats in [0,1] relative to the image, "
    "origin at the TOP-LEFT, as [x0,y0,x1,y1].\n"
    f"- Each blank box bounds the actual blank line/underscore where the "
    f"answer goes. The number of {_BLANK_TOKEN} tokens in 'prompt' MUST equal "
    "the length of 'blanks', in the same left-to-right, top-to-bottom order.\n"
    "- Use kind 'inline' for short fill-in-the-blank lines embedded in a "
    "sentence (most worksheet items). Use 'open' only for questions answered "
    "in a large empty area with no printed line; give a single box covering "
    "that answer area and no " + _BLANK_TOKEN + " token in the prompt.\n"
    "- Ignore header fields you are unsure about (Nombre, Fecha, Hora) only if "
    "they are page chrome; include them as inline items if they are clearly "
    "answer lines.\n"
    "- Do not invent answers. Only locate the blanks and transcribe the "
    "surrounding printed text accurately, including Spanish accents."
)


def _build_client():
    from openai import OpenAI

    api_key = os.environ.get("HCAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No API key found. Set HCAI_API_KEY in .env or environment.")
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://ai.hackclub.com/proxy/v1"),
    )


def _clean_norm_box(box):
    """Sorted normalized [x0,y0,x1,y1] floats, or None if malformed."""
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return [x0, y0, x1, y1]


def _norm_box_to_points(box, page_w: float, page_h: float):
    """Normalized [x0,y0,x1,y1] (top-left origin) -> PDF-point bbox."""
    x0, y0, x1, y1 = box
    return (x0 * page_w, y0 * page_h, x1 * page_w, y1 * page_h)


def _page_is_transposed(blank_boxes: list[list[float]]) -> bool:
    """
    Vision models sometimes return a whole page's coordinates with the x and
    y axes swapped. Real fill-in lines are always wider than tall, so if a
    clear majority of blank boxes come back taller-than-wide, the frame was
    transposed.
    """
    if len(blank_boxes) < 2:
        return False
    portrait = sum(1 for b in blank_boxes if (b[3] - b[1]) > (b[2] - b[0]))
    return portrait >= 0.6 * len(blank_boxes)


def _call_vision(page, client) -> dict:
    pix = page.get_pixmap(dpi=VISION_DPI)
    data_uri = "data:image/png;base64," + base64.b64encode(pix.tobytes("png")).decode()
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Locate every answer blank on this page."},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    return extract_json_object(content)


def detect_scanned_page(page, page_num: int, counter: dict, client=None) -> list[Unit]:
    """Run vision detection on one scanned page and return Unit objects."""
    if client is None:
        client = _build_client()

    parsed = _call_vision(page, client)
    items = parsed.get("items") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        print(f"[vision] page {page_num}: no items in model response")
        return []

    page_w = page.rect.width
    page_h = page.rect.height

    # Pass 1: parse items into normalized boxes (drop malformed ones).
    parsed_items: list[dict] = []
    blank_boxes: list[list[float]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt", "")).strip()
        kind = item.get("kind", "inline")
        boxes = [nb for b in (item.get("blanks") or [])
                 if (nb := _clean_norm_box(b)) is not None]
        if not boxes:
            continue
        if kind != "open":
            blank_boxes.extend(boxes)
        parsed_items.append({"prompt": prompt, "kind": kind, "boxes": boxes})

    # Pass 2: correct a whole-page axis swap if the model transposed it.
    if _page_is_transposed(blank_boxes):
        print(f"[vision] page {page_num}: transposed frame detected, swapping x/y")
        for it in parsed_items:
            it["boxes"] = [[b[1], b[0], b[3], b[2]] for b in it["boxes"]]

    # Pass 3: build units.
    units: list[Unit] = []
    for it in parsed_items:
        prompt, kind, boxes = it["prompt"], it["kind"], it["boxes"]

        if kind == "open":
            region = _norm_box_to_points(boxes[0], page_w, page_h)
            counter["u"] += 1
            units.append(Unit(
                unit_id=f"u{counter['u']}",
                type="open_response",
                page=page_num,
                bbox=region,
                prompt_text=re.sub(r"\s+", " ", prompt).strip(),
                answer_region=region,
            ))
            continue

        # inline_blanks: interleave the prompt's blank tokens with slot ids.
        parts = prompt.split(_BLANK_TOKEN)
        slots: list[Slot] = []
        prompt_parts: list[str] = []
        for i, box in enumerate(boxes):
            prompt_parts.append(parts[i] if i < len(parts) else " ")
            counter["n"] += 1
            slot_id = f"s{counter['n']}"
            sbbox = _norm_box_to_points(box, page_w, page_h)
            slots.append(Slot(
                slot_id=slot_id,
                bbox=sbbox,
                underscore_length=max(1, int(sbbox[2] - sbbox[0])),
            ))
            prompt_parts.append(f"{{{{{slot_id}}}}}")
        prompt_parts.append("".join(parts[len(boxes):]) if len(parts) > len(boxes) else "")
        clean_prompt = re.sub(r"\s+", " ", "".join(prompt_parts)).strip()

        xs0 = [s.bbox[0] for s in slots]
        ys0 = [s.bbox[1] for s in slots]
        xs1 = [s.bbox[2] for s in slots]
        ys1 = [s.bbox[3] for s in slots]
        counter["u"] += 1
        units.append(Unit(
            unit_id=f"u{counter['u']}",
            type="inline_blanks",
            page=page_num,
            bbox=(min(xs0), min(ys0), max(xs1), max(ys1)),
            prompt_text=clean_prompt,
            slots=slots,
        ))

    print(f"[vision] page {page_num}: {len(units)} units, "
          f"{sum(len(u.slots) for u in units)} slots")
    return units
