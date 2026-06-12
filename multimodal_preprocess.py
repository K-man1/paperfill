"""
Multimodal answer-space detector — a parallel path to preprocess.py.

Instead of finding blanks deterministically from underscore runs and vertical
gaps, this module asks a vision model to read the worksheet and list every
place a student writes an answer. Crucially the model never emits coordinates:
for each answer space it returns the printed text the blank attaches to
(`anchor_text`, transcribed verbatim) plus where the blank sits relative to
that anchor. A deterministic resolver then matches each anchor back against the
page's PyMuPDF character map and re-derives the blank/answer geometry, reusing
the very same routines preprocess.py uses (find_underscore_runs, the
open-response answer-region geometry, char bbox helpers).

The output is the SAME structure dict that preprocess_pdf produces, so the
renderer (render.py) consumes it unchanged.

Design seams:
  * The model call is isolated in `_default_detector` and can be swapped by
    passing a `detector` callable to `multimodal_preprocess_pdf` (used by the
    A/B harness to inject recorded responses, and to swap providers).
  * Structured JSON-schema output forces the response shape — no
    "return only JSON" prompt-scraping.
  * The whole PDF is uploaded once via the Files API; pages are addressed by
    index in the returned items.

Anchors that cannot be resolved to a location are dropped and logged — never
guessed into place.
"""

import os
import re
import base64
from dataclasses import asdict

import fitz

from preprocess import (
    Slot,
    Unit,
    ALL_FORMATS,
    find_underscore_runs,
    bbox_of_chars,
    line_is_whitespace,
    lines_in_reading_order,
    MIN_ANSWER_SPACE,
    PAGE_RIGHT_MARGIN,
)


MULTIMODAL_MODEL = os.environ.get(
    "MULTIMODAL_MODEL", os.environ.get("VISION_MODEL", "openai/gpt-5.5")
)

# Width (PDF points) of a synthesized blank when an anchor has no literal
# underscore run to size against — e.g. "definition of bob - ____(empty)".
SYNTH_BLANK_WIDTH = 130.0
# How far past an anchor's end (points) we still accept an underscore run as
# "the blank that attaches to this anchor".
ANCHOR_RUN_GAP = 60.0


# --------------------------------------------------------------------------
# Structured-output schema for the model call.
# --------------------------------------------------------------------------

ANSWER_SPACE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "page": {
                        "type": "integer",
                        "description": "0-based page index the blank is on.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["inline", "open"],
                        "description": (
                            "'inline' for a short fill-in embedded in a line; "
                            "'open' for a question answered in empty space below."
                        ),
                    },
                    "anchor_text": {
                        "type": "string",
                        "description": (
                            "The printed prompt/word the blank attaches to, "
                            "transcribed VERBATIM from the page (exact casing, "
                            "punctuation and accents). For inline this is the "
                            "run of printed text right next to the blank; for "
                            "open it is the question text."
                        ),
                    },
                    "blank_position": {
                        "type": "string",
                        "enum": ["after", "before", "none"],
                        "description": (
                            "For inline: is the blank AFTER or BEFORE the "
                            "anchor_text in reading order. For open: 'none'."
                        ),
                    },
                },
                "required": ["page", "kind", "anchor_text", "blank_position"],
            },
        }
    },
    "required": ["items"],
}


_SYSTEM = (
    "You read a worksheet (provided as a PDF) and list every place a student "
    "is expected to WRITE an answer. Do not solve anything; only locate blanks.\n"
    "\n"
    "For each answer space return:\n"
    "  - page: the 0-based page index it appears on.\n"
    "  - kind: 'inline' for a short fill-in sitting inside a printed line "
    "(a blank line, an underscore run, or empty space after a prompt word/"
    "dash); 'open' for a question answered in the large empty area beneath it.\n"
    "  - anchor_text: the printed text the blank attaches to, transcribed "
    "VERBATIM (exact words, casing, punctuation, accents) so it can be found "
    "again in the page. For inline, give the whole printed phrase on the blank's "
    "side of it (typically 3+ words, e.g. 'The capital of France is') — NOT a "
    "single common word like 'the', which is ambiguous. Do NOT include the "
    "blank itself. For open, give the question.\n"
    "  - blank_position: for inline, whether the blank falls 'after' or "
    "'before' the anchor_text; for open use 'none'.\n"
    "\n"
    "Rules:\n"
    "- A term or prompt followed by a dash or colon and then empty space "
    "(e.g. 'photosynthesis -' or 'Capital:') IS an inline blank: the student "
    "writes the answer in that space, even when no line is printed. Use the "
    "term+dash/colon as anchor_text with blank_position 'after'.\n"
    "- A hyphenated or compound word inside running text (e.g. 'single-eyed', "
    "'well-being', 'follow-up') is NOT a blank. Only list a space where a "
    "student actually writes.\n"
    "- Section headings, titles and instructions are not answer spaces unless "
    "they are themselves a labelled fill-in.\n"
    "- Transcribe anchor_text exactly as printed; a paraphrase will fail to "
    "match and the blank will be dropped.\n"
    "- Each distinct blank is its own item, even when several share a line."
)


def _build_client():
    # Reuse the OpenAI-compatible client wiring from the scanned-page path so
    # provider/base-url/key handling stays in one place.
    from vision_preprocess import _build_client as _bc

    return _bc()


# --------------------------------------------------------------------------
# Model call (isolated + swappable).
# --------------------------------------------------------------------------

def _page_texts(doc) -> list[dict]:
    """Per-page extracted text, given to the model as a transcription aid so
    its anchor_text matches the real character map."""
    return [{"page": i, "text": page.get_text()} for i, page in enumerate(doc)]


# How the worksheet reaches the model. "image" (default) renders each page and
# sends them as vision inputs — the page-image+text approach, which works with
# any vision model. "pdf" uploads the document itself once (Files API, with an
# inline-base64 fallback) for providers with native multi-page PDF support.
MULTIMODAL_INPUT = os.environ.get("MULTIMODAL_INPUT", "image").lower()
MULTIMODAL_DPI = int(os.environ.get("MULTIMODAL_DPI", "150"))


def _upload_pdf(client, pdf_path: str) -> dict:
    """Whole-PDF content part: Files API reference if supported, else an inline
    base64 part so the multi-page document is still sent in one request."""
    try:
        with open(pdf_path, "rb") as fh:
            up = client.files.create(file=fh, purpose="user_data")
        return {"type": "file", "file": {"file_id": up.id}}
    except Exception as e:  # provider has no Files API (e.g. OpenRouter)
        print(f"[multimodal] files API unavailable ({e!r}); inlining PDF")
        with open(pdf_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return {"type": "file", "file": {
            "filename": os.path.basename(pdf_path),
            "file_data": f"data:application/pdf;base64,{b64}",
        }}


def _page_image_parts(pdf_path: str) -> list[dict]:
    """Render each page to a PNG and return interleaved page-marker + image
    content parts (all pages in one message)."""
    doc = fitz.open(pdf_path)
    parts: list[dict] = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=MULTIMODAL_DPI)
        uri = "data:image/png;base64," + base64.b64encode(pix.tobytes("png")).decode()
        parts.append({"type": "text", "text": f"--- PAGE {i} IMAGE ---"})
        parts.append({"type": "image_url", "image_url": {"url": uri}})
    doc.close()
    return parts


def _default_detector(pdf_path: str, pages: list[dict], *,
                      client=None, model: str | None = None) -> list[dict]:
    """
    Live vision call. Sends the worksheet (page images by default, or the PDF
    itself when MULTIMODAL_INPUT='pdf') plus per-page extracted text as a
    transcription aid, and returns the raw `items` list under the structured
    JSON schema. Isolated so a different provider/model can be dropped in.
    """
    if client is None:
        client = _build_client()
    model = model or MULTIMODAL_MODEL

    text_context = "\n\n".join(
        f"--- PAGE {p['page']} TEXT ---\n{p['text']}" for p in pages
    )
    user_text = (
        "Locate every answer blank in this worksheet. Pages are 0-indexed; use "
        "the page numbers shown below. Use the extracted page text only to "
        "transcribe anchor_text accurately; the images are authoritative for "
        "layout.\n\n" + text_context
    )

    extra_body = {}
    if MULTIMODAL_INPUT == "pdf":
        doc_parts = [_upload_pdf(client, pdf_path)]
        # Ask OpenRouter-style providers to rasterize the PDF for vision.
        extra_body = {"plugins": [{"id": "file-parser",
                                   "pdf": {"engine": "pdf-text"}}]}
    else:
        doc_parts = _page_image_parts(pdf_path)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                *doc_parts,
            ]},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer_spaces",
                "schema": ANSWER_SPACE_SCHEMA,
                "strict": True,
            },
        },
        extra_body=extra_body or None,
    )
    content = resp.choices[0].message.content or "{}"
    from json_utils import extract_json_object

    parsed = extract_json_object(content)
    items = parsed.get("items") if isinstance(parsed, dict) else None
    return items if isinstance(items, list) else []


# --------------------------------------------------------------------------
# Anchor -> bbox resolver.
# --------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _flatten_chars(lines: list[dict]) -> list[dict]:
    """
    Flatten a page's reading-order lines into one char list, inserting a single
    space separator between lines so a multi-line anchor still matches. Each
    char keeps its source line index and its index within that line, so the
    blank geometry can be re-derived from the original line later.
    """
    flat: list[dict] = []
    for li, line in enumerate(lines):
        if li > 0:
            flat.append({"c": " ", "bbox": None, "line": li, "ci": -1, "sep": True})
        for ci, c in enumerate(line["chars"]):
            flat.append({"c": c["c"], "bbox": c["bbox"], "line": li, "ci": ci,
                         "sep": False})
    return flat


def _norm_index(flat: list[dict]) -> tuple[str, list[int]]:
    """Build a whitespace-collapsed lowercase string of the page plus a map
    from each normalized-string position back to its index in `flat`."""
    norm: list[str] = []
    idx_map: list[int] = []
    prev_space = True
    for i, c in enumerate(flat):
        ch = c["c"]
        if ch.isspace():
            if not prev_space:
                norm.append(" ")
                idx_map.append(i)
                prev_space = True
            continue
        norm.append(ch.lower())
        idx_map.append(i)
        prev_space = False
    while norm and norm[-1] == " ":
        norm.pop()
        idx_map.pop()
    return "".join(norm), idx_map


def _find_occurrences(haystack: str, needle: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if not needle:
        return out
    start = 0
    while True:
        k = haystack.find(needle, start)
        if k < 0:
            break
        out.append((k, k + len(needle)))
        start = k + 1
    return out


class _PageIndex:
    """Char map + normalized search index for one page, plus the bookkeeping
    used to disambiguate repeated anchors by reading order."""

    def __init__(self, page, lines):
        self.page = page
        self.lines = lines
        self.flat = _flatten_chars(lines)
        self.norm, self.idx_map = _norm_index(self.flat)
        self.used_spans: set[tuple[int, int]] = set()
        self.cursor_src = -1  # last consumed source-char index (reading order)

    def resolve(self, anchor: str) -> tuple[int, int] | None:
        """Return the source-char index span (start, end_inclusive) of the
        chosen occurrence of `anchor`, or None if it isn't on the page.

        Disambiguation: prefer the earliest not-yet-used occurrence at or after
        the reading-order cursor; otherwise the earliest unused occurrence
        anywhere; mark it used so the next identical anchor picks a later one.
        """
        na = _normalize(anchor)
        occ = _find_occurrences(self.norm, na)
        if not occ:
            return None

        chosen = None
        for (s, e) in occ:
            if (s, e) in self.used_spans:
                continue
            if self.idx_map[s] >= self.cursor_src:
                chosen = (s, e)
                break
        if chosen is None:
            for (s, e) in occ:
                if (s, e) not in self.used_spans:
                    chosen = (s, e)
                    break
        if chosen is None:
            return None

        self.used_spans.add(chosen)
        src_start = self.idx_map[chosen[0]]
        src_end = self.idx_map[chosen[1] - 1]
        self.cursor_src = max(self.cursor_src, src_end)
        return (src_start, src_end)


def _anchor_bbox(flat, src_start, src_end):
    """Union bbox over the real (bbox-bearing) chars in a source span."""
    boxes = [flat[i]["bbox"] for i in range(src_start, src_end + 1)
             if flat[i]["bbox"]]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _last_real_char(flat, src_start, src_end):
    for i in range(src_end, src_start - 1, -1):
        if flat[i]["bbox"]:
            return flat[i]
    return None


def _first_real_char(flat, src_start, src_end):
    for i in range(src_start, src_end + 1):
        if flat[i]["bbox"]:
            return flat[i]
    return None


# --------------------------------------------------------------------------
# Geometry derivation (reuses preprocess routines).
# --------------------------------------------------------------------------

def _inline_slot_bbox(line, anchor_ci, position):
    """
    Re-derive the blank geometry for an inline anchor on `line`.

    First tries to reuse an actual underscore run (find_underscore_runs /
    bbox_of_chars) adjacent to the anchor on the side `position` points to.
    If there is no literal run (the blank is bare space, e.g. after a dash),
    synthesize a same-height region of SYNTH_BLANK_WIDTH on that side.

    Returns (bbox, underscore_length).
    """
    chars = line["chars"]
    runs = find_underscore_runs(chars)

    if position == "before":
        # nearest run ending before the anchor's first char
        cand = [(rs, re_) for (rs, re_) in runs if re_ < anchor_ci]
        if cand:
            rs, re_ = max(cand, key=lambda r: r[1])
            anchor_x0 = chars[anchor_ci]["bbox"][0]
            if anchor_x0 - chars[re_]["bbox"][2] <= ANCHOR_RUN_GAP:
                return bbox_of_chars(chars, rs, re_), re_ - rs + 1
        # synthesize to the left of the anchor
        ax0, ay0, ax1, ay1 = chars[anchor_ci]["bbox"]
        return (max(line["bbox"][0], ax0 - SYNTH_BLANK_WIDTH) , ay0,
                ax0 - 2, ay1), 0

    # position == "after" (default): nearest run starting after the anchor end
    cand = [(rs, re_) for (rs, re_) in runs if rs > anchor_ci]
    if cand:
        rs, re_ = min(cand, key=lambda r: r[0])
        anchor_x1 = chars[anchor_ci]["bbox"][2]
        if chars[rs]["bbox"][0] - anchor_x1 <= ANCHOR_RUN_GAP:
            return bbox_of_chars(chars, rs, re_), re_ - rs + 1
    # synthesize to the right of the anchor
    ax0, ay0, ax1, ay1 = chars[anchor_ci]["bbox"]
    return (ax1 + 3, ay0, ax1 + 3 + SYNTH_BLANK_WIDTH, ay1), 0


def _open_region_for_anchor(anchor_bbox, page, lines, obstacle_bboxes):
    """
    Answer region beneath an open-response anchor — the same geometry
    detect_open_response_units uses: span from just under the prompt to the
    next content line (or footer), clamped above any table/image obstacle, and
    only if there's at least MIN_ANSWER_SPACE of blank room.
    """
    p_x0, p_top, p_x1, p_bottom = anchor_bbox
    answer_right = page.rect.x1 - PAGE_RIGHT_MARGIN

    # Only real content lines bound the answer area — whitespace-only lines
    # (stray space glyphs) are the blank space we want to write into, exactly
    # as detect_open_response_units filters them.
    below = [l["bbox"][1] for l in lines
             if not line_is_whitespace(l) and l["bbox"][1] > p_bottom + 1]
    next_top = min(below) if below else page.rect.y1 - 60
    for ob in obstacle_bboxes:
        if p_bottom <= ob[1] < next_top and ob[0] < answer_right and ob[2] > p_x0:
            next_top = ob[1]

    if next_top - p_bottom < MIN_ANSWER_SPACE:
        return None
    return (p_x0 + 4, p_bottom + 4, max(p_x1, answer_right), next_top - 6)


# --------------------------------------------------------------------------
# Main entry point.
# --------------------------------------------------------------------------

def multimodal_preprocess_pdf(path: str, formats=None, detector=None) -> dict:
    """
    Multimodal counterpart to preprocess_pdf. Asks `detector` (the vision model
    by default) for the answer spaces, resolves each anchor to geometry, and
    emits the same structure dict the renderer consumes.

    `formats` filters which kinds are kept (inline_blanks / open_response), the
    same selector preprocess_pdf takes. `detector` is the swap seam: a callable
    (pdf_path, pages) -> list[item dicts]; when None the live model is used.

    Anchor-resolution failures are collected under structure["dropped"] and
    logged, never silently misplaced.
    """
    active = set(formats) & set(ALL_FORMATS) if formats else set()
    if not active:
        active = set(ALL_FORMATS)
    want_inline = "inline_blanks" in active
    want_open = "open_response" in active

    doc = fitz.open(path)
    pages = _page_texts(doc)

    if detector is None:
        detector = _default_detector
    raw_items = detector(path, pages) or []

    # Some models number pages 1..N despite the 0-indexed instruction. If every
    # returned page is >=1 and the max equals the page count (not count-1),
    # treat the whole batch as 1-based and shift it down.
    page_vals = [int(it.get("page", 0) or 0) for it in raw_items
                 if isinstance(it, dict)]
    if (page_vals and min(page_vals) >= 1 and max(page_vals) == len(doc)
            and len(doc) >= 1):
        print("[multimodal] 1-based page indices detected; shifting to 0-based")
        for it in raw_items:
            if isinstance(it, dict) and "page" in it:
                it["page"] = int(it.get("page", 1) or 1) - 1

    # Per-page index, built lazily and cached.
    page_idx_cache: dict[int, _PageIndex] = {}
    obstacle_cache: dict[int, list] = {}

    def page_index(pn: int) -> _PageIndex | None:
        if pn < 0 or pn >= len(doc):
            return None
        if pn not in page_idx_cache:
            lines = lines_in_reading_order(doc[pn])
            page_idx_cache[pn] = _PageIndex(doc[pn], lines)
            obstacle_cache[pn] = [
                b["bbox"] for b in doc[pn].get_text("rawdict")["blocks"]
                if b.get("type") == 1
            ]
        return page_idx_cache[pn]

    counter = {"u": 0, "n": 0}
    units: list[Unit] = []
    dropped: list[dict] = []

    def drop(item, reason):
        rec = {"reason": reason, "anchor_text": item.get("anchor_text", ""),
               "page": item.get("page"), "kind": item.get("kind")}
        dropped.append(rec)
        print(f"[multimodal] dropped anchor (reason={reason}, "
              f"page={rec['page']}): {rec['anchor_text']!r}")

    # Keep items in (page, model order) so reading-order disambiguation is sane.
    indexed = [(i, it) for i, it in enumerate(raw_items) if isinstance(it, dict)]
    indexed.sort(key=lambda t: (t[1].get("page", 0), t[0]))

    for _, item in indexed:
        kind = item.get("kind", "inline")
        anchor = str(item.get("anchor_text", "")).strip()
        pn = int(item.get("page", 0) or 0)
        if not anchor:
            drop(item, "empty_anchor")
            continue
        if kind == "open" and not want_open:
            continue
        if kind != "open" and not want_inline:
            continue

        pidx = page_index(pn)
        if pidx is None:
            drop(item, "bad_page")
            continue

        span = pidx.resolve(anchor)
        if span is None:
            drop(item, "anchor_not_found")
            continue
        src_start, src_end = span
        abbox = _anchor_bbox(pidx.flat, src_start, src_end)
        if abbox is None:
            drop(item, "no_anchor_bbox")
            continue

        if kind == "open":
            region = _open_region_for_anchor(
                abbox, pidx.page, pidx.lines, obstacle_cache.get(pn, [])
            )
            if region is None:
                drop(item, "no_answer_space_below")
                continue
            counter["u"] += 1
            units.append(Unit(
                unit_id=f"u{counter['u']}",
                type="open_response",
                page=pn,
                bbox=abbox,
                prompt_text=_normalize(anchor),
                answer_region=region,
            ))
            continue

        # inline
        position = item.get("blank_position", "after")
        if position not in ("after", "before"):
            position = "after"
        end_char = (_last_real_char(pidx.flat, src_start, src_end)
                    if position == "after"
                    else _first_real_char(pidx.flat, src_start, src_end))
        if end_char is None:
            drop(item, "no_anchor_bbox")
            continue
        line = pidx.lines[end_char["line"]]
        slot_bbox, ulen = _inline_slot_bbox(line, end_char["ci"], position)

        counter["u"] += 1
        counter["n"] += 1
        slot_id = f"s{counter['n']}"
        ptext = _normalize(anchor)
        prompt = (f"{ptext} {{{{{slot_id}}}}}" if position == "after"
                  else f"{{{{{slot_id}}}}} {ptext}")
        units.append(Unit(
            unit_id=f"u{counter['u']}",
            type="inline_blanks",
            page=pn,
            bbox=slot_bbox,
            prompt_text=prompt,
            slots=[Slot(slot_id=slot_id, bbox=slot_bbox,
                        underscore_length=ulen)],
        ))

    doc.close()

    if dropped:
        print(f"[multimodal] {len(dropped)} anchor(s) dropped of "
              f"{len(indexed)} returned")

    return {
        "source": path,
        "detector": "multimodal",
        "unit_count": len(units),
        "slot_count": counter["n"],
        "dropped_count": len(dropped),
        "dropped": dropped,
        "units": [asdict(u) for u in units],
    }


if __name__ == "__main__":
    import sys
    import json

    for path in sys.argv[1:]:
        result = multimodal_preprocess_pdf(path)
        out_path = path.split("/")[-1].replace(".pdf", ".mm.structure.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"{path}: {result['unit_count']} units, "
              f"{result['slot_count']} slots, "
              f"{result['dropped_count']} dropped → {out_path}")
