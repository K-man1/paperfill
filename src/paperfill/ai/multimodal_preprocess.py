import os
import re
import base64
import unicodedata
from dataclasses import asdict

import fitz

from paperfill.ai.preprocess import (
    Slot,
    Unit,
    ALL_FORMATS,
    find_underscore_runs,
    bbox_of_chars,
    line_is_whitespace,
    is_empty_bullet_line,
    lines_in_reading_order,
    text_page_rect,
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
    "You read a worksheet and list every place a student is expected to WRITE "
    "an answer. Do not solve anything; only locate blanks.\n"
    "\n"
    "For each answer space return:\n"
    "  - page: the 0-based page index it appears on.\n"
    "  - kind: 'inline' for a short fill-in sitting inside a printed line "
    "(a blank line, an underscore run, or empty space after a prompt word/"
    "dash); 'open' for a question answered in the large empty area beneath it.\n"
    "  - anchor_text: the printed text DIRECTLY TOUCHING this blank, transcribed "
    "VERBATIM (exact words, casing, punctuation, accents). It is used to find "
    "the blank again by exact text search, so it must be the real words next to "
    "THIS blank and unique enough to land on it.\n"
    "  - blank_position: for inline, whether the blank is 'after' or 'before' "
    "the anchor_text; for open use 'none'.\n"
    "\n"
    "Choosing anchor_text (this is the part models get wrong):\n"
    "- Use the words IMMEDIATELY beside the blank — the word/phrase the blank "
    "physically abuts. If the blank is to the LEFT of a word (numbered lists, "
    "matching columns, '____ Epididymis'), the anchor is THAT word and "
    "blank_position='before'. If the blank follows text ('The capital is ___'), "
    "the anchor is the text before it and blank_position='after'.\n"
    "- NEVER use a shared column header, title, row label, or generic word "
    "(e.g. 'Structure', 'Order', 'Answer', 'Name', 'the') as the anchor for a "
    "blank that actually sits next to specific content. Each row/item has its "
    "OWN distinct text — use that.\n"
    "- Every item's anchor_text MUST be DIFFERENT. If two blanks would get the "
    "same anchor, lengthen each to include neighbouring words until unique. A "
    "good anchor is typically 2-6 words.\n"
    "- Copy the text exactly as printed (a paraphrase will fail to match and the "
    "blank is dropped). For 'open', anchor_text is the full question text.\n"
    "\n"
    "What counts:\n"
    "- A term/prompt followed by a dash or colon then empty space "
    "('photosynthesis -', 'Capital:') IS an inline blank (write in the space, "
    "no printed line needed); anchor='photosynthesis -', position='after'.\n"
    "- A hyphenated/compound word in running text ('single-eyed', 'well-being') "
    "is NOT a blank.\n"
    "- Headings, titles and instructions are not answer spaces unless they are "
    "themselves a labelled fill-in.\n"
    "- Each distinct blank is its own item, even when several share a line."
)


def _build_client():
    # Reuse the OpenAI-compatible client wiring from the scanned-page path so
    # provider/base-url/key handling stays in one place.
    from paperfill.ai.vision_preprocess import _build_client as _bc

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
    try:
        parts: list[dict] = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=MULTIMODAL_DPI)
            uri = "data:image/png;base64," + base64.b64encode(pix.tobytes("png")).decode()
            parts.append({"type": "text", "text": f"--- PAGE {i} IMAGE ---"})
            parts.append({"type": "image_url", "image_url": {"url": uri}})
    finally:
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

    from paperfill.ai.llm_client import call_context
    with call_context("detect"):
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
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
    from paperfill.utils.json_utils import json_from_response

    parsed = json_from_response(resp)
    items = parsed.get("items")
    return items if isinstance(items, list) else []


# --------------------------------------------------------------------------
# Anchor -> bbox resolver.
# --------------------------------------------------------------------------

# Canonicalize characters before matching so the model's transcription and the
# PDF's char map line up despite cosmetic differences. All dash variants (hyphen,
# en/em dash, minus) fold to '-', smart quotes to ASCII, zero-width junk is
# dropped. This is the #1 cause of "anchor_not_found" on dash/colon blanks.
_DASHES = "‐‑‒–—―−﹘﹣－·•"
_ZERO_WIDTH = "​‌‍﻿­"


def _canon_char(ch: str) -> str:
    if ch in _ZERO_WIDTH:
        return ""
    if ch in _DASHES:
        return "-"
    if ch in "‘’ʼ`´":
        return "'"
    if ch in "“”":
        return '"'
    return ch


def _normalize(s: str) -> str:
    # NFC first: the page's char map carries "país" composed while a model
    # often transcribes it decomposed (i + U+0301), and the two never match.
    s = unicodedata.normalize("NFC", s or "")
    s = "".join(_canon_char(ch) for ch in s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _alnum_char(ch: str) -> str:
    """One char reduced to its fuzzy-matching form: the lowercase base letter
    with any combining accent dropped, or "" if it isn't alphanumeric.

    Both the anchor and the page go through this, which is the point. They
    used to disagree — the page kept "í" and "²" while the anchor was stripped
    to [a-z0-9], so 'el país es grande' indexed as 'elpaísesgrande' against an
    anchor of 'elpases' and no accent or superscript could match through the
    fuzzy fallback. Folding the accent off also lets a decomposed page and a
    composed transcription meet in the middle.
    """
    base = unicodedata.normalize("NFD", ch.lower())[:1]
    return base if base.isalnum() else ""


def _alnum(s: str) -> str:
    """Letters/digits only, lowercased — a punctuation/space-insensitive form
    used as a fuzzy fallback when exact text matching fails."""
    return "".join(_alnum_char(ch) for ch in (s or ""))


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
    """Build a canonicalized, whitespace-collapsed lowercase string of the page
    plus a map from each string position back to its index in `flat`."""
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
        cc = _canon_char(ch)
        if not cc:  # zero-width: contributes nothing
            continue
        norm.append(cc.lower())
        idx_map.append(i)
        prev_space = False
    while norm and norm[-1] == " ":
        norm.pop()
        idx_map.pop()
    return "".join(norm), idx_map


def _alnum_index(flat: list[dict]) -> tuple[str, list[int]]:
    """Letters/digits-only string of the page plus a map back to `flat`, for
    punctuation-insensitive fuzzy fallback matching."""
    alnum: list[str] = []
    idx_map: list[int] = []
    for i, c in enumerate(flat):
        ch = _alnum_char(c["c"])
        if ch:
            alnum.append(ch)
            idx_map.append(i)
    return "".join(alnum), idx_map


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


def _token_bounded(hay: str, s: int, e: int) -> bool:
    """True unless hay[s:e] was cut out of the middle of a longer word/number.

    Without this a one-letter anchor ('a', 'b', 'y' — the side labels on a
    geometry diagram) matches the first letter of an ordinary word: 'a' landed
    inside "Special" in a section heading and the answer was stamped across it.
    """
    left_ok = s == 0 or not (hay[s - 1].isalnum() and hay[s].isalnum())
    right_ok = e >= len(hay) or not (hay[e - 1].isalnum() and hay[e].isalnum())
    return left_ok and right_ok


# A one-character anchor cannot identify a location no matter where it lands,
# so it is refused outright rather than resolved to the first such glyph.
MIN_ANCHOR_LEN = 2

# Scattered-token fallback (see _PageIndex.resolve). A cluster is only believed
# when it accounts for most of the anchor and stays inside one item's worth of
# page. The height allowance is generous because an item's own answer space sits
# inside it: on a graphing question the "domain:" label the anchor names is a
# third of a page below the item number it was spliced onto.
MIN_TOKEN_COVERAGE = 0.5
MAX_CLUSTER_WIDTH_FRAC = 0.55
MAX_CLUSTER_HEIGHT_FRAC = 0.45


class _PageIndex:
    """Char map + normalized search index for one page, plus the bookkeeping
    used to disambiguate repeated anchors by reading order."""

    def __init__(self, page, lines):
        self.page = page
        # Char bboxes live in the unrotated frame; page.rect does not.
        self.page_rect = text_page_rect(page)
        self.lines = lines
        self.flat = _flatten_chars(lines)
        self.norm, self.norm_map = _norm_index(self.flat)
        self.alnum, self.alnum_map = _alnum_index(self.flat)
        self.used_src: set[tuple[int, int]] = set()   # chosen source spans
        self.cursor_src = -1                 # reading-order cursor (source idx)

    def _spans(self, needle: str, hay: str, idx_map: list[int], *,
               bounded: bool = False):
        """All source-index spans (start, end_inclusive) where `needle` occurs
        in `hay`. With `bounded`, matches falling inside a longer word are
        discarded — only meaningful on the normalized index, since the
        alphanumeric one has no separators left to bound against."""
        return [(idx_map[s], idx_map[e - 1])
                for (s, e) in _find_occurrences(hay, needle)
                if not bounded or _token_bounded(hay, s, e)]

    def _available(self, spans, *, use_cursor: bool = True):
        """`spans` still eligible to be picked, preferring those at or after the
        reading-order cursor so a repeated anchor advances down the page.

        `use_cursor=False` for the members of a scattered cluster: reading order
        is precisely what has broken for those anchors, so a token's correct
        occurrence often sits "behind" the cursor (a fraction's numerator line
        sorts ahead of the item number it belongs to) and preferring what is
        ahead would reach into the next item instead.
        """
        # Keyed on the whole span, not its start: the prompt tells the model
        # to lengthen an anchor with neighbouring words until it is unique, so
        # two anchors opening on the same word ("La capital" / "La capital de
        # Mexico") are the expected shape and both have to stay resolvable.
        fresh = [sp for sp in spans if sp not in self.used_src]
        if not use_cursor:
            return fresh
        ahead = [sp for sp in fresh if sp[0] >= self.cursor_src]
        return ahead or fresh

    def _take(self, spans):
        """Commit a match. Only the last span is consumed: a cluster's earlier
        spans are shared context (the item number "12." anchors both that item's
        "domain:" and its "range:" blank) and must stay available, while the
        span nearest the blank is what has to advance on the next anchor."""
        spans = sorted(spans)
        self.used_src.add(spans[-1])
        self.cursor_src = max(self.cursor_src, spans[-1][1])
        return spans

    def _cluster(self, anchor: str):
        """Locate an anchor whose text is real but scattered.

        An anchor transcribed from get_text() often cannot appear as one
        contiguous run in reading order. A stacked fraction extracts as separate
        numerator and denominator lines that sort away from their item number
        ("3. 2√8 √200" is laid out as "… 2√8  4. (3 + √6)(3 −√6)  3. √200 …"),
        and the prompt's uniqueness rule makes the model splice a label onto a
        distant item number ("12. domain:"). The words are all on the page, just
        not in that order, so match them individually and keep the spatially
        tightest cluster.
        """
        tokens = [t for t in _normalize(anchor).split(" ") if t]
        if len(tokens) < 2:
            return None

        found = []
        for token in tokens:
            spans = self._available(
                self._spans(token, self.norm, self.norm_map, bounded=True),
                use_cursor=False)
            if spans:
                found.append((token, spans))
        if len(found) < 2:
            return None

        # Seed on the most distinctive token — for these anchors that is the
        # item number, the one piece of text that pins down which item this is.
        # Only the seed honours the reading-order cursor; it is what decides
        # which item the cluster belongs to.
        seed_i = min(range(len(found)),
                     key=lambda i: (len(found[i][1]), -len(found[i][0])))
        seed = self._available(found[seed_i][1])[0]
        seed_box = _anchor_bbox(self.flat, [seed])
        if seed_box is None:
            return None
        cx, cy = (seed_box[0] + seed_box[2]) / 2, (seed_box[1] + seed_box[3]) / 2
        max_dx = MAX_CLUSTER_WIDTH_FRAC * self.page_rect.width
        max_dy = MAX_CLUSTER_HEIGHT_FRAC * self.page_rect.height

        def offset(sp):
            b = _anchor_bbox(self.flat, [sp])
            if b is None:
                return None
            return (abs((b[0] + b[2]) / 2 - cx), abs((b[1] + b[3]) / 2 - cy))

        chosen, kept = [seed], [found[seed_i][0]]
        for i, (token, spans) in enumerate(found):
            if i == seed_i:
                continue
            near = [(o[0] + o[1], sp) for sp in spans
                    for o in [offset(sp)]
                    if o and o[0] <= max_dx and o[1] <= max_dy]
            # A token whose every occurrence is elsewhere on the page belongs to
            # some other item; leave it out rather than let it drag the bbox.
            if not near:
                continue
            chosen.append(min(near)[1])
            kept.append(token)

        # Glyph-mangled tokens (Cambria Math subsets extract as "%√'") are never
        # found; the anchor is only trusted if most of its text was.
        if len(kept) < 2:
            return None
        if sum(len(t) for t in kept) < MIN_TOKEN_COVERAGE * sum(len(t) for t in tokens):
            return None
        # The leading token is the item number the anchor opens with. A cluster
        # that matched everything except that has found some other item's digits,
        # not this item — it is the difference between a real match and a pile of
        # loose numerals that happen to sit near each other.
        if tokens[0] not in kept:
            return None
        return self._take(chosen)

    def resolve(self, anchor: str) -> list[tuple[int, int]] | None:
        """Return the source-char index spans for the chosen occurrence of
        `anchor`, or None if it can't be located. Usually one span; the
        scattered-token fallback returns several.

        Tries an exact (canonicalized) text match first, then a punctuation-
        insensitive alphanumeric fallback, then `_cluster`. Disambiguation:
        prefer the earliest not-yet-used occurrence at or after the reading-order
        cursor; otherwise the earliest unused one; mark it used so a repeated
        anchor advances.
        """
        needle = _normalize(anchor)
        spans = []
        if len(needle) >= MIN_ANCHOR_LEN:
            spans = self._spans(needle, self.norm, self.norm_map, bounded=True)
            if not spans:
                a = _alnum(anchor)
                if len(a) >= 3:  # too-short fuzzy matches are noise
                    spans = self._spans(a, self.alnum, self.alnum_map)
        if not spans:
            return self._cluster(anchor)

        chosen = self._available(sorted(spans))
        if not chosen:
            return self._cluster(anchor)
        return self._take([chosen[0]])


def _anchor_bbox(flat, spans):
    """Union bbox over the real (bbox-bearing) chars in the given source spans.

    Only the spans' own chars count, never the gap between them — a scattered
    anchor's spans can straddle unrelated text that must not widen the box.
    """
    boxes = [flat[i]["bbox"] for (s, e) in spans for i in range(s, e + 1)
             if flat[i]["bbox"]]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _last_real_char(flat, spans):
    """Last real char of the last span — the edge an 'after' blank follows."""
    s, e = spans[-1]
    for i in range(e, s - 1, -1):
        if flat[i]["bbox"]:
            return flat[i]
    return None


def _first_real_char(flat, spans):
    """First real char of the first span — the edge a 'before' blank precedes."""
    s, e = spans[0]
    for i in range(s, e + 1):
        if flat[i]["bbox"]:
            return flat[i]
    return None


# --------------------------------------------------------------------------
# Geometry derivation (reuses preprocess routines).
# --------------------------------------------------------------------------

def _inline_slot_bbox(line, anchor_ci, position, page_rect):
    """
    Re-derive the blank geometry for an inline anchor on `line`.

    First tries to reuse an actual underscore run (find_underscore_runs /
    bbox_of_chars) adjacent to the anchor on the side `position` points to.
    If there is no literal run (the blank is bare space, e.g. after a dash),
    synthesize a region on that side. For an "after" blank the synthesized box
    runs to the next printed text on the line (or the right margin when the
    anchor ends the line) instead of a fixed width — a fixed width here is the
    main reason AI Vision blanks came out too small on whitespace worksheets.
    "before" blanks stay capped at SYNTH_BLANK_WIDTH (they are usually short
    matching-letter blanks where a small box is correct).

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
    # No literal underscore run: synthesize a blank that runs from just past the
    # anchor to the next printed text on the line, or to the right margin if the
    # anchor ends the line. This gives definition/sentence blanks real room.
    ax0, ay0, ax1, ay1 = chars[anchor_ci]["bbox"]
    start_x = ax1 + 3
    right_limit = page_rect.x1 - PAGE_RIGHT_MARGIN
    for c in chars[anchor_ci + 1:]:
        if c["bbox"] and not c["c"].isspace():
            right_limit = min(right_limit, c["bbox"][0] - 4)
            break
    end_x = (right_limit if right_limit - start_x >= 8
             else start_x + SYNTH_BLANK_WIDTH)
    return (start_x, ay0, end_x, ay1), 0


def _open_region_for_anchor(anchor_bbox, page_rect, lines, obstacle_bboxes):
    """
    Answer region beneath an open-response anchor — the same geometry
    detect_open_response_units uses: span from just under the prompt to the
    next content line (or footer), clamped above any table/image obstacle, and
    only if there's at least MIN_ANSWER_SPACE of blank room.
    """
    p_x0, p_top, p_x1, p_bottom = anchor_bbox
    answer_right = page_rect.x1 - PAGE_RIGHT_MARGIN

    # Only real content lines bound the answer area — whitespace-only lines
    # (stray space glyphs) are the blank space we want to write into, exactly
    # as detect_open_response_units filters them.
    below = [l["bbox"][1] for l in lines
             if not line_is_whitespace(l) and l["bbox"][1] > p_bottom + 1]
    next_top = min(below) if below else page_rect.y1 - 60
    for ob in obstacle_bboxes:
        if p_bottom <= ob[1] < next_top and ob[0] < answer_right and ob[2] > p_x0:
            next_top = ob[1]

    if next_top - p_bottom < MIN_ANSWER_SPACE:
        return None
    return (p_x0 + 4, p_bottom + 4, max(p_x1, answer_right), next_top - 6)


def _empty_bullet_regions_below(page_rect, lines, q_bottom, boundary_top):
    """
    Answer regions for empty answer bullets (○/■/•) that sit between a question
    (q_bottom) and the next detected question (boundary_top). Each empty bullet's
    answer is written to the right of the glyph on its own line — the standard
    Google-Docs study-guide layout where an empty bullet IS the answer space and
    there is no printed text to anchor to. Mirrors detect_bullet_answer_units.
    """
    answer_right = page_rect.x1 - PAGE_RIGHT_MARGIN
    span = sorted(
        (l for l in lines
         if q_bottom - 1 < l["bbox"][1] < boundary_top - 1),
        key=lambda l: l["bbox"][1],
    )
    regions = []
    for i, l in enumerate(span):
        if not is_empty_bullet_line(l):
            continue
        x0, y0, x1, y1 = l["bbox"]
        nxt = span[i + 1]["bbox"][1] if i + 1 < len(span) else boundary_top
        regions.append((x1 + 6, y0, answer_right, max(y1, nxt - 2)))
    return regions


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
    try:
        pages = _page_texts(doc)

        if detector is None:
            detector = _default_detector
        raw_items = detector(path, pages) or []

        # Some models number pages 1..N despite the 0-indexed instruction.
        # A batch sitting entirely inside 1..len(doc) is ambiguous: a 5-page
        # packet whose blanks are all on 1-based pages 1-3 (4-5 being an
        # answer key) reads exactly like a 0-based batch. Only max == len(doc)
        # rules 0-based out outright; otherwise the two conventions are told
        # apart by which actually resolves, since reading one page off makes
        # nearly every anchor miss.
        page_vals = [int(it.get("page", 0) or 0) for it in raw_items
                     if isinstance(it, dict)]
        maybe_one_based = (bool(page_vals) and min(page_vals) >= 1
                           and max(page_vals) <= len(doc))

        counter = {"u": 0, "n": 0}
        units: list[Unit] = []

        # Keep items in (page, model order) so reading-order disambiguation is sane.
        indexed = [(i, it) for i, it in enumerate(raw_items) if isinstance(it, dict)]
        indexed.sort(key=lambda t: (t[1].get("page", 0), t[0]))

        def resolve_pass(shift: int):
            """Resolve every anchor to geometry, reading its page as
            (page - shift).

            Each pass builds its own page indexes: resolving consumes
            reading-order and used-span state, so a trial offset cannot share
            them with the pass that ends up being kept.
            """
            idx_cache: dict[int, _PageIndex] = {}
            obs_cache: dict[int, list] = {}

            def page_index(pn: int) -> _PageIndex | None:
                if pn < 0 or pn >= len(doc):
                    return None
                if pn not in idx_cache:
                    lines = lines_in_reading_order(doc[pn])
                    idx_cache[pn] = _PageIndex(doc[pn], lines)
                    obs_cache[pn] = [
                        b["bbox"] for b in doc[pn].get_text("rawdict")["blocks"]
                        if b.get("type") == 1
                    ]
                return idx_cache[pn]

            resolved: list[dict] = []
            drops: list[dict] = []

            def drop(item, reason):
                drops.append({"reason": reason,
                              "anchor_text": item.get("anchor_text", ""),
                              "page": item.get("page"), "kind": item.get("kind")})

            for _, item in indexed:
                kind = "open" if item.get("kind") == "open" else "inline"
                anchor = str(item.get("anchor_text", "")).strip()
                pn = int(item.get("page", 0) or 0) - shift
                if not anchor:
                    drop(item, "empty_anchor")
                    continue
                if kind == "open" and not want_open:
                    continue
                if kind == "inline" and not want_inline:
                    continue
                pidx = page_index(pn)
                if pidx is None:
                    drop(item, "bad_page")
                    continue
                spans = pidx.resolve(anchor)
                if spans is None:
                    drop(item, "anchor_not_found")
                    continue
                abbox = _anchor_bbox(pidx.flat, spans)
                if abbox is None:
                    drop(item, "no_anchor_bbox")
                    continue
                resolved.append({"item": item, "kind": kind, "anchor": anchor,
                                 "pn": pn, "pidx": pidx, "spans": spans,
                                 "abbox": abbox})
            return resolved, drops, page_index, obs_cache

        # Pass 1: resolve every anchor to geometry (disambiguation state lives here).
        resolved, dropped, page_index, obstacle_cache = resolve_pass(0)
        # Only worth trying the other convention when this one lost anchors;
        # a clean 0-based run is already the shape the prompt asks for.
        if maybe_one_based and dropped:
            shifted = resolve_pass(1)
            if len(shifted[0]) > len(resolved):
                print("[multimodal] 1-based page indices detected; shifting "
                      f"to 0-based ({len(shifted[0])} anchors resolve vs "
                      f"{len(resolved)})")
                resolved, dropped, page_index, obstacle_cache = shifted

        for rec in dropped:
            print(f"[multimodal] dropped anchor (reason={rec['reason']}, "
                  f"page={rec['page']}): {rec['anchor_text']!r}")

        def drop(item, reason):
            rec = {"reason": reason, "anchor_text": item.get("anchor_text", ""),
                   "page": item.get("page"), "kind": item.get("kind")}
            dropped.append(rec)
            print(f"[multimodal] dropped anchor (reason={reason}, "
                  f"page={rec['page']}): {rec['anchor_text']!r}")

        # Per-page sorted anchor tops — used to scope a question's empty answer
        # bullets to the vertical band before the NEXT detected question.
        anchor_tops: dict[int, list[float]] = {}
        for r in resolved:
            anchor_tops.setdefault(r["pn"], []).append(r["abbox"][1])
        for pn in anchor_tops:
            anchor_tops[pn].sort()

        def boundary_below(pn: int, y: float) -> float:
            for t in anchor_tops.get(pn, []):
                if t > y + 1:
                    return t
            return page_index(pn).page_rect.y1 - 40

        # Pass 2: build units.
        for r in resolved:
            item, kind, anchor, pn, pidx, abbox = (
                r["item"], r["kind"], r["anchor"], r["pn"], r["pidx"], r["abbox"])
            spans = r["spans"]

            if kind == "open":
                # First try empty answer bullets beneath the question (study-guide
                # layout). Each empty bullet has no text to anchor to, so the model
                # only flags the question — we place the answers on the bullets here.
                bullets = _empty_bullet_regions_below(
                    pidx.page_rect, pidx.lines, abbox[3], boundary_below(pn, abbox[3])
                )
                if bullets:
                    n = len(bullets)
                    for k, reg in enumerate(bullets):
                        ptext = _normalize(anchor)
                        if n > 1:  # multi-bullet question -> distinct points
                            ptext = (f"{ptext} [multi-part answer: give point "
                                     f"{k + 1} of {n}, distinct from the others]")
                        counter["u"] += 1
                        units.append(Unit(
                            unit_id=f"u{counter['u']}", type="open_response",
                            page=pn, bbox=abbox, prompt_text=ptext,
                            answer_region=reg,
                        ))
                    continue
                # Otherwise an open answer area (a vertical gap below the prompt).
                region = _open_region_for_anchor(
                    abbox, pidx.page_rect, pidx.lines, obstacle_cache.get(pn, [])
                )
                if region is None:
                    drop(item, "no_answer_space_below")
                    continue
                counter["u"] += 1
                units.append(Unit(
                    unit_id=f"u{counter['u']}", type="open_response",
                    page=pn, bbox=abbox, prompt_text=_normalize(anchor),
                    answer_region=region,
                ))
                continue

            # inline
            position = item.get("blank_position", "after")
            if position not in ("after", "before"):
                position = "after"
            end_char = (_last_real_char(pidx.flat, spans)
                        if position == "after"
                        else _first_real_char(pidx.flat, spans))
            if end_char is None:
                drop(item, "no_anchor_bbox")
                continue
            line = pidx.lines[end_char["line"]]
            slot_bbox, ulen = _inline_slot_bbox(line, end_char["ci"], position,
                                                pidx.page_rect)

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

        # Multiple-choice questions are located deterministically here, not through
        # the model's blank anchors. Option labels (A/B/C/D…) are reliable in the
        # text layer, and the vision model is told to find BLANKS, not option lists —
        # so an MC worksheet returns almost nothing on this path unless we add it.
        # (This is the same detector the Standard path uses.)
        if "multiple_choice" in active:
            from paperfill.ai.preprocess import detect_multiple_choice_units
            for pn in range(len(doc)):
                pidx = page_index(pn)
                if pidx is None:
                    continue
                mc_units, _ = detect_multiple_choice_units(pidx.lines, pn, counter)
                if not mc_units:
                    continue
                # Drop any model-derived unit that overlaps an MC question's vertical
                # band on this page, so an option line the model mistook for a blank
                # isn't both circled and written into.
                bands = [(u.bbox[1], u.bbox[3]) for u in mc_units]
                def _in_mc_band(u):
                    if u.page != pn:
                        return False
                    cy = (u.bbox[1] + u.bbox[3]) / 2
                    return any(t - 1 <= cy <= b + 1 for t, b in bands)
                kept = []
                for u in units:
                    if _in_mc_band(u):
                        dropped.append({"reason": "overlaps_multiple_choice",
                                        "anchor_text": u.prompt_text,
                                        "page": u.page, "kind": u.type})
                        print(f"[multimodal] dropped unit (reason="
                              f"overlaps_multiple_choice, page={u.page}): "
                              f"{u.prompt_text!r}")
                    else:
                        kept.append(u)
                units = kept
                units.extend(mc_units)
    finally:
        doc.close()

    if dropped:
        print(f"[multimodal] {len(dropped)} anchor(s) dropped of "
              f"{len(indexed)} returned")

    return {
        "source": path,
        "detector": "multimodal",
        "unit_count": len(units),
        # Counted off the surviving units, not the slot counter: a unit dropped
        # for overlapping a choice list took its slot ids with it.
        "slot_count": sum(len(u.slots) for u in units),
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
