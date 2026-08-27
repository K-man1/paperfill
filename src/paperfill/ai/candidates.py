"""
Geometric candidate answer regions — the "code proposes, model selects" path.

The existing AI Vision detector (multimodal_preprocess.py) asks a model where
the blanks are and gets back a quoted piece of printed text, which a resolver
then has to find again. That address format fails in three separate ways: text
that never reaches the text layer (diagram labels), text that extraction
scrambles (stacked fractions), and text the model splices together to satisfy a
uniqueness rule. All three end as dropped or misplaced answers.

Here the direction is reversed. This module enumerates every place on a page a
student *could* write, numbers them, and hands that list to the model along with
the page. The model only ever names a region we already found and can already
draw into, so there is no address to fail to resolve.

That makes over-proposing cheap and under-proposing the only real failure: a
region never offered is a region the model cannot pick. Everything here is
therefore deliberately generous, and the model is expected to reject most of it.

Pure geometry, no model call, so it is testable offline against real worksheets.
"""

import base64
import os
from dataclasses import dataclass, asdict

import fitz

from paperfill.data import models
from paperfill.ai.preprocess import (
    ALL_FORMATS,
    Slot,
    Unit,
    bbox_of_chars,
    find_underscore_runs,
    line_is_whitespace,
    lines_in_reading_order,
    text_page_rect,
    MIN_ANSWER_SPACE,
    PAGE_RIGHT_MARGIN,
)


# A drawn rect counts as a ruled line when it is thin on its short axis and
# spans a real fraction of the page. Worksheets draw fraction bars as tiny
# horizontal rects (4.9pt wide on a 612pt page) that are otherwise identical to
# a grid rule, and those must not become cell boundaries.
RULE_MAX_THICKNESS = 3.0
RULE_MIN_WIDTH_FRAC = 0.25
RULE_MIN_HEIGHT_FRAC = 0.15

# One visual line can arrive as several rects whose coordinates differ by a
# fraction of a point (a column divider gets split wherever a row rule crosses
# it, and the pieces round differently). Anything closer than this is one line.
GRID_TOLERANCE = 2.0

# A band narrower than this is a rounding artifact between two coincident rules,
# not a column or row anyone writes in.
MIN_BAND = 24.0

# Writing room. Height reuses the open-response threshold the deterministic
# detector already uses, so both paths agree on what counts as usable space.
MIN_REGION_HEIGHT = MIN_ANSWER_SPACE
MIN_REGION_WIDTH = 40.0

PAGE_MARGIN = 40.0


@dataclass
class Candidate:
    region_id: str
    page: int
    kind: str          # "area" | "blank" (underscore run) | "graph" (plot grid)
    bbox: tuple
    label: str         # nearest printed text, so the list reads sensibly
    grid: dict | None = None   # graph only: plotted bounds and origin, in points


def _grid_lines(coords: list[float], tolerance: float = GRID_TOLERANCE) -> list[float]:
    """Collapse near-equal rule coordinates into one value per distinct line.

    Sorting first is what makes this cheap: once the values are in order, every
    member of a cluster is adjacent to another member, so a single pass closes
    each group when the gap to the next value exceeds the tolerance. O(n log n)
    for the sort, O(n) for the pass.
    """
    if not coords:
        return []
    ordered = sorted(coords)
    lines: list[float] = []
    group = [ordered[0]]
    for value in ordered[1:]:
        if value - group[-1] <= tolerance:
            group.append(value)
        else:
            lines.append(sum(group) / len(group))
            group = [value]
    lines.append(sum(group) / len(group))
    return lines


def _page_rules(page) -> tuple[list[float], list[float]]:
    """Distinct vertical (x) and horizontal (y) ruled lines on the page."""
    page_rect = text_page_rect(page)
    width, height = page_rect.width, page_rect.height
    xs: list[float] = []
    ys: list[float] = []
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if (rect.height <= RULE_MAX_THICKNESS
                and rect.width >= width * RULE_MIN_WIDTH_FRAC):
            ys.append(rect.y0)
        elif (rect.width <= RULE_MAX_THICKNESS
                and rect.height >= height * RULE_MIN_HEIGHT_FRAC):
            xs.append(rect.x0)
    return _grid_lines(xs), _grid_lines(ys)


def _bands(cuts: list[float], lo: float, hi: float) -> list[tuple[float, float]]:
    """Turn cut positions into the spans between them, bounded by lo/hi."""
    edges = [lo] + [c for c in cuts if lo < c < hi] + [hi]
    spans = [(a, b) for a, b in zip(edges, edges[1:]) if b - a >= MIN_BAND]
    return spans or [(lo, hi)]


def _occupied_rows(cell, boxes) -> list[tuple[float, float]]:
    """Merged vertical spans of everything sitting inside `cell`.

    Only boxes that actually overlap the cell horizontally count, which is what
    keeps the left column's contents from blocking out the right column's
    writing space on a two-column sheet.
    """
    x0, y0, x1, y1 = cell
    spans = []
    for b in boxes:
        if min(b[2], x1) - max(b[0], x0) <= 1:
            continue
        top, bottom = max(b[1], y0), min(b[3], y1)
        if bottom > top:
            spans.append((top, bottom))
    if not spans:
        return []

    spans.sort()
    merged = [spans[0]]
    for top, bottom in spans[1:]:
        if top <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], bottom))
        else:
            merged.append((top, bottom))
    return merged


def _free_strips(cell, boxes) -> list[tuple[float, float, float, float]]:
    """The empty horizontal bands left in a cell once its content is removed.

    Taking the complement rather than just "the gap below the prompt" is what
    lets an answer live *under* a figure: the item number and the diagram merge
    into one occupied span, and the space beneath them is still offered.
    """
    x0, y0, x1, y1 = cell
    strips = []
    cursor = y0
    for top, bottom in _occupied_rows(cell, boxes):
        if top - cursor >= MIN_REGION_HEIGHT:
            strips.append((x0, cursor, x1, top))
        cursor = max(cursor, bottom)
    if y1 - cursor >= MIN_REGION_HEIGHT:
        strips.append((x0, cursor, x1, y1))
    return [s for s in strips if s[2] - s[0] >= MIN_REGION_WIDTH]


# Grid detection. A gridline is dark down (or across) most of the plot, which
# is what separates it from tick labels, axis arrows and the diagonal strokes of
# an ordinary diagram — so finding a grid at all is also how we decide an image
# IS a graph rather than a picture of a triangle.
GRID_DPI = 150
GRID_DARK = 160          # 8-bit grey below this counts as ink
GRID_LINE_COVERAGE = 0.5  # fraction of the axis a line must span
GRID_MIN_LINES = 6
GRAPH_MIN_SIZE = 100.0    # points; smaller images are icons, not plots


def _grid_geometry(page, bbox) -> dict | None:
    """Locate the plotted grid inside an image, in PDF points.

    Returns the grid's bounds and the origin (the axis lines are drawn heavier
    than the gridlines, so they show up as the darkest row and column), or None
    when the image has no grid in it.
    """
    x0, y0, x1, y1 = bbox
    if x1 - x0 < GRAPH_MIN_SIZE or y1 - y0 < GRAPH_MIN_SIZE:
        return None
    pix = page.get_pixmap(dpi=GRID_DPI, clip=fitz.Rect(*bbox),
                          colorspace=fitz.csGRAY)
    if not pix.width or not pix.height:
        return None

    samples, stride = pix.samples, pix.stride
    dark = [[samples[y * stride + x] < GRID_DARK for x in range(pix.width)]
            for y in range(pix.height)]
    col_ink = [sum(dark[y][x] for y in range(pix.height)) / pix.height
               for x in range(pix.width)]
    row_ink = [sum(row) / pix.width for row in dark]

    cols = [x for x, v in enumerate(col_ink) if v > GRID_LINE_COVERAGE]
    rows = [y for y, v in enumerate(row_ink) if v > GRID_LINE_COVERAGE]
    if len(cols) < GRID_MIN_LINES or len(rows) < GRID_MIN_LINES:
        return None

    sx, sy = (x1 - x0) / pix.width, (y1 - y0) / pix.height
    return {
        "left": x0 + cols[0] * sx,
        "right": x0 + cols[-1] * sx,
        "top": y0 + rows[0] * sy,
        "bottom": y0 + rows[-1] * sy,
        "origin_x": x0 + max(range(pix.width), key=lambda x: col_ink[x]) * sx,
        "origin_y": y0 + max(range(pix.height), key=lambda y: row_ink[y]) * sy,
    }


def _label_for(bbox, lines) -> str:
    """Nearest printed text above-or-left of a region, for a readable list."""
    x0, y0, x1, y1 = bbox
    above = [l for l in lines
             if not line_is_whitespace(l)
             and l["bbox"][3] <= y0 + 2
             and min(l["bbox"][2], x1) - max(l["bbox"][0], x0) > 1]
    if not above:
        return ""
    nearest = max(above, key=lambda l: l["bbox"][3])
    return "".join(c["c"] for c in nearest["chars"]).strip()[:60]


def page_candidates(page, page_no: int, counter: dict) -> list[Candidate]:
    """Every region on one page a student could write into."""
    lines = lines_in_reading_order(page)
    images = [b["bbox"] for b in page.get_text("rawdict")["blocks"]
              if b.get("type") == 1]
    content = [l["bbox"] for l in lines if not line_is_whitespace(l)] + images

    out: list[Candidate] = []

    def emit(kind, bbox, grid=None):
        counter["n"] += 1
        out.append(Candidate(region_id=f"r{counter['n']}", page=page_no,
                             kind=kind, bbox=tuple(round(v, 1) for v in bbox),
                             label=_label_for(bbox, lines), grid=grid))

    # Underscore runs are unambiguous answer spaces wherever they appear, so
    # they are emitted directly rather than going through the grid.
    for line in lines:
        chars = line["chars"]
        for start, end in find_underscore_runs(chars):
            emit("blank", bbox_of_chars(chars, start, end))

    for image in images:
        grid = _grid_geometry(page, image)
        if grid is not None:
            emit("graph", image, grid)

    xs, ys = _page_rules(page)
    page_rect = text_page_rect(page)
    left, right = PAGE_MARGIN, page_rect.width - PAGE_RIGHT_MARGIN
    top, bottom = PAGE_MARGIN, page_rect.height - PAGE_MARGIN
    for cx0, cx1 in _bands(xs, left, right):
        for cy0, cy1 in _bands(ys, top, bottom):
            for strip in _free_strips((cx0, cy0, cx1, cy1), content):
                emit("area", strip)
    return out


def document_candidates(path: str) -> dict:
    """Candidate regions for every page, in the shape the model call wants."""
    doc = fitz.open(path)
    counter = {"n": 0}
    regions: list[Candidate] = []
    for page_no, page in enumerate(doc):
        regions.extend(page_candidates(page, page_no, counter))
    doc.close()
    return {"source": path, "region_count": len(regions),
            "regions": [asdict(r) for r in regions]}


def _annotated_doc(path: str, regions: list[dict]):
    """A copy of the PDF with every candidate region drawn and numbered.

    This is the model input as well as the debugging view. Drawing the ids onto
    the page is the whole point of the design: the model answers by reading a
    number off the picture, so no coordinates or quoted text have to survive a
    round trip in either direction.
    """
    doc = fitz.open(path)
    for region in regions:
        page = doc[region["page"]]
        colour = (0.85, 0.2, 0.2) if region["kind"] == "area" else (0.1, 0.4, 0.9)
        rect = fitz.Rect(*region["bbox"])
        page.draw_rect(rect, color=colour, width=0.9)
        label = fitz.Rect(rect.x0, rect.y0, rect.x0 + 26, rect.y0 + 11)
        page.draw_rect(label, color=colour, fill=colour)
        page.insert_text((rect.x0 + 2, rect.y0 + 8.5), region["region_id"],
                         fontsize=8, color=(1, 1, 1))
    return doc


def annotate(path: str, out_path: str) -> dict:
    """Write the annotated PDF to disk. Verification tool for offline runs."""
    found = document_candidates(path)
    doc = _annotated_doc(path, found["regions"])
    doc.save(out_path)
    doc.close()
    return found


# --------------------------------------------------------------------------
# Selection call: the model picks which candidates are real answer spaces.
# --------------------------------------------------------------------------

SELECTION_DPI = int(os.environ.get("REGION_DPI", "150"))


SELECTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "region_id": {
                        "type": "string",
                        "description": "The id printed in the box, e.g. 'r11'.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "The question or label this region is the answer "
                            "space for, e.g. '7. find x and y' or 'domain'. "
                            "Short is fine; it only has to identify the item."
                        ),
                    },
                    "axis_range": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": (
                            "Graph regions ONLY: [xmin, xmax, ymin, ymax] read "
                            "off the tick labels on the axes, e.g. "
                            "[-10, 10, -10, 10]. Omit for every other region."
                        ),
                    },
                },
                "required": ["region_id", "prompt", "axis_range"],
            },
        }
    },
    "required": ["selections"],
}


_SELECT_SYSTEM = (
    "Each worksheet page image has numbered boxes drawn on it. Every box is a "
    "place a student COULD write. Your job is to say which ones a student "
    "actually SHOULD write in, and what question each one answers.\n"
    "\n"
    "The boxes were placed by geometry, not by understanding, so many of them "
    "are wrong. Reject any box that is a margin, a gap between sections, "
    "whitespace under a heading, empty space on an answer-key or instructions "
    "page, or a stray sliver. Keep a box only if a student writes an answer "
    "there.\n"
    "\n"
    "For each box you keep, return its id exactly as printed and a short "
    "prompt naming the item it belongs to (the question number and what is "
    "being asked, or the label beside the blank). The prompt is what a later "
    "step uses to work out the answer, so it must identify the item, but it "
    "does not have to restate the whole question.\n"
    "\n"
    "A box drawn round a coordinate grid is where a graph gets drawn. Keep it "
    "if the question asks the student to graph something, and read the axis "
    "tick labels to fill in axis_range as [xmin, xmax, ymin, ymax]. Every "
    "other box leaves axis_range empty.\n"
    "\n"
    "Do NOT solve anything. Do not invent ids that are not drawn on the page. "
    "If two boxes cover the same answer space, keep the one that fits it "
    "better and drop the other. A question answered in the space below it gets "
    "the box under it, not the box beside it."
)


def _default_selector(pngs: list[bytes], regions: list[dict],
                      client=None, model: str | None = None) -> list[dict]:
    """One vision call over the annotated pages. Isolated so tests and the
    offline harness can inject a recorded response instead."""
    if client is None:
        from paperfill.ai.vision_preprocess import _build_client
        client = _build_client()
    model = model or models.get("regions")

    listing = "\n".join(
        f"{r['region_id']}: page {r['page']}, {r['kind']}"
        + (f", near {r['label']!r}" if r["label"] else "")
        for r in regions
    )
    content: list[dict] = [{"type": "text", "text": (
        "Pick the boxes that are real answer spaces.\n\n"
        "Boxes drawn on the pages:\n" + listing)}]
    for page_no, png in enumerate(pngs):
        uri = "data:image/png;base64," + base64.b64encode(png).decode()
        content.append({"type": "text", "text": f"--- PAGE {page_no} ---"})
        content.append({"type": "image_url", "image_url": {"url": uri}})

    from paperfill.ai.llm_client import call_context
    with call_context("detect"):
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SELECT_SYSTEM},
                {"role": "user", "content": content},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "selections",
                                "schema": SELECTION_SCHEMA, "strict": True},
            },
        )
    from paperfill.utils.json_utils import json_from_response

    parsed = json_from_response(resp)
    picked = parsed.get("selections")
    return picked if isinstance(picked, list) else []


def _axis_range(value) -> tuple[float, float, float, float] | None:
    """Validate a model-supplied [xmin, xmax, ymin, ymax]. A degenerate or
    inverted range would divide by zero or mirror the plot, so it is refused
    rather than corrected into something that looks plotted but is wrong."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x_min, x_max, y_min, y_max = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if x_max - x_min <= 0 or y_max - y_min <= 0:
        return None
    return x_min, x_max, y_min, y_max


def region_preprocess_pdf(path: str, formats=None, selector=None) -> dict:
    """Detect answer spaces by proposing regions and letting a model pick.

    Emits the same structure dict preprocess_pdf does, so the renderer and the
    fill step consume it unchanged. `selector` is the swap seam: a callable
    (page_pngs, regions) -> list of {region_id, prompt}.

    A selection naming an unknown region is dropped rather than guessed at,
    which is the one way this path can still lose an answer.
    """
    active = set(formats) & set(ALL_FORMATS) if formats else set()
    if not active:
        active = set(ALL_FORMATS)
    # Graphs aren't one of the pickable answer formats — a coordinate grid is
    # either there to be drawn on or it isn't, so it is never filtered out.
    wanted = {"blank": "inline_blanks" in active,
              "area": "open_response" in active,
              "graph": True}

    found = document_candidates(path)
    regions = [r for r in found["regions"] if wanted.get(r["kind"])]
    if not regions:
        return {"source": path, "detector": "regions", "unit_count": 0,
                "slot_count": 0, "region_count": 0, "dropped_count": 0,
                "dropped": [], "units": []}

    doc = _annotated_doc(path, regions)
    pngs = [page.get_pixmap(dpi=SELECTION_DPI).tobytes("png") for page in doc]
    doc.close()

    if selector is None:
        selector = _default_selector
    picked = selector(pngs, regions) or []

    by_id = {r["region_id"]: r for r in regions}
    counter = {"u": 0, "n": 0}
    units: list[Unit] = []
    dropped: list[dict] = []
    seen: set[str] = set()

    for choice in picked:
        if not isinstance(choice, dict):
            continue
        rid = str(choice.get("region_id", "")).strip()
        region = by_id.get(rid)
        if region is None:
            dropped.append({"reason": "unknown_region", "region_id": rid})
            print(f"[regions] model named a region that was never drawn: {rid!r}")
            continue
        if rid in seen:
            continue
        seen.add(rid)

        prompt = str(choice.get("prompt", "")).strip() or region["label"]
        bbox = tuple(region["bbox"])
        counter["u"] += 1
        if region["kind"] == "graph":
            axes = _axis_range(choice.get("axis_range"))
            if axes is None:
                counter["u"] -= 1
                dropped.append({"reason": "graph_without_axis_range",
                                "region_id": rid})
                continue
            # The fill step may never see the page (only the vision path sends
            # an image), so the grid's visible range has to travel in the
            # prompt — a point outside it is dropped without a word.
            units.append(Unit(
                unit_id=f"u{counter['u']}", type="graph", page=region["page"],
                bbox=bbox,
                prompt_text=(f"{prompt} [coordinate grid: x from {axes[0]:g} "
                             f"to {axes[1]:g}, y from {axes[2]:g} to "
                             f"{axes[3]:g}; answer with the points to plot]"),
                graph={**region["grid"], "x_min": axes[0], "x_max": axes[1],
                       "y_min": axes[2], "y_max": axes[3]},
            ))
        elif region["kind"] == "blank":
            counter["n"] += 1
            slot_id = f"s{counter['n']}"
            units.append(Unit(
                unit_id=f"u{counter['u']}", type="inline_blanks",
                page=region["page"], bbox=bbox,
                prompt_text=f"{prompt} {{{{{slot_id}}}}}",
                slots=[Slot(slot_id=slot_id, bbox=bbox, underscore_length=0)],
            ))
        else:
            units.append(Unit(
                unit_id=f"u{counter['u']}", type="open_response",
                page=region["page"], bbox=bbox, prompt_text=prompt,
                answer_region=bbox,
            ))

    # Multiple choice is located deterministically, exactly as the AI Vision
    # path does it: option labels (A/B/C…) are reliable in the text layer, and
    # the selector is looking for places to WRITE, so it would not offer an
    # option list on its own.
    if "multiple_choice" in active:
        from paperfill.ai.preprocess import detect_multiple_choice_units

        doc = fitz.open(path)
        try:
            for page_no, page in enumerate(doc):
                mc_units, _ = detect_multiple_choice_units(
                    lines_in_reading_order(page), page_no, counter)
                if not mc_units:
                    continue
                # A region overlapping an MC question's band would get both
                # circled and written into, so the region loses.
                bands = [(u.bbox[1], u.bbox[3]) for u in mc_units]
                units = [u for u in units if u.page != page_no or not any(
                    top - 1 <= (u.bbox[1] + u.bbox[3]) / 2 <= bottom + 1
                    for top, bottom in bands)]
                units.extend(mc_units)
        finally:
            doc.close()

    units.sort(key=lambda u: (u.page, u.bbox[1], u.bbox[0]))
    return {
        "source": path,
        "detector": "regions",
        "unit_count": len(units),
        "slot_count": counter["n"],
        "region_count": len(regions),
        "dropped_count": len(dropped),
        "dropped": dropped,
        "units": [asdict(u) for u in units],
    }


if __name__ == "__main__":
    import sys

    for pdf in sys.argv[1:]:
        out = pdf.rsplit("/", 1)[-1].replace(".pdf", ".regions.pdf")
        result = annotate(pdf, out)
        print(f"{pdf}: {result['region_count']} regions -> {out}")
