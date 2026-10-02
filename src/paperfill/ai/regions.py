"""
Answer-space detection: the model draws the boxes, the page's geometry fixes them.

The previous version of this path had code propose every region a student
could write in and the model pick from a numbered overlay. Its recall was
capped by the proposer's vocabulary (underscore runs, empty strips, gaps in a
text line): ruled write-on lines and anything else outside that list were
never offered, so the model could not pick them, and it also skipped whole
tables it had been offered.

Here the model marks every answer space itself, one call per page, as a
normalized box. Vision models draw these loosely, so each edge is then moved
to the nearest real feature on the page (a drawn rule, a cell border, an
underscore run) within SNAP_TOLERANCE. An edge with nothing nearby keeps the
model's value, which makes a page with no vector features fall back to plain
model coordinates instead of failing.
"""

import base64
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import fitz

from paperfill.data import models
from paperfill.ai.preprocess import (
    ALL_FORMATS,
    Slot,
    Unit,
    bbox_of_chars,
    find_underscore_runs,
    lines_in_reading_order,
)


DETECT_DPI = int(os.environ.get("REGION_DPI", "150"))
# The Hack Club proxy checks its daily budget against max_tokens rather than
# what a call ends up using, so a page batch sent with the model's default
# ceiling gets refused outright. The largest page measured used ~2.7k,
# most of it reasoning.
DETECT_MAX_TOKENS = 8192
PAGE_WORKERS = 8

SNAP_TOLERANCE = 6.0
# Shorter than this is a fraction bar, tick mark or glyph stroke, not a rule.
MIN_RULE_LENGTH = 10.0
RULE_MAX_THICKNESS = 3.0
# Room above a ruled line for writing when the model's box came back thinner.
MIN_LINE_HEIGHT = 12.0
# Printed text narrower than this share of a box is a label or clue number
# inside it, not a question the box swallowed.
TEXT_TRIM_MIN_WIDTH_FRAC = 0.10
# A ruled line, table row or header field. Taller is room for working.
BLANK_MAX_HEIGHT = 36.0
DUPLICATE_IOU = 0.7


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


# --------------------------------------------------------------------------
# Detection call: the model boxes every answer space on one page.
# --------------------------------------------------------------------------

DETECT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["spaces"],
    "properties": {
        "spaces": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["box_2d", "prompt", "axis_range"],
                "properties": {
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "[ymin, xmin, ymax, xmax], 0-1000.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "The question or label this space answers, e.g. "
                            "'3a. 0.269 g = ? kg' or 'Name'. Short is fine."),
                    },
                    "axis_range": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": (
                            "Coordinate grids ONLY: [xmin, xmax, ymin, ymax] "
                            "read off the tick labels. Empty otherwise."),
                    },
                },
            },
        }
    },
}

_DETECT_SYSTEM = (
    "You are shown one page of a school worksheet. Find EVERY place a student "
    "is expected to write an answer: blanks, underscores, ruled write-on "
    "lines, empty table cells, answer boxes, space left under a question for "
    "working, name and date fields, and coordinate grids to graph on.\n"
    "\n"
    "Do not mark printed text, headings, pictures, margins, or a worked "
    "example that is already filled in. Skip crossword and word-search grids.\n"
    "\n"
    "For each space return box_2d as [ymin, xmin, ymax, xmax] normalized to "
    "0-1000, and a short prompt naming the item it answers (the question "
    "number and what is asked, or the label beside the blank). One entry per "
    "separate space: a sentence with three blanks is three entries, a table "
    "is one entry per empty cell. For a coordinate grid, read the axis tick "
    "labels into axis_range as [xmin, xmax, ymin, ymax]; every other entry "
    "leaves axis_range empty.\n"
    "\n"
    "Do NOT solve anything."
)


def _default_detector(png: bytes, client=None, model: str | None = None) -> list:
    """One vision call over one page image. Isolated so tests and the offline
    harness can inject a recorded response instead."""
    if client is None:
        from paperfill.ai.vision_preprocess import _build_client
        client = _build_client()
    model = model or models.get("regions")

    from paperfill.ai.llm_client import call_context
    with call_context("detect"):
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=DETECT_MAX_TOKENS,
            messages=[
                {"role": "system", "content": _DETECT_SYSTEM},
                {"role": "user", "content": [{"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(png).decode()}}]},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "spaces", "schema": DETECT_SCHEMA,
                                "strict": True},
            },
        )
    from paperfill.utils.json_utils import json_from_response

    spaces = json_from_response(resp).get("spaces")
    return spaces if isinstance(spaces, list) else []


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


def _box_to_points(box, page) -> tuple | None:
    """A 0-1000 [ymin, xmin, ymax, xmax] over the rendered page, in the
    unrotated frame the text layer and drawings use. None if malformed."""
    if not (isinstance(box, (list, tuple)) and len(box) == 4
            and all(isinstance(v, (int, float)) for v in box)):
        return None
    y0, x0, y1, x1 = (min(max(float(v), 0.0), 1000.0) for v in box)
    if x1 <= x0 or y1 <= y0:
        return None
    # The image is rendered rotated, so the box lives in page.rect's frame.
    w, h = page.rect.width, page.rect.height
    rect = fitz.Rect(x0 * w / 1000, y0 * h / 1000,
                     x1 * w / 1000, y1 * h / 1000) * page.derotation_matrix
    return (rect.x0, rect.y0, rect.x1, rect.y1)


# --------------------------------------------------------------------------
# Snapping: each model edge moves to the nearest real feature on the page.
# --------------------------------------------------------------------------

def _page_features(page) -> dict:
    """Rules, underscore runs and printed text on a page, in points."""
    horizontal, vertical = [], []   # (coordinate, from, to)

    def add(r):
        if r.height <= RULE_MAX_THICKNESS and r.width >= MIN_RULE_LENGTH:
            horizontal.append(((r.y0 + r.y1) / 2, r.x0, r.x1))
        elif r.width <= RULE_MAX_THICKNESS and r.height >= MIN_RULE_LENGTH:
            vertical.append(((r.x0 + r.x1) / 2, r.y0, r.y1))
        elif r.width >= MIN_RULE_LENGTH and r.height >= MIN_RULE_LENGTH:
            # A cell or box drawn as one rect: each side is a rule.
            horizontal.extend([(r.y0, r.x0, r.x1), (r.y1, r.x0, r.x1)])
            vertical.extend([(r.x0, r.y0, r.y1), (r.x1, r.y0, r.y1)])

    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l":
                p, q = item[1], item[2]
                add(fitz.Rect(min(p.x, q.x), min(p.y, q.y),
                              max(p.x, q.x), max(p.y, q.y)))
            elif item[0] == "re":
                add(item[1])

    underscores, printed = [], []   # underscores: (bbox, line text)
    for line in lines_in_reading_order(page):
        chars = line["chars"]
        for start, end in find_underscore_runs(chars):
            underscores.append((bbox_of_chars(chars, start, end), line["text"]))
        # Split at underscores and word-sized gaps, so "1. 0.005 = ____" gives
        # "1. 0.005 =" and not one box sitting on top of the blank.
        run = []
        for c in chars + [None]:
            glyph = c is not None and c["c"].strip() and c["c"] != "_"
            if glyph and (not run or c["bbox"][0] - run[-1]["bbox"][2] < 6):
                run.append(c)
                continue
            if run:
                printed.append(bbox_of_chars(run, 0, len(run) - 1))
            run = [c] if glyph else []
    return {"horizontal": horizontal, "vertical": vertical,
            "underscores": underscores, "printed": printed}


def _overlap(lo1, hi1, lo2, hi2) -> float:
    return max(0.0, min(hi1, hi2) - max(lo1, lo2))


def _iou(a, b) -> float:
    inter = _overlap(a[0], a[2], b[0], b[2]) * _overlap(a[1], a[3], b[1], b[3])
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _nearest(rules, value, lo, hi, exclude=None):
    """The rule nearest `value` among those spanning most of lo..hi."""
    best = None
    for rule in rules:
        coord, a, b = rule
        if coord == exclude or _overlap(a, b, lo, hi) < 0.5 * (hi - lo):
            continue
        d = abs(coord - value)
        if d <= SNAP_TOLERANCE and (best is None or d < abs(best[0] - value)):
            best = rule
    return best


def _snap(box, features) -> tuple:
    x0, y0, x1, y1 = box
    horizontal, vertical = features["horizontal"], features["vertical"]

    # A line-height box over underscore runs becomes their exact extent. Not
    # split per run: a letter-dash puzzle ("_ _ y _ _") is one word across
    # many runs, and the model already draws "___ + ___ = ___" as three boxes.
    runs = [u for u, _ in features["underscores"]
            if _overlap(u[0], u[2], x0, x1) > 0
            and y0 - 4 <= (u[1] + u[3]) / 2 <= y1 + 4]
    if runs and (y1 - y0) < 2.5 * max(u[3] - u[1] for u in runs):
        # Worksheets often underline a few underscores and let the underline
        # run on past them, so the blank is the runs plus that rule.
        under = [r for r in horizontal
                 if y0 - 4 <= r[0] <= y1 + 4
                 and _overlap(r[1], r[2], x0, x1) > 0.5 * (r[2] - r[1])]
        return (min([u[0] for u in runs] + [r[1] for r in under]),
                min(u[1] for u in runs),
                max([u[2] for u in runs] + [r[2] for r in under]),
                max([u[3] for u in runs] + [r[0] for r in under]))

    bottom = _nearest(horizontal, y1, x0, x1)
    top = _nearest(horizontal, y0, x0, x1, exclude=bottom and bottom[0])
    left = _nearest(vertical, x0, y0, y1)
    right = _nearest(vertical, x1, y0, y1, exclude=left and left[0])
    if bottom:
        y1 = bottom[0]
        if not left and not right:
            # A ruled write-on line: the answer spans the line's own length.
            if abs(bottom[1] - x0) <= SNAP_TOLERANCE * 2:
                x0 = bottom[1]
            if abs(bottom[2] - x1) <= SNAP_TOLERANCE * 2:
                x1 = bottom[2]
    if top:
        y0 = top[0]
    if left:
        x0 = left[0]
    if right:
        x1 = right[0]
    if bottom and not top and y1 - y0 < MIN_LINE_HEIGHT:
        y0 = y1 - MIN_LINE_HEIGHT

    # Printed text the box swallowed (the question above a working box, a
    # prefilled "Uranium -" in a cell): cut it out along whichever side
    # leaves the most room to write.
    for t in sorted(features["printed"], key=lambda t: t[1]):
        if t[2] - t[0] < TEXT_TRIM_MIN_WIDTH_FRAC * (x1 - x0):
            continue
        covered = _overlap(t[0], t[2], x0, x1) * _overlap(t[1], t[3], y0, y1)
        if covered < 0.3 * (t[2] - t[0]) * (t[3] - t[1]):
            continue
        room = [(x1 - x0) * (y1 - t[3]),   # below it
                (x1 - t[2]) * (y1 - y0),   # right of it
                (x1 - x0) * (t[1] - y0),   # above it
                (t[0] - x0) * (y1 - y0)]   # left of it
        side = max(range(4), key=lambda i: room[i])
        if room[side] <= 0:
            continue
        if side == 0:
            y0 = t[3] + 1
        elif side == 1:
            x0 = t[2] + 2
        elif side == 2:
            y1 = t[1] - 1
        else:
            x1 = t[0] - 2
    return (x0, y0, x1, y1)


def region_preprocess_pdf(path: str, formats=None, detector=None) -> dict:
    """Detect answer spaces by having a model box them and snapping the boxes.

    Emits the same structure dict preprocess_pdf does, so the renderer and the
    fill step consume it unchanged. `detector` is the swap seam: a callable
    (page_png) -> list of {box_2d, prompt, axis_range}.
    """
    active = set(formats) & set(ALL_FORMATS) if formats else set()
    if not active:
        active = set(ALL_FORMATS)
    # Graphs aren't one of the pickable answer formats — a coordinate grid is
    # either there to be drawn on or it isn't, so it is never filtered out.
    wanted = {"blank": "inline_blanks" in active,
              "area": "open_response" in active,
              "graph": True}
    if detector is None:
        detector = _default_detector

    doc = fitz.open(path)
    try:
        pngs = [page.get_pixmap(dpi=DETECT_DPI).tobytes("png") for page in doc]
        with ThreadPoolExecutor(max_workers=PAGE_WORKERS) as pool:
            detected = list(pool.map(lambda png: detector(png) or [], pngs))

        counter = {"u": 0, "n": 0}
        units: list[Unit] = []
        dropped: list[dict] = []
        model_boxes: list[dict] = []

        def add_unit(kind, page_no, bbox, prompt, grid=None, axes=None):
            counter["u"] += 1
            unit_id = f"u{counter['u']}"
            if kind == "graph":
                # The fill step may never see the page (only the vision path
                # sends an image), so the grid's visible range has to travel
                # in the prompt — a point outside it is dropped without a word.
                units.append(Unit(
                    unit_id=unit_id, type="graph", page=page_no, bbox=bbox,
                    prompt_text=(f"{prompt} [coordinate grid: x from {axes[0]:g} "
                                 f"to {axes[1]:g}, y from {axes[2]:g} to "
                                 f"{axes[3]:g}; answer with the points to plot]"),
                    graph={**grid, "x_min": axes[0], "x_max": axes[1],
                           "y_min": axes[2], "y_max": axes[3]},
                ))
            elif kind == "blank":
                counter["n"] += 1
                slot_id = f"s{counter['n']}"
                units.append(Unit(
                    unit_id=unit_id, type="inline_blanks", page=page_no,
                    bbox=bbox, prompt_text=f"{prompt} {{{{{slot_id}}}}}",
                    slots=[Slot(slot_id=slot_id, bbox=bbox, underscore_length=0)],
                ))
            else:
                units.append(Unit(
                    unit_id=unit_id, type="open_response", page=page_no,
                    bbox=bbox, prompt_text=prompt, answer_region=bbox,
                ))

        for page_no, (page, spaces) in enumerate(zip(doc, detected)):
            features = _page_features(page)
            graphs = []
            for block in page.get_text("rawdict")["blocks"]:
                if block.get("type") == 1:
                    grid = _grid_geometry(page, block["bbox"])
                    if grid is not None:
                        graphs.append((tuple(block["bbox"]), grid))

            placed: list[tuple] = []
            for space in spaces:
                if not isinstance(space, dict):
                    continue
                model_boxes.append({"page": page_no, **space})
                box = _box_to_points(space.get("box_2d"), page)
                if box is None:
                    dropped.append({"reason": "bad_box", "page": page_no,
                                    "box_2d": space.get("box_2d")})
                    continue
                prompt = str(space.get("prompt", "")).strip()

                graph = next((g for g in graphs if _overlap(g[0][0], g[0][2], box[0], box[2])
                              * _overlap(g[0][1], g[0][3], box[1], box[3])
                              >= 0.5 * (box[2] - box[0]) * (box[3] - box[1])), None)
                if graph is not None:
                    axes = _axis_range(space.get("axis_range"))
                    if axes is None:
                        dropped.append({"reason": "graph_without_axis_range",
                                        "page": page_no, "prompt": prompt})
                        continue
                    add_unit("graph", page_no, graph[0], prompt, graph[1], axes)
                    continue

                bbox = _snap(box, features)
                kind = "blank" if bbox[3] - bbox[1] <= BLANK_MAX_HEIGHT else "area"
                if not wanted[kind] or any(_iou(bbox, p) > DUPLICATE_IOU for p in placed):
                    continue
                placed.append(bbox)
                add_unit(kind, page_no, bbox, prompt)

            # Underscore runs the model never marked. A run with printed text
            # sitting on it is a worked example ("Modelo"), not an open blank.
            if wanted["blank"]:
                for run, line_text in features["underscores"]:
                    if any(_overlap(run[0], run[2], p[0], p[2]) > 0
                           and _overlap(run[1] - 4, run[3], p[1], p[3]) > 0
                           for p in placed):
                        continue
                    if any(_overlap(run[0], run[2], t[0], t[2]) > 0
                           and _overlap(run[1] - 6, run[3], t[1], t[3]) > 0
                           for t in features["printed"]):
                        continue
                    placed.append(run)
                    add_unit("blank", page_no, run, line_text.strip())

        # Multiple choice is located deterministically: option labels (A/B/C…)
        # are reliable in the text layer, and the detector is looking for
        # places to WRITE, so it would not mark an option list on its own.
        if "multiple_choice" in active:
            from paperfill.ai.preprocess import detect_multiple_choice_units

            for page_no, page in enumerate(doc):
                mc_units, _ = detect_multiple_choice_units(
                    lines_in_reading_order(page), page_no, counter)
                # Lettered sub-questions ("a. 3.011 x 10^23 atoms Mg") parse as
                # an option list. The model only boxes places to write, so a
                # box inside the band means it is a write-in, and the circle
                # would land on a question instead of an answer.
                boxed = [(u.bbox[1] + u.bbox[3]) / 2 for u in units if u.page == page_no]
                units.extend([m for m in mc_units if not any(
                    m.bbox[1] - 1 <= mid <= m.bbox[3] + 1 for mid in boxed)])
    finally:
        doc.close()

    units.sort(key=lambda u: (u.page, u.bbox[1], u.bbox[0]))
    return {
        "source": path,
        "detector": "regions",
        "unit_count": len(units),
        "slot_count": counter["n"],
        "dropped_count": len(dropped),
        "dropped": dropped,
        "units": [asdict(u) for u in units],
        # For report bundles: what the model drew before snapping, so a miss
        # can be pinned on the model (never boxed) or on snapping.
        "model_boxes": model_boxes,
    }
