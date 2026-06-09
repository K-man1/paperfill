"""
PDF blank-detection preprocessor.

Outputs a structured JSON representation that an LLM can fill in
without ever seeing coordinates. The renderer uses the JSON's bbox
data deterministically.

Unit types produced:
  - inline_blanks: a sentence with one or more underscore runs to fill
  - table: a grid of cells, each potentially containing inline_blanks
  - open_response: a question followed by empty vertical space
"""

import fitz
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------- helpers ---------------------------------------------------------

UNDERSCORE_RE = re.compile(r"_+")

# Common bullet/list markers that should be stripped from prompts but used
# as logical-unit boundaries.
BULLET_CHARS = set("●○■▪◆◇•·")
NUMBERED_LIST_RE = re.compile(r"^\s*(\d+)\s*[\.\)]\s")


def line_is_whitespace(line: dict) -> bool:
    return not line["text"].strip()


def line_starts_new_logical_unit(line: dict, prev_line: dict | None) -> bool:
    """
    A line starts a new logical unit (bullet, numbered question, heading)
    if it begins with a bullet char or a numbered-list marker, OR there's
    a large vertical gap from the previous line.
    """
    txt = line["text"].lstrip()
    if not txt:
        return False
    # Bullet marker
    if txt[0] in BULLET_CHARS:
        return True
    # Numbered list / question marker
    if NUMBERED_LIST_RE.match(txt):
        return True
    # Large vertical gap
    if prev_line is not None:
        gap = line["bbox"][1] - prev_line["bbox"][3]
        if gap > 14:
            return True
    return False


def strip_bullet_prefix(text: str) -> str:
    """Remove leading bullet char + whitespace (including zero-width)."""
    s = text.lstrip()
    while s and (s[0] in BULLET_CHARS or s[0] in "\u200b\u200c\u200d"):
        s = s[1:].lstrip()
    return s


@dataclass
class Slot:
    """A single fill-in target within a unit."""
    slot_id: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    underscore_length: int  # number of underscore chars; rough size hint only


@dataclass
class Unit:
    """A logical chunk of content the LLM reasons about."""
    unit_id: str
    type: str                # "inline_blanks" | "open_response" | "table"
    page: int
    bbox: tuple[float, float, float, float]
    # For inline_blanks: the sentence with {{slot_id}} placeholders.
    # For open_response: just the question text.
    # For table: a structural description.
    prompt_text: str
    slots: list[Slot] = field(default_factory=list)
    # Open response only: bbox of the empty region where the answer goes.
    answer_region: tuple[float, float, float, float] | None = None
    # Table only: rows of cells, each cell is a sub-unit ref.
    table_cells: list[list[dict]] | None = None


def chars_of_span(span) -> list[dict]:
    """Get character list from a rawdict span."""
    return span.get("chars", [])


def find_underscore_runs(chars: list[dict]) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx_inclusive) for each run of '_' chars."""
    runs, run_start = [], None
    for i, c in enumerate(chars):
        if c["c"] == "_":
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                runs.append((run_start, i - 1))
                run_start = None
    if run_start is not None:
        runs.append((run_start, len(chars) - 1))
    return runs


def bbox_of_chars(chars: list[dict], start: int, end: int) -> tuple[float, ...]:
    """Bounding box covering chars[start..end] inclusive."""
    x0 = min(chars[i]["bbox"][0] for i in range(start, end + 1))
    y0 = min(chars[i]["bbox"][1] for i in range(start, end + 1))
    x1 = max(chars[i]["bbox"][2] for i in range(start, end + 1))
    y1 = max(chars[i]["bbox"][3] for i in range(start, end + 1))
    return (x0, y0, x1, y1)


def text_of_chars(chars: list[dict]) -> str:
    return "".join(c["c"] for c in chars)


# ---------- inline blank extraction ----------------------------------------

def group_lines_into_logical_units(lines: list[dict]) -> list[list[dict]]:
    """
    Group consecutive lines into logical units (bullets, paragraphs).
    A new unit starts on a bullet marker, numbered question, or large gap.
    Whitespace-only lines are dropped.
    """
    content_lines = [l for l in lines if not line_is_whitespace(l)]
    if not content_lines:
        return []
    groups: list[list[dict]] = [[content_lines[0]]]
    for prev, curr in zip(content_lines, content_lines[1:]):
        if line_starts_new_logical_unit(curr, prev):
            groups.append([curr])
        else:
            groups[-1].append(curr)
    return groups


def extract_inline_blanks_from_group(group: list[dict], page_num: int,
                                     counter: dict) -> Unit | None:
    """
    Given a group of lines forming one logical unit (a bullet, paragraph),
    if any line contains underscore runs, emit an inline_blanks unit
    covering all of them. The prompt is the joined text with {{slot_id}}
    placeholders; each slot has its precise per-run bbox.
    """
    # Flatten chars across all lines in this group, in reading order.
    all_chars: list[dict] = []
    for line in group:
        for span in line["spans"]:
            all_chars.extend(chars_of_span(span))

    if not all_chars:
        return None

    runs = find_underscore_runs(all_chars)
    if not runs:
        return None

    slots: list[Slot] = []
    prompt_parts: list[str] = []
    cursor = 0
    for run_start, run_end in runs:
        prompt_parts.append(text_of_chars(all_chars[cursor:run_start]))
        counter["n"] += 1
        slot_id = f"s{counter['n']}"
        bbox = bbox_of_chars(all_chars, run_start, run_end)
        slots.append(Slot(
            slot_id=slot_id,
            bbox=bbox,
            underscore_length=run_end - run_start + 1,
        ))
        prompt_parts.append(f"{{{{{slot_id}}}}}")
        cursor = run_end + 1
    prompt_parts.append(text_of_chars(all_chars[cursor:]))

    # Collapse whitespace and strip bullet prefix for a clean LLM prompt.
    prompt = "".join(prompt_parts)
    prompt = strip_bullet_prefix(prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip()

    # Bounding box: union over all lines in the group.
    x0 = min(l["bbox"][0] for l in group)
    y0 = min(l["bbox"][1] for l in group)
    x1 = max(l["bbox"][2] for l in group)
    y1 = max(l["bbox"][3] for l in group)

    counter["u"] += 1
    return Unit(
        unit_id=f"u{counter['u']}",
        type="inline_blanks",
        page=page_num,
        bbox=(x0, y0, x1, y1),
        prompt_text=prompt,
        slots=slots,
    )


# ---------- multi-line unit grouping ---------------------------------------

def group_continuation_lines(lines_data: list[dict]) -> list[list[dict]]:
    """
    Group consecutive lines that belong to the same logical sentence/bullet.
    Heuristic: lines belong together if vertically close (<= ~1.5 line heights)
    AND the next line doesn't start with a new bullet/number marker.
    """
    if not lines_data:
        return []
    groups = [[lines_data[0]]]
    for prev, curr in zip(lines_data, lines_data[1:]):
        prev_y1 = prev["bbox"][3]
        curr_y0 = curr["bbox"][1]
        gap = curr_y0 - prev_y1
        # If the gap is small, treat as continuation.
        # Otherwise start a new group.
        if gap < 8:  # tuned by inspection; lines are ~14pt
            groups[-1].append(curr)
        else:
            groups.append([curr])
    return groups


def lines_in_reading_order(page) -> list[dict]:
    """
    Extract lines from a page in reading order with their bboxes & chars.
    Each line dict has: bbox, chars (flat list across spans), spans.
    """
    raw = page.get_text("rawdict")
    out = []
    for block in raw["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            chars = []
            for span in line["spans"]:
                chars.extend(chars_of_span(span))
            if not chars:
                continue
            x0 = min(c["bbox"][0] for c in chars)
            y0 = min(c["bbox"][1] for c in chars)
            x1 = max(c["bbox"][2] for c in chars)
            y1 = max(c["bbox"][3] for c in chars)
            out.append({
                "bbox": (x0, y0, x1, y1),
                "chars": chars,
                "spans": line["spans"],
                "text": "".join(c["c"] for c in chars),
            })
    # Sort by y, then x (already mostly true but blocks may interleave)
    out.sort(key=lambda l: (round(l["bbox"][1] / 3) * 3, l["bbox"][0]))
    return out


# ---------- open-response detection ----------------------------------------

NUMBERED_Q_RE = re.compile(r"^\s*(\d+)\s*[\.\)]\s")

# Minimum vertical blank space (in points) below a prompt for it to count as
# a writable answer region. ~2 blank text lines; less than this is ordinary
# paragraph spacing.
MIN_ANSWER_SPACE = 25

# Right-edge margin used when sizing an answer region to the page width.
PAGE_RIGHT_MARGIN = 40


def is_question_start(line: dict) -> bool:
    """True if line starts with a question number like '1.' or '2)'."""
    return bool(NUMBERED_Q_RE.match(line["text"]))


def group_prompt_lines(content_lines: list[dict]) -> list[list[dict]]:
    """
    Group consecutive content lines that wrap together into single prompts.
    Lines join when vertically close AND similarly indented; otherwise a new
    prompt begins. (Whitespace-only lines must already be filtered out.)
    """
    if not content_lines:
        return []
    groups: list[list[dict]] = [[content_lines[0]]]
    for prev, curr in zip(content_lines, content_lines[1:]):
        gap = curr["bbox"][1] - prev["bbox"][3]
        indent_diff = abs(curr["bbox"][0] - prev["bbox"][0])
        if gap < 10 and indent_diff < 30:
            groups[-1].append(curr)
        else:
            groups.append([curr])
    return groups


def detect_open_response_units(lines: list[dict], page_num: int,
                               page_rect, counter: dict,
                               obstacle_bboxes: list | None = None) -> list[Unit]:
    """
    Detect prompts (vocabulary terms, headings, or questions — numbered or
    not) that are followed by enough empty vertical space for a written
    answer. Each such prompt becomes an open_response unit whose answer
    region is the blank area beneath it.

    A prompt is skipped when it contains underscores (handled as inline
    blanks) or when the blank space below it is smaller than MIN_ANSWER_SPACE
    (ordinary line spacing, not an answer area). Tables and images count as
    occupied space, so a heading above a chart or a question above a diagram
    is not mistaken for a writable prompt.
    """
    obstacle_bboxes = obstacle_bboxes or []
    content_lines = [l for l in lines if not line_is_whitespace(l)]
    if not content_lines:
        return []

    prompts = group_prompt_lines(content_lines)
    answer_right = page_rect.x1 - PAGE_RIGHT_MARGIN

    units: list[Unit] = []
    for pi, group in enumerate(prompts):
        joined = re.sub(r"\s+", " ", " ".join(l["text"] for l in group)).strip()
        if not joined or "_" in joined:
            continue  # empty, or an inline-blank prompt handled elsewhere

        p_top = min(l["bbox"][1] for l in group)
        p_bottom = max(l["bbox"][3] for l in group)
        p_x0 = min(l["bbox"][0] for l in group)
        p_x1 = max(l["bbox"][2] for l in group)

        # Bottom of the writable area: the next prompt, or the page footer.
        if pi + 1 < len(prompts):
            next_top = min(l["bbox"][1] for l in prompts[pi + 1])
        else:
            next_top = page_rect.y1 - 60

        # A table or image below the prompt occupies the space — clamp to
        # its top so we don't write an answer over it.
        for ob in obstacle_bboxes:
            if p_bottom <= ob[1] < next_top and ob[0] < answer_right and ob[2] > p_x0:
                next_top = ob[1]

        if next_top - p_bottom < MIN_ANSWER_SPACE:
            continue

        answer_region = (
            p_x0 + 4,
            p_bottom + 4,
            max(p_x1, answer_right),
            next_top - 6,
        )

        counter["u"] += 1
        units.append(Unit(
            unit_id=f"u{counter['u']}",
            type="open_response",
            page=page_num,
            bbox=(p_x0, p_top, p_x1, p_bottom),
            prompt_text=joined,
            answer_region=answer_region,
        ))
    return units


# ---------- table extraction -----------------------------------------------

def extract_table_unit(table, page_num: int, counter: dict) -> Unit | None:
    """
    Build a table unit. Each cell may contain multiple underscore-run slots.
    Slot bboxes are populated in a second pass (populate_table_cell_slots).
    The prompt_text built here is a placeholder; the final LLM-facing
    prompt is rebuilt after slot population so it can reference slot IDs.
    """
    extracted = table.extract()
    if not extracted:
        return None

    row_count = table.row_count
    col_count = table.col_count

    cells_structured: list[list[dict]] = []

    for r in range(row_count):
        row_struct = []
        for c in range(col_count):
            # PyMuPDF's table.cells is column-major: cells[c*row_count + r]
            cell_bbox = table.cells[c * row_count + r]
            if cell_bbox is None:
                row_struct.append(None)
                continue
            cell_text = (extracted[r][c] if r < len(extracted) and c < len(extracted[r])
                         else "") or ""
            row_struct.append({
                "row": r,
                "col": c,
                "bbox": tuple(cell_bbox),
                "raw_text": cell_text,
                "slots": [],  # populated by populate_table_cell_slots
            })
        cells_structured.append(row_struct)

    counter["u"] += 1
    return Unit(
        unit_id=f"u{counter['u']}",
        type="table",
        page=page_num,
        bbox=tuple(table.bbox),
        prompt_text="",  # filled in by build_table_prompt
        table_cells=cells_structured,
    )


def build_table_prompt(unit: Unit) -> str:
    """
    Build a readable prompt that lays out each cell by row/column position,
    with {{slot_id}} placeholders inline in the cell text. Uses the
    rebuilt_text from populate_table_cell_slots which has sentinel-wrapped
    slot IDs in the correct positions relative to surrounding text.
    """
    if unit.type != "table" or not unit.table_cells:
        return ""

    def render_cell(cell):
        text = cell.get("rebuilt_text", cell["raw_text"])
        # Replace sentinel-wrapped slot IDs with {{slot_id}}.
        text = re.sub(r"\x00([a-z]\d+)\x00", r"{{\1}}", text)
        return re.sub(r"\s+", " ", text).strip()

    # Determine if row 0 looks like a header (no blanks in any cell of row 0)
    row0 = unit.table_cells[0] if unit.table_cells else []
    row0_has_blanks = any(cell and cell["slots"] for cell in row0)
    headers = None
    if not row0_has_blanks and row0:
        headers = [render_cell(cell) if cell else "" for cell in row0]

    lines = ["TABLE:"]
    start_row = 1 if headers is not None else 0
    for row in unit.table_cells[start_row:]:
        for cell in row:
            if cell is None:
                continue
            cell_prompt = render_cell(cell)
            if not cell_prompt:
                continue
            if headers is not None and cell["col"] < len(headers):
                lines.append(f"  [under \"{headers[cell['col']]}\"] {cell_prompt}")
            else:
                lines.append(f"  [r{cell['row']}c{cell['col']}] {cell_prompt}")
    return "\n".join(lines)


def populate_table_cell_slots(unit: Unit, page, counter: dict) -> None:
    """
    For each cell in a table unit, find underscore-run bboxes by scanning
    the page's char data restricted to the cell bbox. Underscore runs are
    detected per visual line within the cell.

    Also rebuilds cell["raw_text"] from the chars actually inside the cell
    bbox, with underscore runs replaced by sentinel tokens that match the
    slot IDs we just assigned. This guarantees the prompt text built later
    has slot placeholders in the correct positions relative to surrounding
    cell text.
    """
    if unit.type != "table" or not unit.table_cells:
        return

    raw = page.get_text("rawdict")
    line_chunks = []  # (y_center, [chars])
    for block in raw["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            chars = []
            for span in line["spans"]:
                chars.extend(chars_of_span(span))
            if not chars:
                continue
            y_center = sum((c["bbox"][1] + c["bbox"][3]) / 2 for c in chars) / len(chars)
            line_chunks.append((y_center, chars))

    def in_bbox(c, bbox):
        cx = (c["bbox"][0] + c["bbox"][2]) / 2
        cy = (c["bbox"][1] + c["bbox"][3]) / 2
        return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]

    for row in unit.table_cells:
        for cell in row:
            if cell is None:
                continue
            cell_lines = []
            for y, chars in line_chunks:
                cell_line_chars = [c for c in chars if in_bbox(c, cell["bbox"])]
                if cell_line_chars:
                    cell_line_chars.sort(key=lambda c: c["bbox"][0])
                    cell_lines.append((y, cell_line_chars))
            cell_lines.sort(key=lambda x: x[0])

            # Rebuild cell text with slot placeholders inline, in detection order.
            rebuilt_parts: list[str] = []
            for _, line_chars in cell_lines:
                runs = find_underscore_runs(line_chars)
                # Walk this line's chars, emitting text + slot placeholders.
                cursor = 0
                for rstart, rend in runs:
                    rebuilt_parts.append(text_of_chars(line_chars[cursor:rstart]))
                    counter["n"] += 1
                    slot_id = f"s{counter['n']}"
                    bbox = bbox_of_chars(line_chars, rstart, rend)
                    cell["slots"].append({
                        "slot_id": slot_id,
                        "bbox": bbox,
                        "underscore_length": rend - rstart + 1,
                    })
                    rebuilt_parts.append(f"\x00{slot_id}\x00")
                    cursor = rend + 1
                rebuilt_parts.append(text_of_chars(line_chars[cursor:]))
                rebuilt_parts.append(" ")  # line break → space
            # Store the rebuilt text with sentinel-wrapped slot IDs.
            cell["rebuilt_text"] = "".join(rebuilt_parts)


def table_fill_regions(unit: Unit, page, counter: dict) -> list[Unit]:
    """
    Turn a table that has no underscore slots into one open_response region
    per header cell — the empty body of each column is where the student
    writes (a fill-in-the-chart worksheet). Cells with too little empty space
    below their text are skipped.
    """
    if unit.type != "table" or not unit.table_cells:
        return []

    raw = page.get_text("rawdict")
    chars: list[dict] = []
    for block in raw["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                chars.extend(chars_of_span(span))

    def header_bottom(bbox) -> float:
        ys = [c["bbox"][3] for c in chars
              if bbox[0] <= (c["bbox"][0] + c["bbox"][2]) / 2 <= bbox[2]
              and bbox[1] <= (c["bbox"][1] + c["bbox"][3]) / 2 <= bbox[3]]
        return max(ys) if ys else bbox[1]

    regions: list[Unit] = []
    for row in unit.table_cells:
        for cell in row:
            if cell is None:
                continue
            bx = cell["bbox"]
            h_bottom = header_bottom(bx)
            if bx[3] - h_bottom < 30:
                continue  # no room to write beneath the header
            header_text = (cell.get("raw_text") or "").strip()
            region = (bx[0] + 4, h_bottom + 6, bx[2] - 4, bx[3] - 4)
            counter["u"] += 1
            regions.append(Unit(
                unit_id=f"u{counter['u']}",
                type="open_response",
                page=unit.page,
                bbox=(bx[0], bx[1], bx[2], h_bottom),
                prompt_text=header_text or "Fill in this chart cell.",
                answer_region=region,
            ))
    return regions


# ---------- main preprocessor ----------------------------------------------

def page_has_text_layer(page) -> bool:
    """True if the page exposes real characters (digital PDF), False for a scan."""
    raw = page.get_text("rawdict")
    for block in raw["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("chars"):
                    return True
    return False


def preprocess_pdf(path: str, force_vision: bool = False) -> dict:
    """
    Parse a PDF into fillable Units.

    By default the text-layer pipeline handles digital PDFs and only
    image-only pages fall back to the vision model. Set force_vision=True to
    route *every* page through the vision detector — useful for documents
    that have a text layer but a layout the heuristics miss (e.g. study
    guides with bullet questions and no underscore blanks or answer gaps).
    """
    doc = fitz.open(path)
    counter = {"u": 0, "n": 0}  # unit, slot counters
    all_units: list[Unit] = []
    vision_client = None
    vision_failures = 0

    for page_num, page in enumerate(doc):
        # Scanned (image-only) page, or vision forced for the whole document:
        # the text-layer logic below finds nothing usable, so route it through
        # the vision detector instead.
        if force_vision or not page_has_text_layer(page):
            from vision_preprocess import detect_scanned_page, _build_client
            if vision_client is None:
                vision_client = _build_client()
            # One vision call per page hits the network, so any single page can
            # fail transiently (gateway error, timeout, rate limit). Isolate the
            # failure to that page instead of aborting the whole document — the
            # user still gets every page the model did manage to read.
            try:
                all_units.extend(
                    detect_scanned_page(page, page_num, counter, client=vision_client)
                )
            except Exception as e:
                vision_failures += 1
                print(f"[vision] page {page_num}: detection failed, skipping "
                      f"({type(e).__name__}: {str(e)[:200]})")
            continue

        # Find tables first so we can exclude their content from inline detection
        try:
            tables = page.find_tables()
        except Exception:
            tables = None
        table_bboxes = [t.bbox for t in tables.tables] if tables else []

        # Extract table units
        for t in (tables.tables if tables else []):
            tu = extract_table_unit(t, page_num, counter)
            if not tu:
                continue
            populate_table_cell_slots(tu, page, counter)
            has_slots = any(
                cell and cell["slots"]
                for row in tu.table_cells for cell in row
            )
            if has_slots:
                tu.prompt_text = build_table_prompt(tu)
                all_units.append(tu)
            else:
                # No underscores anywhere — treat as a fill-in-the-chart grid.
                all_units.extend(table_fill_regions(tu, page, counter))

        # Extract lines in reading order
        lines = lines_in_reading_order(page)

        # Image blocks also occupy space — collect them so answers aren't
        # written over diagrams/figures.
        image_bboxes = [
            block["bbox"] for block in page.get_text("rawdict")["blocks"]
            if block.get("type") == 1
        ]
        obstacle_bboxes = table_bboxes + image_bboxes

        def in_any_table(line_bbox):
            cx = (line_bbox[0] + line_bbox[2]) / 2
            cy = (line_bbox[1] + line_bbox[3]) / 2
            for tb in table_bboxes:
                if tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3]:
                    return True
            return False

        non_table_lines = [l for l in lines if not in_any_table(l["bbox"])]

        # Detect open-response question units (numbered questions w/ gaps)
        or_units = detect_open_response_units(
            non_table_lines, page_num, page.rect, counter, obstacle_bboxes
        )
        all_units.extend(or_units)

        # Track which line bboxes belong to open-response questions so we
        # don't re-emit them as inline blanks.
        or_covered = set()
        for u in or_units:
            for l in non_table_lines:
                if (l["bbox"][1] >= u.bbox[1] - 1 and
                    l["bbox"][3] <= u.bbox[3] + 1 and
                    l["bbox"][0] >= u.bbox[0] - 1):
                    or_covered.add(id(l))

        # Inline blanks: group remaining lines into logical units (bullets,
        # paragraphs), then emit one unit per group that contains underscores.
        remaining = [l for l in non_table_lines if id(l) not in or_covered]
        groups = group_lines_into_logical_units(remaining)
        for grp in groups:
            unit = extract_inline_blanks_from_group(grp, page_num, counter)
            if unit:
                all_units.append(unit)

    doc.close()

    # Serialize to dict
    return {
        "source": path,
        "unit_count": len(all_units),
        "slot_count": counter["n"],
        "vision_failures": vision_failures,
        "units": [asdict(u) for u in all_units],
    }


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        result = preprocess_pdf(path)
        out_path = path.split("/")[-1].replace(".pdf", ".structure.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"{path}: {result['unit_count']} units, "
              f"{result['slot_count']} slots → {out_path}")
