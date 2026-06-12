"""
A/B harness: deterministic (preprocess.py) vs multimodal (multimodal_preprocess.py).

Because the multimodal path's model call is isolated behind a `detector`
callable, we drive it here with a RECORDED vision response — the structured
list a competent vision model would return — so the resolver + geometry +
renderer run end to end without a live API. (The bundled proxy key is expired;
with a valid key, `multimodal_preprocess_pdf(path)` makes the same call live.)

It builds a synthetic worksheet that contains the two cases called out in the
goal:
  * "definition of bob -" — an inline blank that is bare space after a dash,
    which the deterministic underscore/gap heuristics MISS.
  * "single-eyed" — a hyphenated word on a line with empty space below it,
    which the deterministic open-response heuristic turns into a FALSE blank.

It then scores each pipeline against the worksheet's known answer spaces and
renders both filled PDFs to prove the multimodal Units render correctly.
"""

import os
import fitz

from preprocess import preprocess_pdf, lines_in_reading_order
from multimodal_preprocess import multimodal_preprocess_pdf
from render import build_overlays_from_structure, render_overlays_pdf

OUT = "ab_samples"


# --------------------------------------------------------------------------
# Synthetic worksheet + its ground truth and recorded vision response.
# --------------------------------------------------------------------------

# (text, baseline-y, expected number of real answer spaces on that line)
SYNTH_LINES = [
    ("Cell Biology  Review Worksheet", 50, 0),
    ("Name: ______________     Date: ______________", 80, 2),
    ("1. The powerhouse of the cell is the ________________.", 110, 1),
    ("2. definition of bob - ", 140, 1),       # bare-space blank after the dash
    ("single-eyed", 170, 0),                    # compound word, NOT a blank
    ("3. Explain why the sky appears blue:", 215, 1),
    ("4. The capital of France is ______________.", 265, 1),
]


def make_synth_worksheet(path: str) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for text, y, _ in SYNTH_LINES:
        page.insert_text((72, y), text, fontsize=11, fontname="helv")
    doc.save(path)
    doc.close()


def synth_recorded_detector(path, pages):
    """What a good vision model returns for the synthetic worksheet: every real
    answer space, transcribed as anchor_text — and crucially NOT 'single-eyed'."""
    return [
        {"page": 0, "kind": "inline", "anchor_text": "Name:", "blank_position": "after"},
        {"page": 0, "kind": "inline", "anchor_text": "Date:", "blank_position": "after"},
        {"page": 0, "kind": "inline",
         "anchor_text": "The powerhouse of the cell is the", "blank_position": "after"},
        {"page": 0, "kind": "inline",
         "anchor_text": "definition of bob -", "blank_position": "after"},
        {"page": 0, "kind": "open",
         "anchor_text": "Explain why the sky appears blue:", "blank_position": "none"},
        {"page": 0, "kind": "inline",
         "anchor_text": "The capital of France is", "blank_position": "after"},
    ]


# --------------------------------------------------------------------------
# Scoring: map each detected answer space to a worksheet line by vertical
# position, then compare per-line detected counts to the expected counts.
# --------------------------------------------------------------------------

def _answer_space_ys(structure: dict) -> list[float]:
    """Y-centre of every answer space a structure encodes (one per inline slot,
    one per open_response / table cell slot)."""
    ys: list[float] = []
    for u in structure["units"]:
        if u["type"] == "inline_blanks":
            for s in u["slots"]:
                b = s["bbox"]
                ys.append((b[1] + b[3]) / 2)
        elif u["type"] == "open_response":
            b = u["bbox"]  # prompt bbox sits on the question line
            ys.append((b[1] + b[3]) / 2)
        elif u["type"] == "table":
            for row in u["table_cells"]:
                for cell in row or []:
                    for s in (cell or {}).get("slots", []):
                        b = s["bbox"]
                        ys.append((b[1] + b[3]) / 2)
    return ys


def score(pdf_path: str, structure: dict) -> dict:
    """Classify a structure's answer spaces against SYNTH ground truth."""
    doc = fitz.open(pdf_path)
    lines = [l for l in lines_in_reading_order(doc[0]) if l["text"].strip()]
    doc.close()

    # Build (y0, y1, expected, label) bands from the real char map, matching
    # each printed line back to its SYNTH_LINES expectation by text prefix.
    bands = []
    for l in lines:
        txt = l["text"].strip()
        expected = 0
        label = txt[:24]
        for s_txt, _, exp in SYNTH_LINES:
            key = s_txt.strip()[:10]
            if key and txt.startswith(key[:8]):
                expected = exp
                break
        bands.append([l["bbox"][1] - 4, l["bbox"][3] + 4, expected, label, 0])

    unmapped = 0
    for y in _answer_space_ys(structure):
        for band in bands:
            if band[0] <= y <= band[1]:
                band[4] += 1
                break
        else:
            unmapped += 1

    correct = miss = false_pos = 0
    detail = []
    for y0, y1, expected, label, detected in bands:
        c = min(expected, detected)
        m = max(0, expected - detected)
        fp = max(0, detected - expected)
        correct += c
        miss += m
        false_pos += fp
        if expected or detected:
            detail.append((label, expected, detected, c, m, fp))
    false_pos += unmapped

    return {"correct": correct, "miss": miss, "false_pos": false_pos,
            "detail": detail}


def render_check(pdf_path: str, structure: dict, out_path: str) -> int:
    """Fill every answer space with a placeholder and render; returns overlay
    count. Proves the Units feed the renderer cleanly."""
    answers = {}
    for u in structure["units"]:
        if u["type"] == "inline_blanks":
            for s in u["slots"]:
                answers[s["slot_id"]] = "ans"
        elif u["type"] == "open_response":
            answers[u["unit_id"]] = "A sample written answer for this prompt."
    overlays = build_overlays_from_structure(structure, answers)
    render_overlays_pdf(pdf_path, overlays, out_path)
    return len(overlays)


def _print_report(name, s):
    print(f"\n[{name}]  correct={s['correct']}  miss={s['miss']}  "
          f"false_positives={s['false_pos']}")
    for label, exp, det, c, m, fp in s["detail"]:
        flag = ""
        if m:
            flag = "  <-- MISS"
        elif fp:
            flag = "  <-- FALSE POSITIVE"
        print(f"    {label:<26} expected={exp} detected={det}{flag}")


def main():
    os.makedirs(OUT, exist_ok=True)
    synth = os.path.join(OUT, "synthetic_worksheet.pdf")
    make_synth_worksheet(synth)

    det_struct = preprocess_pdf(synth)
    mm_struct = multimodal_preprocess_pdf(synth, detector=synth_recorded_detector)

    det_score = score(synth, det_struct)
    mm_score = score(synth, mm_struct)

    print("=" * 68)
    print("SYNTHETIC WORKSHEET A/B  (ground-truth answer spaces = 6)")
    print("=" * 68)
    _print_report("deterministic (preprocess.py)", det_score)
    _print_report("multimodal (anchor-bridge)", mm_score)

    n1 = render_check(synth, det_struct, os.path.join(OUT, "synth_deterministic_filled.pdf"))
    n2 = render_check(synth, mm_struct, os.path.join(OUT, "synth_multimodal_filled.pdf"))
    print(f"\nRendered: deterministic {n1} overlays, multimodal {n2} overlays "
          f"-> {OUT}/synth_*_filled.pdf")
    print(f"Multimodal dropped/logged anchors: {mm_struct['dropped_count']}")

    # ---- Real worksheet: prove the resolver works on a real char map too ----
    real = "uploads/lOMo8tIVqXsnQpTq.pdf"
    if os.path.exists(real):
        def real_detector(path, pages):
            structs = ["Epididymis", "Vas Deferens", "Urethra", "Testes",
                       "Prostate Gland", "Seminal Vesicle"]
            items = [{"page": 0, "kind": "inline", "anchor_text": s,
                      "blank_position": "before"} for s in structs]
            items += [
                {"page": 0, "kind": "open",
                 "anchor_text": "What is the difference between sperm cells and semen?",
                 "blank_position": "none"},
                {"page": 0, "kind": "open",
                 "anchor_text": "Approximately how many sperm cells can the human body produce per second?",
                 "blank_position": "none"},
            ]
            return items

        rmm = multimodal_preprocess_pdf(real, detector=real_detector)
        rdet = preprocess_pdf(real)
        print("\n" + "=" * 68)
        print("REAL WORKSHEET (uploads/lOMo8tIVqXsnQpTq.pdf)")
        print("=" * 68)
        print(f"  deterministic: {rdet['unit_count']} units, {rdet['slot_count']} slots")
        print(f"  multimodal   : {rmm['unit_count']} units, {rmm['slot_count']} slots, "
              f"{rmm['dropped_count']} dropped")
        n3 = render_check(real, rmm, os.path.join(OUT, "real_multimodal_filled.pdf"))
        print(f"  rendered multimodal -> {OUT}/real_multimodal_filled.pdf ({n3} overlays)")


if __name__ == "__main__":
    main()
