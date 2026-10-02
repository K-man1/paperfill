"""
Tests for the AI Vision detector's snapping step.

The shapes come off the Honors Chemistry packet (dxjqn8nVqUda8A4K) and the
Spanish 5A workbook (7QOuEwqRm3UdqHSi) reports, where the old propose-and-pick
path skipped ruled write-on lines and whole tables. The detector is injected, so
the boxes below stand in for what gemini-3.7-flash drew on those pages: roughly
right, and loose by a few points in the ways it was actually loose.
"""

import fitz

from paperfill.ai.regions import region_preprocess_pdf

W, H = 612, 792


def to_box_2d(x0, y0, x1, y1):
    """Points to the model's 0-1000 [ymin, xmin, ymax, xmax]."""
    return [round(y0 / H * 1000), round(x0 / W * 1000),
            round(y1 / H * 1000), round(x1 / W * 1000)]


def detect(pdf_path, spaces):
    return region_preprocess_pdf(
        pdf_path, detector=lambda png: [
            {"box_2d": to_box_2d(*box), "prompt": prompt, "axis_range": []}
            for box, prompt in spaces])


def unit_boxes(structure):
    return [(u["type"], tuple(round(v) for v in u["bbox"]), u["prompt_text"])
            for u in structure["units"]]


def make_pdf(tmp_path, draw):
    doc = fitz.open()
    page = doc.new_page(width=W, height=H)
    draw(page)
    path = tmp_path / "sheet.pdf"
    doc.save(path)
    return str(path)


def test_box_straddling_a_ruled_line_sits_on_it(tmp_path):
    """The model drew each write-on line as a box with the line 2-3pt above
    its bottom edge. Unsnapped, the answer gets struck through."""
    def draw(page):
        page.insert_text((110, 396), "1.", fontsize=11)
        page.draw_line((129, 396), (567, 396), width=0.5)
    pdf = make_pdf(tmp_path, draw)

    kind, box, _ = unit_boxes(detect(pdf, [((130, 386, 566, 399), "1.")]))[0]
    assert kind == "inline_blanks"
    assert box == (129, 384, 567, 396)


def test_table_cell_edges_snap_to_its_borders(tmp_path):
    def draw(page):
        for x in (100, 250, 400):
            for y in (300, 330):
                page.draw_rect(fitz.Rect(x, y, x + 150, y + 30), width=0.5)
        page.insert_text((105, 320), "Mass", fontsize=11)
    pdf = make_pdf(tmp_path, draw)

    units = unit_boxes(detect(pdf, [((253, 302, 398, 328), "Mass unit"),
                                    ((402, 331, 547, 357), "Row 2 abbrev")]))
    assert [box for _, box, _ in units] == [(250, 300, 400, 330),
                                            (400, 330, 550, 360)]


def test_underscore_blank_takes_the_runs_exact_extent(tmp_path):
    def draw(page):
        page.insert_text((80, 150), "1.  0.005 = ________", fontsize=11)
    pdf = make_pdf(tmp_path, draw)
    run = fitz.open(pdf)[0].search_for("________")[0]

    _, box, prompt = unit_boxes(detect(pdf, [((150, 138, 200, 156), "1. 0.005")]))[0]
    assert box[0] == round(run.x0) and box[2] == round(run.x1)
    assert prompt.endswith("{{s1}}")


def test_letter_dash_word_stays_one_slot(tmp_path):
    """Each dash is its own underscore run, but the answer is one word."""
    def draw(page):
        page.insert_text((80, 200), "Llego un __ __ u __ __ __ n a la costa",
                         fontsize=11)
    pdf = make_pdf(tmp_path, draw)
    dashes = fitz.open(pdf)[0].search_for("__")

    units = detect(pdf, [((120, 188, 230, 204), "Llego un ...")])["units"]
    assert len(units) == 1
    assert round(units[0]["bbox"][0]) == round(dashes[0].x0)
    assert round(units[0]["bbox"][2]) == round(dashes[-1].x1)


def test_box_over_a_question_is_trimmed_below_it(tmp_path):
    """A working box drawn from the question's top line down."""
    def draw(page):
        page.insert_text((70, 150), "3a. Convert 0.269 g to kilograms.", fontsize=11)
    pdf = make_pdf(tmp_path, draw)

    kind, box, _ = unit_boxes(detect(pdf, [((70, 138, 300, 220), "3a")]))[0]
    assert kind == "open_response"
    assert box[1] > 150 and box[3] == 220


def test_underscore_the_model_missed_is_added(tmp_path):
    def draw(page):
        page.insert_text((80, 150), "Name ____________", fontsize=11)
        page.insert_text((80, 300), "2. 5,050 = ________", fontsize=11)
    pdf = make_pdf(tmp_path, draw)

    units = unit_boxes(detect(pdf, [((110, 138, 190, 156), "Name")]))
    assert len(units) == 2
    assert units[1][2] == "2. 5,050 = ________ {{s2}}"


def test_worked_example_underscore_is_not_added(tmp_path):
    """The Modelo line: an underscore run with the example answer printed on
    top of it."""
    def draw(page):
        page.insert_text((80, 150), "Modelo  Juanito tiene ______ sed.", fontsize=11)
        page.insert_text((190, 146), "I", fontsize=13)
    pdf = make_pdf(tmp_path, draw)

    assert detect(pdf, [])["units"] == []


def test_malformed_box_is_dropped_not_guessed(tmp_path):
    pdf = make_pdf(tmp_path, lambda page: None)
    structure = region_preprocess_pdf(pdf, detector=lambda png: [
        {"box_2d": [10, 20], "prompt": "x", "axis_range": []},
        {"box_2d": [500, 500, 400, 600], "prompt": "inverted", "axis_range": []},
    ])
    assert structure["units"] == []
    assert [d["reason"] for d in structure["dropped"]] == ["bad_box", "bad_box"]


def test_lettered_subquestions_with_boxes_are_not_multiple_choice(tmp_path):
    """Chem packet Q12: "a. 3.011 X 10^23 atoms Mg / b. 65 g of O2" parsed as
    an option list and got circled over the model's write-in boxes."""
    def draw(page):
        page.insert_text((70, 100), "12. How many moles are in the following?", fontsize=11)
        page.insert_text((90, 118), "a. 3.011 X 10^23 atoms Mg", fontsize=11)
        page.insert_text((90, 134), "b. 65 g of O2", fontsize=11)
        page.insert_text((70, 300), "5. Which is a noble gas?", fontsize=11)
        page.insert_text((90, 318), "a. Neon", fontsize=11)
        page.insert_text((90, 334), "b. Sodium", fontsize=11)
    pdf = make_pdf(tmp_path, draw)

    structure = detect(pdf, [((240, 108, 450, 121), "12a"),
                             ((170, 124, 450, 137), "12b")])
    kinds = sorted((u["type"], round(u["bbox"][1])) for u in structure["units"])
    assert [k for k, _ in kinds].count("inline_blanks") == 2
    assert [k for k, y in kinds if k == "multiple_choice"] == ["multiple_choice"]
    assert all(y > 200 for k, y in kinds if k == "multiple_choice")
