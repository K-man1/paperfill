"""
A table with a merged cell used to crash preprocess_pdf with an IndexError:
table.cells only lists the cells that exist, so it is shorter than
row_count * col_count. From a driver's ed notes sheet whose tables had
double-ruled borders.
"""

import fitz

from paperfill.ai.preprocess import preprocess_pdf


def test_table_with_a_merged_cell(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    left, right, top = 72, 432, 100
    rows = [top, top + 30, top + 60, top + 90]
    cols = [left, left + 120, left + 240, right]
    for y in rows:
        page.draw_line((left, y), (right, y))
    for x in (left, right):
        page.draw_line((x, rows[0]), (x, rows[-1]))
    # Inner column rules stop below the first row, so row 0 is one cell
    # spanning all three columns.
    for x in cols[1:-1]:
        page.draw_line((x, rows[1]), (x, rows[-1]))
    page.insert_text((left + 6, rows[0] + 20), "Car seats", fontsize=11)
    page.insert_text((cols[0] + 6, rows[1] + 20), "Booster", fontsize=11)
    page.insert_text((cols[1] + 6, rows[1] + 20), "Under ______ years", fontsize=11)
    page.insert_text((cols[2] + 6, rows[2] + 20), "Fits ______", fontsize=11)
    path = tmp_path / "merged.pdf"
    doc.save(path)

    table = fitz.open(path)[0].find_tables().tables[0]
    assert len(table.cells) < table.row_count * table.col_count

    structure = preprocess_pdf(str(path))
    tables = [u for u in structure["units"] if u["type"] == "table"]
    assert len(tables) == 1
    prompt = tables[0]["prompt_text"]
    assert prompt.count("{{s") == 2
    assert "Under {{" in prompt and "Fits {{" in prompt
