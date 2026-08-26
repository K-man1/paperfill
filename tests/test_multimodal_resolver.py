"""
Tests for the AI Vision anchor resolver.

Both shapes here came off the Precalculus packet report (QetIi2YvUCUUs7ab),
where 16 of 35 anchors were dropped and 3 more were stamped across a section
heading. A one-character anchor used to match the first such letter anywhere on
the page, and an anchor the model transcribed as a human reads it never matched
at all, because reading order tears stacked math away from its item number.

Fixtures stay ASCII: pages are built with the base-14 font, which has no glyph
for the radicals these sheets are full of.
"""

import fitz
import pytest

from paperfill.ai.multimodal_preprocess import _PageIndex, _anchor_bbox
from paperfill.ai.preprocess import lines_in_reading_order


def page_index(spans):
    """A one-page index built from (text, x, y) spans, in the order given."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for text, x, y in spans:
        page.insert_text((x, y), text, fontsize=11)
    # Round-trip through bytes so the char map matches a real parsed document.
    reopened = fitz.open("pdf", doc.tobytes())[0]
    return _PageIndex(reopened, lines_in_reading_order(reopened))


@pytest.mark.parametrize("anchor", ["a", "b", "y", "x"])
def test_single_character_anchor_is_refused(anchor):
    """'a' must not land inside "Special" — the letters it names are diagram
    labels that live in the image, not the text layer, so the only correct
    outcome is a drop."""
    idx = page_index([
        ("Special Right Triangles (45-45-90 and 30-60-90)", 76, 44),
        ("Find the missing side lengths.  Answers should be in simplest form.",
         76, 59),
        ("Simplifying", 76, 426),
    ])
    assert idx.resolve(anchor) is None


def test_anchor_prefers_the_standalone_word_over_an_embedded_one():
    idx = page_index([
        ("Simplifying radicals", 76, 44),
        ("Write each radical in simplest form.", 76, 59),
    ])
    spans = idx.resolve("radical")
    assert spans is not None
    assert idx.flat[spans[0][0]]["line"] == 1, "matched inside 'radicals'"


def test_stacked_fraction_anchor_resolves_out_of_reading_order():
    """A stacked fraction extracts numerator-line, denominator-line, so "3." and
    its parts sort apart and interleave with the next item. The anchor still has
    to land on item 3, not item 4."""
    idx = page_index([
        ("18", 110, 380),           # numerator sorts ahead of its item number
        ("4. (3 + 6)(3 - 6)", 318, 384),
        ("3.", 82, 392),
        ("200", 110, 396),
    ])
    spans = idx.resolve("3.   \n18\n200")
    assert spans is not None
    assert _anchor_bbox(idx.flat, spans)[2] < 318, "bled into the right column"


def test_item_number_is_shared_between_that_item_s_blanks():
    """"12. domain:" and "12. range:" both anchor through the same item number,
    and 13's pair must then take the second domain/range, not the first."""
    idx = page_index([
        ("12. f(x) = 2x2 - 4x + 1", 82, 120),
        ("domain: ______", 82, 300),
        ("range: ______", 82, 330),
        ("13. f(x) = -(x - 1)2 - 2", 82, 480),
        ("domain: ______", 82, 660),
        ("range: ______", 82, 690),
    ])
    first = idx.resolve("12. domain:")
    second = idx.resolve("13. domain:")
    assert first is not None and second is not None
    assert _anchor_bbox(idx.flat, first)[1] < _anchor_bbox(idx.flat, second)[1]


def test_cluster_without_its_item_number_is_refused():
    """Loose numerals that happen to sit near each other are not item 24."""
    idx = page_index([
        ("Operations with Fractions", 76, 640),
        ("1 6 + 5 3 - 2 9", 82, 700),
    ])
    assert idx.resolve("24.   \n1\n6 + \n5\n3 - \n2\n9") is None
