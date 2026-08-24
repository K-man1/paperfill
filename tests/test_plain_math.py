"""
Tests for the LaTeX-to-plain-text pass on model answers.

Answers are drawn onto the PDF as literal glyphs, so anything left un-converted
lands on a student's worksheet as backslashes and braces. The cases here are
the shapes a math sheet actually produced: radicals, fractions, exponents,
degree marks — plus the ones that must survive untouched, since this runs over
every answer on every sheet, not just the math ones.
"""

import pytest

from paperfill.utils.plain_math import plain_math


@pytest.mark.parametrize("latex, expected", [
    (r"\frac{5\sqrt{6}}{\sqrt{22}}", "5√6/√22"),
    (r"\frac{3+\sqrt{8}}{2-2\sqrt{8}}", "(3+√8)/(2-2√8)"),
    (r"\sqrt[3]{108}", "∛108"),
    (r"3\sqrt[3]{4}", "3∛4"),
    (r"2\sqrt{45} + 2\sqrt{24} - \sqrt{125}", "2√45 + 2√24 - √125"),
    (r"f(x) = -2x^{2} + 16x - 33", "f(x) = -2x² + 16x - 33"),
    (r"30^\circ", "30°"),
    (r"$8 + 2\sqrt{15}$", "8 + 2√15"),
    (r"\( -\infty, \infty \)", "-∞, ∞"),
    (r"\text{no solution}", "no solution"),
    (r"x = 45 \pm 3", "x = 45 ± 3"),
    (r"H_2O", "H₂O"),
    (r"\overline{AB} \parallel \overline{CD}", "AB ∥ CD"),
])
def test_latex_becomes_plain_text(latex, expected):
    assert plain_math(latex) == expected


def test_no_backslashes_or_braces_survive():
    assert plain_math(r"\frac{9\sqrt{2}}{4}").isprintable()
    assert not set(plain_math(r"\frac{9\sqrt{2}}{4}")) & set("\\{}$")


@pytest.mark.parametrize("text", [
    "The cost is $40",
    "snake_case",
    "Photosynthesis converts light into chemical energy.",
    "C",
    "1/2",
    "",
])
def test_non_latex_is_untouched(text):
    assert plain_math(text) == text


def test_exponent_that_has_no_unicode_form_stays_readable():
    # No superscript glyph for a whole expression, so it must not silently drop.
    assert plain_math("2^{a+b}") == "2^(a+b)"
