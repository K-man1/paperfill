"""
Tests for telling a scanned page apart from a digital page with a full-page
image behind it.

From an Arabic chapter test (report wTAfd2qXFSp97OCo): a transparent PNG border
covered the whole page, so both pages were sent to the OCR path, which knows
nothing about multiple choice and ignores the formats the user picked. All 20
questions were lost; the only output was two dashes in the date line.
"""

import fitz
import pytest

from paperfill.ai.preprocess import page_is_scanned


def page_with_cover(text_mode):
    src = fitz.open()
    sheet = src.new_page(width=595, height=842)
    sheet.insert_text((72, 100), "1. Water boils at ______ degrees", fontsize=14)
    raster = sheet.get_pixmap(dpi=72).tobytes("png")

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    line = ((72, 100), "1. Water boils at ______ degrees")
    if text_mode == "under":
        page.insert_text(*line, fontsize=14)
    page.insert_image(page.rect, stream=raster)
    if text_mode == "invisible":
        page.insert_text(*line, fontsize=14, render_mode=3)
    if text_mode == "transparent":
        page.insert_text(*line, fontsize=14, fill_opacity=0)
    if text_mode == "over":
        page.insert_text(*line, fontsize=14)
    return fitz.open("pdf", doc.tobytes())[0]


@pytest.mark.parametrize("text_mode", [None, "invisible", "transparent", "under"])
def test_full_page_image_without_visible_text_on_top_is_a_scan(text_mode):
    assert page_is_scanned(page_with_cover(text_mode))


def test_visible_text_painted_over_a_background_is_not_a_scan():
    assert not page_is_scanned(page_with_cover("over"))


def test_small_image_is_not_a_scan():
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10), False)
    page.insert_image(fitz.Rect(72, 72, 172, 172), pixmap=pix)
    assert not page_is_scanned(page)
