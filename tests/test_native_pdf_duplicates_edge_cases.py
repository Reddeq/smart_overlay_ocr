"""Independent real-PDF regressions for conservative overprint removal."""
import base64
from collections import Counter

import pytest

pymupdf = pytest.importorskip("pymupdf")

from smart_overlay.models import Table, Text
from smart_overlay.native_pdf import native_pdf_project, preview_pdf


def _pdf(tmp_path, draw):
    path = tmp_path / "overprints.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=600, height=400)
        draw(page)
        document.save(path)
    return path


def _texts(project):
    return [obj for obj in project.objects if isinstance(obj, Text)]


def _letters(project):
    return Counter("".join(obj.text for obj in _texts(project)).replace(" ", "").replace("\n", ""))


def _diagnostics(project):
    diagnostics = project.raw["native_pdf"]["deduplication"]
    for key in ("removed_glyphs", "removed_lines", "ambiguous_lines"):
        assert key in diagnostics
    return diagnostics


@pytest.mark.parametrize("jitter,copies", [(0, 1), (0.005, 1), (0.1, 2)])
def test_duplicate_coordinate_tolerance_keeps_real_displacement(tmp_path, jitter, copies):
    def draw(page):
        page.insert_text((70, 90), "LABEL", fontsize=12)
        page.insert_text((70 + jitter, 90 + jitter), "LABEL", fontsize=12)

    project = native_pdf_project(_pdf(tmp_path, draw), detect_tables=False)
    assert _letters(project) == Counter("LABEL" * copies)
    diagnostics = _diagnostics(project)
    assert diagnostics["removed_glyphs"] == (5 if copies == 1 else 0)
    assert diagnostics["removed_lines"] == (1 if copies == 1 else 0)


def test_nearby_repeated_labels_are_not_overprints(tmp_path):
    def draw(page):
        for point in ((70, 90), (125, 90), (70, 108)):
            page.insert_text(point, "RELAY", fontsize=12)

    project = native_pdf_project(_pdf(tmp_path, draw), detect_tables=False)
    assert _letters(project) == Counter("RELAY" * 3)
    assert _diagnostics(project)["removed_glyphs"] == 0


@pytest.mark.parametrize("changed", [
    {"color": (1, 0, 0)},
    {"fontname": "cour"},
    {"fontname": "hebo"},
    {"fontsize": 13},

    {"fill_opacity": 0.5},
    {"render_mode": 1},
], ids=["color", "font", "bold-flags", "size", "alpha", "char-flags"])
def test_same_origin_different_style_is_retained_for_review(tmp_path, changed):
    def draw(page):
        page.insert_text((200, 200), "VALUE", fontsize=12)
        options = {"fontsize": 12, **changed}
        page.insert_text((200, 200), "VALUE", **options)

    project = native_pdf_project(_pdf(tmp_path, draw), detect_tables=False)
    assert _letters(project) == Counter("VALUE" * 2)
    diagnostics = _diagnostics(project)
    assert diagnostics["removed_glyphs"] == 0
    assert diagnostics["ambiguous_lines"] > 0
    assert any(obj.needs_review for obj in _texts(project))


def test_different_directions_sharing_an_anchor_are_distinct_labels(tmp_path):
    def draw(page):
        page.insert_text((200, 200), "VALUE", fontsize=12)
        page.insert_text((200, 200), "VALUE", fontsize=12, rotate=90)

    project = native_pdf_project(_pdf(tmp_path, draw), detect_tables=False)
    assert _letters(project) == Counter("VALUE" * 2)
    assert {abs(obj.rotation) for obj in _texts(project)} == {0, 90}
    assert _diagnostics(project)["removed_glyphs"] == 0
    assert _diagnostics(project)["ambiguous_lines"] == 0


@pytest.mark.parametrize("first,second", [("CONTROL A", "CONTROL B"), ("123.40", "123.49")])
def test_conflicting_overprints_keep_both_common_prefixes(tmp_path, first, second):
    def draw(page):
        page.insert_text((70, 90), first, fontsize=12)
        page.insert_text((70, 90), second, fontsize=12)

    project = native_pdf_project(_pdf(tmp_path, draw), detect_tables=False)
    texts = [obj.text for obj in _texts(project)]
    assert any(first in text for text in texts)
    assert any(second in text for text in texts)
    assert _letters(project) == Counter((first + second).replace(" ", ""))
    diagnostics = _diagnostics(project)
    assert diagnostics["removed_glyphs"] == 0
    assert diagnostics["ambiguous_lines"] > 0
    assert any(obj.needs_review for obj in _texts(project))


@pytest.mark.parametrize("parts", [("ABC", "DEF"), ("DEF",)])
def test_identical_partial_overprints_and_segmented_spans_do_not_corrupt_text(tmp_path, parts):
    def draw(page):
        page.insert_text((70, 90), "ABCDEF", fontname="cour", fontsize=12)
        for part in parts:
            prefix = "ABCDEF"[:"ABCDEF".index(part)]
            x = 70 + pymupdf.get_text_length(prefix, fontname="cour", fontsize=12)
            page.insert_text((x, 90), part, fontname="cour", fontsize=12)

    project = native_pdf_project(_pdf(tmp_path, draw), detect_tables=False)
    assert "".join(obj.text for obj in _texts(project)) == "ABCDEF"
    assert _diagnostics(project)["removed_glyphs"] == sum(map(len, parts))


def test_duplicate_wrapped_paragraph_keeps_logical_text_and_numeric_value(tmp_path):
    def draw(page):
        for _ in range(2):
            page.insert_text((70, 90), "Supply voltage", fontsize=12)
            page.insert_text((70, 107), "is 24.50 volts", fontsize=12)

    project = native_pdf_project(_pdf(tmp_path, draw), detect_tables=False, ignore_numbers=True)
    assert [obj.text for obj in _texts(project)] == ["Supply voltage is 24.50 volts"]
    assert _diagnostics(project)["removed_glyphs"] > 0


@pytest.mark.parametrize("ignore_numbers", [False, True])
def test_table_overprints_dedup_without_losing_repeated_cell_values(tmp_path, ignore_numbers):
    def draw(page):
        for x in (50, 150, 250):
            page.draw_line((x, 80), (x, 170))
        for y in (80, 110, 140, 170):
            page.draw_line((50, y), (250, y))
        for x, y, value in ((60, 100, "ITEM"), (160, 100, "VALUE"),
                            (60, 130, "12.50"), (160, 130, "12.50"),
                            (60, 160, "001"), (160, 160, "001")):
            for _ in range(2):
                page.insert_text((x, y), value, fontsize=10)

    project = native_pdf_project(_pdf(tmp_path, draw), ignore_numbers=ignore_numbers)
    tables = [obj for obj in project.objects if isinstance(obj, Table)]
    assert len(tables) == 1
    cells = {(cell.row, cell.col): cell.text for cell in tables[0].cells}
    assert cells == {(0, 0): "ITEM", (0, 1): "VALUE", (1, 0): "12.50",
                     (1, 1): "12.50", (2, 0): "001", (2, 1): "001"}
    assert _diagnostics(project)["removed_glyphs"] == 25


def test_original_repeated_characters_survive_overprint_removal(tmp_path):
    def draw(page):
        for _ in range(2):
            page.insert_text((70, 90), "BOOKKEEPER 1001", fontsize=12)

    project = native_pdf_project(_pdf(tmp_path, draw), detect_tables=False)
    assert [obj.text for obj in _texts(project)] == ["BOOKKEEPER 1001"]


def test_overprint_cleanup_does_not_modify_source_pdf_or_png(tmp_path):
    def draw(page):
        for _ in range(2):
            page.insert_text((70, 90), "UNCHANGED SOURCE", fontsize=12)
        page.draw_line((20, 90), (400, 90))

    path = _pdf(tmp_path, draw)
    original = path.read_bytes()
    preview = preview_pdf(path)
    project = native_pdf_project(path, detect_tables=False)
    assert path.read_bytes() == original
    assert base64.b64decode(project.source.original_b64) == original
    assert project.source.raster_b64 == preview.source.raster_b64
    assert [obj.text for obj in _texts(project)] == ["UNCHANGED SOURCE"]
