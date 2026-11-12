from PIL import Image, ImageDraw

from smart_overlay.input import source_image
from smart_overlay.models import Box
from smart_overlay.tables import geometric_tables, tables_from_result


def test_opencv_finds_missed_table_and_merged_cells(project):
    tables = geometric_tables(source_image(project), project.fragments, [])
    assert len(tables) == 1
    table = tables[0]
    assert (table.rows, table.columns) == (4, 3)
    assert next(c for c in table.cells if c.row == 0 and c.col == 0).colspan == 3
    assert next(c for c in table.cells if c.row == 2 and c.col == 0).rowspan == 2
    assert len(table.ocr_ids) == 9
    assert table.needs_review
    assert geometric_tables(source_image(project), project.fragments, tables) == []


def test_rectangular_diagram_is_not_a_table():
    image = Image.new("RGB", (300, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 280, 180), outline="black", width=2)
    draw.line((20, 100, 280, 100), fill="black", width=2)
    assert geometric_tables(image, [], []) == []


def test_malformed_html_retains_reviewable_geometry_and_text(project):
    raw = [{"table_res_list": [{"cell_box_list": [[500,360,1120,640]], "pred_html": "broken"}]}]
    tables = tables_from_result(raw, source_image(project), project.fragments)
    assert tables[0].needs_review
    assert tables[0].cells[0].text
