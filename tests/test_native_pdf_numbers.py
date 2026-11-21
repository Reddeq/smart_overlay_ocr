import pytest
import pymupdf
from smart_overlay.models import Text, Table
from smart_overlay.native_pdf import native_pdf_project


@pytest.mark.parametrize('label', ['1200', '−25.5', '±0,2', '10×20', '⌀50', 'Ø50', '50%', '10–20', '1/2', '(25)', '20°',
                                 '2^3', '10≠20', '3·4', '5∓0.2', '１２３', '١٢٣', '½'])
def test_numeric_labels(label):
    from smart_overlay.native_numbers import is_numeric_label
    assert is_numeric_label(label)


@pytest.mark.parametrize('label', ['', '±', 'DN50', 'M12', '10 мм', 'ГОСТ', '1. Install', '1.\nInstall', '50 apples', '1) item', '2\nprose', 'Ⅳ', '1.'])
def test_non_numeric_labels_are_never_partially_deleted(label):
    from smart_overlay.native_numbers import is_numeric_label
    assert not is_numeric_label(label)


def test_ignore_numbers_only_omits_standalone_text_not_table_cells(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=400)
    page.insert_text((40, 50), '1200', fontsize=12)
    page.insert_text((300, 50), 'DN50', fontsize=12)
    for x in (50, 150, 250):
        page.draw_line((x, 100), (x, 180))
    for y in (100, 140, 180):
        page.draw_line((50, y), (250, y))
    for x, y, text in [(60, 125, '123'), (160, 125, '456'), (60, 165, '789')]:
        page.insert_text((x, y), text, fontsize=10)
    path = tmp_path / 'numbers.pdf'
    doc.save(path)
    doc.close()
    default = native_pdf_project(path)
    project = native_pdf_project(path, ignore_numbers=True)
    assert any(isinstance(o, Text) and o.text == '1200' for o in default.objects)
    assert [o.text for o in project.objects if isinstance(o, Text)] == ['DN50']
    assert {c.text for o in project.objects if isinstance(o, Table) for c in o.cells} >= {'123', '456', '789'}
    ignored = next(f for f in project.fragments if f.text == '1200')
    assert 'numeric' in project.ignored[ignored.id]
    assert project.raw['native_pdf']['ignore_numbers'] is True


def test_numeric_filter_runs_after_mixed_label_grouping(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=400)
    page.insert_text((40, 60), '1200', fontsize=12)
    page.insert_text((40, 77), 'Install here', fontsize=12)
    path = tmp_path / 'mixed-number.pdf'
    doc.save(path)
    doc.close()
    project = native_pdf_project(path, detect_tables=False, ignore_numbers=True)
    assert [o.text for o in project.objects if isinstance(o, Text)] == ['1200 Install here']
    assert not project.ignored


def test_all_numeric_page_is_valid_and_original_raster_survives(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((40, 60), '1200', fontsize=12)
    path = tmp_path / 'only-number.pdf'
    doc.save(path)
    doc.close()
    default = native_pdf_project(path, detect_tables=False)
    project = native_pdf_project(path, detect_tables=False, ignore_numbers=True)
    assert not project.objects
    assert len(project.fragments) == len(project.ignored) == 1
    assert project.source.raster_b64 == default.source.raster_b64

