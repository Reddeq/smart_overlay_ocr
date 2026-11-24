"""Wired CAD labels are individual shapes; ordinary text and cells keep their layout."""
from collections import Counter
from types import SimpleNamespace
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pymupdf
import pytest

from smart_overlay.models import Table, Text
from smart_overlay.native_pdf import native_pdf_project
from smart_overlay.settings import Settings, load_settings


@pytest.fixture
def drawing(tmp_path):
    path = tmp_path / 'row-labels.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page(width=580, height=380)
        # A one-column grid, like the terminal strip in the reported drawing.
        for x in (130, 225):
            page.draw_line((x, 90), (x, 162))
        for y in (90, 108, 126, 144, 162):
            page.draw_line((130, y), (225, y))
        for i, y in enumerate((103, 121, 139, 157)):
            page.insert_text((68, y), 'Wire B' if i == 2 else 'Wire A', fontsize=12)
            page.draw_line((108, y-4), (130, y-4))
            page.insert_text((155, y), str(80+i), fontsize=12)
            # A real four-line note on the other side of the same grid; aligned
            # with its rows, but without connectors from the text to the grid.
            page.insert_text((235, y), ('Power supply', 'must remain', 'connected', 'during service')[i], fontsize=12)
        # A separately recoverable two-column table with multiline cell text.
        for x in (300, 395, 510):
            page.draw_line((x, 205), (x, 290))
        for y in (205, 250, 290):
            page.draw_line((300, y), (510, y))
        page.insert_text((310, 224), 'Keep this\ncell together', fontsize=12)
        for x, y, text in ((405, 224, 'Port A'), (310, 270, 'Outlet'), (405, 270, 'Port B')):
            page.insert_text((x, y), text, fontsize=12)
        doc.save(path)
    return path


@pytest.mark.parametrize('detect_tables', [True, False])
def test_automatic_separation_uses_wires_not_just_alignment(drawing, detect_tables):
    project = native_pdf_project(drawing, detect_tables=detect_tables)
    free = [obj for obj in project.objects if isinstance(obj, Text)]
    wired = [obj for obj in free if 'Wire' in obj.text]
    assert Counter(obj.text for obj in wired) == {'Wire A': 3, 'Wire B': 1}
    assert all(len(obj.ocr_ids) == 1 for obj in wired)
    assert any(obj.text == 'Power supply must remain connected during service' for obj in free)


def test_split_lines_option_does_not_change_table_cells(drawing):
    normal = native_pdf_project(drawing)
    split = native_pdf_project(drawing, split_lines=True)
    texts = [obj for obj in split.objects if isinstance(obj, Text)]
    assert all(len(obj.ocr_ids) == 1 for obj in texts)
    assert {'Power supply', 'must remain', 'connected', 'during service'} <= {obj.text for obj in texts}
    def cells(project):
        return [[(cell.text, cell.row, cell.col, cell.rowspan, cell.colspan) for cell in obj.cells]
                for obj in project.objects if isinstance(obj, Table)]
    assert cells(split) == cells(normal)
    assert any('Keep this cell together' == cell[0] for table in cells(split) for cell in table)
    assert split.source.raster_b64 == normal.source.raster_b64


def test_batch_exports_distinct_pptx_shapes(drawing, monkeypatch):
    from smart_overlay import pdf_batch
    monkeypatch.setattr(pdf_batch, 'data_dir', lambda: drawing.parent / 'work')
    result = pdf_batch.convert_pdf_pages(drawing)
    with ZipFile(result.groups[0].pptx) as pptx:
        slide = ET.fromstring(pptx.read('ppt/slides/slide1.xml'))
        texts = [''.join(t.text or '' for t in shape.findall('.//{*}t'))
                 for shape in slide.findall('.//{*}sp')]
        assert Counter(text for text in texts if 'Wire' in text) == {'Wire A': 3, 'Wire B': 1}
        assert 'Power supply must remain connected during service' in texts


def test_checkbox_persists_and_is_passed_to_batch(gui_app, monkeypatch, tmp_path):
    from smart_overlay import pdf_batch, quick_app
    monkeypatch.setattr('smart_overlay.settings.data_dir', lambda: tmp_path)
    monkeypatch.setattr(quick_app.ConvertWorker, 'start', lambda _: None)
    calls = []
    monkeypatch.setattr(pdf_batch, 'convert_pdf_pages', lambda path, **kw: calls.append(kw['settings']))
    window = quick_app.QuickWindow()
    try:
        box = window.pdf_split_lines_checkbox
        assert box.text() == 'Не объединять строки' and not box.isChecked()
        parent = box.parentWidget()
        assert parent is not None
        layout = parent.layout()
        assert layout is not None
        assert layout.indexOf(box) == layout.indexOf(window.pdf_ignore_numbers_checkbox) + 1
        box.click()
        assert load_settings().native_pdf_split_lines
        window.path = tmp_path / 'drawing.pdf'
        window.recognize()
        assert window.worker is not None
        assert not box.isEnabled()
        window.worker.work(lambda _: None, lambda: False)
        assert calls[0].native_pdf_split_lines
    finally:
        window.finished()
        window.close()
        window.deleteLater()


def test_cli_can_enable_or_override_split_lines(monkeypatch, tmp_path):
    from smart_overlay.__main__ import main
    config = tmp_path / 'settings.json'
    calls = []
    def convert(*args, **kw):
        calls.append(kw['settings'])
        return SimpleNamespace(pptx=tmp_path / 'output.pptx', warnings=[])
    monkeypatch.setattr('smart_overlay.one_click.convert_drawing', convert)
    for saved, flag in ((False, '--pdf-split-lines'), (True, '--no-pdf-split-lines')):
        config.write_text(Settings(native_pdf_split_lines=saved).model_dump_json(), encoding='utf-8')
        monkeypatch.setattr('sys.argv', ['smart_overlay', 'convert', 'drawing.pdf', '--settings', str(config), flag])
        assert main() == 0
        assert calls[-1].native_pdf_split_lines is not saved
