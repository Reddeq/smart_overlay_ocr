"""Oversize native PDF sheets retain the full layout without exceeding the raster budget."""
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pytest
import pymupdf

from smart_overlay import native_pdf, one_click, pdf_batch
from smart_overlay.input import open_project
from smart_overlay.models import Table, Text


def make_pdf(path, *sizes):
    with pymupdf.open() as document:
        for number, (width, height) in enumerate(sizes, 1):
            page = document.new_page(width=width, height=height)
            page.insert_text((30, 40), f'Control panel {number}', fontsize=12)
            for x in (30, 100, 170):
                page.draw_line((x, 75), (x, 135))
            for y in (75, 105, 135):
                page.draw_line((30, y), (170, y))
            for x, y, text in ((40, 95, 'Power'), (110, 95, '8'),
                               (40, 125, 'Ground'), (110, 125, '7')):
                page.insert_text((x, y), text, fontsize=12)
        document.save(path)
    return path


def test_preview_adapts_at_real_wide_sheet_size(tmp_path):
    path = make_pdf(tmp_path / 'wide.pdf', (4768, 1684))
    preview = native_pdf.preview_pdf(path)
    w, h = preview.source.width, preview.source.height
    assert w * h <= native_pdf.MAX_PREVIEW_PIXELS
    assert w > 9480  # Use close to the original render quality, not a fixed lower DPI.
    assert w / h == pytest.approx(4768 / 1684, rel=0.001)
    assert preview.raw['native_pdf_preview']['zoom'] < native_pdf.PDF_ZOOM


def test_batch_keeps_common_resolution_and_live_geometry(tmp_path, monkeypatch):
    # Two near-identical sheets either side of the raster limit. A small limit
    # lets the test exercise real rendering and PPTX generation cheaply.
    path = make_pdf(tmp_path / 'batch.pdf', (400, 200), (400.4, 200))
    monkeypatch.setattr(native_pdf, 'MAX_PREVIEW_PIXELS', 320_000)
    monkeypatch.setattr(pdf_batch, 'data_dir', lambda: tmp_path / 'work')
    result = pdf_batch.convert_pdf_pages(path)
    assert [group.pages for group in result.groups] == [[1, 2]]
    group = result.groups[0]
    projects = [open_project(file) for file in group.project_files]
    zooms = {p.raw['native_pdf']['zoom'] for p in projects}
    assert len(zooms) == 1
    zoom = zooms.pop()
    assert zoom < native_pdf.PDF_ZOOM
    for p in projects:
        assert p.source.width * p.source.height <= native_pdf.MAX_PREVIEW_PIXELS
        assert p.settings['inches_per_pixel'] * zoom * 72 == pytest.approx(1)
        table = next(obj for obj in p.objects if isinstance(obj, Table))
        assert table.bbox.x / zoom == pytest.approx(30, abs=0.5)
        assert {cell.text for cell in table.cells} == {'Power', '8', 'Ground', '7'}
        assert any(isinstance(obj, Text) and obj.text.startswith('Control panel') for obj in p.objects)
    with ZipFile(group.pptx) as archive:
        deck = ET.fromstring(archive.read('ppt/presentation.xml'))
        size = deck.find('{*}sldSz')
        assert size is not None
        assert int(size.attrib['cx']) / 12700 == pytest.approx(400, abs=1)
        for number in (1, 2):
            slide = ET.fromstring(archive.read(f'ppt/slides/slide{number}.xml'))
            assert slide.find('.//{*}tbl') is not None
            assert f'Control panel {number}' in {node.text for node in slide.iterfind('.//{*}t')}


def test_single_page_export_uses_effective_raster_resolution(tmp_path, monkeypatch):
    path = make_pdf(tmp_path / 'single.pdf', (400, 200))
    monkeypatch.setattr(native_pdf, 'MAX_PREVIEW_PIXELS', 160_000)
    monkeypatch.setattr(one_click, 'data_dir', lambda: tmp_path / 'work')
    result = one_click.convert_drawing(path)
    zoom = result.project.raw['native_pdf']['zoom']
    assert zoom < native_pdf.PDF_ZOOM
    assert result.project.source.width * result.project.source.height <= native_pdf.MAX_PREVIEW_PIXELS
    assert result.project.settings['inches_per_pixel'] * zoom * 72 == pytest.approx(1)
    with ZipFile(result.pptx) as archive:
        size = ET.fromstring(archive.read('ppt/presentation.xml')).find('{*}sldSz')
        assert size is not None
        assert int(size.attrib['cx']) / 12700 == pytest.approx(400, abs=1)
