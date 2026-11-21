from pathlib import Path
import pytest
import pymupdf as fitz
from smart_overlay import pdf_batch as batch
from smart_overlay.input import open_project
from smart_overlay.settings import Settings
from smart_overlay.one_click import ProcessingCancelled


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'drawing.pdf'
    doc = fitz.open()
    for w,h in [(200,300),(300,200),(200.4,300),(200.8,300)]:
        page = doc.new_page(width=w, height=h)
        page.insert_text((20,40), f'Page {len(doc)} text')
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def exports(monkeypatch):
    records = []
    def export(projects, output, settings, **kwargs):
        assert iter(projects) is projects
        records.append(([p.source.width for p in projects], settings.model_copy(), kwargs))
        Path(output).write_bytes(b'exported')
    monkeypatch.setattr(batch, 'export_pptx_pages', export)
    return records


def test_groups_real_extraction_spooling_settings(source, exports, monkeypatch):
    monkeypatch.setattr(batch, 'data_dir', lambda: source.parent / 'data')
    settings = Settings(native_pdf_detect_tables=False, native_pdf_ignore_numbers=True)
    messages = []
    result = batch.convert_pdf_pages(source, settings=settings, progress=messages.append)
    assert result.selected_pages == [1,2,3,4]
    assert [g.pages for g in result.groups] == [[1,3],[2],[4]]
    assert [g.pptx.name for g in result.groups] == ['drawing_01,03.pptx','drawing_02.pptx','drawing_04.pptx']
    assert len(exports) == 3
    assert settings.ocr_engine == 'smart'
    for group in result.groups:
        assert group.pptx.read_bytes() == b'exported'
        assert len(group.project_files) == len(group.pages)
        for path in group.project_files:
            project = open_project(path)
            assert project.settings['native_pdf_ignore_numbers'] is True
            assert project.settings['native_pdf_detect_tables'] is False
            assert project.settings['ocr_engine'] == 'native_pdf'
        assert not hasattr(group, 'projects')
    assert any('3' in message for message in messages)
    assert sum(g.needs_review for g in result.groups) == result.needs_review


def test_near_square_orientation_stays_separate(source, exports, monkeypatch):
    monkeypatch.setattr(batch, 'data_dir', lambda: source.parent / 'data')
    monkeypatch.setattr(batch, 'pdf_info', lambda _: {'pages': 2, 'page_sizes': [[200,200.2],[200.2,200]]})
    result = batch.convert_pdf_pages(source)
    assert [group.pages for group in result.groups] == [[1],[2]]


def test_collision_and_explicit_destination(source, exports, monkeypatch):
    monkeypatch.setattr(batch, 'data_dir', lambda: source.parent / 'data')
    existing = source.parent / 'custom_01.pptx'
    existing.write_bytes(b'keep')
    result = batch.convert_pdf_pages(source, pages='1', destination=source.parent / 'custom.pptx')
    assert result.groups[0].pptx != existing
    assert '01' in result.groups[0].pptx.stem
    assert existing.read_bytes() == b'keep'


@pytest.mark.parametrize('mode', ['cancel','export','publish_cancel'])
def test_failure_leaves_no_deliverables(source, exports, monkeypatch, mode):
    root = source.parent / 'data'
    monkeypatch.setattr(batch, 'data_dir', lambda: root)
    state = {'cancel': False, 'calls': 0}
    real = batch.export_pptx_pages
    def export(*args, **kwargs):
        real(*args, **kwargs)
        state['calls'] += 1
        if mode == 'export' and state['calls'] == 2:
            raise RuntimeError('export failed')
        if mode == 'publish_cancel':
            state['cancel'] = True
    monkeypatch.setattr(batch, 'export_pptx_pages', export)
    def progress(message):
        if mode == 'cancel' and '2' in message:
            state['cancel'] = True
    with pytest.raises((ProcessingCancelled, RuntimeError)):
        batch.convert_pdf_pages(source, progress=progress, cancelled=lambda: state['cancel'])
    assert not list(source.parent.glob('*.pptx'))
    assert not list(root.rglob('*.socr.json'))


def test_publication_failure_rolls_back_only_own_files(source, exports, monkeypatch):
    monkeypatch.setattr(batch, 'data_dir', lambda: source.parent / 'data')
    existing = source.parent / 'drawing_01,03.pptx'
    existing.write_bytes(b'keep')
    real_link = batch.os.link
    published = []
    def link(src, dst):
        if published:
            raise OSError('publication failure')
        real_link(src, dst)
        published.append(dst)
    monkeypatch.setattr(batch.os, 'link', link)
    with pytest.raises(OSError, match='publication failure'):
        batch.convert_pdf_pages(source)
    assert existing.read_bytes() == b'keep'
    assert not any(path.exists() for path in published)
    assert not list((source.parent / 'data').rglob('*.socr.json'))


def test_real_node_export_xml_page_counts(source, monkeypatch):
    import zipfile
    from xml.etree import ElementTree as ET
    if not hasattr(batch.export, 'export_pptx_pages'):
        pytest.skip('Parent-owned export_pptx_pages API not yet available')
    monkeypatch.setattr(batch, 'data_dir', lambda: source.parent / 'data')
    result = batch.convert_pdf_pages(source, settings=Settings(native_pdf_detect_tables=False))
    ns = {'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
          'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
    for group in result.groups:
        with zipfile.ZipFile(group.pptx) as archive:
            presentation = ET.fromstring(archive.read('ppt/presentation.xml'))
            assert len(presentation.findall('p:sldIdLst/p:sldId', ns)) == len(group.pages)
            for i, page in enumerate(group.pages, 1):
                slide = ET.fromstring(archive.read(f'ppt/slides/slide{i}.xml'))
                text = ' '.join(node.text or '' for node in slide.findall('.//a:t', ns))
                assert f'Page {page} text' in text


def test_preflight_selection_before_extraction(source, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('Extraction happened before validation')
    monkeypatch.setattr(batch, 'native_pdf_project', unexpected)
    with pytest.raises(ValueError):
        batch.convert_pdf_pages(source, pages='1,5')


def test_page_failure_names_original_pdf_page(source,monkeypatch):
    monkeypatch.setattr(batch,'data_dir',lambda:source.parent/'data')
    def fail(*args,**kwargs):
        raise ValueError('Bad native layer')
    monkeypatch.setattr(batch,'native_pdf_project',fail)
    with pytest.raises(ValueError,match='3.*Bad native layer'):
        batch.convert_pdf_pages(source,pages='3')
    assert not list(source.parent.glob('*.pptx'))
