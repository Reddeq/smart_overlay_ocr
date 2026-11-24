"""Real Qt controls and CLI boundary contracts for native PDF batches."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from smart_overlay.settings import Settings


@pytest.fixture
def window(gui_app, monkeypatch):
    from smart_overlay import quick_app
    monkeypatch.setattr(quick_app, 'load_settings', lambda: Settings())
    monkeypatch.setattr(quick_app, 'save_settings', lambda _: None)
    widget = quick_app.QuickWindow()
    yield widget
    widget.worker = None
    widget.close()
    widget.deleteLater()


def test_selection_is_separate_from_preview_and_busy_scoped(window):
    window.path = Path('drawing.pdf')
    assert window.pdf_pages_input.text() == ''
    assert '1, 2, 6-9' in window.pdf_pages_input.placeholderText()
    assert window.pdf_pages_label.text()
    window.pdf_pages_input.setText('2-3')
    window.pdf_page.blockSignals(True)
    window.pdf_page.setRange(1, 9)
    window.pdf_page.setValue(8)
    assert window.pdf_pages_input.text() == '2-3'
    window.set_busy(True)
    assert not window.pdf_pages_input.isEnabled()
    window.set_busy(False)
    assert window.pdf_pages_input.isEnabled()
    window.path = Path('drawing.png')
    window.set_busy(False)
    assert not window.pdf_pages_input.isEnabled()


def test_gui_invalid_selection_never_starts_worker(window, monkeypatch):
    from smart_overlay import quick_app
    window.path = Path('drawing.pdf')
    window.pdf_page.setRange(1, 3)
    window.pdf_pages_input.setText('3-1')
    warnings = []
    monkeypatch.setattr(quick_app.QMessageBox, 'warning', lambda *a: warnings.append(a[-1]))
    window.recognize()
    assert window.worker is None
    assert warnings and '3-1' in warnings[0]


def test_gui_batch_completion_lists_all_outputs_and_disables_deck_editor(window, project, monkeypatch):
    from smart_overlay import quick_app
    seen = []
    monkeypatch.setattr(quick_app, 'open_project', lambda path: seen.append(path) or project)
    monkeypatch.setattr(window.preview, 'show_project', lambda _: None)
    groups = [SimpleNamespace(pptx=Path('wide.pptx'), pages=[1, 3],
                              project_files=[Path('one.socr'), Path('three.socr')]),
              SimpleNamespace(pptx=Path('tall.pptx'), pages=[2], project_files=[Path('two.socr')])]
    window.done(SimpleNamespace(groups=groups, selected_pages=[1, 2, 3], warnings=['review fonts'], needs_review=4))
    assert window.output_combo.count() == 2
    assert seen == [Path('one.socr')]
    assert not window.review.isEnabled()
    assert 'wide.pptx' in window.summary.text() and 'tall.pptx' in window.summary.text()
    opened = []
    monkeypatch.setattr(quick_app.QDesktopServices, 'openUrl', lambda url: opened.append(url.toLocalFile()))
    window.output_combo.setCurrentIndex(1)
    window.open_result()
    assert opened[-1].endswith('tall.pptx')
    window.set_busy(False)
    assert not window.review.isEnabled()


def test_cli_pdf_selection_flags_are_mutually_exclusive(monkeypatch):
    from smart_overlay.__main__ import main
    monkeypatch.setattr('sys.argv', ['smart_overlay', 'convert', 'drawing.pdf',
                                    '--pdf-page', '1', '--pdf-pages', 'all'])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_cli_batch_routes_locally_prints_every_output(tmp_path, monkeypatch, capsys, caplog):
    from smart_overlay.__main__ import main
    from smart_overlay import pdf_batch
    calls = []
    def convert(source, **kwargs):
        calls.append((source, kwargs))
        return SimpleNamespace(groups=[SimpleNamespace(pptx=Path('a.pptx')),
                                       SimpleNamespace(pptx=Path('b.pptx'))], warnings=['font warning'])
    monkeypatch.setattr(pdf_batch, 'convert_pdf_pages', convert)
    monkeypatch.setattr('smart_overlay.one_click.convert_drawing',
                        lambda *a, **k: pytest.fail('single-page route called'))
    monkeypatch.setattr('sys.argv', ['smart_overlay', 'convert', 'drawing.pdf', '--pdf-pages',
                                    '1, 3-4', '--output', str(tmp_path / 'base.pptx'),
                                    '--no-pdf-tables', '--pdf-ignore-numbers'])
    assert main() == 0
    source, kwargs = calls[0]
    assert source == Path('drawing.pdf')
    assert kwargs['pages'] == '1, 3-4'
    assert kwargs['destination'] == tmp_path / 'base.pptx'
    assert not kwargs['settings'].native_pdf_detect_tables
    assert kwargs['settings'].native_pdf_ignore_numbers
    assert 'api_key' not in kwargs and 'consent' not in kwargs
    output = capsys.readouterr().out
    assert 'a.pptx' in output and 'b.pptx' in output
    assert 'font warning' in caplog.text


def test_cli_default_stays_single_page(monkeypatch):
    from smart_overlay.__main__ import main
    calls = []
    monkeypatch.setattr('smart_overlay.one_click.convert_drawing',
                        lambda *a, **k: calls.append(k) or SimpleNamespace(pptx=Path('a.pptx'), warnings=[]))
    monkeypatch.setattr('sys.argv', ['smart_overlay', 'convert', 'drawing.pdf'])
    assert main() == 0
    assert calls[0]['pdf_page'] == 1


@pytest.mark.parametrize('selection', ['', '1, 3'])
def test_gui_worker_routes_selection_not_preview_locally(window, monkeypatch, selection):
    from smart_overlay import quick_app, pdf_batch
    window.path = Path('drawing.pdf')
    window.pdf_page.blockSignals(True)
    window.pdf_page.setRange(1, 4)
    window.pdf_page.setValue(4)
    window.pdf_pages_input.setText(selection)
    window.pdf_tables_checkbox.setChecked(False)
    window.pdf_ignore_numbers_checkbox.setChecked(True)
    window.settings.ocr_engine = 'cloud_full'  # Settings from a previous release.
    calls = []
    monkeypatch.setattr(pdf_batch, 'convert_pdf_pages', lambda source, **kw: calls.append((source, kw)))
    monkeypatch.setattr('smart_overlay.one_click.convert_drawing', lambda *a, **k: pytest.fail('OCR route'))
    monkeypatch.setattr(quick_app.ConvertWorker, 'start', lambda _: None)
    window.recognize()
    assert window.worker is not None
    assert not window.pdf_pages_input.isEnabled()
    report, stop = lambda _: None, lambda: False
    window.worker.work(report, stop)
    assert calls[0][0] == Path('drawing.pdf')
    kwargs = calls[0][1]
    assert kwargs['pages'] == selection
    assert kwargs['progress'] is report and kwargs['cancelled'] is stop
    assert not kwargs['settings'].native_pdf_detect_tables
    assert kwargs['settings'].native_pdf_ignore_numbers
    assert 'api_key' not in kwargs
    window.finished()
    assert window.pdf_pages_input.isEnabled()


def test_gui_new_input_resets_selection(window, project, monkeypatch):
    from smart_overlay import quick_app
    window.pdf_pages_input.setText('2-3')
    monkeypatch.setattr(quick_app.QFileDialog, 'getOpenFileName', lambda *a: ('drawing.pdf', ''))
    monkeypatch.setattr(quick_app, 'pdf_info', lambda _: {'pages': 3})
    monkeypatch.setattr(quick_app, 'preview_pdf', lambda *_: project)
    window.choose()
    assert window.pdf_pages_input.text() == ''


def test_cli_batch_overrides_saved_cloud_engine(tmp_path, monkeypatch):
    from smart_overlay import pdf_batch
    from smart_overlay.__main__ import main
    settings = tmp_path / 'settings.json'
    settings.write_text(Settings(ocr_engine='cloud_full').model_dump_json(), encoding='utf-8')
    monkeypatch.setattr(pdf_batch, 'convert_pdf_pages', lambda *a, **k: SimpleNamespace(groups=[], warnings=[]))
    monkeypatch.setattr('sys.argv', ['smart_overlay', 'convert', 'drawing.pdf', '--pdf-pages', 'all',
                                    '--settings', str(settings)])
    assert main() == 0


def test_page_selection_field_is_wide_enough_for_real_input(window, gui_app):
    from PySide6.QtGui import QFontDatabase
    font=Path('C:/Windows/Fonts/segoeui.ttf')
    if font.exists():
        QFontDatabase.addApplicationFont(str(font))
    window.path=Path('drawing.pdf')
    window.set_busy(False)
    window.show()
    gui_app.processEvents()
    assert window.pdf_pages_input.width()>=200


def test_single_selected_page_keeps_existing_project_editor(window, project, monkeypatch):
    from smart_overlay import quick_app
    monkeypatch.setattr(quick_app,'open_project',lambda _:project)
    monkeypatch.setattr(window.preview,'show_project',lambda _:None)
    group=SimpleNamespace(pptx=Path('one_03.pptx'),pages=[3],
                          project_files=[Path('one.socr')],warnings=[],needs_review=0)
    window.done(SimpleNamespace(groups=[group],selected_pages=[3],warnings=[],needs_review=0))
    assert not window.review.isHidden()
    assert window.review.isEnabled()
    assert window.result.project is project
