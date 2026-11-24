from pathlib import Path

from smart_overlay.settings import Settings


def test_quick_window_uses_native_pdf_regardless_of_legacy_settings(gui_app, monkeypatch):
    from PySide6.QtWidgets import QCheckBox, QPushButton, QVBoxLayout
    from smart_overlay import quick_app

    monkeypatch.setattr(quick_app, 'load_settings', lambda: Settings(ocr_engine='cloud_full'))
    window = quick_app.QuickWindow()
    try:
        tables, numbers = window.pdf_tables_checkbox, window.pdf_ignore_numbers_checkbox
        assert isinstance(tables, QCheckBox) and isinstance(numbers, QCheckBox)
        assert [tables.text(), numbers.text()] == ['Реконструировать таблицы', 'Игнорировать числа']
        options = tables.parentWidget().layout()
        assert isinstance(options, QVBoxLayout)
        assert options.indexOf(numbers) == options.indexOf(tables) + 1
        assert not hasattr(window, 'mode_combo')
        assert not any(button.text() == 'Настройки' for button in window.findChildren(QPushButton))
        assert not window.run_button.isEnabled()
    finally:
        window.close()
        window.deleteLater()


def test_quick_window_accepts_only_pdf(gui_app, monkeypatch):
    from smart_overlay import quick_app

    monkeypatch.setattr(quick_app, 'load_settings', Settings)
    filters, warnings = [], []
    monkeypatch.setattr(quick_app.QFileDialog, 'getOpenFileName',
                        lambda *args: (filters.append(args[-1]) or 'drawing.png', ''))
    monkeypatch.setattr(quick_app.QMessageBox, 'warning', lambda *args: warnings.append(args[-1]))
    window = quick_app.QuickWindow()
    try:
        window.choose()
        assert filters[0].endswith('(*.pdf)')
        assert window.path is None
        assert warnings and 'PDF' in warnings[0]
    finally:
        window.close()
        window.deleteLater()


def test_native_correction_editor_has_no_ocr_or_vl_controls(gui_app, project, monkeypatch):
    from PySide6.QtWidgets import QPushButton, QToolBar
    from smart_overlay import app, quick_app
    from smart_overlay.one_click import OneClickResult

    monkeypatch.setattr(quick_app, 'load_settings', Settings)
    monkeypatch.setattr(app, 'load_settings', Settings)
    project.source.mode = 'PDF'
    project.settings = Settings(inches_per_pixel=1/144).model_dump()
    window = quick_app.QuickWindow()
    try:
        window.result = OneClickResult(project=project, project_file=Path('drawing.socr.json'),
                                       pptx=Path('drawing.pptx'), warnings=[])
        window.open_editor()
        editor = window.editor
        assert editor is not None
        assert editor.settings.inches_per_pixel == 1/144
        labels = [action.text() for bar in editor.findChildren(QToolBar) for action in bar.actions()]
        labels += [button.text() for button in editor.findChildren(QPushButton) if button.isVisibleTo(editor)]
        assert 'Сохранить проект' in labels and 'Экспорт в PPTX' in labels
        assert not any(any(word in label for word in ('OCR', 'VL', 'Настройки', 'Открыть изображение'))
                       for label in labels)
    finally:
        if window.editor:
            window.editor.close()
            window.editor.deleteLater()
        window.close()
        window.deleteLater()
