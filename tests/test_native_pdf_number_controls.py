"""Numeric-label filtering controls, without invoking OCR or PowerPoint."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from smart_overlay.settings import Settings, load_settings, save_settings


def test_ignore_numbers_defaults_off_and_persists(tmp_path, monkeypatch):
    monkeypatch.setattr("smart_overlay.settings.data_dir", lambda: tmp_path)
    assert Settings().native_pdf_ignore_numbers is False
    for enabled in (True, False):
        save_settings(Settings(native_pdf_ignore_numbers=enabled))
        assert load_settings().native_pdf_ignore_numbers is enabled


@pytest.mark.parametrize("flag,saved,expected", [
    ([], False, False), ([], True, True),
    (["--pdf-ignore-numbers"], False, True),
    (["--no-pdf-ignore-numbers"], True, False),
])
def test_cli_ignore_numbers_explicit_overrides(flag, saved, expected, tmp_path, monkeypatch):
    from smart_overlay.__main__ import main
    path = tmp_path / "settings.json"
    path.write_text(Settings(native_pdf_ignore_numbers=saved).model_dump_json(), encoding="utf-8")
    calls = []
    def convert(*args, **kwargs):
        calls.append(kwargs["settings"])
        return SimpleNamespace(pptx=Path("result.pptx"), warnings=[])
    monkeypatch.setattr("smart_overlay.one_click.convert_drawing", convert)
    monkeypatch.setattr("sys.argv", ["smart_overlay", "convert", "drawing.pdf", "--settings", str(path), *flag])
    assert main() == 0
    assert calls[0].native_pdf_ignore_numbers is expected


def test_cli_ignore_numbers_help_describes_scope(monkeypatch, capsys):
    from smart_overlay.__main__ import main
    monkeypatch.setattr("sys.argv", ["smart_overlay", "convert", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--pdf-ignore-numbers" in help_text
    assert "standalone" in help_text
    assert "PNG" in help_text
    assert "table" in help_text


@pytest.mark.parametrize("saved_value", [False, True])
def test_gui_ignore_numbers_persistence_and_availability(gui_app, monkeypatch, saved_value):
    from smart_overlay import quick_app
    settings = Settings(native_pdf_ignore_numbers=saved_value)
    monkeypatch.setattr(quick_app, "load_settings", lambda: settings)
    saved = []
    monkeypatch.setattr(quick_app, "save_settings", lambda value: saved.append(value.model_copy(deep=True)))
    window = quick_app.QuickWindow()
    try:
        checkbox = window.pdf_ignore_numbers_checkbox
        assert checkbox.text() == "Игнорировать числа"
        assert checkbox.isChecked() is saved_value
        assert checkbox.isHidden()
        assert not checkbox.isEnabled()
        window.path = Path("drawing.pdf")
        window.mode_combo.setCurrentIndex(window.mode_combo.findData("native_pdf"))
        assert not checkbox.isHidden()
        assert checkbox.isEnabled()
        tooltip = checkbox.toolTip()
        assert "PNG" in tooltip
        assert "отдельные" in tooltip
        assert "таблиц" in tooltip
        checkbox.setChecked(not saved_value)
        assert saved[-1].native_pdf_ignore_numbers is (not saved_value)
        window.set_busy(True)
        assert not checkbox.isEnabled()
        window.set_busy(False)
        assert checkbox.isEnabled()
        window.path = Path("drawing.png")
        window.mode_changed()
        assert checkbox.isHidden()
        assert not checkbox.isEnabled()
        window.set_busy(True)
        window.set_busy(False)
        assert not checkbox.isEnabled()
        window.path = Path("drawing.pdf")
        window.mode_combo.setCurrentIndex(window.mode_combo.findData("smart"))
        assert checkbox.isHidden()
        assert not checkbox.isEnabled()
        assert window.pdf_tables_combo.isEnabled() == checkbox.isEnabled()
    finally:
        window.close()
        window.deleteLater()
