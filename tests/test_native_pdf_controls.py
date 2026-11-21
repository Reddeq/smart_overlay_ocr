"""Settings and entry-point contracts for digital PDF conversion."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from smart_overlay.settings import Settings, load_settings, save_settings


def test_pdf_settings_default_and_legacy_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("smart_overlay.settings.data_dir", lambda: tmp_path)
    assert Settings().native_pdf_overlay_mode == "editable"
    assert Settings().native_pdf_detect_tables is True
    for mode in ("editable", "ocr_style", "source_exact"):
        settings = Settings(native_pdf_overlay_mode=mode, native_pdf_detect_tables=False)
        save_settings(settings)
        assert load_settings() == settings


@pytest.mark.parametrize("flag,saved,expected", [
    ([], False, False), ([], True, True),
    (["--pdf-tables"], False, True), (["--no-pdf-tables"], True, False),
])
def test_cli_pdf_tables_override_only_when_explicit(flag, saved, expected, tmp_path, monkeypatch):
    from smart_overlay.__main__ import main
    path = tmp_path / "settings.json"
    path.write_text(Settings().model_dump_json(), encoding="utf-8")
    import json
    raw = json.loads(path.read_text("utf-8"))
    raw["native_pdf_detect_tables"] = saved
    path.write_text(json.dumps(raw), encoding="utf-8")
    calls = []
    def convert(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(pptx=Path("result.pptx"), warnings=[])
    monkeypatch.setattr("smart_overlay.one_click.convert_drawing", convert)
    monkeypatch.setattr("sys.argv", ["smart_overlay", "convert", "drawing.pdf", "--settings", str(path), *flag])
    assert main() == 0
    assert calls[0]["settings"].native_pdf_detect_tables is expected


@pytest.mark.parametrize("ignore_numbers", [True, False])
@pytest.mark.parametrize("detect_tables", [True, False])
@pytest.mark.parametrize("legacy_mode", ["ocr_style", "source_exact"])
def test_new_conversion_keeps_native_editable_mode(ignore_numbers, detect_tables, legacy_mode, tmp_path, monkeypatch):
    from smart_overlay.diagnose_vl import synthetic
    from smart_overlay.one_click import convert_drawing
    project, _ = synthetic()
    project.raw["native_pdf"] = {"overlay_mode": "editable",
                                 "font_warnings": ["Source font must be installed"]}
    calls = []
    def native(source, page_number, *, zoom, progress, **kwargs):
        calls.append(kwargs)
        return project
    monkeypatch.setattr("smart_overlay.native_pdf.native_pdf_project", native)
    monkeypatch.setattr("smart_overlay.one_click.data_dir", lambda: tmp_path)
    monkeypatch.setattr("smart_overlay.one_click.export_pptx",
                        lambda project, path, *a, **kw: path.write_bytes(b"integration export double"))
    result = convert_drawing(tmp_path / "drawing.pdf", settings=Settings(
        native_pdf_overlay_mode=legacy_mode, native_pdf_detect_tables=detect_tables,
        native_pdf_ignore_numbers=ignore_numbers))
    assert calls == [{"detect_tables": detect_tables, "ignore_numbers": ignore_numbers}]
    assert result.project.raw["native_pdf"]["overlay_mode"] == "editable"
    assert result.warnings == ["Source font must be installed"]


def test_gui_pdf_tables_toggle_persists_and_respects_busy(gui_app, monkeypatch):
    from smart_overlay import quick_app
    settings = Settings.model_validate({"native_pdf_detect_tables": False,
                                        "native_pdf_overlay_mode": "source_exact"})
    monkeypatch.setattr(quick_app, "load_settings", lambda: settings)
    saved = []
    monkeypatch.setattr(quick_app, "save_settings", lambda value: saved.append(value.model_copy(deep=True)))
    window = quick_app.QuickWindow()
    try:
        window.path = Path("drawing.pdf")
        window.mode_combo.setCurrentIndex(window.mode_combo.findData("native_pdf"))
        combo = window.pdf_tables_combo
        assert combo.count() == 2
        assert combo.currentData() is False
        assert not combo.isHidden()
        combo.setCurrentIndex(combo.findData(True))
        assert saved[-1].native_pdf_detect_tables is True
        window.set_busy(True)
        assert not combo.isEnabled()
        window.set_busy(False)
        assert combo.isEnabled()
        window.path = Path("drawing.png")
        window.mode_changed()
        assert combo.isHidden()
    finally:
        window.close()
        window.deleteLater()


def test_gui_new_pdf_completion_describes_visible_editable_text(gui_app, project, monkeypatch):
    from smart_overlay import quick_app
    monkeypatch.setattr(quick_app, "load_settings", lambda: Settings())
    monkeypatch.setattr(quick_app, "save_settings", lambda _: None)
    project.source.mode = "PDF"
    project.raw["native_pdf"] = {"overlay_mode": "editable"}
    window = quick_app.QuickWindow()
    monkeypatch.setattr(window.preview, "show_project", lambda _: None)
    try:
        window.done(SimpleNamespace(project=project, pptx=Path("drawing.pptx"),
                                    needs_review=0, warnings=[]))
        assert "видим" in window.summary.text().lower()
        assert "невидим" not in window.summary.text().lower()
    finally:
        window.close()
        window.deleteLater()