"""Pure multilingual routing tests; no model downloads or HTTP requests."""
from types import SimpleNamespace

import pytest
from PIL import Image

from smart_overlay.diagnose_vl import synthetic
from smart_overlay.languages import (
    choose, families, native_evidence, recognize_lines, representative_indices,
    resolve, Reading,
)
from smart_overlay.settings import Settings


def test_recognizer_families_are_deduplicated():
    assert families(["ch", "en", "japan"]) == ("cjk",)
    assert families(["ru", "en", "uk"]) == ("eslav",)
    assert families(["fr", "de", "en"]) == ("latin",)
    assert families(["ar", "en"]) == ("arabic", "en")
    assert families(["ch", "ru"]) == ("cjk", "eslav")


@pytest.mark.parametrize("codes,primary,extra", [
    (["ch", "en"], "ch", ()),
    (["ru", "en"], "ru", ()),
    (["en"], "en", ()),
    (["ch", "ru"], "ch", ("eslav",)),
    (["ar", "en"], "ar", ("en",)),
    (["de", "fr"], "fr", ()),
    (["korean", "ru"], "ru", ("korean",)),
])
def test_manual_routes_to_recognizer_families(codes, primary, extra):
    settings = Settings(ocr_language_mode="manual", ocr_languages=codes)
    assert resolve(settings) == (primary, extra)


def test_auto_covers_bounded_scripts_without_requesting_legacy_lang_auto():
    settings = Settings(ocr_language_mode="auto")
    assert resolve(settings) == ("ch", ("eslav", "korean", "arabic", "latin"))


def test_settings_backward_compatibility_and_validation():
    assert resolve(Settings(language="ru")) == ("ru", ())
    with pytest.raises(ValueError, match="Выберите"):
        Settings(ocr_language_mode="manual")
    with pytest.raises(ValueError, match="Неизвестный"):
        Settings(ocr_language_mode="manual", ocr_languages=["xx"])
    with pytest.raises(ValueError, match="разных"):
        Settings(ocr_language_mode="manual", ocr_languages=["ru", "ru"])
    with pytest.raises(ValueError, match="PP-OCRv5"):
        Settings(ocr_language_mode="auto", ocr_version="PP-OCRv3")


def test_script_evidence_not_inferred_from_latin_codes():
    assert native_evidence("Основной блок", "eslav", .91)
    assert native_evidence("电源开关", "cjk", .96)
    assert native_evidence("한국어", "korean", .90)
    assert native_evidence("المحرك", "arabic", .90)
    assert not native_evidence("XS1 24V", "eslav", .99)
    assert not native_evidence("XS1 24V", "cjk", .99)


def test_candidate_selection_uses_script_and_marks_conflicts():
    best, conflict = choose([
        Reading("cjk", "KONTPOLЬ", .70),
        Reading("eslav", "КОНТРОЛЬ", .93),
    ])
    assert best.text == "КОНТРОЛЬ"
    assert not conflict
    best, conflict = choose([
        Reading("cjk", "电源开关", .93),
        Reading("eslav", "Основной", .92),
    ])
    assert conflict
    assert best.group in ("cjk", "eslav")


class FakeRecognizer:
    def __init__(self, predictions):
        self.predictions = predictions
        self.calls = []

    def predict(self, input, batch_size=8):
        self.calls.append(len(input))
        return [SimpleNamespace(json={"res": {"rec_text": text, "rec_score": score}})
                for text, score in self.predictions[:len(input)]]


def test_manual_secondary_ocr_preserves_geometry_and_ids():
    project, _ = synthetic()
    original = project.fragments
    fake = FakeRecognizer([("Основной", .99), ("CONTROL", .90),
                           ("PANEL 1", .85), ("VALVE 1", .90), ("PUMP 1", .91)])
    result, audit = recognize_lines(
        Image.new("RGB", (600, 180)), original, "cjk", ("eslav",), auto=False,
        model_factory=lambda _: fake)
    assert len(fake.calls) == 1
    assert audit["active"] == ["cjk", "eslav"]
    assert [f.id for f in result] == [f.id for f in original]
    assert [f.bbox for f in result] == [f.bbox for f in original]
    assert result[0].text == "Основной"
    assert original[0].text == "MAIN"


def test_failed_secondary_does_not_drop_original_text():
    project, _ = synthetic()
    def missing(_):
        raise RuntimeError("not installed")
    result, audit = recognize_lines(
        Image.new("RGB", (600, 180)), project.fragments, "cjk", ("eslav",),
        auto=False, model_factory=missing)
    assert [f.text for f in result] == [f.text for f in project.fragments]
    assert audit["warnings"]


def test_auto_probe_skips_missing_scripts_without_full_extra_pass():
    project, _ = synthetic()
    fake = FakeRecognizer([("CONTROL", .99)] * len(project.fragments))
    result, audit = recognize_lines(
        Image.new("RGB", (600, 180)), project.fragments, "cjk", ("eslav",),
        auto=True, model_factory=lambda _: fake)
    assert fake.calls == [5]
    assert audit["active"] == ["cjk"]
    assert [f.text for f in result] == [f.text for f in project.fragments]


def test_auto_probe_can_activate_cyrillic_model():
    project, _ = synthetic()
    fake = FakeRecognizer([("КОНТРОЛЬ", .95)] * len(project.fragments))
    result, audit = recognize_lines(
        Image.new("RGB", (600, 180)), project.fragments, "cjk", ("eslav",),
        auto=True, model_factory=lambda _: fake)
    assert audit["active"] == ["cjk", "eslav"]
    assert all(f.text == "КОНТРОЛЬ" for f in result)


def test_representative_sample_includes_bottom_of_page():
    project, _ = synthetic(30)
    picks = representative_indices(project.fragments, limit=8)
    assert len(picks) == 8
    assert max(picks) >= 25


def test_language_chooser_presets_and_auto_default(gui_app, monkeypatch, tmp_path):
    from smart_overlay.quick_app import LanguageDialog, QuickWindow
    monkeypatch.setattr("smart_overlay.quick_app.load_settings", Settings)
    monkeypatch.setattr("smart_overlay.quick_app.data_dir", lambda: tmp_path)
    window = QuickWindow()
    dialog = LanguageDialog("auto", [], window)
    try:
        assert window.settings.ocr_language_mode == "auto"
        assert "Авто" in window.language_button.text()
        dialog.use_preset(["ch", "en"])
        assert dialog.selected() == ("manual", ["en", "ch"])
    finally:
        dialog.close()
        window.close()
        dialog.deleteLater()
        window.deleteLater()


def test_unread_primary_detection_can_be_recovered_by_secondary():
    from smart_overlay.ocr import additional_detected_lines
    project, _ = synthetic()
    known = project.fragments[:1]
    raw = [{"dt_polys": [
        [[known[0].bbox.x, known[0].bbox.y],
         [known[0].bbox.right, known[0].bbox.y],
         [known[0].bbox.right, known[0].bbox.bottom],
         [known[0].bbox.x, known[0].bbox.bottom]],
        [[350, 120], [445, 120], [445, 144], [350, 144]]
    ]}]
    missing = additional_detected_lines(raw, known)
    assert len(missing) == 1
    assert missing[0].text == ""
    fake = FakeRecognizer([("Основной", .96), ("Питание", .97)])
    combined, audit = recognize_lines(
        Image.new("RGB", (600, 180)), known + missing, "cjk", ("eslav",),
        auto=False, model_factory=lambda _: fake)
    assert combined[-1].text == "Питание"
    assert combined[-1].bbox == missing[-1].bbox


def test_auto_confusable_script_keeps_original_and_requests_review():
    project, _ = synthetic()
    fake = FakeRecognizer([("КОНТРОЛЬ", .85)] * len(project.fragments))
    result, audit = recognize_lines(
        Image.new("RGB", (600, 180)), project.fragments,
        "cjk", ("eslav",), auto=True, model_factory=lambda _: fake)
    assert result[0].text == project.fragments[0].text
    assert result[0].id in audit["uncertain_ids"]
