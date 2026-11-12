from PIL import Image
import pytest

from smart_overlay.input import load_image, save_project, open_project
from smart_overlay.models import Box, Fragment
from smart_overlay.pipeline import SmartFailure, local_ocr, smart_ocr
from smart_overlay.settings import Settings
from smart_overlay.structure import validate_project


@pytest.mark.parametrize("format, extension", [("TIFF", "tif"), ("BMP", "bmp"), ("JPEG", "jpg"), ("PNG", "png")])
def test_input_formats_preserve_original(tmp_path, format, extension):
    path = tmp_path / ("input." + extension)
    Image.new("RGB", (200, 100), "white").save(path, format=format)
    project = load_image(path)
    assert (project.source.width, project.source.height) == (200, 100)
    save_project(project, tmp_path / "nested" / "project.json")
    assert open_project(tmp_path / "nested" / "project.json").source == project.source


def test_exif_orientation_is_shared_by_ocr_preview_and_export(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif(); exif[274] = 6
    Image.new("RGB", (200, 100), "white").save(path, exif=exif)
    project = load_image(path)
    assert (project.source.width, project.source.height) == (100, 200)
    assert project.source.orientation == 6
    assert project.source.raster_mime == "image/png"


def test_local_mode_never_calls_external_api(project, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("local mode contacted VL")
    monkeypatch.setattr("smart_overlay.vision.VisionClient.request", forbidden)
    class Engine:
        def recognize(self, image):
            return project.fragments[:1], [{"example": "local fixture"}]
        def tables(self, image):
            return [{"table_res_list": []}]
    result = local_ocr(project, Settings(), Engine())
    assert len(result.objects) == 1
    validate_project(result)


def test_table_failure_retains_text_for_review(project):
    class Engine:
        def recognize(self, image):
            return project.fragments[:1], []
        def tables(self, image):
            raise RuntimeError("model unavailable")
    result = local_ocr(project, Settings(), Engine())
    assert result.raw["table_detection_failed"]
    assert all(o.needs_review for o in result.objects)
    validate_project(result)


def test_smart_failure_preserves_local_project(project, tmp_path):
    with pytest.raises(SmartFailure) as exc:
        smart_ocr(project, Settings(model="test"), "secret", tmp_path, False)
    validate_project(exc.value.project)
    assert exc.value.project.fragments == project.fragments
    assert exc.value.project.objects == project.objects


def test_smart_pipeline_keeps_omitted_table_review_without_failing(project, tmp_path, monkeypatch):
    from smart_overlay.models import Text, Table
    from smart_overlay.vision import Analysis, Group
    analysis = Analysis(groups=[Group(ocr_ids=o.ocr_ids, text=o.text, confidence=1, reason="same")
                                for o in project.objects if isinstance(o, Text)],
                        tables=[], ignored={}, regions=[])
    monkeypatch.setattr("smart_overlay.vision.VisionClient.request", lambda *a, **kw: analysis)
    result = smart_ocr(project, Settings(model="test"), "secret", tmp_path, True)
    table = next(o for o in result.objects if isinstance(o, Table))
    assert table.needs_review
    assert table.cells[0].text == project.objects[-1].cells[0].text
    assert any("VL не проверила таблицу" in message for message in result.messages)
    validate_project(result)


def test_cancelled_smart_keeps_valid_local_project(project, tmp_path):
    from smart_overlay.pipeline import SmartCancelled
    with pytest.raises(SmartCancelled) as exc:
        smart_ocr(project, Settings(model="test"), "secret", tmp_path, True, cancelled=lambda: True)
    assert exc.value.project.objects == project.objects
    assert exc.value.project.fragments == project.fragments
    validate_project(exc.value.project)


def test_partial_analysis_continues_smart_pipeline(project, tmp_path, monkeypatch):
    from smart_overlay.models import Text
    from smart_overlay.vision import Analysis, Group
    label = next(o for o in project.objects if o.id == "drain")
    partial = Analysis(groups=[Group(ocr_ids=label.ocr_ids, text=label.text, confidence=1, reason="visible")],
                       tables=[], ignored={}, regions=[])
    monkeypatch.setattr("smart_overlay.vision.VisionClient.request", lambda *a, **kw: partial)
    result = smart_ocr(project, Settings(model="test"), "secret", tmp_path, True)
    validate_project(result)
    assert any("VL пропустила OCR-фрагменты: 4" in message for message in result.messages)
    assert all(o.needs_review for o in result.objects if isinstance(o, Text) and "ocr_4" not in o.ocr_ids)
    restored = next(o for o in result.objects if o.id == "pressure")
    assert restored.bbox == next(o for o in project.objects if o.id == "pressure").bbox
    assert any(u.get("stage") == "smart" for u in result.usage)


def test_refinement_connection_failure_keeps_accepted_results(project, tmp_path, monkeypatch):
    from smart_overlay.models import Text
    from smart_overlay.vision import Analysis, Reading, VisionError
    original_second = project.objects[1].text
    project.objects[0].confidence = project.objects[1].confidence = 0.3
    monkeypatch.setattr("smart_overlay.pipeline.analyze", lambda p, client: (p.model_copy(deep=True), Analysis(groups=[], tables=[], ignored={}, regions=[])))
    calls = []
    def request(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            return Reading(text="Accepted correction", confidence=0.99, reason="visible")
        raise VisionError("Тайм-аут подключения TLS")
    monkeypatch.setattr("smart_overlay.vision.VisionClient.request", request)
    result = smart_ocr(project, Settings(model="test"), "secret", tmp_path, True)
    validate_project(result)
    assert next(o for o in result.objects if o.id == "pressure").text == "Accepted correction"
    unverified = next(o for o in result.objects if o.id == "drain")
    assert unverified.text == original_second and unverified.needs_review
    assert len(calls) == 2
    assert any("получено 1/2" in m for m in result.messages)
    assert any(u.get("stage") == "refine" and u["unverified"] == 1 for u in result.usage)


def test_cancellation_during_refinement_still_cancels_smart(project, tmp_path, monkeypatch):
    from smart_overlay.pipeline import SmartCancelled
    from smart_overlay.vision import Analysis, VisionCancelled
    project.objects[0].confidence = 0.3
    monkeypatch.setattr("smart_overlay.pipeline.analyze", lambda p, client: (p.model_copy(deep=True), Analysis(groups=[], tables=[], ignored={}, regions=[])))
    def cancel(*args, **kwargs):
        raise VisionCancelled("Smart OCR отменён пользователем")
    monkeypatch.setattr("smart_overlay.vision.VisionClient.request", cancel)
    with pytest.raises(SmartCancelled):
        smart_ocr(project, Settings(model="test"), "secret", tmp_path, True)
