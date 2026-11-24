"""Local PaddleOCR-VL contract tests (offline; no 0.9B model downloaded)."""
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import numpy as np
import pytest

from smart_overlay.input import load_image
from smart_overlay.models import Table, Text
from smart_overlay.one_click import convert_drawing
from smart_overlay.settings import Settings
from smart_overlay.structure import validate_project
from smart_overlay.vl_local import (
    LocalVLError, block_box, local_vl_ocr, parse_page,
)


@pytest.fixture
def drawing(tmp_path):
    source = tmp_path / "drawing.png"
    Image.new("RGB", (640, 480), "#FFFFFF").save(source)
    return load_image(source), source


def page(*blocks):
    return {
        "width": 640, "height": 480,
        "parsing_res_list": list(blocks),
    }


def text_box(x=30, y=25, w=180, h=60, text="Электрическая\nсхема"):
    return {"block_label": "text", "block_bbox": [x, y, x+w, y+h],
            "block_content": text}


def table_box():
    return {"block_label": "table", "block_bbox": [300, 120, 620, 420],
            "block_content": (
                "<table><tr><th colspan=\"2\">Питание</th></tr>"
                "<tr><td>24 V DC</td><td>Выход</td></tr></table>")}


def test_vl_page_to_editable_frames_and_native_merged_table(drawing):
    source, _ = drawing
    project = parse_page(source, Settings(fill_mode="auto"),
                         page(text_box(), table_box()))
    validate_project(project)
    assert len(project.fragments) == 1
    assert [type(o) for o in project.objects] == [Text, Table]
    assert project.objects[0].text == "Электрическая\nсхема"
    assert project.objects[0].needs_review
    table = project.objects[1]
    assert table.rows == 2 and table.columns == 2
    assert table.cells[0].colspan == 2
    assert table.cells[0].text == "Питание"
    assert table.cells[1].text == "24 V DC"
    assert table.needs_review
    assert project.source == source.source
    assert project.raw["local_vl"]["converted_tables"] == 1


def test_block_coordinates_stay_in_source_pixels(drawing):
    project, _ = drawing
    bbox = block_box([10.25, 20.5, 80.4, 60.2], 640, 480)
    assert bbox.x == 10.25 and bbox.y == 20.5
    assert bbox.w == pytest.approx(70.15)
    for wrong in ([10, 20, 5, 30], [0, 0, 999, 200], [0, 0, 0, 2],
                  [[1, 2], [3, 4], [5, 6], [7, 8]]):
        with pytest.raises(ValueError):
            block_box(wrong, 640, 480)
    with pytest.raises(LocalVLError, match="ширину"):
        parse_page(project, Settings(), {**page(text_box()), "width": 800})


def test_unknown_image_regions_and_bad_table_are_not_silently_hidden(drawing):
    source, _ = drawing
    result = parse_page(source, Settings(), page(
        {"block_label": "image", "block_bbox": [0, 0, 600, 450],
         "block_content": "image"},
        {"block_label": "table", "block_bbox": [300, 120, 600, 200],
         "block_content": "Сетка с неопределённой структурой"}
    ))
    assert len(result.objects) == 1
    assert isinstance(result.objects[0], Text)
    assert "Сетка" in result.objects[0].text
    assert any("таблица без проверенной" in m for m in result.messages)
    assert result.objects[0].needs_review


def test_empty_model_result_fails_no_false_success(drawing):
    source, _ = drawing
    with pytest.raises(LocalVLError, match="Нет пригодных"):
        parse_page(source, Settings(), page())
    with pytest.raises(LocalVLError, match="parsing_res_list"):
        parse_page(source, Settings(), {"width":640, "height":480})


class MockLocalPipeline:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def predict(self, *, input, **kwargs):
        self.calls.append((input.shape, kwargs))
        yield SimpleNamespace(json={"res": self.response})


def test_full_local_pipeline_no_paddleocr_classic_or_cloud(drawing, monkeypatch, tmp_path):
    source, _ = drawing
    pipeline = MockLocalPipeline(page(text_box(), table_box()))
    monkeypatch.setattr("smart_overlay.vl_local.default_pipeline",
                        lambda *, device: pytest.fail("unexpected model download"))
    checked = []
    result = local_vl_ocr(
        source, Settings(fill_mode="auto"), device="cpu",
        pipeline_factory=lambda *, device: pipeline,
        diagnostics_dir=tmp_path/"diagnostics",
        check=lambda: checked.append(True))
    assert checked
    assert pipeline.calls[0][0] == (480, 640, 3)
    assert pipeline.calls[0][1]["layout_shape_mode"] == "rect"
    assert result.raw["local_vl"]["device"] == "cpu"
    assert (tmp_path/"diagnostics"/"paddle-vl-result.json").exists()
    assert (tmp_path/"diagnostics"/"paddle-vl-boxes.png").exists()
    saved = json.loads((tmp_path/"diagnostics"/"paddle-vl-result.json").read_text("utf-8"))
    assert saved["parsing_res_list"][0]["block_content"].startswith("Электрическая")


def test_one_click_vl_mode_cannot_contact_openrouter(drawing, monkeypatch, tmp_path):
    source, path = drawing
    monkeypatch.setattr("smart_overlay.one_click.local_ocr",
                        lambda *a, **kw: pytest.fail("Classic PaddleOCR must not run"))
    monkeypatch.setattr("smart_overlay.one_click._smart",
                        lambda *a, **kw: pytest.fail("OpenRouter must not run"))
    calls = []
    def isolated(source_path, settings, device, diagnostics_dir, **kwargs):
        calls.append((Path(source_path), device))
        return page(text_box()), {"tiles": 1, "finished": 1,
                                  "missing_or_failed": [], "seam_skips": [], "blocks": 1}
    monkeypatch.setattr("smart_overlay.vl_tiled.run_isolated", isolated)
    monkeypatch.setattr("smart_overlay.one_click.data_dir", lambda: tmp_path/"userdata")
    exports = []

    def export(p, output, settings, *, allow_unreviewed, inline_text_fill):
        validate_project(p)
        assert inline_text_fill and allow_unreviewed
        exports.append(p)
        Path(output).write_bytes(b"PPTX mock")

    monkeypatch.setattr("smart_overlay.one_click.export_pptx", export)
    result = convert_drawing(path, ocr_engine="vl_local", api_key="secret",
                             consent=True, vl_device="cpu",
                             vl_diagnostics=tmp_path/"diagnostics")
    assert len(exports) == 1
    assert result.pptx.read_bytes() == b"PPTX mock"
    assert result.needs_review >= 1
    assert calls == [(path.resolve(), "cpu")]
    assert result.project_file.exists()


def test_model_failure_never_creates_empty_pptx(drawing, monkeypatch, tmp_path):
    _, path = drawing
    monkeypatch.setattr("smart_overlay.vl_tiled.run_isolated",
                        lambda *args, **kwargs: (
                            page(), {"tiles": 1, "finished": 1,
                                     "missing_or_failed": [], "seam_skips": [], "blocks": 0}))
    monkeypatch.setattr("smart_overlay.one_click.export_pptx",
                        lambda *a, **k: pytest.fail("Must not export empty layout"))
    with pytest.raises(LocalVLError, match="Нет пригодных"):
        convert_drawing(path, ocr_engine="vl_local", vl_device="cpu")
    assert not path.with_suffix(".pptx").exists()



def test_image_only_vl_failure_saves_raw_output_and_layout(drawing, tmp_path):
    """Real drawings classified as one 'image' must not erase the raster."""
    source, _ = drawing
    from smart_overlay.vl_local import diagnostic_summary
    raw = page({
        "block_label": "image",
        "block_bbox": [0, 0, 640, 480],
        "block_content": "Some text found in the diagram",
    })
    pipeline = MockLocalPipeline(raw)
    out = tmp_path / "image-only"
    with pytest.raises(LocalVLError, match="Классы VL: image: 1"):
        local_vl_ocr(
            source, Settings(), device="cpu",
            pipeline_factory=lambda *, device: pipeline,
            diagnostics_dir=out)
    assert (out / "paddle-vl-result.json").exists()
    assert (out / "paddle-vl-summary.json").exists()
    assert (out / "paddle-vl-boxes.png").exists()
    saved = json.loads((out / "paddle-vl-result.json").read_text("utf-8"))
    assert saved["parsing_res_list"][0]["block_content"].startswith("Some text")
    summary = json.loads((out / "paddle-vl-summary.json").read_text("utf-8"))
    assert summary["status"] == "failed"
    assert summary["block_labels"] == {"image": 1}
    assert summary["regions"][0]["has_content"]
    assert diagnostic_summary(raw)["layout_block_count"] == 1
