import base64
import shutil
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pytest

from smart_overlay.export import export_pptx
from smart_overlay.export import resolve_node
from smart_overlay.models import Box, Text
from smart_overlay.settings import Settings

NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


def test_native_pptx_geometry_image_and_merges(project, tmp_path):
    if not shutil.which("node"):
        pytest.skip("Node.js not installed")
    output = tmp_path / "result.pptx"
    export_pptx(project, output, Settings())
    with ZipFile(output) as archive:
        slide = ET.fromstring(archive.read("ppt/slides/slide1.xml"))
        presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
        size = presentation.find("p:sldSz", NS)
        assert int(size.attrib["cx"])*800 == int(size.attrib["cy"])*1200
        tree = slide.find("p:cSld/p:spTree", NS)
        assert tree[2].tag.endswith("pic")  # original is the first visible layer
        assert len(slide.findall(".//p:sp", NS)) == 8  # five source-line masks + three text frames
        assert len(slide.findall(".//a:tbl", NS)) == 1
        assert slide.find('.//a:tc[@gridSpan="3"]', NS) is not None
        assert slide.find('.//a:tc[@rowSpan="2"]', NS) is not None
        assert len(slide.findall(".//a:tr", NS)) == 4
        for row, height in zip(slide.findall(".//a:tr", NS), [55, 75, 75, 75]):
            assert abs(int(row.attrib["h"]) - height/150*914400) <= 1
        for shape in slide.findall(".//p:sp", NS):
            assert shape.find("p:spPr/a:solidFill", NS) is not None
        media = [n for n in archive.namelist() if n.startswith("ppt/media/") and n.endswith(".png")]
        assert any(archive.read(n) == base64.b64decode(project.source.original_b64) for n in media)


def test_export_rejects_invalid_slide_size(project, tmp_path):
    with pytest.raises(ValueError, match="Размер слайда"):
        export_pptx(project, tmp_path / "bad.pptx", Settings(inches_per_pixel=0.1))


def test_table_cells_use_valid_drawingml_vertical_anchors(project,tmp_path):
    output=tmp_path/"anchors.pptx"
    export_pptx(project,output,Settings())
    with ZipFile(output) as archive:
        slide=ET.fromstring(archive.read("ppt/slides/slide1.xml"))
        for props in slide.findall(".//a:tcPr",NS):
            assert props.get("anchor","t") in {"t","ctr","b","just","dist"}


def test_rotated_frame_has_rotation(project, tmp_path):
    obj = project.objects[0]
    obj.rotation = 90
    export_pptx(project, tmp_path / "rotated.pptx", Settings())
    with ZipFile(tmp_path / "rotated.pptx") as archive:
        slide = ET.fromstring(archive.read("ppt/slides/slide1.xml"))
        assert slide.find('.//a:xfrm[@rot="5400000"]', NS) is not None


def test_export_unreviewed_when_explicitly_allowed(project, tmp_path):
    project.objects[0].needs_review = True
    project.objects[-1].needs_review = True
    output = tmp_path / "unreviewed.pptx"
    with pytest.raises(ValueError, match="проверка"):
        export_pptx(project, output, Settings())
    export_pptx(project, output, Settings(), allow_unreviewed=True)
    with ZipFile(output) as archive:
        slide = ET.fromstring(archive.read("ppt/slides/slide1.xml"))
        assert len(slide.findall(".//p:sp", NS)) == 8
        assert len(slide.findall(".//a:tbl", NS)) == 1
    assert project.objects[0].needs_review and project.objects[-1].needs_review


def test_allow_unreviewed_keeps_structural_validation(project, tmp_path):
    project.objects[0].ocr_ids.append(project.objects[1].ocr_ids[0])
    with pytest.raises(ValueError, match="несколько раз"):
        export_pptx(project, tmp_path / "invalid.pptx", Settings(), allow_unreviewed=True)


def test_find_portable_node_without_path(tmp_path, monkeypatch):
    import os
    portable = tmp_path / ".runtime" / "node" / ("node.exe" if os.name == "nt" else "bin/node")
    portable.parent.mkdir(parents=True)
    portable.touch()
    monkeypatch.delenv("SMART_OVERLAY_NODE", raising=False)
    monkeypatch.setattr("smart_overlay.export.shutil.which", lambda _: None)
    monkeypatch.setattr("smart_overlay.export.PROJECT_ROOT", tmp_path)
    assert resolve_node() == str(portable.resolve())


def test_explicit_node_override_and_invalid_override(tmp_path, monkeypatch):
    executable = tmp_path / "custom node.exe"
    executable.touch()
    monkeypatch.setenv("SMART_OVERLAY_NODE", str(executable))
    assert resolve_node() == str(executable.resolve())
    monkeypatch.setenv("SMART_OVERLAY_NODE", str(tmp_path / "missing.exe"))
    with pytest.raises(RuntimeError, match="SMART_OVERLAY_NODE"):
        resolve_node()


def test_node_not_found_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.delenv("SMART_OVERLAY_NODE", raising=False)
    monkeypatch.delenv("ProgramFiles", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr("smart_overlay.export.shutil.which", lambda _: None)
    monkeypatch.setattr("smart_overlay.export.PROJECT_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match=r"\.runtime/node"):
        resolve_node()


def test_inline_text_fill_creates_only_editable_text_frames(project, tmp_path):
    """One-click output has no separate white mask rectangles."""
    if not shutil.which("node"):
        pytest.skip("Node.js not installed")
    output = tmp_path / "one-click.pptx"
    export_pptx(project, output, Settings(fill_mode="auto"), inline_text_fill=True)
    with ZipFile(output) as archive:
        slide = ET.fromstring(archive.read("ppt/slides/slide1.xml"))
        text_shapes = slide.findall(".//p:sp", NS)
        assert len(text_shapes) == 3
        assert len(slide.findall(".//a:tbl", NS)) == 1
        for shape in text_shapes:
            assert shape.find("p:spPr/a:solidFill", NS) is not None
            name = shape.find("p:nvSpPr/p:cNvPr", NS).attrib["name"]
            assert not name.endswith("_mask")
