"""Native table typography is tested through the real Node exporter."""
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pytest

from smart_overlay.export import export_pptx
from smart_overlay.models import Box, Cell, NativeLayout, NativeLine, NativeRun, Style, Table, Text
from smart_overlay.settings import Settings

NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main"}


def _project(project, *, text="Source width", advance=20, height=50, kind="table"):
    style = Style(font="Arial", font_size=12)
    run = NativeRun(text=text, font="Arial", font_size=12, color="#000000", advance=advance)
    cell = Cell(row=0, col=0, text=text, style=style,
                native_layout=NativeLayout(frame=Box(x=10, y=10, w=200, h=24),
                                           lines=[NativeLine(runs=[run])],
                                           reference_style=style))
    project.source.mode = "PDF"
    project.raw = {"native_pdf": {"background_mode": "full_original", "overlay_mode": "editable", "zoom": 2}}
    project.fragments = []
    project.ignored = {}
    project.objects = [Table(id="native", bbox=Box(x=10, y=10, w=200, h=height),
                             rows=1, columns=1, row_heights=[height], col_widths=[200], cells=[cell])]
    if kind=="text":
        project.objects=[Text(id="native",bbox=project.objects[0].bbox,text=text,
                              style=cell.style,native_layout=cell.native_layout)]
    return project


def _slide(project, path):
    export_pptx(project, path, Settings())
    with ZipFile(path) as archive:
        return ET.fromstring(archive.read("ppt/slides/slide1.xml"))


@pytest.mark.parametrize("kind",["table","text"])
def test_pdf_run_advances_never_compress_visible_text(project, tmp_path, kind):
    project = _project(project,kind=kind)
    slide = _slide(project, tmp_path / "advances.pptx")
    path=".//a:tbl/a:tr/a:tc/a:txBody" if kind=="table" else ".//p:sp/p:txBody"
    run = slide.find(path+"/a:p/a:r/a:rPr", NS)
    assert run is not None
    assert run.get("sz") == "1152"  # 12 PDF pt * native scale 0.96
    assert int(run.get("spc", "0")) >= 0
    assert slide.find(".//a:normAutofit", NS) is None


def test_small_native_cell_does_not_force_tiny_line_spacing(project, tmp_path):
    project=_project(project,height=4)
    slide=_slide(project,tmp_path/"natural-leading.pptx")
    spacing=slide.find(".//a:tbl//a:lnSpc/a:spcPts",NS)
    assert spacing is None or int(spacing.get("val","0"))>=1152


def test_native_label_wraps_and_grows_without_font_autoshrink(project,tmp_path):
    project=_project(project,kind="text",height=4,text="Natural size text needs room to wrap")
    slide=_slide(project,tmp_path/"growing-label.pptx")
    body=slide.find(".//p:sp/p:txBody/a:bodyPr",NS)
    assert body is not None and body.get("wrap")!="none"
    assert body.find("a:spAutoFit",NS) is not None
    assert body.find("a:normAutofit",NS) is None


def test_same_baseline_pdf_spans_keep_their_horizontal_gap(project, tmp_path):
    project = _project(project, text="A\nB", advance=10)
    cell = project.objects[0].cells[0]
    cell.native_layout.lines = [NativeLine(runs=[NativeRun(text=text, font="Arial", font_size=12,
                                                         color="#000000", advance=10)])
                                for text in ("A", "B")]
    slide = _slide(project, tmp_path / "same-baseline.pptx")
    paragraphs = slide.findall(".//a:tbl/a:tr/a:tc/a:txBody/a:p", NS)
    assert len(paragraphs) == 1
    tab = paragraphs[0].find("a:pPr/a:tabLst/a:tab", NS)
    assert tab is not None
    assert int(tab.attrib["pos"]) / 12700 == 86.4
    assert [run.text for run in paragraphs[0].findall("a:r/a:t", NS)] == ["A", "\tB"]


def test_empty_native_cell_has_no_default_font_row_minimum(project, tmp_path):
    project = _project(project, text="", advance=0, height=4)
    project.objects[0].cells[0].native_layout = None
    slide = _slide(project, tmp_path / "empty.pptx")
    end = slide.find(".//a:tbl/a:tr/a:tc/a:txBody/a:p/a:endParaRPr", NS)
    assert end is not None
    assert int(end.get("sz", "0")) <= 100


@pytest.mark.parametrize("kind",["table","text"])
def test_edited_cell_does_not_recover_original_pdf_advances(project, tmp_path, kind):
    project = _project(project, advance=1,kind=kind)
    obj=project.objects[0]
    if isinstance(obj,Table):
        obj.cells[0].text="Edited"
    else:
        obj.text="Edited"
    slide = _slide(project, tmp_path / "edited.pptx")
    assert slide.findtext(".//a:t", default="", namespaces=NS) == "Edited"
    properties = slide.find(".//a:rPr", NS)
    assert properties is not None
    assert properties.get("spc", "0") == "0"


@pytest.mark.parametrize("mode", ["source_exact", "ocr_style"])
@pytest.mark.parametrize("kind",["table","text"])
def test_saved_noneditable_pdf_modes_keep_original_table_typography(project, tmp_path, mode, kind):
    project = _project(project,kind=kind)
    project.raw["native_pdf"]["overlay_mode"] = mode
    slide = _slide(project, tmp_path / (mode + ".pptx"))
    assert slide.find(".//a:tbl//a:noAutofit", NS) is None
    properties = slide.find(".//a:rPr", NS)
    assert properties is not None
    assert properties.get("spc", "0") == "0"


def test_slide_serialization_preserves_prefix_valued_namespace_declarations():
    from smart_overlay.export.native_tables import _parse_slide
    data = b'<p:sld xmlns:p="urn:p" xmlns:mc="urn:mc" xmlns:p14="urn:p14" mc:Ignorable="p14"/>'
    rendered = ET.tostring(_parse_slide(data))
    assert b'xmlns:p14="urn:p14"' in rendered
    assert b'mc:Ignorable="p14"' in rendered