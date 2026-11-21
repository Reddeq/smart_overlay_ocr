"""Opt-in acceptance test against the real Office table layout engine.

Set SMART_OVERLAY_POWERPOINT_PDF to a native PDF drawing to exercise the
installed PowerPoint; normal platform-independent tests do not launch Office.
"""
import json
import os
from pathlib import Path
import base64
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pytest

from smart_overlay.export import export_pptx
from smart_overlay.settings import Settings
from smart_overlay.models import Box, Cell, NativeLayout, NativeLine, NativeRun, Style, Table, Text
from office_pdf_probe import measure_in_powerpoint


NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


def _text(value):
    return "".join(value.split())


def test_native_pdf_structure_and_natural_fonts_survive_real_powerpoint(tmp_path):
    source = os.environ.get("SMART_OVERLAY_POWERPOINT_PDF")
    if not source:
        pytest.skip("Set SMART_OVERLAY_POWERPOINT_PDF for real Office verification")

    from smart_overlay.native_pdf import native_pdf_project

    project = native_pdf_project(Path(source),ignore_numbers=True)
    output = tmp_path / "native-tables.pptx"
    settings = Settings()
    export_pptx(project, output, settings, allow_unreviewed=True)
    scale = settings.inches_per_pixel * 72
    expected = {obj.id: [height * scale for height in obj.row_heights]
                for obj in project.objects if obj.type == "table" and not obj.excluded}

    tables = {obj.id: obj for obj in project.objects if obj.type == "table"}

    with ZipFile(output) as archive:
        slide = ET.fromstring(archive.read("ppt/slides/slide1.xml"))
        for frame in slide.findall(".//p:graphicFrame", NS):
            table = frame.find(".//a:tbl", NS)
            if table is None:
                continue
            properties = frame.find("p:nvGraphicFramePr/p:cNvPr", NS)
            assert properties is not None
            name = properties.attrib["name"]
            declared = [int(row.attrib["h"]) / 12700 for row in table.findall("a:tr", NS)]
            assert len(declared)==tables[name].rows
            assert all(after>=before-.001 for after,before in zip(declared,expected[name]))
            assert len(table.findall("a:tblGrid/a:gridCol",NS))==tables[name].columns
            # Natural text may expand rows; topology, content and visible
            # point sizes must remain intact, including numeric table cells.
            for cell in tables[name].cells:
                xmlcell = table.findall("a:tr", NS)[cell.row].findall("a:tc", NS)[cell.col]
                assert int(xmlcell.get("rowSpan","1"))==cell.rowspan
                assert int(xmlcell.get("gridSpan","1"))==cell.colspan
                assert _text("".join(t.text or "" for t in xmlcell.findall(".//a:t",NS)))==_text(cell.text)
                if not cell.native_layout:
                    continue
                runs = xmlcell.findall("a:txBody/a:p/a:r", NS)
                paragraphs=cell.native_layout.paragraphs
                lines=paragraphs if paragraphs is not None else cell.native_layout.lines
                original = [run for line in lines for run in line.runs]
                assert [run.findtext("a:t", default="", namespaces=NS).strip() for run in runs] == [run.text.strip() for run in original]
                properties = [run.find("a:rPr", NS) for run in runs]
                assert all(prop is not None for prop in properties)
                assert [int(prop.attrib["sz"]) / 100 for prop in properties if prop is not None] == pytest.approx(
                    [run.font_size * project.raw["native_pdf"]["zoom"] * scale for run in original], abs=0.011)
        assert slide.find(".//a:normAutofit", NS) is None
        assert all(int(prop.get("spc","0"))>=0 for prop in slide.findall(".//a:rPr",NS))
        assert slide.find(".//a:lnSpc/a:spcPts",NS) is None
        assert any(archive.read(n)==base64.b64decode(project.source.raster_b64)
                   for n in archive.namelist() if n.startswith("ppt/media/"))
    assert expected, "The fixture must contain native editable tables"
    report=measure_in_powerpoint(output,project,tmp_path)
    actual=report["tables"]
    assert {item["name"] for item in actual} == set(expected)
    for item in actual:
        before = expected[item["name"]]
        model=tables[item["name"]]
        assert (item["rows"],item["columns"])==(model.rows,model.columns)
        assert all(opened>=declared-.15 for declared,opened in zip(before,item["heights"]))
        assert item["height"]>=sum(before)-.15
        opened={(c["row"]-1,c["col"]-1):c["text"] for c in item["cells"]}
        assert len(opened)==len(model.cells)
        for cell in model.cells:
            assert _text(opened[cell.row,cell.col])==_text(cell.text)


def test_native_text_frames_grow_in_real_powerpoint(project,tmp_path):
    style=Style(font="Arial",font_size=12)
    text="This sentence needs several naturally wrapped lines. "*8
    frame=Box(x=40,y=40,w=240,h=8)
    run=NativeRun(text=text,font="Arial",font_size=12,color="#000000")
    layout=NativeLayout(frame=frame,lines=[NativeLine(runs=[run])],reference_style=style)
    project.source.mode="PDF"
    project.raw={"native_pdf":{"background_mode":"full_original","overlay_mode":"editable","zoom":2}}
    project.fragments=[]
    project.ignored={}
    project.objects=[Text(id="growing",bbox=frame,text=text,style=style,native_layout=layout)]
    output=tmp_path/"growing.pptx"
    settings=Settings()
    export_pptx(project,output,settings)
    report=measure_in_powerpoint(output,project,tmp_path)
    label=next(item for item in report["labels"] if item["name"]=="growing")
    assert _text(label["text"])==_text(text)
    assert label["font_size"]==pytest.approx(12*2*settings.inches_per_pixel*72,abs=.011)
    assert label["height"]>frame.h*settings.inches_per_pixel*72+10
    import pymupdf
    with pymupdf.open(tmp_path/"powerpoint.pdf") as document:
        assert _text(text) in _text(document[0].get_text())
