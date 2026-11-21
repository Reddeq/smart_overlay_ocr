"""Exercise the logical-paragraph DTO through the real Node generator."""
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pytest

from smart_overlay.export import resolve_node
from smart_overlay.models import Box, Cell, NativeLayout, NativeLine, NativeRun, Style, Table, Text
from smart_overlay.settings import Settings

NS={"a":"http://schemas.openxmlformats.org/drawingml/2006/main",
    "p":"http://schemas.openxmlformats.org/presentationml/2006/main"}


@pytest.mark.parametrize("kind",["text","table"])
@pytest.mark.parametrize("edited",[False,True])
def test_logical_paragraphs_keep_styles_without_pdf_line_breaks(project,tmp_path,kind,edited):
    style=Style(font="Arial",font_size=12)
    runs=[NativeRun(text="First ",font="Arial",font_size=12,color="#000000"),
          NativeRun(text="styled",font="Arial",font_size=12,color="#FF0000",bold=True),
          NativeRun(text=" paragraph.",font="Arial",font_size=12,color="#000000")]
    last=NativeRun(text="Second paragraph.",font="Arial",font_size=12,color="#000000")
    frame=Box(x=40,y=40,w=300,h=100)
    layout=NativeLayout(frame=frame,lines=[NativeLine(runs=runs[:2]),
        NativeLine(runs=runs[2:]),NativeLine(runs=[last])],reference_style=style)
    text="First styled paragraph.\nSecond paragraph."
    if kind=="text":
        obj=Text(id="logical",bbox=frame,text=text,style=style,native_layout=layout)
    else:
        cell=Cell(row=0,col=0,text=text,style=style,native_layout=layout)
        obj=Table(id="logical",bbox=frame,rows=1,columns=1,row_heights=[100],
                  col_widths=[300],cells=[cell])
    project.objects=[obj]
    payload=project.model_dump()
    payload["source"]["mode"]="PDF"
    payload.update(native_pdf_background_mode="full_original",native_pdf_overlay_mode="editable",
                   native_pdf_zoom=2,settings=Settings().model_dump(),allow_unreviewed=True)
    target=payload["objects"][0]
    if kind=="table":target=target["cells"][0]
    # Generator contract is independent of Python model migration. These are
    # test inputs, not fabricated exporter output or stand-in dependencies.
    target["native_layout"]["paragraphs"]=[{"runs":[r.model_dump() for r in runs]},
                                          {"runs":[last.model_dump()]}]
    if edited:target["text"]="Edited text"
    data=tmp_path/"payload.json"
    data.write_text(json.dumps(payload),encoding="utf8")
    output=tmp_path/"paragraphs.pptx"
    generator=Path(__file__).resolve().parents[1]/"smart_overlay/export/generate.cjs"
    subprocess.run([resolve_node(),str(generator),str(data),str(output)],check=True,
                   capture_output=True,text=True,timeout=30)
    with ZipFile(output) as archive:
        slide=ET.fromstring(archive.read("ppt/slides/slide1.xml"))
    body=slide.find(".//p:sp/p:txBody" if kind=="text" else ".//a:tc/a:txBody",NS)
    assert body is not None
    paragraphs=body.findall("a:p",NS)
    assert len(paragraphs)==(1 if edited else 2)
    actual=["".join(t.text or "" for t in p.findall("a:r/a:t",NS)) for p in paragraphs]
    assert actual==(["Edited text"] if edited else text.split("\n"))
    assert body.find(".//a:br",NS) is None
    if not edited:
        styled=next(r for r in body.findall(".//a:r",NS) if r.findtext("a:t",namespaces=NS)=="styled")
        properties=styled.find("a:rPr",NS)
        assert properties is not None and properties.get("b")=="1"
        color=properties.find("a:solidFill/a:srgbClr",NS)
        assert color is not None and color.get("val")=="FF0000"
        assert body.find(".//a:lnSpc/a:spcPts",NS) is None
