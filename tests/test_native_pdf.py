"""Born-digital PDF -> PPTX: vector background, real text, native merged table.

These tests synthesize *genuine PDF text operators and vector lines*.
There is no OCR network or pretrained-model dependency.
"""
import base64
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image, ImageChops

pymupdf=pytest.importorskip("pymupdf")

from smart_overlay.input import source_image, open_project, save_project
from smart_overlay.models import Table, Text
from smart_overlay.native_pdf import native_pdf_project, preview_pdf, pdf_info, NativePDFError
from smart_overlay.one_click import convert_drawing
from smart_overlay.settings import Settings
from smart_overlay.structure import validate_project


@pytest.fixture
def live_pdf(tmp_path):
    doc=pymupdf.open()
    page=doc.new_page(width=600,height=400)
    # An actual line crosses a printed label; it MUST survive removal of text.
    page.draw_line((20,70),(580,70),color=(0,0,0),width=1)
    page.insert_text((22,69),"CONTROL PANEL",fontsize=14)
    # A two-column table whose header cell is physically merged.
    x0,x1,x2=60,185,350
    y0,y1,y2,y3=110,143,176,209
    for y in (y0,y1,y2,y3):
        page.draw_line((x0,y),(x2,y),color=(0,0,0),width=1)
    page.draw_line((x0,y0),(x0,y3),color=(0,0,0),width=1)
    page.draw_line((x2,y0),(x2,y3),color=(0,0,0),width=1)
    page.draw_line((x1,y1),(x1,y3),color=(0,0,0),width=1)
    for x,y,text in [(72,130,"TERMINALS"),(72,163,"8"),
                     (198,163,"POWER"),(72,195,"7"),(198,195,"GROUND")]:
        page.insert_text((x,y),text,fontsize=10)
    page.insert_text((90,275),"Power supply",fontsize=11)
    page.insert_text((90,290),"Module A",fontsize=11)
    # Keep two pages to verify one-based page addressing.
    page2=doc.new_page(width=600,height=400)
    page2.insert_text((30,60),"PAGE TWO",fontsize=13)
    target=tmp_path/"electrical.pdf"
    doc.save(target)
    doc.close()
    return target


def test_pdf_page_info_and_preview(live_pdf):
    info=pdf_info(live_pdf)
    assert info["pages"]==2
    assert info["page_sizes"][0]==[600,400]
    preview=preview_pdf(live_pdf,1)
    assert preview.source.mode=="PDF"
    assert (preview.source.width,preview.source.height)==(1200,800)
    assert not preview.objects
    assert Image.open(__import__("io").BytesIO(base64.b64decode(
        preview.source.raster_b64))).size==(1200,800)


def test_native_pdf_extracts_real_multiline_labels_and_merged_cells(live_pdf):
    project=native_pdf_project(live_pdf)
    assert project.source.mode=="PDF"
    assert len(project.fragments)>=7
    assert project.raw["native_pdf"]["page"]==1
    assert project.raw["native_pdf"]["pages"]==2
    assert project.raw["native_pdf"]["text_redacted_background"] is False
    assert project.raw["native_pdf"]["background_mode"] == "full_original"
    assert project.raw["native_pdf"]["vector_paths"]>0
    tables=[o for o in project.objects if isinstance(o,Table)]
    assert len(tables)>=1
    table=tables[0]
    assert (table.rows,table.columns)==(3,2)
    assert next(c for c in table.cells if c.row==0 and c.col==0).colspan==2
    assert any(c.text=="POWER" for c in table.cells)
    blocks=[o for o in project.objects if isinstance(o,Text)]
    assert any("CONTROL PANEL" in o.text for o in blocks)
    assert all(o.style.border_pt==0 for o in tables)
    validate_project(project)
    assert sum(len(o.ocr_ids) for o in project.objects)==len(project.fragments)


def test_background_keeps_every_pixel_of_original_drawing_including_glyphs(live_pdf):
    project=native_pdf_project(live_pdf)
    result=source_image(project)
    full=source_image(preview_pdf(live_pdf))
    assert result.size==full.size==(1200,800)
    # The full original image is the PPTX underlay, not a redacted PDF.
    assert ImageChops.difference(result,full).getbbox() is None
    assert min(result.getpixel((80,140)))<180  # crossing vector line survives
    assert any(min(result.getpixel((x,y)))<180
               for y in range(105,135) for x in range(47,260))
    assert project.raw["native_pdf"]["text_redacted_background"] is False
    assert project.raw["native_pdf"]["background_mode"] == "full_original"


def test_pdf_second_page_and_invalid_rotation_are_explicit(live_pdf,tmp_path):
    second=native_pdf_project(live_pdf,page_number=2)
    assert "PAGE TWO" in "\n".join(o.text for o in second.objects if isinstance(o,Text))
    assert second.raw["native_pdf"]["page"]==2
    with pytest.raises(NativePDFError,match="отсутствует"):
        native_pdf_project(live_pdf,page_number=3)
    document=pymupdf.open()
    page=document.new_page(width=300,height=200)
    page.insert_text((20,60),"ROTATED",fontsize=12)
    page.set_rotation(90)
    rotated=tmp_path/"rotated.pdf"
    document.save(rotated)
    with pytest.raises(NativePDFError,match="Rotate"):
        native_pdf_project(rotated)


def test_scanned_pdf_rejected_before_ocr(tmp_path):
    from PIL import ImageDraw
    import io
    image=Image.new("RGB",(400,200),"white")
    ImageDraw.Draw(image).text((20,50),"SCANNED PICTURE",fill="black")
    buf=io.BytesIO()
    image.save(buf,format="PNG")
    document=pymupdf.open()
    page=document.new_page(width=400,height=200)
    page.insert_image(pymupdf.Rect(0,0,400,200),stream=buf.getvalue())
    path=tmp_path/"scan.pdf"
    document.save(path)
    with pytest.raises(NativePDFError,match="нет извлекаемого текста"):
        native_pdf_project(path)


def test_legacy_text_cleaned_pdf_project_still_opens(live_pdf,tmp_path):
    from smart_overlay.native_pdf import _clean_background
    project=native_pdf_project(live_pdf)
    original=base64.b64decode(project.source.original_b64)
    cleaned=_clean_background(original,1,2.0)
    project.source.raster_b64=base64.b64encode(cleaned).decode("ascii")
    project.raw["native_pdf"]["background_mode"]="legacy_cleaned"
    project.raw["native_pdf"]["text_redacted_background"]=True
    dest=tmp_path/"legacy.socr.json"
    save_project(project,dest)
    loaded=open_project(dest)
    assert loaded.raw["native_pdf"]["background_mode"]=="legacy_cleaned"
    assert ImageChops.difference(source_image(loaded),
                                 source_image(preview_pdf(live_pdf))).getbbox()


def test_native_pdf_project_roundtrip_validates_source_pdf(live_pdf,tmp_path):
    project=native_pdf_project(live_pdf)
    file=tmp_path/"editable.socr.json"
    save_project(project,file)
    loaded=open_project(file)
    assert loaded==project
    assert any(o.text=="POWER" for o in loaded.objects
               if isinstance(o,Table) for o in o.cells)


def test_native_pdf_one_click_does_not_contact_ocr_or_cloud(live_pdf,tmp_path,monkeypatch):
    monkeypatch.setattr("smart_overlay.one_click.load_image",
                        lambda *a,**kw:pytest.fail("Raster OCR path forbidden"))
    monkeypatch.setattr("smart_overlay.one_click.local_ocr",
                        lambda *a,**kw:pytest.fail("Paddle OCR forbidden"))
    monkeypatch.setattr("smart_overlay.one_click._smart",
                        lambda *a,**kw:pytest.fail("Network forbidden"))
    monkeypatch.setattr("smart_overlay.one_click.data_dir",lambda:tmp_path/"app")
    exported=[]
    def fake_export(project,output,settings,*,allow_unreviewed,inline_text_fill):
        validate_project(project)
        exported.append((project,settings))
        Path(output).write_bytes(b"native pptx stub")
    monkeypatch.setattr("smart_overlay.one_click.export_pptx",fake_export)
    result=convert_drawing(live_pdf,settings=Settings(),pdf_page=1)
    assert result.pptx.read_bytes()==b"native pptx stub"
    assert result.project.source.mode=="PDF"
    assert exported[0][1].inches_per_pixel==pytest.approx(1/144)
    assert any(o.text=="POWER" for o in result.project.objects
               if isinstance(o,Table) for o in o.cells)
    assert result.project_file.exists()
    assert result.warnings==[]


def test_bad_pdf_header_and_empty_table_do_not_guess(tmp_path):
    fake=tmp_path/"drawing.pdf"
    fake.write_bytes(b"this is not a PDF")
    with pytest.raises(NativePDFError,match="заголовка PDF"):
        native_pdf_project(fake)


def test_real_pptx_has_editable_pdf_text_and_native_merged_table(live_pdf,tmp_path):
    """Inspects real PresentationML, not a mocked PowerPoint export."""
    import shutil
    import xml.etree.ElementTree as ET
    from smart_overlay.export import export_pptx
    if not shutil.which("node"):
        pytest.skip("Node.js is not available")
    project=native_pdf_project(live_pdf)
    native_texts=[o for o in project.objects if isinstance(o,Text)]
    native_tables=[o for o in project.objects if isinstance(o,Table)]
    dest=tmp_path/"digital.pptx"
    export_pptx(project,dest,Settings(inches_per_pixel=1/144),
                allow_unreviewed=True,inline_text_fill=True)
    namespaces={"p":"http://schemas.openxmlformats.org/presentationml/2006/main",
                "a":"http://schemas.openxmlformats.org/drawingml/2006/main"}
    with ZipFile(dest) as zipfile:
        root=ET.fromstring(zipfile.read("ppt/slides/slide1.xml"))
        shape_nodes=root.findall(".//p:sp",namespaces)
        table_nodes=root.findall(".//a:tbl",namespaces)
        object_names=root.xpath(".//p:sp/p:nvSpPr/p:cNvPr/@name",namespaces)
        masks=[name for name in object_names if "_mask_" in name]
        # Editable text shapes PLUS per-source-glyph masks, with no
        # cleaned background or opaque whole-cell rectangle.
        assert len(shape_nodes)==len(native_texts)+len(masks)
        assert len(masks)>=len(native_texts)
        assert len(table_nodes)==len(native_tables)
        assert table_nodes[0].find(".//a:tc[@gridSpan='2']",namespaces) is not None
        source=zipfile.read("ppt/slides/slide1.xml").decode("utf-8")
        assert "CONTROL PANEL" in source
        assert "POWER" in source
        assert "_mask_" in source
        # PowerPoint keeps the actual PDF font family name, but the full
        # unmodified source glyphs also remain in its picture underlay.
        assert "<a:solidFill>" in source
        assert any(name.startswith("ppt/media/") and name.lower().endswith(".png")
                   for name in zipfile.namelist())
