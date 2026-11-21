"""Translation-friendly native PDF -> genuine PowerPoint XML contract."""
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import pytest

pymupdf=pytest.importorskip('pymupdf')

from smart_overlay.export import export_pptx
from smart_overlay.models import Table,Text
from smart_overlay.native_pdf import native_pdf_project
from smart_overlay.settings import Settings
from smart_overlay.structure import validate_project

NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'p':'http://schemas.openxmlformats.org/presentationml/2006/main'}


@pytest.mark.parametrize('detect_tables',[False,True])
def test_numeric_labels_stay_png_only_while_cells_and_paragraphs_remain_live(tmp_path,detect_tables):
    document=pymupdf.open()
    page=document.new_page(width=600,height=500)
    for x,y,text in ((30,50,'987654321'),(30,85,'DN50'),
                     (30,125,'Apply pressure in two'),(30,141,'steps. Check the valve.'),
                     (30,180,'Repeat after inspection.')):
        page.insert_text((x,y),text,fontsize=12)
    for x in (60,220,380):page.draw_line((x,250),(x,340))
    for y in (250,280,310,340):page.draw_line((60,y),(380,y))
    for x,y,text in ((70,270,'Item'),(230,270,'Quantity'),
                     (70,300,'Valve'),(230,300,'1234'),
                     (70,330,'Pipe'),(230,330,'5678')):
        page.insert_text((x,y),text,fontsize=10)
    source=tmp_path/'cat-friendly.pdf'
    document.save(source)
    document.close()
    plain=native_pdf_project(source,detect_tables=detect_tables)
    project=native_pdf_project(source,detect_tables=detect_tables,ignore_numbers=True)
    assert plain.source.raster_b64==project.source.raster_b64
    assert any(isinstance(obj,Text) and obj.text=='987654321' for obj in plain.objects)
    assert not any(isinstance(obj,Text) and obj.text=='987654321' for obj in project.objects)
    paragraph=next(obj for obj in project.objects if isinstance(obj,Text) and 'Apply pressure' in obj.text)
    assert paragraph.text=='Apply pressure in two steps. Check the valve.'
    assert paragraph.native_layout is not None
    assert len(paragraph.native_layout.lines)==2
    assert paragraph.native_layout.paragraphs is not None
    assert len(paragraph.native_layout.paragraphs)==1
    validate_project(project)
    output=tmp_path/'cat-friendly.pptx'
    export_pptx(project,output,Settings(inches_per_pixel=1/144),allow_unreviewed=True)
    with ZipFile(output) as archive:
        slide=ET.fromstring(archive.read('ppt/slides/slide1.xml'))
    texts=[node.text for node in slide.findall('.//a:t',NS)]
    assert '987654321' not in texts
    assert 'DN50' in texts
    assert slide.find('.//a:br',NS) is None
    assert slide.find('.//a:normAutofit',NS) is None
    assert all(int(props.get('spc','0'))>=0 for props in slide.findall('.//a:rPr',NS))
    shapes={name.get('name'):s for s in slide.findall('.//p:sp',NS)
            if (name:=s.find('p:nvSpPr/p:cNvPr',NS)) is not None}
    shape=shapes[paragraph.id]
    assert len(shape.findall('p:txBody/a:p',NS))==1
    if detect_tables:
        tables=[obj for obj in project.objects if isinstance(obj,Table)]
        assert len(tables)==1
        assert {cell.text for cell in tables[0].cells}>={'1234','5678'}
        xmltable=slide.find('.//a:tbl',NS)
        assert xmltable is not None
        assert {'1234','5678'}<=set(xmltable.itertext())
