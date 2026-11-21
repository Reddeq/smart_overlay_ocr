"""Duplicate PDF drawing operations must not duplicate editable text."""
import base64
from collections import Counter
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pytest

pymupdf=pytest.importorskip('pymupdf')

from smart_overlay.export import export_pptx
from smart_overlay.models import Table,Text
from smart_overlay.native_pdf import native_pdf_project,preview_pdf
from smart_overlay.settings import Settings
from smart_overlay.structure import validate_project


@pytest.mark.parametrize('detect_tables',[False,True])
@pytest.mark.parametrize('ignore_numbers',[False,True])
def test_duplicate_layers_clean_labels_and_cells_but_never_source(tmp_path,detect_tables,ignore_numbers):
    doc=pymupdf.open()
    page=doc.new_page(width=600,height=500)
    for copies,x,y,text in ((2,40,50,'VALVE'),(1,240,50,'VALVE'),
                             (2,40,100,'987654321'),(1,40,150,'book 1000')):
        for _ in range(copies):page.insert_text((x,y),text,fontsize=12)
    for x in (50,150,250):page.draw_line((x,240),(x,320))
    for y in (240,280,320):page.draw_line((50,y),(250,y))
    for x,y,text in ((60,265,'CODE'),(160,265,'VALUE'),(60,305,'1000'),(160,305,'1000')):
        for _ in range(2):page.insert_text((x,y),text,fontsize=10)
    source=tmp_path/'duplicate-layer.pdf'
    doc.save(source)
    doc.close()
    before=source.read_bytes()
    raster=preview_pdf(source).source.raster_b64
    project=native_pdf_project(source,detect_tables=detect_tables,ignore_numbers=ignore_numbers)
    assert source.read_bytes()==before
    assert base64.b64decode(project.source.original_b64)==before
    assert project.source.raster_b64==raster
    labels=[o for o in project.objects if isinstance(o,Text)]
    assert sum(o.text=='VALVE' for o in labels)==2
    assert sum(o.text=='987654321' for o in labels)==(0 if ignore_numbers else 1)
    assert any(o.text=='book 1000' for o in labels)
    diagnostics=project.raw['native_pdf']['deduplication']
    assert diagnostics['removed_glyphs']>0
    assert diagnostics['removed_lines']>0
    validate_project(project)
    if detect_tables:
        tables=[o for o in project.objects if isinstance(o,Table)]
        assert len(tables)==1
        assert Counter(c.text for c in tables[0].cells)==Counter(['CODE','VALUE','1000','1000'])
    output=tmp_path/'deduplicated.pptx'
    export_pptx(project,output,Settings(inches_per_pixel=1/144),allow_unreviewed=True)
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
    with ZipFile(output) as archive:
        slide=ET.fromstring(archive.read('ppt/slides/slide1.xml'))
        texts=[''.join(t.text or '' for t in shape.findall('.//a:t',ns)) for shape in slide.findall('.//p:sp',ns)]
        assert texts.count('VALVE')==2
        assert texts.count('987654321')==(0 if ignore_numbers else 1)
        assert any(archive.read(n)==base64.b64decode(raster) for n in archive.namelist() if n.startswith('ppt/media/'))


def test_duplicates_do_not_repeat_across_pdf_pages(tmp_path):
    document=pymupdf.open()
    for _ in range(2):
        page=document.new_page(width=300,height=200)
        for _ in range(2):page.insert_text((40,60),'PAGE LABEL',fontsize=12)
    source=tmp_path/'pages.pdf'
    document.save(source)
    document.close()
    for number in (1,2):
        project=native_pdf_project(source,number,detect_tables=False)
        assert [o.text for o in project.objects if isinstance(o,Text)]==['PAGE LABEL']


def test_exact_duplicates_with_changed_span_segmentation(tmp_path):
    document=pymupdf.open()
    page=document.new_page(width=400,height=200)
    page.insert_text((40,60),'SAME LABEL',fontsize=12)
    page.insert_text((40,60),'SAME ',fontsize=12)
    width=pymupdf.get_text_length('SAME ',fontsize=12)
    page.insert_text((40+width,60),'LABEL',fontsize=12)
    source=tmp_path/'segmented.pdf'
    document.save(source)
    document.close()
    project=native_pdf_project(source,detect_tables=False)
    assert [o.text for o in project.objects if isinstance(o,Text)]==['SAME LABEL']


def test_partial_overlap_preserves_and_marks_both_complete_texts(tmp_path):
    document=pymupdf.open()
    page=document.new_page(width=400,height=200)
    page.insert_text((40,60),'ABCD',fontname='cour',fontsize=12)
    width=pymupdf.get_text_length('AB',fontname='cour',fontsize=12)
    page.insert_text((40+width,60),'CDEF',fontname='cour',fontsize=12)
    source=tmp_path/'partial-overlap.pdf'
    document.save(source)
    document.close()
    project=native_pdf_project(source,detect_tables=False)
    labels=[o for o in project.objects if isinstance(o,Text)]
    assert Counter(o.text for o in labels)==Counter(['ABCD','CDEF'])
    assert all(o.needs_review for o in labels)
    assert project.raw['native_pdf']['deduplication']['removed_glyphs']==0
