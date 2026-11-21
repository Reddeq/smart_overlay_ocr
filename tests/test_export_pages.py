"""Multi-slide exports use each page's own image, objects and native layout."""
import base64
import posixpath
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pytest

pymupdf=pytest.importorskip('pymupdf')

from smart_overlay.native_pdf import native_pdf_project
from smart_overlay.models import Table
from smart_overlay.settings import Settings

NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}


def _projects(tmp_path,sizes=((400,300),(400,300),(400,300))):
    source=tmp_path/'pages.pdf'
    with pymupdf.open() as doc:
        for number,(width,height) in enumerate(sizes,1):
            page=doc.new_page(width=width,height=height)
            page.insert_text((30,50),f'PAGE {number}',fontsize=14+number)
            for x in (30,150,270):page.draw_line((x,120),(x,150+30*number))
            for y in range(120,151+30*number,30):page.draw_line((30,y),(270,y))
            for row in range(number+1):
                page.insert_text((40,140+30*row),f'Item {row}',fontsize=10)
                page.insert_text((160,140+30*row),f'{number}00{row}',fontsize=10)
        doc.save(source)
    projects=[native_pdf_project(source,n,ignore_numbers=True) for n in range(1,len(sizes)+1)]
    # Object names are slide-local. Distinct page payloads must never be merged
    # into a global name -> object map during native typography processing.
    for project in projects:
        for obj in project.objects:
            obj.id='same-table' if isinstance(obj,Table) else 'same-label'
    return projects


def test_same_format_pages_share_one_deck_without_cross_page_layout(tmp_path):
    from smart_overlay.export import export_pptx_pages
    projects=_projects(tmp_path)
    output=tmp_path/'pages.pptx'
    export_pptx_pages((p for p in projects),output,Settings(inches_per_pixel=1/144),allow_unreviewed=True)
    with ZipFile(output) as archive:
        presentation=ET.fromstring(archive.read('ppt/presentation.xml'))
        assert len(presentation.findall('p:sldIdLst/p:sldId',NS))==len(projects)
        for number,project in enumerate(projects,1):
            slide=ET.fromstring(archive.read(f'ppt/slides/slide{number}.xml'))
            assert len(slide.findall('.//a:tbl',NS))==1
            assert len(slide.findall('.//a:tbl/a:tr',NS))==number+1
            assert f'PAGE {number}' in ''.join(slide.itertext())
            assert all(int(props.get('spc','0'))>=0 for props in slide.findall('.//a:rPr',NS))
            assert slide.find('.//a:normAutofit',NS) is None
            face=slide.find('.//p:sp/p:txBody/a:p/a:r/a:rPr/a:latin',NS)
            assert face is not None and face.get('typeface')=='Arial'
            texts=[''.join(cell.itertext()) for cell in slide.findall('.//a:tbl/a:tr/a:tc/a:txBody',NS)]
            for table in (o for o in project.objects if isinstance(o,Table)):
                assert texts==[cell.text for cell in table.cells]
            image=slide.find('.//p:pic/p:blipFill/a:blip',NS)
            assert image is not None
            rid=image.get('{'+NS['r']+'}embed')
            assert rid is not None
            rels=ET.fromstring(archive.read(f'ppt/slides/_rels/slide{number}.xml.rels'))
            target=next(rel.get('Target') for rel in rels if rel.get('Id')==rid)
            assert target is not None
            image_name=posixpath.normpath(posixpath.join('ppt/slides',target))
            assert archive.read(image_name)==base64.b64decode(project.source.raster_b64)


def test_multi_export_rejects_mixed_formats_before_replacing_output(tmp_path):
    from smart_overlay.export import export_pptx_pages
    projects=_projects(tmp_path,((400,300),(300,400)))
    output=tmp_path/'keep.pptx'
    output.write_bytes(b'Existing user file')
    with pytest.raises(ValueError,match='формат|размер'):
        export_pptx_pages(projects,output,Settings(inches_per_pixel=1/144),allow_unreviewed=True)
    assert output.read_bytes()==b'Existing user file'


def test_multi_export_requires_at_least_one_page(tmp_path):
    from smart_overlay.export import export_pptx_pages
    output=tmp_path/'empty.pptx'
    with pytest.raises(ValueError):
        export_pptx_pages(iter(()),output,Settings())
    assert not output.exists()
