from zipfile import ZipFile
import xml.etree.ElementTree as ET

from PIL import Image
import pytest

from smart_overlay.export import export_pptx
from smart_overlay.formatting import fitted_size, font_path
from smart_overlay.input import load_image
from smart_overlay.layout import text_layout
from smart_overlay.models import Box, Fragment, Text
from smart_overlay.pipeline import smart_ocr
from smart_overlay.settings import Settings
from smart_overlay.structure import base_texts, validate_project
from smart_overlay.tables import assign_fragments, new_table
from smart_overlay.vision import Analysis, Region, analysis_images


def fragment(identifier, text, x, y, w=30, h=12):
    return Fragment(id=identifier, text=text, bbox=Box(x=x, y=y, w=w, h=h),
                    polygon=[(x,y), (x+w,y), (x+w,y+h), (x,y+h)], confidence=.99)


@pytest.mark.parametrize('manual', [False, True])
def test_new_table_ocr_is_exported_without_losing_edits(tmp_path, monkeypatch, manual):
    path = tmp_path/'sheet.png'
    Image.new('RGB', (400,300), 'white').save(path)
    project = load_image(path)
    old = fragment('old', 'OLD', 65,65)
    project.fragments = [old]
    table = new_table(Box(x=50,y=50,w=200,h=100),1,1)
    assign_fragments(table, project.fragments)
    table.cells[0].text = 'MANUAL' if manual else 'OLD'
    project.objects = [table]
    replies = iter([Analysis(groups=[],tables=[],ignored={},regions=[
        Region(bbox=Box(x=60,y=85,w=80,h=25),reason='missed')]),
        Analysis(groups=[],tables=[],ignored={},regions=[])])
    monkeypatch.setattr('smart_overlay.pipeline.analyze', lambda p,c: (p.model_copy(deep=True),next(replies)))
    monkeypatch.setattr('smart_overlay.pipeline.refine', lambda *args: None)
    class Engine:
        def recognize(self, *args):
            return [fragment('new','NEW',65,90)], []
    result = smart_ocr(project,Settings(),'unused',tmp_path,True,engine=Engine())
    assert result.objects[0].cells[0].text == ('MANUAL\nNEW' if manual else 'OLD\nNEW')
    export_pptx(result,tmp_path/'table.pptx',Settings(),allow_unreviewed=True)
    with ZipFile(tmp_path/'table.pptx') as archive:
        xml=archive.read('ppt/slides/slide1.xml').decode()
        assert 'NEW' in xml and ('MANUAL' if manual else 'OLD') in xml


def test_empty_review_placeholder_does_not_hide_source(project,tmp_path):
    project.objects = [Text(id='missing',bbox=Box(x=10,y=10,w=40,h=20),text='',needs_review=True)]
    project.fragments=[]
    export_pptx(project,tmp_path/'empty.pptx',Settings(),allow_unreviewed=True)
    with ZipFile(tmp_path/'empty.pptx') as archive:
        root=ET.fromstring(archive.read('ppt/slides/slide1.xml'))
        assert not root.findall('.//{http://schemas.openxmlformats.org/presentationml/2006/main}sp')


def test_minority_overlap_stays_outside_table(project):
    f=fragment('neighbor','NEXT',249,60)
    table=new_table(Box(x=50,y=50,w=200,h=100),1,1)
    assign_fragments(table,[f])
    assert not table.ocr_ids and table.needs_review
    project.fragments=[f]
    project.objects=[table]
    base_texts(project)
    validate_project(project)
    obj=next(o for o in project.objects if isinstance(o,Text))
    _,masks=text_layout(project,obj)
    assert all(b.x >= table.bbox.right for b in masks)


def test_real_cyrillic_font_metrics_and_missing_font():
    assert font_path('Arial')
    settings=Settings(min_font=10,max_font=10)
    assert fitted_size('ШШШШШ',Box(x=0,y=0,w=70,h=28),10,settings)[1]
    assert fitted_size('ШШШШШ',Box(x=0,y=0,w=120,h=28),10,settings) == (10,False)
    assert fitted_size('text',Box(x=0,y=0,w=120,h=28),10,settings,font_face='Nonexistent Font 123')[1]


def test_masks_preserve_gap_between_merged_lines(project):
    obj=project.objects[0]
    _,masks=text_layout(project,obj)
    assert len(masks)==3
    assert masks[0].bottom < masks[1].y


def test_equal_overlap_between_adjacent_tables_has_no_duplicate_owner(project):
    f=fragment('edge','EDGE',90,60,w=20)
    tables=[new_table(Box(x=x,y=50,w=50,h=100),1,1) for x in (50,100)]
    for table in tables:
        assign_fragments(table,[f])
        assert not table.ocr_ids
    project.fragments=[f]
    project.objects=tables
    base_texts(project)
    validate_project(project)


def test_tile_description_matches_actual_crop(project):
    parts=analysis_images(project,Settings(tile_size=512))
    descriptions=[p['text'] for p in parts if p['type']=='text']
    assert any('(448, 0, 1088, 576)' in text for text in descriptions)


def test_formatting_preview_can_render(gui_app,project):
    from smart_overlay.app import MainWindow
    window=MainWindow()
    window.project=project
    window.formatting_preview.setChecked(True)
    assert window.preview.show_formatting
    assert len(window.preview.scene().items()) > len(project.objects)
    window.close()
