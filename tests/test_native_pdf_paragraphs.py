import pytest
import pymupdf

from smart_overlay.models import Text, NativeLayout, Box
from smart_overlay.native_pdf import native_pdf_project


def make_pdf(tmp_path, lines):
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=400)
    for x, y, text in lines:
        page.insert_text((x, y), text, fontsize=12)
    path = tmp_path / 'paragraphs.pdf'
    doc.save(path)
    doc.close()
    return path


def test_soft_wrap_is_a_space_and_physical_lines_survive(tmp_path):
    path = make_pdf(tmp_path, [(40, 60, 'A wrapped sentence'), (40, 77, 'continues here.')])
    project = native_pdf_project(path, detect_tables=False)
    obj = project.objects[0]
    assert isinstance(obj, Text)
    assert obj.native_layout is not None
    assert obj.native_layout.paragraphs is not None
    assert obj.text == 'A wrapped sentence continues here.'
    assert len(obj.native_layout.lines) == 2
    assert [''.join(r.text for r in p.runs) for p in obj.native_layout.paragraphs] == [obj.text]
    assert NativeLayout(frame=Box(x=0, y=0, w=10, h=10), lines=[]).paragraphs is None


def test_list_items_indents_and_large_gaps_are_real_paragraphs():
    from smart_overlay.models import NativeLine, NativeRun
    from smart_overlay.native_paragraphs import normalize_layout
    def line(text, spacing=14, indent=0):
        return NativeLine(runs=[NativeRun(text=text, font='Arial', font_size=12,
                          color='#000000')], line_spacing=spacing, indent=indent)
    layout = NativeLayout(frame=Box(x=0, y=0, w=200, h=200), lines=[
        line('1. First item'), line('continued'), line('2. Second item', spacing=30),
        line('New paragraph'), line('Indented paragraph', indent=15)])
    assert normalize_layout(layout) == '1. First item continued\n2. Second item\nNew paragraph\nIndented paragraph'


@pytest.mark.parametrize(('before', 'after', 'expected'), [
    ('discre\u00ad', 'tionary', 'discretionary'),
    ('well-', 'known', 'well-known'), ('AB-', '123', 'AB-123'),
    ('ordinary-', 'word', 'ordinary-word')])
def test_only_explicit_discretionary_hyphen_is_removed(before, after, expected):
    from smart_overlay.models import NativeLine, NativeRun
    from smart_overlay.native_paragraphs import normalize_layout
    lines = [NativeLine(runs=[NativeRun(text=t, font='Arial', font_size=12, color='#000000')])
             for t in (before, after)]
    layout = NativeLayout(frame=Box(x=0, y=0, w=100, h=100), lines=lines)
    assert normalize_layout(layout) == expected


def test_hanging_list_indent_is_a_wrap_not_a_new_paragraph():
    from smart_overlay.models import NativeLine, NativeRun
    from smart_overlay.native_paragraphs import normalize_layout
    def line(text, indent):
        return NativeLine(runs=[NativeRun(text=text, font='Arial', font_size=12,
                          color='#000000')], line_spacing=14, indent=indent)
    layout = NativeLayout(frame=Box(x=0, y=0, w=200, h=100), lines=[
        line('1. First item', 0), line('continued here', 15), line('2. Next item', 0)])
    assert normalize_layout(layout) == '1. First item continued here\n2. Next item'


def test_pdf_hanging_list_wraps_into_its_item(tmp_path):
    path = make_pdf(tmp_path, [(40, 60, '1. First item'),
                               (55, 77, 'continuation of first item'),
                               (40, 94, '2. Next item')])
    project = native_pdf_project(path, detect_tables=False)
    assert len(project.objects) == 1
    obj = project.objects[0]
    assert isinstance(obj, Text)
    assert obj.text == '1. First item continuation of first item\n2. Next item'


def test_columns_gaps_list_items_and_rotations_stay_separate(tmp_path):
    path = make_pdf(tmp_path, [(40, 60, 'Left column'), (40, 77, 'continues'),
                               (300, 60, 'Right column'), (300, 77, 'continues'),
                               (40, 150, '1. First item'), (40, 167, '2. Second item'),
                               (40, 240, 'Separate paragraph')])
    project = native_pdf_project(path, detect_tables=False)
    objects = [obj for obj in project.objects if isinstance(obj, Text)]
    texts = [obj.text for obj in objects]
    assert 'Left column continues' in texts
    assert 'Right column continues' in texts
    assert '1. First item\n2. Second item' in texts
    assert 'Separate paragraph' in texts
    assert all(obj.native_layout is not None and obj.native_layout.paragraphs is not None
               for obj in objects)


def test_normalization_keeps_rich_runs_and_physical_text(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=400)
    page.insert_text((40, 60), 'Normal ', fontsize=12)
    page.insert_text((81.34, 60), 'Bold', fontsize=12, fontname='hebo', color=(1, 0, 0))
    page.insert_text((40, 77), 'continuation', fontsize=12, fontname='heit')
    path = tmp_path / 'rich-paragraph.pdf'
    doc.save(path)
    doc.close()
    obj = native_pdf_project(path, detect_tables=False).objects[0]
    assert isinstance(obj, Text)
    assert obj.native_layout is not None
    assert obj.native_layout.paragraphs is not None
    assert obj.text == 'Normal Bold continuation'
    assert len(obj.native_layout.paragraphs) == 1
    runs = obj.native_layout.paragraphs[0].runs
    assert any(r.text.strip() == 'Bold' and r.bold and r.color == '#FF0000' for r in runs)
    assert runs[-1].italic
    assert ''.join(r.text for r in obj.native_layout.lines[0].runs) == 'Normal Bold'
    assert ''.join(r.text for r in obj.native_layout.lines[1].runs) == 'continuation'


def test_bare_numbered_notes_are_paragraphs_but_measurements_are_not():
    from smart_overlay.models import NativeLine,NativeRun
    from smart_overlay.native_paragraphs import normalize_layout
    def layout(texts):
        return NativeLayout(frame=Box(x=0,y=0,w=200,h=200),lines=[
            NativeLine(runs=[NativeRun(text=text,font='Arial',font_size=12,color='#000000')],
                       line_spacing=16) for text in texts])
    notes=layout(['1 Dimensions are for reference.', '2 **Check the general plan.',
                  '3 Manufacture and test', 'in accordance with standards.'])
    assert normalize_layout(notes)==('1 Dimensions are for reference.\n2 **Check the general plan.\n'
                                     '3 Manufacture and test in accordance with standards.')
    measures=layout(['Length 10 mm', '20 mm minimum'])
    assert normalize_layout(measures)=='Length 10 mm 20 mm minimum'


def test_pdf_bare_number_hanging_list_keeps_its_continuation(tmp_path):
    path=make_pdf(tmp_path,[(40,60,'1 First item'),(55,77,'continuation of first item'),
                          (40,94,'2 Next item')])
    project=native_pdf_project(path,detect_tables=False)
    assert len(project.objects)==1
    assert isinstance(project.objects[0],Text)
    assert project.objects[0].text=='1 First item continuation of first item\n2 Next item'


def test_font_size_change_preserves_heading_paragraph():
    from smart_overlay.models import NativeLine,NativeRun
    from smart_overlay.native_paragraphs import normalize_layout
    layout=NativeLayout(frame=Box(x=0,y=0,w=200,h=100),lines=[
        NativeLine(runs=[NativeRun(text='Heading',font='Arial',font_size=18,color='#000000')],line_spacing=20),
        NativeLine(runs=[NativeRun(text='Body sentence',font='Arial',font_size=12,color='#000000')],line_spacing=16),
        NativeLine(runs=[NativeRun(text='continues.',font='Arial',font_size=12,color='#000000')])])
    assert normalize_layout(layout)=='Heading\nBody sentence continues.'

