"""CAD vector-grid tests: no raster table OCR and no client PDF fixture."""
import pymupdf
import pytest

from smart_overlay.native_pdf import _font_name, native_pdf_project
from smart_overlay.models import Table
from smart_overlay.vector_grid import discover_vector_grids


def _cad_table(path):
    doc=pymupdf.open()
    page=doc.new_page(width=1300,height=900)
    xs=[250,285,445,490,535,585,780,995]  # 7 actual columns
    ys=[80,145,165,185,205,225,245,265,285,305,325]  # 10 real rows
    for y in ys:
        for i in range(len(xs)-1):
            page.draw_line((xs[i],y),(xs[i+1],y),
                           color=(0,0,0),width=.5)
    for i,x in enumerate(xs):
        for j in range(len(ys)-1):
            if i==1 and j==0:
                continue  # physical merged header across columns 0 and 1
            page.draw_line((x,ys[j]),(x,ys[j+1]),
                           color=(0,0,0),width=.5)
    page.insert_text((260,112),"Number",fontsize=11)
    page.insert_text((295,112),"Designation",fontsize=11)
    for row in range(1,10):
        page.insert_text((260,ys[row]+13),str(row),fontsize=11)
        page.insert_text((300,ys[row]+13),"Valve",fontsize=11)
        page.insert_text((457,ys[row]+13),"1",fontsize=11)
        page.insert_text((500,ys[row]+13),"DN80",fontsize=11)
        page.insert_text((610,ys[row]+13),"Steel",fontsize=11)
    doc.save(path)
    doc.close()


def test_cad_row_and_column_boundaries_not_misidentified(tmp_path):
    path=tmp_path/"vector.pdf"
    _cad_table(path)
    with pymupdf.open(path) as doc:
        grids=discover_vector_grids(doc[0])
    assert grids
    first=next(grid for grid in grids
               if len(grid.xedges)==8 and len(grid.yedges)==11)
    assert len(first.cells)==69  # 70 base cells minus 1 horizontal merge
    assert next(c for c in first.cells
                if c.row==0 and c.col==0).colspan==2
    assert first.bbox==pytest.approx((250,80,995,325),abs=1.0)


def test_vector_geometry_builds_native_table_and_keeps_original_font_names(tmp_path):
    path=tmp_path/"vector.pdf"
    _cad_table(path)
    project=native_pdf_project(path)
    grid=next(obj for obj in project.objects
              if isinstance(obj,Table) and obj.columns==7)
    assert (grid.rows,grid.columns)==(10,7)
    assert any(c.text=="Valve" for c in grid.cells)
    assert all(c.style.font=="Helvetica" for c in grid.cells if c.text)
    assert _font_name("HPDFAA+GOSTTypeA")=="GOSTTypeA"
    assert _font_name("HPDFAC+Calibri-Italic,Italic")=="Calibri-Italic"
    assert project.raw["native_pdf"]["table_detection"]["vector_candidates"]>=1


def test_narrow_side_table_is_not_rejected_by_page_width(tmp_path):
    doc=pymupdf.open()
    page=doc.new_page(width=900,height=600)
    xs=[20,34,54]
    ys=[180,250,325,400,510]
    for x in xs:
        page.draw_line((x,ys[0]),(x,ys[-1]))
    for y in ys:
        page.draw_line((xs[0],y),(xs[-1],y))
    for y,text in ((240,"Inventory"),(315,"Signature"),(390,"Date"),(500,"Archive")):
        page.insert_text((31,y),text,fontsize=8,rotate=90)
    pdf=tmp_path/"side-table.pdf"
    doc.save(pdf)
    doc.close()
    with pymupdf.open(pdf) as document:
        grids=discover_vector_grids(document[0])
    assert any(grid.bbox==pytest.approx((20,180,54,510),abs=.8)
               and len(grid.xedges)==3 and len(grid.yedges)==5 for grid in grids)


def test_adjacent_specification_and_irregular_stamp_stay_separate(tmp_path):
    doc=pymupdf.open()
    page=doc.new_page(width=900,height=600)
    for y in (80,110,130,150):
        page.draw_line((300,y),(700,y))
    for x in (300,340,480,700):
        page.draw_line((x,80),(x,150))
    for y in (165,200,230):
        page.draw_line((300,y),(700,y))
    page.draw_line((300,180),(400,180))
    for x in (300,350,400,700):
        page.draw_line((x,150),(x,230))
    for x in (500,580):
        page.draw_line((x,165),(x,200))
    pdf=tmp_path/"spec-and-stamp.pdf"
    doc.save(pdf)
    doc.close()
    with pymupdf.open(pdf) as document:
        grids=discover_vector_grids(document[0])
    assert len(grids)==2
    assert grids[0].bbox==pytest.approx((300,80,700,150),abs=.8)
    assert grids[1].bbox==pytest.approx((300,150,700,230),abs=.8)
    assert any(c.rowspan>1 or c.colspan>1 for c in grids[1].cells)


def test_dense_hatching_does_not_partition_a_tall_header(tmp_path):
    doc=pymupdf.open()
    page=doc.new_page(width=900,height=600)
    xs=[100,140,270,430,600]
    ys=[50,105,125,145,165,185,205]
    for x in xs:
        page.draw_line((x,ys[0]),(x,ys[-1]))
    for y in ys:
        page.draw_line((xs[0],y),(xs[-1],y))
    # Dense strokes touch column borders but are fill, NOT 130 table rows.
    for i in range(130):
        y=52+i*.3
        page.draw_line((270,y),(430,y),width=.1)
    page.insert_text((124,98),"FORMAT",fontsize=8,rotate=90)
    page.insert_text((285,82),"NAME",fontsize=10)
    pdf=tmp_path/"hatched-header.pdf"
    doc.save(pdf)
    doc.close()
    with pymupdf.open(pdf) as document:
        grids=discover_vector_grids(document[0])
    assert len(grids)==1
    assert grids[0].bbox==pytest.approx((100,50,600,205),abs=.8)
    assert grids[0].yedges==pytest.approx(ys,abs=.8)


def test_stroked_rectangles_form_tables_but_filled_rectangles_do_not(tmp_path):
    doc=pymupdf.open()
    page=doc.new_page(width=600,height=400)
    for row in range(3):
        for col in range(2):
            page.draw_rect(pymupdf.Rect(40+col*70,60+row*30,
                                        110+col*70,90+row*30))
            page.draw_rect(pymupdf.Rect(300+col*70,60+row*30,
                                        370+col*70,90+row*30),
                           color=None,fill=(.8,.8,.8))
    pdf=tmp_path/"rectangle-tables.pdf"
    doc.save(pdf)
    doc.close()
    with pymupdf.open(pdf) as document:
        grids=discover_vector_grids(document[0])
    assert len(grids)==1
    assert grids[0].bbox==pytest.approx((40,60,180,150),abs=.8)
