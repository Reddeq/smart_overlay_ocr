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
