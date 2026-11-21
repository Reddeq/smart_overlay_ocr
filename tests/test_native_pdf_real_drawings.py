"""Optional local acceptance checks; client drawings are never test assets."""
from pathlib import Path

import pytest

pytest.importorskip("pymupdf")

from smart_overlay.models import Table, Text
from smart_overlay.native_pdf import native_pdf_project
from smart_overlay.structure import validate_project

SOURCE = Path(__file__).resolve().parents[1] / "drawings" / (
    "Арматурный блок НГС 2400-40 Сборочный чертеж _ ОТММ 186.00.00.000 СБ № 22418-1 _ ver21.pdf")
SCM = SOURCE.parent / "СЦМ 2.965.000 - Блок подготовки топливного газа.pdf"


@pytest.mark.skipif(not SOURCE.exists(), reason="Optional local client drawing is not present")
def test_ngs_side_title_specification_header_and_main_title_are_native_tables():
    project = native_pdf_project(SOURCE)
    zoom = project.raw["native_pdf"]["zoom"]
    tables = [obj for obj in project.objects if isinstance(obj, Table)]

    def at(bounds):
        found = [table for table in tables if (
            table.bbox.x / zoom, table.bbox.y / zoom,
            table.bbox.right / zoom, table.bbox.bottom / zoom) == pytest.approx(bounds, abs=1)]
        assert len(found) == 1, f"Expected complete table at {bounds}"
        return found[0]

    side = at((24, 1260, 58.1, 1671.1))
    spec = at((3034.6, 860.4, 3559, 1515.1))
    stamp = at((3034.6, 1515.1, 3559, 1671.1))
    assert (side.rows, side.columns) == (5, 2)
    assert all(cell.native_layout is not None and cell.native_layout.rotation == -90
               for cell in side.cells if cell.text.strip())
    header = [cell for cell in spec.cells if cell.row == 0]
    header_text = "\n".join(cell.text for cell in header)
    for label in ("Формат", "Зона", "Поз.", "Обозначение", "Наименование", "Кол."):
        assert label in header_text
        assert not any(isinstance(obj, Text) and obj.text == label for obj in project.objects)
    stamp_text = "\n".join(cell.text for cell in stamp.cells)
    for label in ("OTM/M 186.00.00.000 SB", "Developed by", "OILTEAMMASH LLC"):
        assert label in stamp_text
    assert any("Valve assembly NGS 2400-40 Assembly drawing" in " ".join(cell.text.split())
               for cell in stamp.cells)
    assert any(cell.rowspan > 1 or cell.colspan > 1 for cell in stamp.cells)
    validate_project(project)


@pytest.mark.skipif(not SCM.exists(), reason="Optional local client drawing is not present")
@pytest.mark.parametrize("detect_tables", [False, True])
@pytest.mark.parametrize("page,lines,glyphs,tables", [(1,52,158,5),(2,82,156,2),(3,13,74,2)])
def test_scm_overprinted_source_does_not_duplicate_editable_labels(page,lines,glyphs,tables,detect_tables):
    project = native_pdf_project(SCM,page,detect_tables=detect_tables)
    diagnostics = project.raw["native_pdf"]["deduplication"]
    assert diagnostics["removed_lines"] == lines
    assert diagnostics["removed_glyphs"] == glyphs
    assert diagnostics["ambiguous_lines"] == 0
    assert sum(isinstance(obj,Table) for obj in project.objects) == (tables if detect_tables else 0)
    seen=set()
    for obj in project.objects:
        if not isinstance(obj,Text):
            continue
        frame=obj.native_layout.frame
        key=(obj.text,obj.rotation,*(round(v,3) for v in (frame.x,frame.y,frame.w,frame.h)))
        assert key not in seen, f"Overprinted editable label: {obj.text}"
        seen.add(key)
    validate_project(project)
