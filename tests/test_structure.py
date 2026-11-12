import base64
import hashlib

import pytest

from smart_overlay.geometry import union
from smart_overlay.input import open_project, save_project
from smart_overlay.models import Box, Text
from smart_overlay.ocr import fragments_from_result
from smart_overlay.structure import merge_texts, split_text, validate_project
from smart_overlay.tables import merge_cells, parse_html, split_cell
from smart_overlay.vision import Analysis, Group, TableReview, apply_analysis


def test_portable_round_trip_and_integrity(project, tmp_path):
    path = tmp_path / "project.json"
    save_project(project, path)
    reopened = open_project(path)
    assert reopened == project
    assert hashlib.sha256(base64.b64decode(reopened.source.original_b64)).hexdigest() == project.source.sha256
    reopened.source.sha256 = "0"*64
    path.write_text(reopened.model_dump_json(), "utf-8")
    with pytest.raises(ValueError, match="сумма"):
        open_project(path)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "unknown", "bounds", "overlap"])
def test_reject_invalid_structure(project, defect):
    if defect == "missing":
        project.objects[0].ocr_ids.pop()
    elif defect == "duplicate":
        project.objects[1].ocr_ids.append("ocr_1")
    elif defect == "unknown":
        project.objects[0].ocr_ids.append("invented")
    elif defect == "bounds":
        project.objects[0].bbox.x = -10
    else:
        project.objects[0].bbox = project.objects[-1].bbox
    with pytest.raises(ValueError):
        validate_project(project)


def test_manual_merge_split_retains_provenance(project):
    merged = merge_texts(project, ["drain", "outlet"])
    assert merged.text == "Drain valve\nOutlet"
    b = merged.bbox
    split_text(project, merged.id, [Box(x=b.x, y=b.y, w=b.w/2, h=b.h),
               Box(x=b.x+b.w/2, y=b.y, w=b.w/2, h=b.h)], ["Drain valve", "Outlet"])
    validate_project(project)
    assert project.ignored["ocr_4"]
    assert sum(f.origin == "manual_split" for f in project.fragments) == 2


def test_review_blocks_export_but_exclusion_is_recorded(project):
    project.objects[0].needs_review = True
    with pytest.raises(ValueError, match="проверка"):
        validate_project(project, for_export=True)
    project.objects[0].excluded = True
    validate_project(project, for_export=True)


def test_table_spans_and_topology(project):
    rows, cols, cells = parse_html('<table><tr><td colspan="2">Head</td></tr><tr><td>A</td><td>B</td></tr></table>')
    assert (rows, cols) == (2, 2)
    assert cells[0].colspan == 2
    table = project.objects[-1]
    split = split_cell(table, 0, 0)
    assert len(split.cells) == len(table.cells)+2
    merged = merge_cells(split, 0, 0, 0, 2)
    assert next(c for c in merged.cells if c.row == 0).colspan == 3
    with pytest.raises(ValueError):
        merge_cells(table, 2, 0, 2, 1)


def analysis_for(project):
    table = project.objects[-1]
    return Analysis(groups=[Group(ocr_ids=o.ocr_ids, text=o.text, confidence=1, reason="same")
                            for o in project.objects if isinstance(o, Text)],
                    tables=[TableReview(id=table.id, rows=table.rows, columns=table.columns,
                                        cells=table.cells, confidence=1, issues=[])], ignored={}, regions=[])


def test_vl_correction_threshold_and_atomic_validation(project):
    analysis = analysis_for(project)
    analysis.groups[0].text = "Uncertain reading"
    analysis.groups[0].confidence = 0.7
    result = apply_analysis(project, analysis)
    assert result.objects[1].text == project.objects[0].text
    assert result.objects[1].needs_review
    analysis.groups[1].ocr_ids = analysis.groups[0].ocr_ids
    with pytest.raises(ValueError):
        apply_analysis(project, analysis)
    validate_project(project)


def test_vl_cannot_move_table_text_to_label(project):
    analysis = analysis_for(project)
    analysis.groups[0].ocr_ids.append(project.objects[-1].ocr_ids[0])
    with pytest.raises(ValueError, match="табличный"):
        apply_analysis(project, analysis)


def test_vl_missing_table_review_preserves_local_table_and_smart_text(project):
    analysis = analysis_for(project)
    analysis.tables = []
    analysis.groups[1].text = "Corrected drain"
    result = apply_analysis(project, analysis)
    table = next(o for o in result.objects if o.type == "table")
    assert table.cells == project.objects[-1].cells
    assert table.bbox == project.objects[-1].bbox
    assert table.needs_review
    assert any("VL не проверила таблицу" in note for note in table.review_notes)
    assert next(o for o in result.objects if o.id != table.id and "ocr_4" in o.ocr_ids).text == "Corrected drain"
    assert not project.objects[-1].needs_review  # atomic, no changes to the original
    validate_project(result)


def test_vl_table_lines_are_not_exported_as_duplicate_labels(project):
    analysis = analysis_for(project)
    analysis.tables = []
    table_ids = project.objects[-1].ocr_ids
    analysis.groups.append(Group(ocr_ids=table_ids, text="Untrusted reconstructed table", confidence=1, reason="table"))
    result = apply_analysis(project, analysis)
    assert len(result.objects) == len(project.objects)
    assert not any(set(o.ocr_ids).intersection(table_ids) for o in result.objects if isinstance(o, Text))
    assert any("табличный текст" in message for message in result.messages)
    validate_project(result)


@pytest.mark.parametrize("defect, message", [("unknown", "неизвестные ID"), ("duplicate", "повторяющиеся ID")])
def test_vl_invalid_table_references_still_rejected(project, defect, message):
    analysis = analysis_for(project)
    if defect == "unknown":
        analysis.tables[0].id = "invented_table"
    else:
        analysis.tables.append(analysis.tables[0].model_copy(deep=True))
    with pytest.raises(ValueError, match=message):
        apply_analysis(project, analysis)


def test_partial_vl_restores_missing_fragments_and_keeps_valid_corrections(project):
    analysis = analysis_for(project)
    analysis.groups = analysis.groups[1:]  # omit the three-line pressure label
    analysis.groups[0].text = "Verified drain valve"
    result = apply_analysis(project, analysis)
    recovered = [o for o in result.objects if isinstance(o, Text) and set(o.ocr_ids).intersection({"ocr_1", "ocr_2", "ocr_3"})]
    assert len(recovered) == 1
    assert recovered[0].text == project.objects[0].text
    assert recovered[0].bbox == project.objects[0].bbox
    assert recovered[0].ocr_ids == ["ocr_1", "ocr_2", "ocr_3"]
    assert recovered[0].needs_review
    corrected = next(o for o in result.objects if "ocr_4" in o.ocr_ids)
    assert corrected.text == "Verified drain valve"
    assert any("3" in m and "VL пропустила" in m for m in result.messages)
    validate_project(result)
    assert len(project.objects) == 4


def test_empty_vl_groups_preserve_every_fragment_without_table_duplicates(project):
    result = apply_analysis(project, Analysis(groups=[], tables=[], ignored={}, regions=[]))
    validate_project(result)
    table_ids = set(project.objects[-1].ocr_ids)
    labels = [o for o in result.objects if isinstance(o, Text)]
    assert len(labels) == 3
    assert sum(len(o.ocr_ids) for o in labels) == 5
    assert all(o.needs_review for o in labels)
    assert all(not table_ids.intersection(o.ocr_ids) for o in labels)
    assert any("пустой структурный анализ" in message for message in result.messages)


def test_coverage_recovery_respects_ignored_and_excluded_fragments(project):
    project.ignored["ocr_5"] = "Explicit exclusion"
    project.objects = [o for o in project.objects if o.id != "outlet"]
    next(o for o in project.objects if o.id == "drain").excluded = True
    result = apply_analysis(project, Analysis(groups=[], tables=[], ignored={}, regions=[]))
    assert not any("ocr_5" in o.ocr_ids for o in result.objects)
    assert next(o for o in result.objects if "ocr_4" in o.ocr_ids).excluded
    assert result.ignored == project.ignored
    validate_project(result)


@pytest.mark.parametrize("defect", ["duplicate", "unknown", "outside"])
def test_partial_coverage_does_not_mask_other_invalid_vl_output(project, defect):
    analysis = analysis_for(project)
    analysis.groups.pop()  # leave an omitted fragment to recover
    if defect == "duplicate":
        analysis.groups.append(analysis.groups[0].model_copy(deep=True))
    elif defect == "unknown":
        analysis.groups[0].ocr_ids = ["invented"]
    else:
        from smart_overlay.vision import Region
        analysis.regions = [Region(bbox=Box(x=-10,y=0,w=30,h=30), reason="outside")]
    with pytest.raises(ValueError):
        apply_analysis(project, analysis)


def test_missing_whole_block_preserves_manual_corrections(project):
    from smart_overlay.structure import manual_text
    original = next(o for o in project.objects if o.id == "pressure")
    manual_text(original, "Manually corrected pressure label")
    analysis = analysis_for(project)
    analysis.groups = analysis.groups[1:]
    result = apply_analysis(project, analysis)
    restored = next(o for o in result.objects if o.id == "pressure")
    assert restored.text == original.text
    assert restored.corrections == original.corrections
    assert restored.style == original.style
    assert restored.bbox == original.bbox
    assert restored.needs_review


def test_partial_block_uses_measured_fragment_not_character_split(project):
    analysis = analysis_for(project)
    analysis.groups[0].ocr_ids = ["ocr_1", "ocr_2"]
    analysis.groups[0].text = "Maximum permissible\noperating pressure"
    result = apply_analysis(project, analysis)
    restored = next(o for o in result.objects if o.ocr_ids == ["ocr_3"])
    fragment = next(f for f in project.fragments if f.id == "ocr_3")
    assert restored.bbox == fragment.bbox
    assert restored.text == fragment.text
    assert restored.rotation == fragment.rotation
    assert restored.confidence == fragment.confidence
    assert restored.needs_review
    validate_project(result)


def test_paddle_3_result_coordinates():
    raw = {"rec_texts": ["Valve"], "rec_scores": [0.93], "rec_polys": [[[10, 20], [70, 20], [70, 40], [10, 40]]]}
    result = fragments_from_result(raw, (100, 200))
    assert result[0].bbox == Box(x=110, y=220, w=60, h=20)
    assert result[0].rotation == 0
    raw["rec_scores"] = []
    with pytest.raises(ValueError):
        fragments_from_result(raw)
