import time

from PIL import Image, ImageDraw
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QDialog

from smart_overlay.app import MainWindow, TablesResultDialog
from smart_overlay.grouping import build_plan, find_project_tables, fingerprint, mark_manual
from smart_overlay.input import load_image, open_project, save_project, source_image
from smart_overlay.models import Box, Fragment, Project, Table, Text
from smart_overlay.pipeline import format_project, local_ocr
from smart_overlay.settings import Settings
from smart_overlay.structure import base_texts, validate_project
from smart_overlay.tables import geometric_tables


@pytest.fixture
def wired_project(tmp_path):
    image = Image.new('RGB', (600, 400), 'white')
    draw = ImageDraw.Draw(image)
    fragments = []
    # Both tables belong to one connected wire network. The second has one column.
    for number, (left, right, rows, cols) in enumerate([(60, 180, 5, 2), (350, 430, 6, 1)]):
        for row in range(rows+1):
            y = 50+row*40
            draw.line((left, y, right, y), fill='black', width=2)
        for col in range(cols+1):
            x = left+(right-left)*col//cols
            draw.line((x, 50, x, 50+rows*40), fill='black', width=2)
        for row in range(rows):
            for col in range(cols):
                x, y = left+(right-left)*col//cols+12, 64+row*40
                text = f'{number}{row}{col}'
                draw.text((x, y), text, fill='black')
                box = Box(x=x, y=y, w=24, h=12)
                fragments.append(Fragment(id=f'ocr_{text}', text=text, bbox=box,
                    polygon=[(x,y), (x+24,y), (x+24,y+12), (x,y+12)], confidence=.95))
    for y in [90, 130, 170, 210]:
        draw.line((180, y, 350, y), fill='black', width=2)
    draw.line((20, 30, 20, 350), fill='black', width=2)
    draw.line((20, 90, 60, 90), fill='black', width=2)
    path = tmp_path/'wired.png'
    image.save(path)
    project = load_image(path)
    project.fragments = fragments
    base_texts(project)
    format_project(project, Settings())
    validate_project(project)
    return project


def test_connected_grids_and_single_column_are_detected_without_wire_corridor(wired_project):
    tables = geometric_tables(source_image(wired_project), wired_project.fragments, [])
    assert sorted((t.rows, t.columns) for t in tables) == [(5, 2), (6, 1)]
    assert sorted(len(t.ocr_ids) for t in tables) == [6, 10]
    assert all(t.needs_review for t in tables)
    assert geometric_tables(source_image(wired_project), wired_project.fragments, tables) == []


def test_existing_project_detection_is_immutable_repeatable_and_persists(wired_project, tmp_path):
    before = fingerprint(wired_project)
    result = find_project_tables(wired_project, Settings())
    assert result.found == 2 and result.skipped == 0
    assert result.signature == before == fingerprint(wired_project)
    assert result.candidate.source == wired_project.source
    assert result.candidate.fragments == wired_project.fragments
    assert all(isinstance(o, Table) for o in result.candidate.objects)
    validate_project(result.candidate)
    assert find_project_tables(result.candidate, Settings()).found == 0
    path = tmp_path/'tables.socr.json'
    save_project(result.candidate, path)
    assert open_project(path) == result.candidate
    # Native tables automatically protect their OCR from VL grouping/reading.
    plan = build_plan(result.candidate, Settings(), [], review_ocr=True)
    assert not plan.regions and not plan.reading_ids


def test_changed_font_setting_does_not_protect_automatic_objects(wired_project):
    result = find_project_tables(wired_project, Settings(font='Courier New'))
    assert result.found == 2 and result.skipped == 0


@pytest.mark.parametrize('edit', ['text', 'marked', 'excluded', 'geometry', 'style', 'ignored'])
def test_manual_or_ignored_objects_are_preserved(wired_project, edit):
    obj = wired_project.objects[0]
    if edit == 'text':
        obj.text = 'manual correction'
    elif edit == 'marked':
        mark_manual(wired_project, obj.id)
    elif edit == 'excluded':
        obj.excluded = True
    elif edit == 'geometry':
        obj.bbox.x += 2
    elif edit == 'style':
        obj.style.font = 'Courier New'
    else:
        wired_project.ignored[obj.ocr_ids[0]] = 'user ignored'
        wired_project.objects.remove(obj)
    before = fingerprint(wired_project)
    result = find_project_tables(wired_project, Settings())
    assert result.found == 1
    assert result.candidate.ignored == wired_project.ignored
    if edit != 'ignored':
        assert next(o for o in result.candidate.objects if o.id == obj.id) == obj
    assert fingerprint(wired_project) == before
    validate_project(result.candidate)


def test_regular_local_ocr_automatically_uses_connected_grid_detection(wired_project):
    class Engine:
        def recognize(self, image):
            return wired_project.fragments, []

        def tables(self, image):
            return []

    result = local_ocr(wired_project, Settings(), engine=Engine())
    assert len(result.objects) == len(result.raw['opencv_tables']) == 2
    assert all(isinstance(o, Table) for o in result.objects)
    validate_project(result)


def test_automatic_text_padding_outside_new_table_is_clipped(wired_project):
    box = Box(x=183, y=105, w=14, h=12)
    f = Fragment(id='near_table', text='X', bbox=box,
                 polygon=[(183,105), (197,105), (197,117), (183,117)], confidence=.95)
    text = Text(id='near_text', text=f.text, ocr_ids=[f.id], bbox=box.model_copy())
    padding = Project(source=wired_project.source, fragments=[f], objects=[text])
    format_project(padding, Settings())
    wired_project.fragments.append(f)
    wired_project.objects.append(text)
    before = fingerprint(wired_project)
    result = find_project_tables(wired_project, Settings())
    assert result.found == 2
    restored = next(o for o in result.candidate.objects if o.id == text.id)
    assert restored.bbox == box and restored.text == text.text and restored.style == text.style
    assert fingerprint(wired_project) == before
    validate_project(result.candidate)


@pytest.mark.parametrize('accepted', [True, False])
def test_table_preview_apply_and_undo(gui_app, wired_project, monkeypatch, accepted):
    result = find_project_tables(wired_project, Settings())
    window = MainWindow()
    window.project = wired_project
    before = fingerprint(wired_project)
    monkeypatch.setattr(TablesResultDialog, 'exec', lambda self:
        QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected)
    window.show_tables_result(result)
    if accepted:
        assert len(window.project.objects) == 2 and window.dirty
        assert window.undo_grouping_button.isEnabled()
        window.undo_grouping()
    assert fingerprint(window.project) == before and not window.dirty
    assert not window.undo_grouping_button.isEnabled()
    window.close()


def test_table_preview_rejects_stale_result(gui_app, wired_project, monkeypatch):
    result = find_project_tables(wired_project, Settings())
    window = MainWindow()
    window.project = wired_project
    window.project.objects[0].text = 'newer edit'
    errors = []
    monkeypatch.setattr(window, 'error', errors.append)
    monkeypatch.setattr(TablesResultDialog, 'exec', lambda self: pytest.fail('stale preview'))
    window.show_tables_result(result)
    assert errors and window.project.objects[0].text == 'newer edit'
    window.close()


def test_detection_worker_opens_preview_after_reenabling_window(gui_app, wired_project, monkeypatch):
    window = MainWindow()
    window.project = wired_project
    viewed = []

    def preview(result):
        assert window.worker is None and window.find_tables_button.isEnabled()
        viewed.append(result)

    monkeypatch.setattr(window, 'show_tables_result', preview)
    window.find_tables()
    assert not window.find_tables_button.isEnabled()
    deadline = time.monotonic()+4
    while window.worker and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.01)
    assert len(viewed) == 1 and viewed[0].found == 2
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    window.close()
