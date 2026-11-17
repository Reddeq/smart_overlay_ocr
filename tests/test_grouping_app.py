import os
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from smart_overlay.app import GroupingResultDialog, GroupingSetupDialog, MainWindow
from smart_overlay.diagnose_vl import synthetic
from smart_overlay.grouping import (GroupingResult, build_plan, candidate_project, fingerprint, finish)
from smart_overlay.models import Box
from smart_overlay.settings import Settings


@pytest.fixture(scope='module', autouse=True)
def qapp(gui_app):
    app = gui_app
    app.setQuitOnLastWindowClosed(False)
    yield app
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def grouped_result():
    p, _ = synthetic()
    plan = build_plan(p, Settings(), [])
    records = []
    for region in plan.regions:
        groups = [['id_1','id_2','id_3']] if {'id_1','id_2','id_3'} <= set(region.ocr_ids) else []
        records.append({'test': region.name, 'http_status': 200, 'normalized': {'applied_merges': groups, 'review_groups': []}})
    return finish(GroupingResult(plan, results=records))


def test_setup_requires_table_review_and_prepares_budget_without_network(monkeypatch):
    p, _ = synthetic()
    before = fingerprint(p)
    monkeypatch.setattr('smart_overlay.grouping.preflight', lambda *a: pytest.fail('Unexpected API'))
    dialog = GroupingSetupDialog(p, Settings())
    dialog.prepare()
    assert dialog.plan and not dialog.send_button.isEnabled()
    dialog.tables_checked.setChecked(True)
    assert dialog.send_button.isEnabled()
    assert str(dialog.plan.requests) in dialog.send_button.text()
    dialog.add_mask(Box(x=450,y=40,w=130,h=60))
    assert dialog.plan is None and not dialog.tables_checked.isChecked() and len(dialog.masks) == 1
    dialog.zones.setCurrentRow(0)
    dialog.remove_mask()
    assert not dialog.masks and fingerprint(p) == before
    dialog.reject()


def test_preview_is_immutable_and_checkboxes_control_candidate(grouped_result):
    before = fingerprint(grouped_result.plan.project)
    dialog = GroupingResultDialog(grouped_result)
    assert dialog.groups.rowCount() == 1
    # A single-view proposal is not selected automatically.
    assert not dialog.selected_groups()
    item = dialog.groups.item(0,0)
    item.setCheckState(Qt.CheckState.Checked)
    assert len(dialog.candidate.objects) == 3
    assert fingerprint(grouped_result.plan.project) == before
    item.setCheckState(Qt.CheckState.Unchecked)
    assert len(dialog.candidate.objects) == 5
    dialog.reject()


def test_apply_and_undo_restore_exact_project_and_dirty_state(grouped_result, monkeypatch):
    window = MainWindow()
    window.project = grouped_result.plan.project.model_copy(deep=True)
    window.dirty = False
    before = fingerprint(window.project)
    def accept(dialog):
        dialog.groups.item(0,0).setCheckState(Qt.CheckState.Checked)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(GroupingResultDialog, 'exec', accept)
    window.show_grouping_result(grouped_result)
    assert len(window.project.objects) == 3 and window.dirty
    assert window.undo_grouping_button.isEnabled()
    window.undo_grouping()
    assert fingerprint(window.project) == before and not window.dirty
    assert not window.undo_grouping_button.isEnabled()
    window.close()


def test_discard_preview_changes_nothing(grouped_result, monkeypatch):
    window = MainWindow()
    window.project = grouped_result.plan.project.model_copy(deep=True)
    before = fingerprint(window.project)
    monkeypatch.setattr(GroupingResultDialog, 'exec', lambda self: QDialog.DialogCode.Rejected)
    window.show_grouping_result(grouped_result)
    assert fingerprint(window.project) == before and not window.dirty
    window.close()


def test_stale_plan_and_later_manual_edits_cannot_be_overwritten(grouped_result, monkeypatch):
    window = MainWindow()
    window.project = grouped_result.plan.project.model_copy(deep=True)
    errors = []
    monkeypatch.setattr(window, 'error', errors.append)
    monkeypatch.setattr(GroupingResultDialog, 'exec', lambda self: pytest.fail('Unexpected preview'))
    window.project.objects[-1].text = 'Later edit'
    window.show_grouping_result(grouped_result)
    assert errors and window.project.objects[-1].text == 'Later edit'
    candidate = candidate_project(grouped_result, [['id_1','id_2','id_3']])
    before = window.project.model_copy(deep=True)
    window.project = candidate
    window.grouping_undo = (before, fingerprint(candidate), False)
    window.project.objects[-1].text = 'Edited after applying'
    window.undo_grouping()
    assert window.project.objects[-1].text == 'Edited after applying' and window.grouping_undo is None
    window.dirty = False
    window.close()


def test_new_operation_worker_cancel_returns_partial_preview(grouped_result, monkeypatch):
    window = MainWindow()
    window.project = grouped_result.plan.project.model_copy(deep=True)
    viewed = []
    def operation(progress):
        while not window.cancellation_requested():
            time.sleep(.01)
        grouped_result.status = 'cancelled'
        return grouped_result
    monkeypatch.setattr(window, 'show_grouping_result', viewed.append)
    window.active_grouping = True
    window.start(operation, cancellable=True)
    assert not window.grouping_button.isEnabled()
    window.cancel_smart()
    deadline = time.monotonic()+3
    while window.worker and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.01)
    assert window.worker is None and viewed == [grouped_result]
    assert window.grouping_button.isEnabled() and not window.cancel_button.isEnabled()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    window.close()


def test_toolbar_flow_prepares_sends_previews_applies_and_undoes(grouped_result, monkeypatch):
    window = MainWindow()
    window.project = grouped_result.plan.project.model_copy(deep=True)
    window.api_key = 'session-secret'
    before = fingerprint(window.project)
    calls = []
    def setup(dialog):
        dialog.tables_checked.setChecked(True)
        dialog.prepare()
        dialog.accept()
        return dialog.result()
    def send(plan, key, proxy, **kwargs):
        assert key == 'session-secret' and plan.signature == before
        assert kwargs['cancelled']() is False
        calls.append(plan.requests)
        return grouped_result
    def preview(dialog):
        dialog.groups.item(0,0).setCheckState(Qt.CheckState.Checked)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(GroupingSetupDialog, 'exec', setup)
    monkeypatch.setattr('smart_overlay.app.run_grouping', send)
    monkeypatch.setattr(GroupingResultDialog, 'exec', preview)
    window.run_grouping()
    deadline = time.monotonic()+3
    while window.worker and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.01)
    assert calls == [grouped_result.plan.requests]
    assert len(window.project.objects) == 3 and window.undo_grouping_button.isEnabled()
    assert 'session-secret' not in window.project.model_dump_json()
    window.undo_grouping()
    assert fingerprint(window.project) == before
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    window.close()


def test_setup_cancel_prevents_worker_and_leaves_project_unchanged(grouped_result, monkeypatch):
    window = MainWindow()
    window.project = grouped_result.plan.project.model_copy(deep=True)
    window.api_key = 'secret'
    before = fingerprint(window.project)
    monkeypatch.setattr(GroupingSetupDialog, 'exec', lambda self: QDialog.DialogCode.Rejected)
    monkeypatch.setattr(window, 'start', lambda *a, **kw: pytest.fail('Unexpected worker'))
    window.run_grouping()
    assert fingerprint(window.project) == before and not window.dirty
    window.close()


def test_reading_changes_are_unchecked_until_explicitly_selected(grouped_result):
    grouped_result.readings = [{'ocr_id':'id_1','reading':{'text':'CHANGED','confidence':1,'reason':'visible'}}]
    before=fingerprint(grouped_result.plan.project)
    dialog=GroupingResultDialog(grouped_result)
    assert dialog.readings.rowCount()==1
    assert dialog.candidate.objects[0].text=='MAIN'
    dialog.readings.item(0,0).setCheckState(Qt.CheckState.Checked)
    assert dialog.candidate.objects[0].text=='CHANGED'
    assert fingerprint(grouped_result.plan.project)==before
    dialog.reject()
