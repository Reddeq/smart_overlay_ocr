import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
from PySide6.QtCore import QCoreApplication, QEvent

from smart_overlay.app import MainWindow, TableDialog


@pytest.fixture(scope="module", autouse=True)
def qapp(gui_app):
    # Keep one QApplication alive across tests and queued QThread deletions.
    app = gui_app
    app.setQuitOnLastWindowClosed(False)
    yield app
    app.processEvents()


def test_gui_load_inspect_edit_and_table(project, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr("smart_overlay.app.load_settings", lambda: __import__("smart_overlay.settings", fromlist=["Settings"]).Settings())
    window = MainWindow()
    window.project = project
    window.refresh(fit=True, selected="drain")
    assert window.content.toPlainText() == "Drain valve"
    window.content.setPlainText("Drain valve edited")
    window.apply_properties()
    assert next(o for o in window.project.objects if o.id == "drain").text == "Drain valve edited"
    dialog = TableDialog(project.objects[-1])
    assert dialog.grid.columnSpan(0, 0) == 3
    assert dialog.grid.rowSpan(2, 0) == 2
    dialog.grid.setRangeSelected(__import__("PySide6.QtWidgets", fromlist=["QTableWidgetSelectionRange"]).QTableWidgetSelectionRange(0, 0, 0, 2), True)
    dialog.cell_fill.setText("#ABCDEF")
    dialog.style_cells()
    dialog.accept()
    assert not dialog.table.needs_review
    assert next(c for c in dialog.table.cells if c.row == 0).style.fill == "#ABCDEF"
    window.dirty = False
    window.close()
    app.processEvents()


def test_smart_worker_can_be_cancelled_and_gui_recovers(project, tmp_path, monkeypatch):
    import asyncio
    import threading
    import time
    import httpx
    from smart_overlay.pipeline import smart_ocr
    from smart_overlay.settings import Settings
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.project = project
    entered = threading.Event()
    async def handler(request):
        entered.set()
        await asyncio.sleep(60)
    original = httpx.AsyncClient
    monkeypatch.setattr("smart_overlay.vision_transport.httpx.AsyncClient",
                        lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    def operation(progress):
        return smart_ocr(project, Settings(model="test/model"), "secret", tmp_path, True,
                         progress=progress, cancelled=window.cancellation_requested)
    window.start(operation, cancellable=True)
    assert window.cancel_button.isEnabled()
    deadline = time.monotonic()+3
    while not entered.is_set() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    window.cancel_smart()
    deadline = time.monotonic()+3
    while window.worker and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert window.worker is None
    assert entered.is_set()
    assert not window.cancel_button.isEnabled()
    assert window.export_button.isEnabled()
    assert "отменён" in window.statusBar().currentMessage()
    assert window.project.objects == project.objects
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    window.dirty = False
    window.close()
    app.processEvents()


@pytest.mark.parametrize("answer, exported", [(QMessageBox.StandardButton.Yes, True),
                                            (QMessageBox.StandardButton.Cancel, False)])
def test_gui_can_export_unreviewed_after_warning(project, tmp_path, monkeypatch, answer, exported):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.project = project
    project.objects[0].needs_review = True
    warnings, calls = [], []
    def warning(*args):
        warnings.append(args[2])
        return answer
    monkeypatch.setattr(QMessageBox, "warning", warning)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(tmp_path / "out.pptx"), ""))
    monkeypatch.setattr("smart_overlay.app.export_pptx", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(window, "start", lambda operation: operation(lambda _: None))
    window.export()
    assert len(warnings) == 1
    assert bool(calls) == exported
    if exported:
        assert calls[0]["allow_unreviewed"] is True
    assert project.objects[0].needs_review
    window.close()
    app.processEvents()
