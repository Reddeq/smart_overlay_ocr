from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.create_example import create


@pytest.fixture
def project(tmp_path):
    return create(tmp_path)


@pytest.fixture(scope="session")
def gui_app():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QCoreApplication, QEvent
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    yield app
    # Widgets/signals can form Python cycles: destroy native widgets while the
    # single QApplication still exists, before pytest's final cyclic collection.
    for widget in app.topLevelWidgets():
        if hasattr(widget, "dirty"):
            widget.dirty = False
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


@pytest.fixture(autouse=True)
def isolated_gui_settings(monkeypatch):
    from smart_overlay.settings import Settings
    monkeypatch.setattr("smart_overlay.app.load_settings", lambda: Settings())

