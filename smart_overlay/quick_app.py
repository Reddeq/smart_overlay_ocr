"""Minimal Windows UI: choose a drawing, recognize, receive an editable PPTX.

The legacy object editor is kept as an optional correction tool, not the default
workflow. Advanced experiments and diagnostics remain CLI-only.
"""
import os
from pathlib import Path

from PySide6.QtCore import QThread, Signal, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from .app import Preview
from .input import load_image
from .one_click import OneClickResult, ProcessingCancelled, convert_drawing
from .settings import data_dir, load_settings, save_settings


class ConvertWorker(QThread):
    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)
    stopped = Signal()

    def __init__(self, work, parent=None):
        super().__init__(parent)
        self.work = work

    def run(self):
        try:
            self.completed.emit(self.work(self.progress.emit, self.isInterruptionRequested))
        except ProcessingCancelled:
            self.stopped.emit()
        except Exception as exc:
            # GUI boundary: never echo potentially sensitive network response bodies.
            self.failed.emit(type(exc).__name__ + ": " + str(exc)[:500])


class ConnectionDialog(QDialog):
    def __init__(self, settings, key, proxy, enabled, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки распознавания")
        self.resize(520, 200)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.language = QLineEdit(settings.language)
        self.language.setPlaceholderText("ru или en")
        form.addRow("Язык чертежа", self.language)
        self.cloud = QCheckBox("Улучшать группировку и таблицы через OpenRouter")
        self.cloud.setChecked(enabled)
        form.addRow("", self.cloud)
        self.key = QLineEdit(key)
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText("sk-or-… (не сохраняется на диск)")
        form.addRow("API-ключ", self.key)
        self.proxy = QLineEdit(proxy)
        self.proxy.setPlaceholderText("http://127.0.0.1:10809")
        form.addRow("Прокси Xray", self.proxy)
        layout.addLayout(form)
        notice = QLabel("Локальный OCR работает без ключа и сети. При облачном анализе части чертежа передаются провайдеру и могут тарифицироваться. Перед отправкой каждого файла будет запрос согласия.")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        controls = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        controls.accepted.connect(self.accept)
        controls.rejected.connect(self.reject)
        layout.addWidget(controls)

    def accept(self):
        if self.cloud.isChecked() and not self.key.text().strip():
            QMessageBox.warning(self, "OpenRouter", "Для облачного распознавания укажите API-ключ или отключите облачный режим.")
            return
        if not self.language.text().strip():
            QMessageBox.warning(self, "Язык", "Укажите язык OCR, например ru или en.")
            return
        super().accept()


class QuickWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Overlay OCR")
        self.resize(1100, 760)
        self.settings = load_settings()
        # Automatic background sampling retains more of the source's appearance.
        self.settings.fill_mode = "auto"
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.proxy = os.environ.get("SMART_OVERLAY_PROXY", "http://127.0.0.1:10809")
        self.cloud_enabled = bool(self.api_key)
        self.path = None
        self.result: OneClickResult | None = None
        self.worker = None
        self.editor = None
        self.setFont(QFont("Segoe UI", 10))

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)

        head = QHBoxLayout()
        title_group = QVBoxLayout()
        title = QLabel("Чертёж → PowerPoint")
        title.setStyleSheet("font-size: 23px; font-weight: 600;")
        subtitle = QLabel("Редактируемые надписи и таблицы поверх исходного изображения. Один файл — один слайд.")
        subtitle.setStyleSheet("color: #777;")
        title_group.addWidget(title)
        title_group.addWidget(subtitle)
        head.addLayout(title_group, 1)
        self.config = QPushButton("Настройки")
        self.config.clicked.connect(self.configure)
        head.addWidget(self.config)
        layout.addLayout(head)

        chooser = QHBoxLayout()
        self.file_label = QLabel("Чертёж не выбран")
        self.file_label.setStyleSheet("padding: 10px;")
        chooser.addWidget(self.file_label, 1)
        self.pick = QPushButton("Выбрать чертёж")
        self.pick.clicked.connect(self.choose)
        chooser.addWidget(self.pick)
        self.run_button = QPushButton("Распознать → PPTX")
        self.run_button.setMinimumHeight(43)
        self.run_button.setEnabled(False)
        self.run_button.setStyleSheet("font-weight: 600; padding: 8px 18px;")
        self.run_button.clicked.connect(self.recognize)
        chooser.addWidget(self.run_button)
        layout.addLayout(chooser)

        self.preview = Preview()
        self.preview.show_formatting = True
        layout.addWidget(self.preview, 1)

        status = QHBoxLayout()
        self.status_text = QLabel("Откройте изображение PNG, JPG, TIFF или BMP")
        self.status_text.setWordWrap(True)
        status.addWidget(self.status_text, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(160)
        self.progress.hide()
        status.addWidget(self.progress)
        self.cancel = QPushButton("Отменить")
        self.cancel.hide()
        self.cancel.clicked.connect(self.cancel_work)
        status.addWidget(self.cancel)
        layout.addLayout(status)

        actions = QHBoxLayout()
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        actions.addWidget(self.summary, 1)
        self.open_pptx = QPushButton("Открыть PPTX")
        self.open_pptx.clicked.connect(self.open_result)
        self.open_pptx.hide()
        actions.addWidget(self.open_pptx)
        self.review = QPushButton("Исправить результат…")
        self.review.clicked.connect(self.open_editor)
        self.review.hide()
        actions.addWidget(self.review)
        layout.addLayout(actions)
        self.statusBar().showMessage("Готово к работе")

    def choose(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать чертёж", str(Path.home()),
            "Чертежи (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if not path:
            return
        try:
            project = load_image(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Открытие файла", str(exc))
            return
        self.path = Path(path)
        self.file_label.setText(str(self.path))
        self.preview.show_project(project, fit=True)
        self.result = None
        self.open_pptx.hide()
        self.review.hide()
        self.summary.setText("")
        self.run_button.setEnabled(True)
        self.status_text.setText("Можно распознавать")

    def configure(self):
        dialog = ConnectionDialog(self.settings, self.api_key, self.proxy, self.cloud_enabled, self)
        if not dialog.exec():
            return
        self.api_key = dialog.key.text().strip()
        self.proxy = dialog.proxy.text().strip()
        self.cloud_enabled = dialog.cloud.isChecked()
        self.settings.language = dialog.language.text().strip()
        try:
            save_settings(self.settings)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Настройки", str(exc))
        self.status_text.setText("Параметры сохранены. Ключ хранится только в памяти приложения.")

    def recognize(self):
        if not self.path or self.worker:
            return
        consent = False
        if self.cloud_enabled:
            answer = QMessageBox.question(
                self, "Облачное распознавание",
                "Отправить части этого чертежа и распознанный текст в OpenRouter через настроенный VPN?\n"
                "Модель Qwen3-VL-8B, провайдер Parasail. Запросы могут быть платными.\n"
                "Нет — создать PPTX полностью локально.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Yes)
            if answer == QMessageBox.StandardButton.Cancel:
                return
            consent = answer == QMessageBox.StandardButton.Yes

        source, settings, key, proxy = self.path, self.settings.model_copy(deep=True), self.api_key, self.proxy
        self.set_busy(True)
        self.summary.setText("")
        self.status_text.setText("Начинаю распознавание…")
        self.worker = ConvertWorker(
            lambda report, stop: convert_drawing(source, settings=settings, api_key=key,
                                                 consent=consent, proxy=proxy,
                                                 progress=report, cancelled=stop), self)
        self.worker.progress.connect(self.status_text.setText)
        self.worker.completed.connect(self.done)
        self.worker.failed.connect(self.failed)
        self.worker.stopped.connect(self.stopped)
        self.worker.finished.connect(self.finished)
        self.worker.start()

    def set_busy(self, busy):
        for widget in (self.pick, self.run_button, self.config):
            widget.setEnabled(not busy)
        self.progress.setVisible(busy)
        self.cancel.setVisible(busy)
        self.open_pptx.setEnabled(not busy)
        self.review.setEnabled(not busy)

    def cancel_work(self):
        if self.worker:
            self.worker.requestInterruption()
            self.status_text.setText("Остановка после текущего этапа…")
            self.cancel.setEnabled(False)

    def done(self, result):
        self.result = result
        self.preview.show_project(result.project)
        self.open_pptx.show()
        self.review.show()
        notes = [f"Готово: {result.pptx.name}",
                 f"Объектов: {len(result.project.objects)}; на проверку: {result.needs_review}."]
        if result.warnings:
            notes.append("Есть замечания — откройте «Исправить результат».")
        self.summary.setText(" ".join(notes))
        self.status_text.setText("PowerPoint создан")
        self.statusBar().showMessage(str(result.pptx))

    def failed(self, reason):
        QMessageBox.warning(self, "Не удалось создать PPTX", reason)
        self.status_text.setText("Не удалось завершить обработку. Исходник не изменён.")

    def stopped(self):
        self.status_text.setText("Обработка отменена.")

    def finished(self):
        worker = self.worker
        self.worker = None
        self.set_busy(False)
        self.cancel.setEnabled(True)
        if worker:
            worker.deleteLater()

    def open_result(self):
        if self.result:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result.pptx)))

    def open_editor(self):
        if not self.result:
            return
        from .app import MainWindow
        self.editor = MainWindow()
        self.editor.project = self.result.project.model_copy(deep=True)
        self.editor.project_path = (str(self.result.project_file)
                                   if self.result.project_file.exists() else None)
        self.editor.dirty = False
        self.editor.refresh(fit=True)
        self.editor.show()

    def closeEvent(self, event):
        if self.worker:
            QMessageBox.information(self, "Распознавание", "Сначала отмените текущую обработку.")
            event.ignore()
        else:
            event.accept()


def run_quick_app():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setQuitOnLastWindowClosed(True)
    window = QuickWindow()
    window.show()
    return app.exec()
