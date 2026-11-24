"""Desktop workflow for converting born-digital PDF drawings to PowerPoint."""
import re
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, QThread, Signal, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from .app import MainWindow, Preview
from .input import open_project
from .native_pdf import NativePDFError, pdf_info, preview_pdf
from .one_click import OneClickResult, ProcessingCancelled
from .pdf_batch import PdfBatchResult
from .pdf_pages import parse_pdf_pages
from .settings import Settings, load_settings, save_settings


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
            self.failed.emit(f"{type(exc).__name__}: {str(exc)[:500]}")


class QuickWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Overlay · Живые чертежи")
        self.resize(1100, 760)
        try:
            self.settings = load_settings()
        except (ValueError, OSError):
            self.settings = Settings()
            self.statusBar().showMessage("Не удалось прочитать настройки; использованы значения по умолчанию")
        self.path: Path | None = None
        self.result: OneClickResult | PdfBatchResult | None = None
        self.worker: ConvertWorker | None = None
        self.editor: MainWindow | None = None
        self.setFont(QFont("Segoe UI", 10))

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)

        title = QLabel("Живые чертежи → PowerPoint")
        title.setStyleSheet("font-size: 23px; font-weight: 600;")
        layout.addWidget(title)
        subtitle = QLabel("Цифровые PDF с текстовым слоем. Редактируемые надписи и таблицы в PPTX.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #777;")
        layout.addWidget(subtitle)

        chooser = QHBoxLayout()
        self.file_label = QLabel("Чертёж не выбран")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("padding: 10px;")
        chooser.addWidget(self.file_label, 1)
        self.pick = QPushButton("Выбрать PDF")
        self.pick.clicked.connect(self.choose)
        chooser.addWidget(self.pick)
        self.run_button = QPushButton("Создать PPTX")
        self.run_button.setMinimumHeight(43)
        self.run_button.setStyleSheet("font-weight: 600; padding: 8px 18px;")
        self.run_button.clicked.connect(self.recognize)
        chooser.addWidget(self.run_button)
        layout.addLayout(chooser)

        pdf_row = QHBoxLayout()
        self.pdf_pages_label = QLabel("Обработать страницы:")
        self.pdf_pages_input = QLineEdit()
        self.pdf_pages_input.setMinimumWidth(200)
        self.pdf_pages_input.setPlaceholderText("Пусто = все; например: 1, 2, 6-9")
        self.pdf_pages_input.setToolTip("Страницы для обработки, независимо от страницы предпросмотра.")
        self.pdf_pages_label.setBuddy(self.pdf_pages_input)
        pdf_row.addWidget(self.pdf_pages_label)
        pdf_row.addWidget(self.pdf_pages_input, 1)
        self.pdf_page_label = QLabel("Предпросмотр:")
        self.pdf_page = QSpinBox()
        self.pdf_page.setRange(1, 1)
        self.pdf_page.setToolTip("Только предпросмотр. Страницы для обработки задаются отдельно.")
        self.pdf_page.valueChanged.connect(self.preview_pdf_page)
        self.pdf_page_label.setBuddy(self.pdf_page)
        pdf_row.addWidget(self.pdf_page_label)
        pdf_row.addWidget(self.pdf_page)
        layout.addLayout(pdf_row)

        options = QWidget()
        options_layout = QVBoxLayout(options)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(4)
        self.pdf_tables_checkbox = QCheckBox("Реконструировать таблицы")
        self.pdf_tables_checkbox.setChecked(self.settings.native_pdf_detect_tables)
        self.pdf_tables_checkbox.setToolTip(
            "Восстанавливать редактируемые таблицы и объединения ячеек. "
            "Без галочки программа извлекает только текстовые блоки.")
        self.pdf_tables_checkbox.toggled.connect(self.save_pdf_options)
        options_layout.addWidget(self.pdf_tables_checkbox)
        self.pdf_ignore_numbers_checkbox = QCheckBox("Игнорировать числа")
        self.pdf_ignore_numbers_checkbox.setChecked(self.settings.native_pdf_ignore_numbers)
        self.pdf_ignore_numbers_checkbox.setToolTip(
            "Отдельные числовые надписи остаются только на PNG, без редактируемого текста. "
            "В реконструированных таблицах все числа сохраняются.")
        self.pdf_ignore_numbers_checkbox.toggled.connect(self.save_pdf_options)
        options_layout.addWidget(self.pdf_ignore_numbers_checkbox)
        self.pdf_split_lines_checkbox = QCheckBox("Не объединять строки")
        self.pdf_split_lines_checkbox.setChecked(self.settings.native_pdf_split_lines)
        self.pdf_split_lines_checkbox.setToolTip(
            "Каждая строка вне реконструированных таблиц станет отдельным фреймом. "
            "Содержимое ячеек не меняется.")
        self.pdf_split_lines_checkbox.toggled.connect(self.save_pdf_options)
        options_layout.addWidget(self.pdf_split_lines_checkbox)
        layout.addWidget(options)

        self.preview = Preview()
        self.preview.show_formatting = True
        layout.addWidget(self.preview, 1)

        status = QHBoxLayout()
        self.status_text = QLabel("Выберите PDF, из которого можно скопировать текст. Сканы не обрабатываются.")
        self.status_text.setWordWrap(True)
        status.addWidget(self.status_text, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(160)
        status.addWidget(self.progress)
        self.cancel = QPushButton("Отменить")
        self.cancel.clicked.connect(self.cancel_work)
        status.addWidget(self.cancel)
        layout.addLayout(status)

        actions = QHBoxLayout()
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        actions.addWidget(self.summary, 1)
        self.output_combo = QComboBox()
        self.output_combo.setMinimumContentsLength(25)
        self.output_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.output_combo.hide()
        actions.addWidget(self.output_combo)
        self.open_pptx = QPushButton("Открыть PPTX")
        self.open_pptx.clicked.connect(self.open_result)
        self.open_pptx.hide()
        actions.addWidget(self.open_pptx)
        self.review = QPushButton("Исправить результат…")
        self.review.clicked.connect(self.open_editor)
        self.review.hide()
        actions.addWidget(self.review)
        layout.addLayout(actions)
        self.set_busy(False)

    def save_pdf_options(self, _checked):
        self.settings.native_pdf_detect_tables = self.pdf_tables_checkbox.isChecked()
        self.settings.native_pdf_ignore_numbers = self.pdf_ignore_numbers_checkbox.isChecked()
        self.settings.native_pdf_split_lines = self.pdf_split_lines_checkbox.isChecked()
        try:
            save_settings(self.settings)
        except OSError:
            self.statusBar().showMessage("Не удалось сохранить галочки для следующего запуска")

    def preview_pdf_page(self):
        if self.path is None:
            return
        try:
            self.preview.show_project(preview_pdf(self.path, self.pdf_page.value()), fit=True)
        except (OSError, ValueError, NativePDFError) as exc:
            QMessageBox.warning(self, "Страница PDF", str(exc))

    def choose(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать чертёж", str(Path.home()), "Цифровые чертежи (*.pdf)")
        if not path:
            return
        selected = Path(path)
        if selected.suffix.lower() != ".pdf":
            QMessageBox.warning(self, "Открытие файла", "Выберите цифровой PDF с текстовым слоем.")
            return
        try:
            details = pdf_info(selected)
            project = preview_pdf(selected, 1)
        except (OSError, ValueError, NativePDFError) as exc:
            QMessageBox.warning(self, "Открытие файла", str(exc))
            return
        with QSignalBlocker(self.pdf_page):
            self.pdf_page.setRange(1, details["pages"])
            self.pdf_page.setValue(1)
        self.path = selected
        self.pdf_pages_input.clear()
        self.output_combo.clear()
        self.output_combo.hide()
        self.file_label.setText(str(selected))
        self.preview.show_project(project, fit=True)
        self.result = None
        self.open_pptx.hide()
        self.review.hide()
        self.summary.clear()
        self.statusBar().clearMessage()
        self.set_busy(False)
        self.status_text.setText("Презентации будут сохранены рядом с PDF, отдельно для каждого размера листа.")

    def recognize(self):
        source = self.path
        if source is None or source.suffix.lower() != ".pdf" or self.worker:
            return
        from .pdf_batch import convert_pdf_pages

        pages_spec = self.pdf_pages_input.text()
        try:
            parse_pdf_pages(pages_spec, self.pdf_page.maximum())
        except ValueError as exc:
            QMessageBox.warning(self, "Страницы PDF",
                                f"Неверный выбор {pages_spec!r}: {exc}\n"
                                "Укажите 1, 2, 6-9 или оставьте поле пустым для всех страниц.")
            return
        settings = self.settings.model_copy(deep=True)
        self.set_busy(True)
        self.summary.clear()
        self.status_text.setText("Начинаю обработку PDF…")

        def work(report, stop):
            return convert_pdf_pages(source, pages=pages_spec, settings=settings,
                                     progress=report, cancelled=stop)

        self.worker = ConvertWorker(work, self)
        self.worker.progress.connect(self.status_text.setText)
        self.worker.completed.connect(self.done)
        self.worker.failed.connect(self.failed)
        self.worker.stopped.connect(self.stopped)
        self.worker.finished.connect(self.finished)
        self.worker.start()

    def set_busy(self, busy):
        for widget in (self.pick, self.pdf_tables_checkbox, self.pdf_ignore_numbers_checkbox,
                       self.pdf_split_lines_checkbox,
                       self.open_pptx, self.output_combo):
            widget.setEnabled(not busy)
        has_pdf = self.path is not None and self.path.suffix.lower() == ".pdf"
        for widget in (self.run_button, self.pdf_page, self.pdf_pages_input):
            widget.setEnabled(not busy and has_pdf)
        self.progress.setVisible(busy)
        self.cancel.setVisible(busy)
        self.review.setEnabled(not busy and isinstance(self.result, OneClickResult))

    def cancel_work(self):
        if self.worker:
            self.worker.requestInterruption()
            self.status_text.setText("Отменяю обработку PDF…")
            self.cancel.setEnabled(False)

    def done(self, result):
        self.output_combo.clear()
        if len(result.selected_pages) != 1:
            self.result = result
            for group in result.groups:
                self.output_combo.addItem(
                    f"{group.pptx.name} · страницы {', '.join(map(str, group.pages))}", group.pptx)
            self.output_combo.setVisible(bool(result.groups))
            self.open_pptx.setVisible(bool(result.groups))
            self.review.hide()
            self.review.setEnabled(False)
            if result.groups and result.groups[0].project_files:
                self.preview.show_project(open_project(result.groups[0].project_files[0]))
            notes = [f"Готово: {len(result.selected_pages)} страниц; на проверку: {result.needs_review}."]
            notes.extend(str(group.pptx) for group in result.groups[:2])
            if len(result.groups) > 2:
                notes.append(f"Ещё файлов: {len(result.groups) - 2}; все доступны в списке.")
            notes.extend(result.warnings[:2])
            if len(result.warnings) > 2:
                notes.append(f"Других замечаний: {len(result.warnings) - 2}; подробности в проектах страниц.")
            notes.append("Предпросмотр: первая страница первой презентации.")
            self.summary.setText("\n".join(notes))
            self.status_text.setText("PowerPoint создан: выберите файл для открытия")
            return

        group = result.groups[0]
        single = OneClickResult(project=open_project(group.project_files[0]), pptx=group.pptx,
                                project_file=group.project_files[0], warnings=result.warnings)
        self.result = single
        self.output_combo.hide()
        self.preview.show_project(single.project)
        self.open_pptx.show()
        self.review.show()
        self.review.setEnabled(self.worker is None)
        notes = [f"Готово: {single.pptx.name}",
                 f"Объектов: {len(single.project.objects)}; на проверку: {single.needs_review}.",
                 "Видимый редактируемый текст; исходные надписи закрыты заливкой фреймов и ячеек."]
        detected = single.project.raw.get("native_pdf", {}).get("source_fonts", [])
        installed = {re.sub(r"[^a-z0-9]", "", family.lower()) for family in QFontDatabase.families()}
        missing = [family for family in detected if re.sub(r"[^a-z0-9]", "", family.lower()) not in installed]
        if missing:
            notes.append("Без исходных шрифтов PowerPoint может подставить другие: " + ", ".join(missing[:4]) + ".")
        if single.warnings:
            notes.append("Замечания: " + "; ".join(single.warnings[:2]))
            if len(single.warnings) > 2:
                notes.append(f"И ещё {len(single.warnings) - 2}.")
        self.summary.setText(" ".join(notes))
        self.status_text.setText("PowerPoint создан")
        self.statusBar().showMessage(str(single.pptx))

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
            path = self.result.pptx if isinstance(self.result, OneClickResult) else self.output_combo.currentData()
            if path:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def open_editor(self):
        if not isinstance(self.result, OneClickResult):
            return
        if self.editor and self.editor.isVisible() and not self.editor.close():
            return
        self.editor = MainWindow(native_only=True)
        self.editor.project = self.result.project.model_copy(deep=True)
        self.editor.settings = Settings.model_validate(self.editor.project.settings)
        self.editor.project_path = str(self.result.project_file) if self.result.project_file.exists() else None
        self.editor.refresh(fit=True)
        self.editor.show()

    def closeEvent(self, event):
        if self.worker:
            QMessageBox.information(self, "Обработка PDF", "Сначала отмените текущую обработку.")
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
