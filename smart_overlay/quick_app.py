"""Minimal Windows UI: choose a drawing, recognize, receive an editable PPTX.

The legacy object editor is kept as an optional correction tool, not the default
workflow. Advanced experiments and diagnostics remain CLI-only.
"""
import os
import json
import re
from uuid import uuid4
from pathlib import Path

from PySide6.QtCore import QThread, Signal, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QVBoxLayout, QWidget, QGridLayout, QRadioButton,
    QSpinBox,
)

from .app import Preview
from .input import load_image
from .languages import LANGUAGES
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
        self.cloud = QCheckBox("Для обычного Smart OCR разрешить облачную группировку")
        self.cloud.setChecked(enabled)
        form.addRow("", self.cloud)
        self.key = QLineEdit(key)
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText("sk-or-… (не сохраняется на диск)")
        form.addRow("API-ключ OpenRouter", self.key)
        self.cloud_model = QLineEdit(settings.cloud_ocr_model)
        self.cloud_model.setPlaceholderText("google/gemini-2.5-pro")
        self.cloud_model.setToolTip("Модель, которой будет отправлен весь чертёж в режиме облачного OCR")
        form.addRow("Модель полного облачного OCR", self.cloud_model)
        self.proxy = QLineEdit(proxy)
        self.proxy.setPlaceholderText("http://127.0.0.1:10809")
        form.addRow("Прокси Xray", self.proxy)
        layout.addLayout(form)
        notice = QLabel("Режим «Облако · весь чертёж» отправляет исходные изображения в OpenRouter и может тарифицироваться. Для него потребуется подтверждение каждого запуска. Локальные режимы не передают изображение без отдельного согласия.")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        controls = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        controls.accepted.connect(self.accept)
        controls.rejected.connect(self.reject)
        layout.addWidget(controls)

    def accept(self):
        if (self.cloud.isChecked() or self.parent() and
                getattr(self.parent(), "mode_combo", None) and
                self.parent().mode_combo.currentData() == "cloud_full") and not self.key.text().strip():
            QMessageBox.warning(self, "OpenRouter", "Для облачного распознавания укажите API-ключ или отключите облачный режим.")
            return
        super().accept()


class LanguageDialog(QDialog):
    """One compact chooser for automatic routing, presets and manual combinations."""

    PRESETS = [
        ("Русский + английский", ["ru", "en"]),
        ("Китайский + английский", ["ch", "en"]),
        ("Японский + английский", ["japan", "en"]),
        ("Китайский + русский", ["ch", "ru"]),
    ]

    def __init__(self, mode, languages, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Языки распознавания")
        self.setMinimumWidth(540)
        layout = QVBoxLayout(self)
        self.auto = QRadioButton("Определять автоматически")
        self.manual = QRadioButton("Указать языки")
        self.auto.setChecked(mode != "manual")
        self.manual.setChecked(mode == "manual")
        layout.addWidget(self.auto)
        layout.addWidget(self.manual)

        preset_grid = QGridLayout()
        for index, (title, codes) in enumerate(self.PRESETS):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, choice=codes: self.use_preset(choice))
            preset_grid.addWidget(button, index // 2, index % 2)
        layout.addLayout(preset_grid)

        self.checks = {}
        grid = QGridLayout()
        for i, (code, (title, _)) in enumerate(LANGUAGES.items()):
            check = QCheckBox(title)
            check.setChecked(code in languages)
            check.toggled.connect(self.on_check)
            self.checks[code] = check
            grid.addWidget(check, i // 3, i % 3)
        layout.addLayout(grid)
        self.hint = QLabel("Автоопределение проверяет несколько письменностей и может работать дольше.")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        self.auto.toggled.connect(self.sync)
        self.manual.toggled.connect(self.sync)
        self.sync()

        actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        actions.accepted.connect(self.accept)
        actions.rejected.connect(self.reject)
        layout.addWidget(actions)

    def on_check(self):
        if self.sender().isChecked():
            self.manual.setChecked(True)

    def use_preset(self, codes):
        for code, check in self.checks.items():
            check.blockSignals(True)
            check.setChecked(code in codes)
            check.blockSignals(False)
        self.manual.setChecked(True)
        self.sync()

    def sync(self):
        manual = self.manual.isChecked()
        for check in self.checks.values():
            check.setEnabled(manual)
        self.hint.setText(
            "Ручной набор: будут использованы только указанные семейства OCR-моделей."
            if manual else "Авто: китайский/японский, английский, кириллица, корейский и арабский. "
            "Неизвестные письменности могут потребовать ручного выбора.")

    def selected(self):
        return ("manual" if self.manual.isChecked() else "auto",
                [code for code, check in self.checks.items() if check.isChecked()])

    def accept(self):
        mode, selected = self.selected()
        if mode == "manual" and not selected:
            QMessageBox.warning(self, "Языки OCR", "Выберите хотя бы один язык.")
            return
        super().accept()


class QuickWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Overlay OCR")
        self.resize(1100, 760)
        self.settings = load_settings()
        config = data_dir() / "settings.json"
        explicit_mode = False
        if config.exists():
            try:
                explicit_mode = "ocr_language_mode" in json.loads(config.read_text("utf-8"))
            except (ValueError, OSError):
                pass
        if not explicit_mode or self.settings.ocr_language_mode == "single":
            # Older settings had only one language. The new quick UI defaults to auto.
            self.settings.ocr_version = "PP-OCRv5"
            self.settings.ocr_language_mode = "auto"
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

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Способ распознавания:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Smart OCR · обычный", "smart")
        self.mode_combo.addItem("Smart OCR · два распознавателя + VL", "hybrid_local")
        self.mode_combo.addItem("Облако · весь чертёж (эксперимент)", "cloud_full")
        self.mode_combo.addItem("Цифровой PDF · структура без OCR", "native_pdf")
        self.mode_combo.addItem("PaddleOCR-VL 1.6 · целиком (эксперимент)", "vl_local")
        engine_index = self.mode_combo.findData(self.settings.ocr_engine)
        self.mode_combo.setCurrentIndex(max(0, engine_index))
        self.mode_combo.currentIndexChanged.connect(self.mode_changed)
        mode_row.addWidget(self.mode_combo)
        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("GPU", "gpu:0")
        self.device_combo.setCurrentIndex(0 if self.settings.vl_device == "cpu" else 1)
        self.device_combo.setToolTip("GPU может потребовать более 6 ГБ видеопамяти. CPU медленнее.")
        self.device_combo.currentIndexChanged.connect(self.device_changed)
        mode_row.addWidget(self.device_combo)
        self.pdf_page_label = QLabel("Страница:")
        self.pdf_page = QSpinBox()
        self.pdf_page.setRange(1,1)
        self.pdf_page.setToolTip("Пока одна страница PDF конвертируется в один PPTX")
        self.pdf_page.valueChanged.connect(self.preview_pdf_page)
        mode_row.addWidget(self.pdf_page_label)
        mode_row.addWidget(self.pdf_page)
        self.pdf_overlay_combo = QComboBox()
        self.pdf_overlay_combo.addItem("Видимые надписи · как OCR", "ocr_style")
        self.pdf_overlay_combo.addItem("Точный оригинал · невидимые редактируемые слои", "source_exact")
        index = self.pdf_overlay_combo.findData(self.settings.native_pdf_overlay_mode)
        self.pdf_overlay_combo.setCurrentIndex(max(0,index))
        self.pdf_overlay_combo.setToolTip(
            "Точный оригинал: редактируемые текстовые объекты присутствуют, "
            "но их цвет полностью прозрачен. Для видимого перевода "
            "выбирайте «как OCR».")
        self.pdf_overlay_combo.currentIndexChanged.connect(self.pdf_overlay_changed)
        mode_row.addWidget(self.pdf_overlay_combo)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        language_row = QHBoxLayout()
        language_row.addWidget(QLabel("Языки OCR:"))
        self.language_button = QPushButton()
        self.language_button.clicked.connect(self.choose_languages)
        language_row.addWidget(self.language_button)
        language_row.addStretch(1)
        layout.addLayout(language_row)
        self.refresh_languages()
        self.mode_changed()

        self.preview = Preview()
        self.preview.show_formatting = True
        layout.addWidget(self.preview, 1)

        status = QHBoxLayout()
        self.status_text = QLabel("Откройте изображение или цифровой PDF")
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

    def mode_changed(self, *_):
        engine = self.mode_combo.currentData()
        local = engine in ("vl_local", "hybrid_local")
        self.settings.ocr_engine = engine
        self.language_button.setEnabled(engine not in ("vl_local", "cloud_full", "native_pdf") and self.worker is None)
        self.device_combo.setEnabled(local and self.worker is None)
        self.device_combo.setVisible(local)
        pdf_selected = bool(self.path and self.path.suffix.lower() == ".pdf")
        self.pdf_page_label.setVisible(engine == "native_pdf" and pdf_selected)
        self.pdf_page.setVisible(engine == "native_pdf" and pdf_selected)
        self.pdf_overlay_combo.setVisible(engine == "native_pdf" and pdf_selected)
        self.run_button.setText("Извлечь PDF → PPTX" if engine == "native_pdf"
                                else "Распознать → PPTX")
        if self.path and self.worker is None:
            self.run_button.setEnabled((engine == "native_pdf") == pdf_selected)
        self.statusBar().showMessage(
            "Цифровой PDF: прямое извлечение текста и таблиц без OCR и без облака"
            if engine == "native_pdf" else
            "Полностью облачное OCR: модель читает изображение напрямую, запросы платные"
            if engine == "cloud_full" else
            "Гибрид: PP-OCRv5 + совместимые модели PP-OCRv6, затем пакетная VL-проверка; без облака"
            if engine == "hybrid_local" else
            "VL полностью локальный: экспериментальный режим для небольших листов"
            if engine == "vl_local" else "Smart OCR готов")
        self._save_mode()

    def pdf_overlay_changed(self, *_):
        self.settings.native_pdf_overlay_mode = self.pdf_overlay_combo.currentData()
        self._save_mode()

    def device_changed(self, *_):
        self.settings.vl_device = self.device_combo.currentData()
        self._save_mode()

    def _save_mode(self):
        try:
            save_settings(self.settings)
        except OSError:
            self.statusBar().showMessage("Не удалось сохранить настройки запуска")

    def refresh_languages(self):
        if self.settings.ocr_language_mode != "manual":
            label = "Автоопределение"
        else:
            codes = self.settings.ocr_languages
            label = " + ".join(LANGUAGES[code][0] for code in codes if code in LANGUAGES)
        self.language_button.setText(label + "  ▾")
        self.language_button.setToolTip("Выбрать языковую пару или несколько языков OCR")

    def choose_languages(self):
        dialog = LanguageDialog(self.settings.ocr_language_mode, self.settings.ocr_languages, self)
        if not dialog.exec():
            return
        mode, codes = dialog.selected()
        previous = self.settings.model_copy(deep=True)
        try:
            self.settings.ocr_version = "PP-OCRv5"
            self.settings.ocr_language_mode = mode
            self.settings.ocr_languages = codes
            self.settings = type(self.settings).model_validate(self.settings.model_dump())
            save_settings(self.settings)
        except (ValueError, OSError) as exc:
            self.settings = previous
            QMessageBox.warning(self, "Языки OCR", str(exc))
        self.refresh_languages()

    def preview_pdf_page(self, *_):
        if not self.path or self.path.suffix.lower() != ".pdf":
            return
        from .native_pdf import preview_pdf, NativePDFError
        try:
            project = preview_pdf(self.path,self.pdf_page.value())
            self.preview.show_project(project,fit=True)
        except (OSError,ValueError,NativePDFError) as exc:
            QMessageBox.warning(self,"Страница PDF",str(exc))

    def choose(self):
        path, _ = QFileDialog.getOpenFileName(
            self,"Выбрать чертёж",str(Path.home()),
            "Чертежи и цифровые PDF (*.pdf *.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if not path:
            return
        selected=Path(path)
        try:
            if selected.suffix.lower() == ".pdf":
                from .native_pdf import pdf_info, preview_pdf
                details=pdf_info(path)
                self.pdf_page.blockSignals(True)
                self.pdf_page.setRange(1,details["pages"])
                self.pdf_page.setValue(1)
                self.pdf_page.blockSignals(False)
                project=preview_pdf(path,1)
            else:
                project=load_image(path)
                self.pdf_page.blockSignals(True)
                self.pdf_page.setRange(1,1)
                self.pdf_page.blockSignals(False)
        except (OSError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self,"Открытие файла",str(exc))
            return
        self.path=selected
        if selected.suffix.lower()==".pdf":
            native_index=self.mode_combo.findData("native_pdf")
            self.mode_combo.setCurrentIndex(native_index)
        elif self.mode_combo.currentData()=="native_pdf":
            self.mode_combo.setCurrentIndex(self.mode_combo.findData("smart"))
        self.mode_changed()
        self.file_label.setText(str(self.path))
        self.preview.show_project(project,fit=True)
        self.result=None
        self.open_pptx.hide()
        self.review.hide()
        self.summary.setText("")
        self.run_button.setEnabled(True)
        self.status_text.setText(
            "Цифровой PDF: текст и таблицы будут извлечены напрямую; "
            "исходные линии сохранятся на фоне"
            if selected.suffix.lower()==".pdf" else "Можно распознавать")

    def configure(self):
        dialog = ConnectionDialog(self.settings, self.api_key, self.proxy, self.cloud_enabled, self)
        if not dialog.exec():
            return
        self.api_key = dialog.key.text().strip()
        self.proxy = dialog.proxy.text().strip()
        self.cloud_enabled = dialog.cloud.isChecked()
        self.settings.cloud_ocr_model = dialog.cloud_model.text().strip()
        try:
            self.settings = type(self.settings).model_validate(self.settings.model_dump())
            save_settings(self.settings)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Настройки", str(exc))
        self.status_text.setText("Параметры сохранены. Ключ хранится только в памяти приложения.")

    def recognize(self):
        if not self.path or self.worker:
            return
        consent = False
        engine = self.mode_combo.currentData()
        local_vl = engine == "vl_local"
        if engine == "cloud_full":
            if not self.api_key:
                QMessageBox.warning(
                    self, "Облачное OCR",
                    "Для полного облачного распознавания введите API-ключ в «Настройки».")
                return
            from .cloud_ocr import balanced_cloud_tiles, CloudOCRError
            try:
                source_info = load_image(self.path).source
                plan = balanced_cloud_tiles(
                    source_info.width, source_info.height,
                    size=self.settings.cloud_ocr_tile_size,
                    overlap=self.settings.cloud_ocr_overlap,
                    max_tiles=self.settings.cloud_ocr_max_requests)
            except (OSError, ValueError, CloudOCRError) as exc:
                QMessageBox.warning(self, "План облачного OCR", str(exc))
                return
            answer = QMessageBox.question(
                self, "Подтверждение платного облачного OCR",
                f"Отправить исходный чертёж в OpenRouter для полного распознавания?\n"
                f"Модель: {self.settings.cloud_ocr_model}\n"
                f"План: {len(plan)} запросов; области до "
                f"{self.settings.cloud_ocr_tile_size} пикселей, "
                f"перекрытие ~{self.settings.cloud_ocr_overlap} пикселей.\n"
                f"Лимит ответа: {self.settings.cloud_ocr_max_tokens} токенов на запрос.\n"
                "Запросы могут быть платными. После отправки оплата возможна даже при отмене.\n"
                "Автоматических повторов нет; локальный OCR не используется.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
            consent = True
        elif self.cloud_enabled and engine == "smart":
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
        pdf_page = self.pdf_page.value()
        device = self.device_combo.currentData()
        diagnostics = (data_dir() / "diagnostics" /
                       ("cloud-full" if engine == "cloud_full" else
                        "paddle-vl-hybrid" if engine == "hybrid_local" else "paddle-vl")
                       / (self.path.stem + "-" + uuid4().hex[:8])
                       if engine in ("vl_local", "hybrid_local", "cloud_full") else None)
        self.set_busy(True)
        self.summary.setText("")
        self.status_text.setText("Начинаю распознавание…")
        self.worker = ConvertWorker(
            lambda report, stop: convert_drawing(source, settings=settings, api_key=key,
                                                 consent=consent, proxy=proxy,
                                                 ocr_engine=engine, vl_device=device,
                                                 vl_diagnostics=diagnostics,
                                                 pdf_page=pdf_page,
                                                 progress=report, cancelled=stop), self)
        self.worker.progress.connect(self.status_text.setText)
        self.worker.completed.connect(self.done)
        self.worker.failed.connect(self.failed)
        self.worker.stopped.connect(self.stopped)
        self.worker.finished.connect(self.finished)
        self.worker.start()

    def set_busy(self, busy):
        for widget in (self.pick, self.run_button, self.config,
                       self.language_button, self.mode_combo, self.device_combo,
                       self.pdf_page, self.pdf_overlay_combo):
            widget.setEnabled(not busy)
        self.progress.setVisible(busy)
        self.cancel.setVisible(busy)
        self.open_pptx.setEnabled(not busy)
        self.review.setEnabled(not busy)

    def cancel_work(self):
        if self.worker:
            self.worker.requestInterruption()
            self.status_text.setText("Останавливаю отдельный процесс VL…")
            self.cancel.setEnabled(False)

    def done(self, result):
        self.result = result
        self.preview.show_project(result.project)
        self.open_pptx.show()
        self.review.show()
        notes = [f"Готово: {result.pptx.name}",
                 f"Объектов: {len(result.project.objects)}; на проверку: {result.needs_review}."]
        if result.project.source.mode == "PDF":
            if result.project.raw.get("native_pdf",{}).get("overlay_mode")=="source_exact":
                notes.append("Точный оригинал: редактируемый текст невидим. "
                             "Для видимого перевода требуется экспорт «как OCR».")
            detected=result.project.raw.get("native_pdf",{}).get("source_fonts",[])
            installed={re.sub(r"[^a-z0-9]","",family.lower())
                       for family in QFontDatabase.families()}
            missing=[family for family in detected
                     if re.sub(r"[^a-z0-9]","",family.lower()) not in installed]
            if missing:
                notes.append("Для точного совпадения шрифтов установите "
                             "их лицензированные версии в Windows: "+
                             ", ".join(missing[:4])+
                             ". Без них PowerPoint подставляет другой шрифт.")
        if result.warnings:
            notes.append("Замечания: " + "; ".join(result.warnings[:2]))
            if len(result.warnings) > 2:
                notes.append(f"И ещё {len(result.warnings) - 2}.")
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
        self.mode_changed()
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
