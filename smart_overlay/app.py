import base64
import logging
import os
from pathlib import Path

from pydantic import ValidationError
from PySide6.QtCore import Qt, QThread, Signal, QRectF
from PySide6.QtGui import QAction, QColor, QFont, QFontDatabase, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGraphicsRectItem,
    QGraphicsScene, QGraphicsView, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTextEdit,
    QToolBar, QVBoxLayout, QWidget,
)

from .export import export_pptx
from .geometry import contains, union
from .input import load_image, open_project, save_project
from .models import Box, Project, Table, Text, uid
from .ocr import PaddleEngine
from .pipeline import SmartCancelled, SmartFailure, local_ocr, reocr_text, smart_ocr
from .settings import Settings, data_dir, load_settings, save_settings
from .structure import manual_text, merge_texts, split_text, validate_project
from .tables import assign_fragments, cell_box, merge_cells, new_table, split_cell

LOG = logging.getLogger(__name__)


class Worker(QThread):
    result = Signal(object)
    failed = Signal(str)
    progress = Signal(str)
    fallback = Signal(object, str)
    cancelled = Signal(object)

    def __init__(self, operation, secret="", parent=None):
        super().__init__(parent)
        self.operation, self.secret = operation, secret

    def run(self):
        try:
            self.result.emit(self.operation(self.progress.emit))
        except SmartCancelled as exc:
            self.cancelled.emit(exc.project)
        except SmartFailure as exc:
            self.fallback.emit(exc.project, str(exc))
        except Exception as exc:
            # UI boundary: report unexpected failures without leaking provider data/keys.
            message = f"{type(exc).__name__}: {exc}"
            if self.secret:
                message = message.replace(self.secret, "[API KEY]")
            LOG.error("Background operation failed: %s", type(exc).__name__)
            self.failed.emit(message[:1500])


class Preview(QGraphicsView):
    selected = Signal(str)
    region = Signal(object)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.drawing = False
        self.start = None
        self.rubber = None
        self.setBackgroundBrush(QColor("#242a34"))

    def show_project(self, project: Project, fit=False):
        self.scene().clear()
        pixmap = QPixmap()
        pixmap.loadFromData(base64.b64decode(project.source.raster_b64))
        self.scene().addPixmap(pixmap)
        self.setSceneRect(0, 0, project.source.width, project.source.height)
        for obj in project.objects:
            if obj.excluded:
                continue
            b = obj.bbox
            item = QGraphicsRectItem(b.x, b.y, b.w, b.h)
            color = QColor("#e5b92e" if obj.needs_review else "#3e9fe8" if isinstance(obj, Table) else "#37bd7d")
            item.setPen(QPen(color, 2))
            color.setAlpha(30)
            item.setBrush(color)
            item.setData(0, obj.id)
            item.setZValue(1)
            item.setToolTip((obj.text if isinstance(obj, Text) else f"Таблица {obj.rows} × {obj.columns}") +
                            "\n" + "\n".join(obj.review_notes))
            self.scene().addItem(item)
            if isinstance(obj, Table):
                for cell in obj.cells:
                    cb = cell_box(obj, cell)
                    grid = self.scene().addRect(cb.x, cb.y, cb.w, cb.h, QPen(QColor("#3e9fe8"), 1))
                    grid.setData(0, obj.id)
                    grid.setZValue(2)
        if fit:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        if 0.02 <= self.transform().m11()*factor <= 20:
            self.scale(factor, factor)

    def draw_region(self):
        self.drawing = True
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, event):
        if self.drawing:
            self.start = self.mapToScene(event.position().toPoint())
            self.rubber = self.scene().addRect(QRectF(self.start, self.start), QPen(QColor("#ffffff"), 2))
            return
        item = self.itemAt(event.position().toPoint())
        if item and item.data(0):
            self.selected.emit(item.data(0))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.start is not None:
            self.rubber.setRect(QRectF(self.start, self.mapToScene(event.position().toPoint())).normalized())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing and self.start is not None:
            rect = self.rubber.rect().intersected(self.sceneRect())
            self.scene().removeItem(self.rubber)
            self.rubber = self.start = None
            self.drawing = False
            self.unsetCursor()
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            if rect.width() > 2 and rect.height() > 2:
                self.region.emit(Box(x=rect.x(), y=rect.y(), w=rect.width(), h=rect.height()))
            return
        super().mouseReleaseEvent(event)


def button(text, callback, layout):
    widget = QPushButton(text)
    widget.clicked.connect(callback)
    layout.addWidget(widget)
    return widget


def spin(value, low=0, high=100000, decimals=2):
    widget = QDoubleSpinBox()
    widget.setRange(low, high)
    widget.setDecimals(decimals)
    widget.setValue(value)
    return widget


class SettingsDialog(QDialog):
    def __init__(self, settings, api_key, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки OCR и OpenRouter")
        self.setMinimumWidth(520)
        form = QFormLayout(self)
        self.fields = {}
        labels = {"language": "Язык OCR (en, ru, …)", "ocr_version": "Профиль OCR",
                  "endpoint": "API endpoint (без /chat/completions)", "model": "ID мультимодальной модели",
                  "retries": "Повторы запроса", "padding_x": "Запас X, px", "padding_y": "Запас Y, px",
                  "connect_timeout": "Тайм-аут подключения, с", "request_timeout": "Общий лимит попытки API, с",
                  "stream_responses": "Потоковый ответ VL", "max_output_tokens": "Лимит выходных токенов VL",
                  "fill_mode": "Заливка", "font": "Шрифт", "min_font": "Минимальный шрифт, pt",
                  "max_font": "Максимальный шрифт, pt", "confidence_threshold": "Порог уверенности",
                  "inches_per_pixel": "Дюймов на пиксель (1 / DPI)", "tile_size": "Размер фрагмента VL, px"}
        for name, value in settings.model_dump().items():
            if isinstance(value, bool):
                field = QCheckBox()
                field.setChecked(value)
            elif name in {"ocr_version", "fill_mode"}:
                field = QComboBox()
                field.addItems(["PP-OCRv5", "PP-OCRv4", "PP-OCRv3"] if name == "ocr_version" else ["white", "auto"])
                field.setCurrentText(value)
            else:
                field = QLineEdit(str(value))
            self.fields[name] = field
            form.addRow(labels[name], field)
        self.key = QLineEdit(api_key)
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("API-ключ (только на время сеанса)", self.key)
        form.addRow(QLabel("Ключ не сохраняется в настройках и проектах.\nМожно задать переменную OPENROUTER_API_KEY."))
        controls = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        controls.accepted.connect(self.accept)
        controls.rejected.connect(self.reject)
        form.addRow(controls)

    def value(self):
        return Settings.model_validate({k: w.isChecked() if isinstance(w, QCheckBox) else w.currentText() if isinstance(w, QComboBox) else w.text()
                                        for k, w in self.fields.items()})

    def accept(self):
        try:
            self.value()
        except ValidationError as exc:
            QMessageBox.warning(self, "Проверьте настройки", str(exc))
            return
        super().accept()


class TableDialog(QDialog):
    def __init__(self, table, parent=None):
        super().__init__(parent)
        self.table = table.model_copy(deep=True)
        self.setWindowTitle("Структура таблицы — выделите ячейки для объединения")
        self.resize(800, 560)
        layout = QVBoxLayout(self)
        dimensions = QHBoxLayout()
        self.rows, self.cols = QSpinBox(), QSpinBox()
        self.rows.setRange(1, 500)
        self.cols.setRange(1, 100)
        self.rows.setValue(table.rows)
        self.cols.setValue(table.columns)
        dimensions.addWidget(QLabel("Строки")); dimensions.addWidget(self.rows)
        dimensions.addWidget(QLabel("Столбцы")); dimensions.addWidget(self.cols)
        button("Перестроить сетку", self.regrid, dimensions)
        layout.addLayout(dimensions)
        self.grid = QTableWidget()
        self.grid.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        layout.addWidget(self.grid)
        form = QFormLayout()
        self.widths, self.heights = QLineEdit(), QLineEdit()
        form.addRow("Ширина столбцов, px (через ;)", self.widths)
        form.addRow("Высота строк, px (через ;)", self.heights)
        layout.addLayout(form)
        actions = QHBoxLayout()
        button("Объединить ячейки", self.merge, actions)
        button("Разделить ячейку", self.split, actions)
        layout.addLayout(actions)
        cell_style = QHBoxLayout()
        self.cell_fill = QLineEdit("#FFFFFF")
        self.cell_border = spin(0.5, 0, 20)
        self.cell_border_color = QLineEdit("#000000")
        cell_style.addWidget(QLabel("Заливка / граница, pt / цвет"))
        cell_style.addWidget(self.cell_fill)
        cell_style.addWidget(self.cell_border)
        cell_style.addWidget(self.cell_border_color)
        button("К выбранным", self.style_cells, cell_style)
        layout.addLayout(cell_style)
        self.review = QCheckBox("Структура и содержимое таблицы проверены")
        self.review.setChecked(not table.needs_review)
        layout.addWidget(self.review)
        controls = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        controls.accepted.connect(self.accept)
        controls.rejected.connect(self.reject)
        layout.addWidget(controls)
        self.render()

    def render(self):
        self.grid.clearSpans()
        self.grid.setRowCount(self.table.rows)
        self.grid.setColumnCount(self.table.columns)
        self.grid.clearContents()
        for cell in self.table.cells:
            self.grid.setItem(cell.row, cell.col, QTableWidgetItem(cell.text))
            if cell.rowspan > 1 or cell.colspan > 1:
                self.grid.setSpan(cell.row, cell.col, cell.rowspan, cell.colspan)
        self.widths.setText("; ".join(f"{v:.4f}" for v in self.table.col_widths))
        self.heights.setText("; ".join(f"{v:.4f}" for v in self.table.row_heights))

    def capture(self):
        for cell in self.table.cells:
            item = self.grid.item(cell.row, cell.col)
            cell.text = item.text() if item else ""

    def regrid(self):
        if QMessageBox.question(self, "Перестроение", "Перестроить сетку? Текст будет сохранён в первой ячейке для ручного распределения.") != QMessageBox.StandardButton.Yes:
            return
        self.capture()
        previous = self.table
        self.table = new_table(previous.bbox, self.rows.value(), self.cols.value(), previous.style)
        self.table.id, self.table.ocr_ids = previous.id, previous.ocr_ids
        self.table.corrections = previous.corrections
        self.table.excluded = previous.excluded
        self.table.cells[0].text = "\n".join(c.text for c in previous.cells if c.text)
        self.table.cells[0].ocr_ids = previous.ocr_ids
        self.review.setChecked(False)
        self.render()

    def merge(self):
        ranges = self.grid.selectedRanges()
        if not ranges:
            return
        self.capture()
        r = ranges[0]
        try:
            self.table = merge_cells(self.table, r.topRow(), r.leftColumn(), r.bottomRow(), r.rightColumn())
            self.review.setChecked(False)
            self.render()
        except ValueError as exc:
            QMessageBox.warning(self, "Объединение", str(exc))

    def split(self):
        r, c = self.grid.currentRow(), self.grid.currentColumn()
        self.capture()
        if any(cell.row == r and cell.col == c for cell in self.table.cells):
            self.table = split_cell(self.table, r, c)
            self.review.setChecked(False)
            self.render()

    def style_cells(self):
        ranges = self.grid.selectedRanges()
        if not ranges:
            return
        r = ranges[0]
        for cell in self.table.cells:
            if r.topRow() <= cell.row <= r.bottomRow() and r.leftColumn() <= cell.col <= r.rightColumn():
                cell.style.fill = self.cell_fill.text()
                cell.style.border_pt = self.cell_border.value()
                cell.style.border_color = self.cell_border_color.text()

    def accept(self):
        try:
            self.capture()
            widths = [float(v.strip()) for v in self.widths.text().split(";")]
            heights = [float(v.strip()) for v in self.heights.text().split(";")]
            self.table.col_widths = widths
            self.table.row_heights = heights
            self.table.needs_review = not self.review.isChecked()
            self.table = Table.model_validate(self.table.model_dump())
        except ValueError as exc:
            QMessageBox.warning(self, "Некорректная сетка", str(exc))
            return
        super().accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Overlay OCR · MVP 0.1")
        self.resize(1320, 850)
        self.project = None
        self.project_path = None
        self.dirty = False
        self.worker = None
        self.pending_smart = False
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        # The offscreen Qt platform has no system font database on Windows.
        # Loading the installed UI font also makes headless visual QA meaningful.
        if not QFontDatabase.families():
            system_font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
            if system_font.exists():
                QFontDatabase.addApplicationFont(str(system_font))
        self.setFont(QFont("Segoe UI", 9))
        try:
            self.settings = load_settings()
        except (ValueError, OSError) as exc:
            self.settings = Settings()
            QMessageBox.warning(self, "Настройки", "Не удалось прочитать настройки: " + str(exc))
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self.actions = []
        for title, callback in [("Открыть изображение", self.open_image), ("Открыть проект", self.open_saved),
                                ("Сохранить проект", self.save), ("Настройки", self.configure)]:
            action = QAction(title, self)
            action.triggered.connect(callback)
            toolbar.addAction(action)
            self.actions.append(action)
        self.mode = QComboBox()
        self.mode.addItems(["Smart OCR", "OCR Only"])
        toolbar.addWidget(self.mode)
        self.recognize_button = QPushButton("Распознать")
        self.recognize_button.clicked.connect(self.recognize)
        toolbar.addWidget(self.recognize_button)
        self.smart_button = QPushButton("VL для текущего OCR")
        self.smart_button.clicked.connect(self.run_smart)
        toolbar.addWidget(self.smart_button)
        self.cancel_button = QPushButton("Отменить Smart OCR")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_smart)
        toolbar.addWidget(self.cancel_button)
        self.export_button = QPushButton("Экспорт в PPTX")
        self.export_button.clicked.connect(self.export)
        toolbar.addWidget(self.export_button)
        splitter = QSplitter()
        self.setCentralWidget(splitter)
        left = QWidget()
        left.setMinimumWidth(200)
        left.setMaximumWidth(285)
        layout = QVBoxLayout(left)
        legend = QLabel("Объекты · Ctrl для выбора нескольких\nЗелёный: текст · Синий: таблица · Жёлтый: проверить")
        legend.setWordWrap(True)
        layout.addWidget(legend)
        self.objects = QListWidget()
        self.objects.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.objects.itemSelectionChanged.connect(self.inspect)
        layout.addWidget(self.objects)
        for title, fn in [("Добавить текст (область)", self.add_region),
                          ("Объединить блоки", self.merge), ("Разделить блок", self.split),
                          ("Таблица ↔ текст", self.convert), ("Повторное OCR области", self.reocr)]:
            button(title, fn, layout)
        splitter.addWidget(left)
        self.preview = Preview()
        self.preview.selected.connect(self.select)
        self.preview.region.connect(self.add_text)
        splitter.addWidget(self.preview)
        right = QWidget()
        right.setMinimumWidth(290)
        right.setMaximumWidth(355)
        props = QVBoxLayout(right)
        self.content = QTextEdit()
        self.content.setPlaceholderText("Выберите объект. Таблица открывается отдельной кнопкой.")
        props.addWidget(self.content)
        form = QFormLayout()
        self.coords = {k: spin(0, 0.01 if k in {"w", "h"} else 0) for k in ("x", "y", "w", "h")}
        for k, w in self.coords.items():
            form.addRow(k + ", px", w)
        self.rotation = spin(0, -360, 360)
        self.fill = QLineEdit("#FFFFFF")
        self.color = QLineEdit("#000000")
        self.font = QLineEdit("Arial")
        self.font_size = spin(12, 1, 256)
        self.align = QComboBox(); self.align.addItems(["left", "center", "right"])
        self.border = spin(0.5, 0, 20)
        self.border_color = QLineEdit("#000000")
        for label, w in [("Поворот, °", self.rotation), ("Заливка #RRGGBB", self.fill),
                         ("Цвет текста", self.color), ("Шрифт", self.font), ("Размер, pt", self.font_size),
                         ("Выравнивание", self.align), ("Граница таблицы, pt", self.border),
                         ("Цвет границы", self.border_color)]:
            form.addRow(label, w)
        props.addLayout(form)
        self.excluded = QCheckBox("Исключить из экспорта")
        self.reviewed = QCheckBox("Объект проверен")
        props.addWidget(self.excluded); props.addWidget(self.reviewed)
        button("Применить свойства", self.apply_properties, props)
        button("Редактировать таблицу", self.edit_table, props)
        self.notes = QLabel()
        self.notes.setWordWrap(True)
        props.addWidget(self.notes)
        self.journal = QTextEdit()
        self.journal.setReadOnly(True)
        self.journal.setMaximumHeight(140)
        props.addWidget(self.journal)
        splitter.addWidget(right)
        splitter.setSizes([230, 760, 320])
        self.statusBar().showMessage("Откройте изображение или сохранённый проект. Колесо — масштаб; перетаскивание — перемещение вида.")

    def error(self, text):
        QMessageBox.warning(self, "Smart Overlay OCR", str(text))

    def selected_ids(self):
        return [item.data(Qt.ItemDataRole.UserRole) for item in self.objects.selectedItems()]

    def current(self):
        ids = self.selected_ids()
        return next((o for o in self.project.objects if ids and o.id == ids[0]), None) if self.project else None

    def confirm_discard(self):
        if not self.dirty:
            return True
        answer = QMessageBox.question(self, "Несохранённые изменения", "Сохранить изменения проекта?",
                                      QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        return self.save() if answer == QMessageBox.StandardButton.Save else answer == QMessageBox.StandardButton.Discard

    def open_image(self):
        if not self.confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Открыть изображение", "", "Изображения (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if path:
            try:
                self.project = load_image(path)
                self.project_path, self.dirty = None, True
                self.refresh(fit=True)
            except (ValueError, OSError) as exc:
                self.error(exc)

    def open_saved(self):
        if not self.confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Открыть проект", "", "Проект (*.socr.json)")
        if path:
            try:
                project = open_project(path)
                settings = Settings.model_validate(project.settings) if project.settings else self.settings
                self.project, self.settings = project, settings
                self.project_path, self.dirty = path, False
                self.refresh(fit=True)
            except (ValueError, OSError) as exc:
                self.error(exc)

    def save(self):
        if not self.project:
            return False
        path = self.project_path
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", str(Path(self.project.source.filename).with_suffix(".socr.json")), "Проект (*.socr.json)")
        if not path:
            return False
        try:
            self.project.settings = self.settings.model_dump()
            save_project(self.project, path)
            self.project_path, self.dirty = path, False
            self.statusBar().showMessage("Проект сохранён: " + path)
            return True
        except (ValueError, OSError) as exc:
            self.error(exc)
            return False

    def configure(self):
        dialog = SettingsDialog(self.settings, self.api_key, self)
        if dialog.exec():
            self.settings, self.api_key = dialog.value(), dialog.key.text()
            try:
                save_settings(self.settings)
            except OSError as exc:
                self.error(exc)
            self.dirty = self.project is not None

    def set_busy(self, busy):
        for widget in [self.centralWidget(), self.recognize_button, self.smart_button, self.export_button, self.mode]:
            widget.setEnabled(not busy)
        for action in self.actions:
            action.setEnabled(not busy)

    def start(self, operation, *, cancellable=False):
        if self.worker:
            return
        self.set_busy(True)
        self.worker = Worker(operation, self.api_key, self)
        self.cancel_button.setEnabled(cancellable)
        self.worker.progress.connect(self.statusBar().showMessage)
        self.worker.result.connect(self.received)
        self.worker.failed.connect(self.worker_error)
        self.worker.fallback.connect(self.fallback)
        self.worker.cancelled.connect(self.operation_cancelled)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()

    def cancel_smart(self):
        if self.worker and self.cancel_button.isEnabled():
            self.worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.statusBar().showMessage("Отмена Smart OCR; локальный результат будет сохранён…")

    def cancellation_requested(self):
        return bool(self.worker and self.worker.isInterruptionRequested())

    def operation_cancelled(self, project):
        self.received(project)
        self.statusBar().showMessage("Smart OCR отменён; локальные результаты сохранены")

    def worker_error(self, message):
        self.pending_smart = False
        self.error(message)

    def worker_finished(self):
        worker = self.worker
        self.worker = None
        self.cancel_button.setEnabled(False)
        self.set_busy(False)
        worker.deleteLater()
        if self.pending_smart:
            self.pending_smart = False
            self.run_smart()

    def recognize(self):
        if not self.project:
            return
        if self.project.objects and QMessageBox.question(self, "Новое OCR", "Повторное распознавание заменит текущие объекты. Продолжить?") != QMessageBox.StandardButton.Yes:
            return
        self.pending_smart = self.mode.currentText() == "Smart OCR"
        project, settings = self.project.model_copy(deep=True), self.settings.model_copy()
        self.start(lambda progress: local_ocr(project, settings, progress=progress))

    def run_smart(self):
        if not self.project or not self.project.fragments:
            return
        if not self.settings.model or not self.api_key:
            self.error("Укажите VL-модель и API-ключ в настройках. Локальный OCR сохранён.")
            return
        consent = QMessageBox.question(self, "Отправка во внешний API",
                    f"Отправить исходное изображение, фрагменты и OCR-текст на {self.settings.endpoint}\n"
                    f"Модель: {self.settings.model}. Запросы могут быть платными.\n"
                    "Текущая группировка и оформление будут пересчитаны.\n"
                    "Кеш ответов и распознанный текст хранятся локально. Продолжить?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No)
        if consent != QMessageBox.StandardButton.Yes:
            return
        project, settings, key = self.project.model_copy(deep=True), self.settings.model_copy(), self.api_key
        self.start(lambda progress: smart_ocr(project, settings, key, data_dir()/"cache", True,
                                             progress=progress, cancelled=self.cancellation_requested), cancellable=True)

    def received(self, result):
        if isinstance(result, Project):
            self.project, self.dirty = result, True
            self.refresh()
            self.journal.setPlainText("\n".join(result.messages) + "\n" + "\n".join(str(u) for u in result.usage))
            self.statusBar().showMessage(f"Объектов: {len(result.objects)}; требуют проверки: {sum(o.needs_review for o in result.objects)}")
        elif isinstance(result, str):
            self.statusBar().showMessage(result)

    def fallback(self, project, reason):
        self.received(project)
        QMessageBox.information(self, "Smart OCR не завершён", "Обработка VL не завершена: " + reason +
                                "\nЛокальные результаты сохранены. Можно продолжить в OCR Only или повторить VL для текущего OCR.")
        self.mode.setCurrentText("OCR Only")

    def refresh(self, fit=False, selected=None):
        if not self.project:
            return
        self.objects.blockSignals(True)
        self.objects.clear()
        for obj in self.project.objects:
            title = obj.text[:55].replace("\n", " / ") if isinstance(obj, Text) else f"Таблица {obj.rows} × {obj.columns}"
            item = QListWidgetItem(("[x] " if obj.excluded else "[?] " if obj.needs_review else "[OK] ") + title)
            item.setData(Qt.ItemDataRole.UserRole, obj.id)
            self.objects.addItem(item)
            if selected == obj.id:
                item.setSelected(True)
        self.objects.blockSignals(False)
        self.preview.show_project(self.project, fit=fit)
        self.inspect()

    def select(self, object_id):
        modifiers = QApplication.keyboardModifiers()
        if not modifiers & Qt.KeyboardModifier.ControlModifier:
            self.objects.clearSelection()
        for i in range(self.objects.count()):
            item = self.objects.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == object_id:
                item.setSelected(True)
                self.objects.scrollToItem(item)

    def inspect(self):
        obj = self.current()
        if not obj:
            self.content.clear()
            return
        self.content.setReadOnly(isinstance(obj, Table))
        self.content.setPlainText(obj.text if isinstance(obj, Text) else "\n".join(c.text for c in obj.cells))
        for name, widget in self.coords.items():
            widget.setValue(getattr(obj.bbox, name))
        self.rotation.setValue(obj.rotation if isinstance(obj, Text) else 0)
        self.fill.setText(obj.style.fill); self.color.setText(obj.style.color)
        self.font.setText(obj.style.font); self.font_size.setValue(obj.style.font_size)
        self.align.setCurrentText(obj.style.align)
        self.border.setValue(obj.style.border_pt); self.border_color.setText(obj.style.border_color)
        self.excluded.setChecked(obj.excluded); self.reviewed.setChecked(not obj.needs_review)
        notes = obj.review_notes + [f"{c.source}: {c.before} → {c.after} ({c.confidence:.2f}; применено: {c.applied})" for c in obj.corrections]
        self.notes.setText("\n".join(notes[-10:]))

    def apply_properties(self):
        obj = self.current()
        if not obj:
            return
        candidate = self.project.model_copy(deep=True)
        new = next(o for o in candidate.objects if o.id == obj.id)
        try:
            new.bbox = Box(**{k: w.value() for k, w in self.coords.items()})
            new.style = new.style.model_copy(update={"fill": self.fill.text(), "color": self.color.text(),
                         "font": self.font.text(), "font_size": self.font_size.value(), "align": self.align.currentText(),
                         "border_pt": self.border.value(), "border_color": self.border_color.text()})
            new.needs_review, new.excluded = not self.reviewed.isChecked(), self.excluded.isChecked()
            if isinstance(new, Text):
                manual_text(new, self.content.toPlainText())
                new.rotation = self.rotation.value()
            else:
                new.col_widths = [v*new.bbox.w/obj.bbox.w for v in new.col_widths]
                new.row_heights = [v*new.bbox.h/obj.bbox.h for v in new.row_heights]
                for cell in new.cells:
                    cell.style = new.style.model_copy()
            validate_project(candidate)
            self.project, self.dirty = candidate, True
            self.refresh(selected=new.id)
        except ValueError as exc:
            self.error(exc)

    def add_region(self):
        if self.project:
            self.preview.draw_region()
            self.statusBar().showMessage("Нарисуйте прямоугольник для новой надписи")

    def add_text(self, box):
        text, ok = QInputDialog.getMultiLineText(self, "Добавить надпись", "Текст")
        if not ok:
            return
        candidate = self.project.model_copy(deep=True)
        obj = Text(id=uid("text"), bbox=box, text=text)
        candidate.objects.append(obj)
        try:
            validate_project(candidate)
            self.project, self.dirty = candidate, True
            self.refresh(selected=obj.id)
        except ValueError as exc:
            self.error(exc)

    def merge(self):
        if not self.project:
            return
        try:
            obj = merge_texts(self.project, self.selected_ids())
            self.dirty = True
            self.refresh(selected=obj.id)
        except ValueError as exc:
            self.error(exc)

    def split(self):
        obj = self.current()
        if not isinstance(obj, Text):
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Разделить блок с заданием геометрии")
        form = QFormLayout(dialog)
        axis = QComboBox(); axis.addItems(["Горизонтально (верх / низ)", "Вертикально (лево / право)"])
        ratio = spin(50, 1, 99)
        first, second = QTextEdit(), QTextEdit()
        first.setPlainText(obj.text)
        form.addRow("Направление", axis); form.addRow("Граница, %", ratio)
        form.addRow("Текст первой части", first); form.addRow("Текст второй части", second)
        controls = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        controls.accepted.connect(dialog.accept); controls.rejected.connect(dialog.reject)
        form.addRow(controls)
        if not dialog.exec():
            return
        b, part = obj.bbox, ratio.value()/100
        if axis.currentIndex() == 0:
            boxes = [Box(x=b.x, y=b.y, w=b.w, h=b.h*part), Box(x=b.x, y=b.y+b.h*part, w=b.w, h=b.h*(1-part))]
        else:
            boxes = [Box(x=b.x, y=b.y, w=b.w*part, h=b.h), Box(x=b.x+b.w*part, y=b.y, w=b.w*(1-part), h=b.h)]
        try:
            split_text(self.project, obj.id, boxes, [first.toPlainText(), second.toPlainText()])
            self.dirty = True
            self.refresh()
        except ValueError as exc:
            self.error(exc)

    def convert(self):
        obj = self.current()
        if not obj:
            return
        candidate = self.project.model_copy(deep=True)
        if isinstance(obj, Table):
            new = Text(id=uid("text"), bbox=obj.bbox.model_copy(), ocr_ids=obj.ocr_ids,
                       text="\n".join(c.text for c in sorted(obj.cells, key=lambda c: (c.row, c.col))),
                       style=obj.style.model_copy(), needs_review=True)
            candidate.objects = [o for o in candidate.objects if o.id != obj.id] + [new]
        else:
            selected = [o for o in candidate.objects if o.id in self.selected_ids()]
            if any(isinstance(o, Table) for o in selected):
                self.error("Выберите только текстовые блоки")
                return
            box = union([o.bbox for o in selected])
            table = new_table(box, 1, 1, obj.style)
            # Claim every intersecting text object so no raster text is duplicated.
            inside = [o for o in candidate.objects if isinstance(o, Text) and contains(box, o.bbox, 1)]
            table.ocr_ids = [i for o in inside for i in o.ocr_ids]
            table.cells[0].ocr_ids = table.ocr_ids
            table.cells[0].text = "\n".join(o.text for o in sorted(inside, key=lambda o: (o.bbox.y, o.bbox.x)))
            table.needs_review = True
            candidate.objects = [o for o in candidate.objects if o.id not in {t.id for t in inside}] + [table]
            new = table
        try:
            validate_project(candidate)
            self.project, self.dirty = candidate, True
            self.refresh(selected=new.id)
            if isinstance(new, Table):
                self.edit_table()
        except ValueError as exc:
            self.error(exc)

    def edit_table(self):
        obj = self.current()
        if not isinstance(obj, Table):
            return
        dialog = TableDialog(obj, self)
        if dialog.exec():
            candidate = self.project.model_copy(deep=True)
            table = dialog.table
            before = "\n".join(f"[{c.row},{c.col}] {c.text}" for c in obj.cells)
            after = "\n".join(f"[{c.row},{c.col}] {c.text}" for c in table.cells)
            if before != after:
                from .models import Correction
                table.corrections.append(Correction(source="manual", before=before, after=after,
                                                     confidence=1, applied=True, reason="Редактирование таблицы"))
            # Reassign IDs by measured geometry after grid edits.
            assign_fragments(table, [f for f in candidate.fragments if f.id in obj.ocr_ids])
            if dialog.review.isChecked():
                table.needs_review = False
            candidate.objects = [o for o in candidate.objects if o.id != obj.id] + [table]
            try:
                validate_project(candidate)
                self.project, self.dirty = candidate, True
                self.refresh(selected=table.id)
            except ValueError as exc:
                self.error(exc)

    def reocr(self):
        obj = self.current()
        if not isinstance(obj, Text):
            self.error("Выберите текстовый блок")
            return
        project, settings = self.project.model_copy(deep=True), self.settings.model_copy()
        self.start(lambda progress: reocr_text(project, obj.id, PaddleEngine(settings)))

    def export(self):
        if not self.project:
            return
        try:
            validate_project(self.project)
        except ValueError as exc:
            self.error(exc)
            return
        unreviewed = sum(o.needs_review and not o.excluded for o in self.project.objects)
        if unreviewed:
            answer = QMessageBox.warning(self, "Экспорт непроверенных объектов",
                         f"Непроверенных объектов: {unreviewed}. В PPTX могут остаться ошибки распознавания и структуры таблиц.\n"
                         "Экспортировать без обязательной проверки?",
                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                         QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Yes:
                return
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт в PowerPoint", str(Path(self.project.source.filename).with_suffix(".pptx")), "PowerPoint (*.pptx)")
        if path:
            project, settings = self.project.model_copy(deep=True), self.settings.model_copy()
            def run(progress):
                export_pptx(project, path, settings, allow_unreviewed=bool(unreviewed))
                return "Создан PPTX: " + path
            self.start(run)

    def closeEvent(self, event):
        if self.worker:
            self.error("Дождитесь завершения текущей операции")
            event.ignore()
        elif self.confirm_discard():
            event.accept()
        else:
            event.ignore()


def run_app():
    application = QApplication.instance() or QApplication([])
    application.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return application.exec()

