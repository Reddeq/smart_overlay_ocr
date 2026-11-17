import base64
import logging
import os
from pathlib import Path

from pydantic import ValidationError
from PySide6.QtCore import Qt, QThread, Signal, QRectF
from PySide6.QtGui import QAction, QColor, QFont, QFontDatabase, QPainter, QPen, QPixmap, QTextOption
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGraphicsRectItem,
    QGraphicsScene, QGraphicsView, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
    QToolBar, QVBoxLayout, QWidget,
)

from .export import export_pptx
from .geometry import contains, union
from .grouping import (MODEL as GROUPING_MODEL, GroupingResult, TableDetectionResult,
                       build_plan, candidate_project, find_project_tables, fingerprint,
                       mark_manual, run_grouping, saved_masks)
from .input import load_image, open_project, save_project
from .layout import text_layout
from .models import Box, Project, Table, Text, uid
from .ocr import PaddleEngine
from .pipeline import SmartCancelled, SmartFailure, local_ocr, reocr_text, smart_ocr
from .settings import Settings, data_dir, load_settings, save_settings
from .table_review import (TableReviewResult, build_table_plan, protection_reason,
                           run_table_review, table_candidate_project)
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
        self.show_formatting = False
        self.setBackgroundBrush(QColor("#242a34"))

    def show_project(self, project: Project, fit=False):
        self.scene().clear()
        pixmap = QPixmap()
        pixmap.loadFromData(base64.b64decode(project.source.raster_b64))
        self.scene().addPixmap(pixmap)
        self.setSceneRect(0, 0, project.source.width, project.source.height)
        if self.show_formatting:
            self.draw_formatting(project)
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

    def draw_formatting(self, project):
        scale = project.settings.get('inches_per_pixel', 1/150)

        def label(text, box, style, rotation=0):
            item = self.scene().addText(text)
            item.document().setDocumentMargin(0)
            option = item.document().defaultTextOption()
            option.setAlignment({'left': Qt.AlignmentFlag.AlignLeft, 'center': Qt.AlignmentFlag.AlignHCenter,
                                 'right': Qt.AlignmentFlag.AlignRight}[style.align])
            option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
            item.document().setDefaultTextOption(option)
            font = QFont(style.font)
            font.setPixelSize(max(1, round(style.font_size/(72*scale))))
            item.setFont(font)
            item.setDefaultTextColor(QColor(style.color))
            item.setTextWidth(box.w)
            item.setPos(box.x, box.y + max(0, (box.h-item.boundingRect().height())/2))
            item.setTransformOriginPoint(box.w/2, item.boundingRect().height()/2)
            item.setRotation(rotation)

        for obj in project.objects:
            if obj.excluded:
                continue
            if isinstance(obj, Text):
                if not obj.text.strip():
                    continue
                frame, masks = text_layout(project, obj)
                for b in masks:
                    self.scene().addRect(b.x, b.y, b.w, b.h, QPen(Qt.PenStyle.NoPen), QColor(obj.style.fill))
                label(obj.text, frame, obj.style, obj.rotation)
            else:
                for cell in obj.cells:
                    b = cell_box(obj, cell)
                    pen = QPen(QColor(cell.style.border_color))
                    pen.setWidthF(cell.style.border_pt/(72*scale))
                    if not cell.style.border_pt:
                        pen.setStyle(Qt.PenStyle.NoPen)
                    self.scene().addRect(b.x, b.y, b.w, b.h, pen, QColor(cell.style.fill))
                    label(cell.text, b, cell.style)

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


class GroupingSetupDialog(QDialog):
    def __init__(self, project, settings, parent=None):
        super().__init__(parent)
        self.project, self.settings, self.plan = project, settings, None
        self.masks = saved_masks(project)
        self.setWindowTitle("VL-группировка · экспериментальный режим")
        self.resize(1120, 800)
        layout = QVBoxLayout(self)
        intro = QLabel("Отметьте пропущенные таблицы прямоугольниками. Найденные таблицы уже исключены.\n"
                       "Ручные правки, исключённые и составные блоки защищены от перегруппировки.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.view = Preview()
        self.view.region.connect(self.add_mask)
        layout.addWidget(self.view, 1)
        actions = QHBoxLayout()
        button("Исключить область таблицы", self.view.draw_region, actions)
        button("Удалить выбранную зону", self.remove_mask, actions)
        layout.addLayout(actions)
        self.zones = QListWidget()
        self.zones.setMaximumHeight(85)
        layout.addWidget(self.zones)
        self.tables_checked = QCheckBox("Я проверил границы таблиц; их содержимое не должно участвовать в группировке")
        self.tables_checked.toggled.connect(self.update_start)
        layout.addWidget(self.tables_checked)
        form = QFormLayout()
        self.proxy = QLineEdit("http://127.0.0.1:10809")
        form.addRow("HTTP-прокси активного VPN", self.proxy)
        self.limit = QSpinBox(); self.limit.setRange(1, 100); self.limit.setValue(20)
        self.limit.valueChanged.connect(self.invalidate)
        form.addRow("Разрешённый максимум запросов", self.limit)
        self.review_ocr = QCheckBox('Уточнять неуверенный нетабличный OCR по увеличенным фрагментам')
        self.review_ocr.setChecked(True)
        self.review_ocr.toggled.connect(self.invalidate)
        form.addRow(self.review_ocr)
        self.ocr_threshold = spin(settings.confidence_threshold, 0, 1)
        self.ocr_threshold.valueChanged.connect(self.invalidate)
        form.addRow('Проверять OCR с уверенностью ниже', self.ocr_threshold)
        layout.addLayout(form)
        self.summary = QLabel("Модель: "+GROUPING_MODEL+" · OpenRouter / Parasail\n"
                              "Сначала подготовьте области. Ключ берётся из настроек; в проект не записывается.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        controls = QHBoxLayout()
        button("Подготовить области и бюджет", self.prepare, controls)
        self.send_button = button("Отправить", self.accept, controls)
        self.send_button.setEnabled(False)
        button("Отмена", self.reject, controls)
        layout.addLayout(controls)
        self.redraw()

    def invalidate(self):
        self.plan = None
        self.send_button.setEnabled(False)
        self.send_button.setText("Отправить")
        self.summary.setText("Параметры или зоны изменены. Подготовьте области и бюджет заново.")

    def update_start(self):
        self.send_button.setEnabled(bool(self.plan and self.plan.requests and self.tables_checked.isChecked()))

    def redraw(self):
        display = self.project.model_copy(deep=True)
        display.objects = [o for o in display.objects if isinstance(o, Table)]
        self.view.show_project(display, fit=True)
        self.zones.clear()
        for index, b in enumerate(self.masks, 1):
            self.zones.addItem(f"Зона {index}: {b.x:.0f}, {b.y:.0f} · {b.w:.0f} × {b.h:.0f} px")
            item = self.view.scene().addRect(b.x, b.y, b.w, b.h, QPen(QColor("#da7750"), 2))
            item.setToolTip("Табличная область: исключена из VL-группировки")

    def add_mask(self, box):
        self.masks.append(box)
        self.tables_checked.setChecked(False)
        self.invalidate()
        self.redraw()

    def remove_mask(self):
        row = self.zones.currentRow()
        if row >= 0:
            self.masks.pop(row)
            self.tables_checked.setChecked(False)
            self.invalidate()
            self.redraw()

    def prepare(self):
        try:
            settings = self.settings.model_copy(update={'confidence_threshold': self.ocr_threshold.value()})
            self.plan = build_plan(self.project, settings, self.masks, max_requests=self.limit.value(),
                                   review_ocr=self.review_ocr.isChecked())
            count = self.plan.requests
            self.summary.setText(f"OpenRouter / Parasail · {GROUPING_MODEL}\n"
                f"{len(self.plan.regions)} областей; OCR: {len(self.plan.selection['eligible_ids'])}; "
                f"защищено объектов: {len(self.plan.protected_objects)}.\n"
                f"Запросы: группировка {self.plan.grouping_requests}, уточнение OCR {len(self.plan.reading_ids)}, "
                f"всего {count}; не более {count*1024} выходных токенов. "
                "Входные токены также оплачиваются. Повторов нет.\n"
                "Передаются изображения областей и OCR-текст через указанный VPN. "
                "Для денежного лимита используйте ограничение ключа на стороне API.")
            self.send_button.setText(f"Отправить {count} запросов")
            self.update_start()
            self.layout().activate()
            self.view.fitInView(self.view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        except ValueError as exc:
            self.invalidate()
            QMessageBox.warning(self, "Подготовка группировки", str(exc))

    def accept(self):
        if not self.plan or not self.plan.requests or not self.tables_checked.isChecked():
            return
        from .diagnose_real import check_proxy
        try:
            check_proxy(self.proxy.text().strip())
        except ValueError:
            QMessageBox.warning(self, "VPN", "Укажите HTTP-адрес работающего локального VPN-прокси без логина и пароля")
            return
        super().accept()

    def showEvent(self, event):
        super().showEvent(event)
        self.layout().activate()
        self.view.fitInView(self.view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class GroupingResultDialog(QDialog):
    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.result, self.candidate = result, None
        self.drawn = False
        self.setWindowTitle("Предпросмотр VL-группировки · проект ещё не изменён")
        self.resize(1120, 820)
        layout = QVBoxLayout(self)
        received = sum(not r.get("local_only") for r in result.results)+len(result.readings)
        summary = QLabel(f"Ответов: {received}/{result.plan.requests}; предложений: "
                         f"{len(result.consensus['accepted_merges'])}; конфликтов: {len(result.consensus['conflicts'])}.\n"
                         +result.message+"\nОтметьте объединения для применения. Пунктир — невыбранное предложение; красный — конфликт. "
                         "Исправления OCR выбираются отдельно; уверенность модели не гарантирует точность.")
        summary.setWordWrap(True)
        layout.addWidget(summary)
        self.view = Preview()
        layout.addWidget(self.view, 1)
        self.groups = QTableWidget(0, 3)
        self.groups.setHorizontalHeaderLabels(["Применить", "Объединяемые строки", "Проверка"])
        self.groups.setColumnWidth(0, 85); self.groups.setColumnWidth(1, 650); self.groups.setColumnWidth(2, 260)
        self.groups.setMaximumHeight(190)
        fragments = {f.id: f for f in result.plan.project.fragments}
        for group in result.consensus['accepted_merges']:
            row = self.groups.rowCount(); self.groups.insertRow(row)
            item = QTableWidgetItem()
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked if group['needs_review'] else Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, group['ocr_ids'])
            self.groups.setItem(row, 0, item)
            text = QTableWidgetItem(" / ".join(fragments[i].text for i in group['ocr_ids']))
            text.setFlags(Qt.ItemFlag.ItemIsEnabled)
            text.setToolTip(text.text())
            self.groups.setItem(row, 1, text)
            status = QTableWidgetItem("Проверьте; одно наблюдение или неполный проход" if group['needs_review'] else "Согласовано между областями")
            status.setFlags(Qt.ItemFlag.ItemIsEnabled)
            status.setToolTip(status.text())
            self.groups.setItem(row, 2, status)
        self.readings = QTableWidget(0, 4)
        self.readings.setHorizontalHeaderLabels(['Применить', 'OCR Paddle', 'Предложение VL', 'Уверенность модели'])
        for col, width in enumerate((85, 390, 390, 170)):
            self.readings.setColumnWidth(col, width)
        for record in result.readings:
            if 'reading' not in record:
                continue
            reading = record['reading']; identifier = record['ocr_id']
            row = self.readings.rowCount(); self.readings.insertRow(row)
            item = QTableWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, identifier)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | (Qt.ItemFlag.ItemIsUserCheckable
                          if reading['text'] != fragments[identifier].text else Qt.ItemFlag.NoItemFlags))
            item.setCheckState(Qt.CheckState.Unchecked)
            self.readings.setItem(row, 0, item)
            for col, value in enumerate((fragments[identifier].text, reading['text'], f"{reading['confidence']:.2f}"), 1):
                cell = QTableWidgetItem(value); cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
                cell.setToolTip(reading['reason']); self.readings.setItem(row, col, cell)
        tabs = QTabWidget()
        tabs.addTab(self.groups, 'Объединения')
        tabs.addTab(self.readings, f'Уточнение OCR ({self.readings.rowCount()})')
        tabs.setMaximumHeight(220)
        layout.addWidget(tabs)
        controls = QHBoxLayout()
        self.apply_button = button("Применить выбранное", self.accept, controls)
        button("Оставить проект без изменений", self.reject, controls)
        layout.addLayout(controls)
        self.groups.itemChanged.connect(self.redraw)
        self.readings.itemChanged.connect(self.redraw)
        self.redraw()

    def selected_groups(self):
        return [self.groups.item(row, 0).data(Qt.ItemDataRole.UserRole) for row in range(self.groups.rowCount())
                if self.groups.item(row, 0).checkState() == Qt.CheckState.Checked]

    def redraw(self):
        try:
            selected_readings = [self.readings.item(row, 0).data(Qt.ItemDataRole.UserRole)
                                 for row in range(self.readings.rowCount())
                                 if self.readings.item(row, 0).checkState() == Qt.CheckState.Checked]
            self.candidate = candidate_project(self.result, self.selected_groups(), selected_readings)
            self.view.show_project(self.candidate, fit=not self.drawn)
            self.drawn = True
            fragments = {f.id: f for f in self.result.plan.project.fragments}
            selected = self.selected_groups()
            for group in self.result.consensus['accepted_merges']:
                if group['ocr_ids'] not in selected:
                    b = union([fragments[i].bbox for i in group['ocr_ids']])
                    self.view.scene().addRect(b.x, b.y, b.w, b.h, QPen(QColor("#bbbbbb"), 2, Qt.PenStyle.DashLine))
            for group in self.result.consensus['conflicts']:
                b = union([fragments[i].bbox for i in group['ocr_ids']])
                item = self.view.scene().addRect(b.x, b.y, b.w, b.h, QPen(QColor("#f04c62"), 3))
                item.setToolTip("Конфликт: оставлен для ручной проверки")
            self.apply_button.setEnabled(True)
        except ValueError:
            self.candidate = None
            self.apply_button.setEnabled(False)

    def accept(self):
        if self.candidate is not None:
            super().accept()

    def showEvent(self, event):
        super().showEvent(event)
        self.layout().activate()
        self.view.fitInView(self.view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class TablesResultDialog(QDialog):
    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Найденные таблицы · проект ещё не изменён')
        self.resize(1000, 760)
        layout = QVBoxLayout(self)
        label = QLabel(f'Новых таблиц: {result.found}; отложено из-за пересечений или ручных правок: {result.skipped}.\n'
                       'Проверьте границы. После применения сетки и ячейки можно исправить в редакторе таблицы.')
        label.setWordWrap(True); layout.addWidget(label)
        self.view = Preview(); layout.addWidget(self.view, 1)
        self.view.show_project(result.candidate, fit=True)
        controls = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        controls.button(QDialogButtonBox.StandardButton.Apply).setText('Применить найденные таблицы')
        controls.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.accept)
        controls.rejected.connect(self.reject); layout.addWidget(controls)

    def showEvent(self, event):
        super().showEvent(event)
        self.layout().activate()
        self.view.fitInView(self.view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class TableReviewSetupDialog(QDialog):
    def __init__(self, project, settings, parent=None):
        super().__init__(parent)
        self.project, self.settings, self.plan = project, settings, None
        self.setWindowTitle('VL-проверка таблиц · подготовка')
        self.resize(1050, 790)
        layout=QVBoxLayout(self)
        intro=QLabel('Выберите таблицы. Текст проверяется по неуверенным OCR-фрагментам, сетка — отдельным запросом.\n'
                     'Ручные и ранее принятые правки защищены. Все предложения потребуют отдельного выбора.')
        intro.setWordWrap(True); layout.addWidget(intro)
        self.view=Preview(); self.view.show_project(project,fit=True); layout.addWidget(self.view,1)
        self.tables=QListWidget(); self.tables.setMaximumHeight(145); layout.addWidget(self.tables)
        for number,table in enumerate((o for o in project.objects if isinstance(o,Table)),1):
            reason=protection_reason(project,table)
            item=QListWidgetItem(f'Таблица {number}: {table.rows} × {table.columns}'+(' · '+reason if reason else ''))
            item.setData(Qt.ItemDataRole.UserRole,table.id)
            item.setFlags(item.flags()|Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked if reason or not table.needs_review else Qt.CheckState.Checked)
            if reason: item.setFlags(item.flags()&~Qt.ItemFlag.ItemIsEnabled)
            self.tables.addItem(item)
        self.tables.itemChanged.connect(self.invalidate)
        form=QFormLayout()
        self.review_text=QCheckBox('Уточнять OCR с низкой уверенностью'); self.review_text.setChecked(True)
        self.review_structure=QCheckBox('Проверять строки, столбцы и объединённые ячейки'); self.review_structure.setChecked(True)
        self.threshold=spin(settings.confidence_threshold,0,1)
        self.limit=QSpinBox(); self.limit.setRange(1,100); self.limit.setValue(20)
        self.proxy=QLineEdit('http://127.0.0.1:10809')
        for check in (self.review_text,self.review_structure):
            form.addRow(check); check.toggled.connect(self.invalidate)
        form.addRow('Порог уверенности OCR',self.threshold); self.threshold.valueChanged.connect(self.invalidate)
        form.addRow('Максимум запросов',self.limit); self.limit.valueChanged.connect(self.invalidate)
        form.addRow('HTTP-прокси активного VPN',self.proxy); layout.addLayout(form)
        self.summary=QLabel('Подготовьте бюджет. Модель '+GROUPING_MODEL+' / Parasail. Запросы могут быть платными.')
        self.summary.setWordWrap(True); layout.addWidget(self.summary)
        controls=QHBoxLayout(); button('Подготовить бюджет',self.prepare,controls)
        self.send_button=button('Отправить',self.accept,controls); self.send_button.setEnabled(False)
        button('Отмена',self.reject,controls); layout.addLayout(controls)

    def invalidate(self,*_):
        self.plan=None; self.send_button.setEnabled(False)
        self.send_button.setText('Отправить'); self.summary.setText('Выбор изменён. Подготовьте бюджет заново.')

    def prepare(self):
        self.invalidate()
        ids=[self.tables.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.tables.count())
             if self.tables.item(i).checkState()==Qt.CheckState.Checked]
        try:
            settings=self.settings.model_copy(update={'confidence_threshold':self.threshold.value()})
            self.plan=build_table_plan(self.project,settings,ids,review_text=self.review_text.isChecked(),
                review_structure=self.review_structure.isChecked(),max_requests=self.limit.value())
            p=self.plan
            self.summary.setText(f'Структура: {len(p.structure_ids)}; OCR: {len(p.reading_targets)}; '
                f'всего {p.requests} запросов, до {p.output_tokens} выходных токенов. Без повторов.\n'
                'Будут переданы выбранные таблицы, увеличенные OCR-фрагменты, тексты и координаты. Проект до применения не изменится.')
            self.send_button.setText(f'Отправить {p.requests} запросов'); self.send_button.setEnabled(p.requests>0)
        except ValueError as exc:
            self.summary.setText(str(exc))

    def showEvent(self,event):
        super().showEvent(event); self.layout().activate()
        self.view.fitInView(self.view.sceneRect(),Qt.AspectRatioMode.KeepAspectRatio)


class TableReviewResultDialog(QDialog):
    def __init__(self,result,parent=None):
        super().__init__(parent)
        self.result=result; self.candidate=None; self.keys=[]
        self.setWindowTitle('VL-проверка таблиц · проект ещё не изменён'); self.resize(1160,840)
        layout=QVBoxLayout(self)
        label=QLabel((result.message or 'Проверка завершена.')+' Выберите нужные исправления. Уверенность модели не подтверждает правильность.')
        label.setWordWrap(True); layout.addWidget(label)
        self.view=Preview(); layout.addWidget(self.view,1)
        self.proposals=QTableWidget(0,5)
        self.proposals.setHorizontalHeaderLabels(['Применить','Таблица / OCR','Сейчас','Предложение','Статус / причина'])
        self.proposals.setMaximumHeight(240); layout.addWidget(self.proposals)
        tables={o.id:o for o in result.plan.project.objects if isinstance(o,Table)}
        fragments={f.id:f for f in result.plan.project.fragments}
        for record in result.records:
            row=self.proposals.rowCount(); self.proposals.insertRow(row)
            table=tables[record['table_id']]; reply=record.get('reply')
            structure=record['kind']=='structure'; identifier=record.get('ocr_id')
            self.keys.append((record['table_id'],identifier,structure))
            item=QTableWidgetItem(); item.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            if not reply or (not structure and reply['text']==fragments[identifier].text):
                item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.proposals.setItem(row,0,item)
            current=f'{table.rows} × {table.columns}, ячеек {len(table.cells)}' if structure else fragments[identifier].text
            proposed=(f"{len(reply['y_lines'])-1} × {len(reply['x_lines'])-1}, ячеек {len(reply['cells'])}" if structure else reply['text']) if reply else 'Не принято'
            reason=(reply['reason'] if reply else record.get('error','Ответ не получен'))
            if reply and not structure:
                reason=f"Оценка VL: {reply['confidence']:.2f}. "+reason
            for col,value in enumerate([table.id+(' · сетка' if structure else ' · '+identifier),current,proposed,reason],1):
                cell=QTableWidgetItem(value); cell.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsSelectable)
                cell.setToolTip(value); self.proposals.setItem(row,col,cell)
        for col,width in enumerate([75,230,220,250,300]):self.proposals.setColumnWidth(col,width)
        self.proposals.itemChanged.connect(self.update_candidate)
        self.error_label=QLabel(); self.error_label.setWordWrap(True); layout.addWidget(self.error_label)
        controls=QHBoxLayout(); self.apply_button=button('Применить выбранное',self.accept,controls)
        button('Оставить проект без изменений',self.reject,controls); layout.addLayout(controls)
        self.update_candidate()

    def selected(self):
        selected=[key for row,key in enumerate(self.keys) if self.proposals.item(row,0).checkState()==Qt.CheckState.Checked]
        return [t for t,i,s in selected if s],[(t,i) for t,i,s in selected if not s]

    def update_candidate(self,*_):
        structures,readings=self.selected()
        try:
            self.candidate=table_candidate_project(self.result,structures,readings)
            self.view.show_project(self.candidate)
            self.error_label.setText('Сетки и OCR выбираются независимо. Исходный растр и OCR-фрагменты сохраняются.')
            self.apply_button.setEnabled(bool(structures or readings))
        except ValueError as exc:
            self.candidate=None; self.apply_button.setEnabled(False); self.error_label.setText(str(exc))

    def showEvent(self,event):
        super().showEvent(event); self.layout().activate()
        self.view.fitInView(self.view.sceneRect(),Qt.AspectRatioMode.KeepAspectRatio)


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
        self.pending_grouping_result = None
        self.pending_tables_result = None
        self.pending_table_review_result = None
        self.active_grouping = False
        self.grouping_undo = None
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
        self.addToolBarBreak()
        grouping_toolbar = QToolBar("Экспериментальная группировка")
        grouping_toolbar.setMovable(False)
        self.addToolBar(grouping_toolbar)
        self.grouping_button = QPushButton("VL-группировка · эксперимент")
        self.grouping_button.clicked.connect(self.run_grouping)
        grouping_toolbar.addWidget(self.grouping_button)
        self.find_tables_button = QPushButton('Найти таблицы')
        self.find_tables_button.clicked.connect(self.find_tables)
        grouping_toolbar.addWidget(self.find_tables_button)
        self.table_review_button=QPushButton('VL-проверка таблиц')
        self.table_review_button.clicked.connect(self.review_tables)
        grouping_toolbar.addWidget(self.table_review_button)
        self.undo_grouping_button = QPushButton("Отменить применение")
        self.undo_grouping_button.setEnabled(False)
        self.undo_grouping_button.clicked.connect(self.undo_grouping)
        grouping_toolbar.addWidget(self.undo_grouping_button)
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
        self.formatting_preview = QCheckBox("Оформление (приблизительно)")
        self.formatting_preview.setToolTip("Маски и рамки экспорта; переносы PowerPoint могут отличаться от Qt")
        toolbar.addWidget(self.formatting_preview)
        self.formatting_preview.toggled.connect(self.toggle_formatting_preview)
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
            self.grouping_undo = None
            self.undo_grouping_button.setEnabled(False)

    def set_busy(self, busy):
        for widget in [self.centralWidget(), self.recognize_button, self.smart_button, self.export_button, self.mode,
                       self.grouping_button, self.find_tables_button, self.table_review_button]:
            widget.setEnabled(not busy)
        for action in self.actions:
            action.setEnabled(not busy)
        self.undo_grouping_button.setEnabled(not busy and self.grouping_undo is not None)

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
            self.statusBar().showMessage("Отмена VL; полученные предложения будут сохранены…" if self.active_grouping else
                                        "Отмена Smart OCR; локальный результат будет сохранён…")

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
        self.cancel_button.setText("Отменить Smart OCR")
        self.set_busy(False)
        worker.deleteLater()
        self.active_grouping = False
        if self.pending_table_review_result is not None:
            result,self.pending_table_review_result=self.pending_table_review_result,None
            self.show_table_review_result(result)
        if self.pending_tables_result is not None:
            result, self.pending_tables_result = self.pending_tables_result, None
            self.show_tables_result(result)
        if self.pending_grouping_result is not None:
            result, self.pending_grouping_result = self.pending_grouping_result, None
            self.show_grouping_result(result)
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

    def run_grouping(self):
        if self.worker or not self.project or not self.project.fragments:
            self.error("Сначала выполните локальное OCR или откройте проект с результатом OCR")
            return
        if not self.api_key:
            self.error("Укажите API-ключ OpenRouter в настройках")
            return
        dialog = GroupingSetupDialog(self.project.model_copy(deep=True), self.settings.model_copy(), self)
        if not dialog.exec():
            return
        plan, proxy, key = dialog.plan, dialog.proxy.text().strip(), self.api_key
        if plan.signature != fingerprint(self.project):
            self.error("Проект изменился после подготовки; подготовьте области заново")
            return
        self.active_grouping = True
        self.cancel_button.setText("Отменить VL-группировку")
        self.start(lambda progress: run_grouping(plan, key, proxy, progress=progress,
                                               cancelled=self.cancellation_requested), cancellable=True)

    def find_tables(self):
        if not self.project or not self.project.fragments or self.worker:
            self.error('Сначала выполните OCR или откройте проект с OCR')
            return
        snapshot, settings = self.project.model_copy(deep=True), self.settings.model_copy()
        def detect(progress):
            progress('Автоматический поиск табличных сеток…')
            return find_project_tables(snapshot, settings)
        self.start(detect)

    def review_tables(self):
        if self.worker or not self.project or not any(isinstance(o,Table) for o in self.project.objects):
            self.error('Сначала найдите таблицы или откройте проект с таблицами'); return
        if not self.api_key:
            self.error('Укажите API-ключ OpenRouter в настройках'); return
        dialog=TableReviewSetupDialog(self.project.model_copy(deep=True),self.settings.model_copy(),self)
        if not dialog.exec():return
        if dialog.plan.signature!=fingerprint(self.project):
            self.error('Проект изменился после подготовки'); return
        self.active_grouping=True
        self.cancel_button.setText('Отменить проверку таблиц')
        plan,proxy,key=dialog.plan,dialog.proxy.text().strip(),self.api_key
        self.start(lambda progress:run_table_review(plan,key,proxy,progress=progress,cancelled=self.cancellation_requested),cancellable=True)

    def show_table_review_result(self,result):
        if not self.project or fingerprint(self.project)!=result.plan.signature:
            self.error('Проект изменился; предложения таблиц не применены'); return
        if not result.records:
            self.statusBar().showMessage(result.message or 'Предложений нет; проект не изменён'); return
        dialog=TableReviewResultDialog(result,self)
        if dialog.exec() and dialog.candidate is not None:
            before,dirty=self.project.model_copy(deep=True),self.dirty
            self.project=dialog.candidate
            self.grouping_undo=(before,fingerprint(self.project),dirty)
            self.dirty=True; self.refresh(); self.undo_grouping_button.setEnabled(True)
            self.statusBar().showMessage('Предложения таблиц применены. Проверьте текст, сетку и оформление')

    def show_tables_result(self, result):
        if not self.project or fingerprint(self.project) != result.signature:
            self.error('Проект изменился; найденные таблицы не применены')
            return
        if not result.found:
            self.statusBar().showMessage(f'Новых таблиц не найдено; отложено: {result.skipped}')
            return
        if TablesResultDialog(result, self).exec():
            before, dirty = self.project.model_copy(deep=True), self.dirty
            self.project = result.candidate
            self.grouping_undo = (before, fingerprint(self.project), dirty)
            self.dirty = True
            self.refresh()
            self.undo_grouping_button.setEnabled(True)
            self.statusBar().showMessage(f'Добавлено таблиц: {result.found}. Проверьте сетки и содержимое')

    def show_grouping_result(self, result):
        self.cancel_button.setText("Отменить Smart OCR")
        if not self.project or fingerprint(self.project) != result.plan.signature:
            self.error("Проект изменился. Предложения группировки не применены")
            return
        if not result.results:
            if result.status == "cancelled":
                self.statusBar().showMessage("Группировка отменена; проект не изменён")
            else:
                self.error(result.message or "Ответов группировки не получено; проект не изменён")
            return
        dialog = GroupingResultDialog(result, self)
        if dialog.exec() and dialog.candidate is not None:
            before, previous_dirty = self.project.model_copy(deep=True), self.dirty
            self.project = dialog.candidate
            self.grouping_undo = (before, fingerprint(self.project), previous_dirty)
            self.dirty = True
            self.refresh()
            self.undo_grouping_button.setEnabled(True)
            self.journal.setPlainText(result.message+"\n"+self.project.messages[-1])
            self.statusBar().showMessage("Группировка применена. Отмена доступна до следующего редактирования")

    def undo_grouping(self):
        if not self.grouping_undo or self.worker:
            return
        before, after_signature, previous_dirty = self.grouping_undo
        if self.project is None or fingerprint(self.project) != after_signature:
            self.grouping_undo = None
            self.undo_grouping_button.setEnabled(False)
            self.error("После применения проект изменён; отмена не будет стирать новые правки")
            return
        self.project, self.dirty = before, previous_dirty
        self.grouping_undo = None
        self.undo_grouping_button.setEnabled(False)
        self.refresh()
        self.statusBar().showMessage("Последнее применение отменено")

    def received(self, result):
        if isinstance(result, TableReviewResult):
            self.pending_table_review_result=result
        elif isinstance(result, TableDetectionResult):
            self.pending_tables_result = result
        elif isinstance(result, GroupingResult):
            self.pending_grouping_result = result
        elif isinstance(result, Project):
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

    def toggle_formatting_preview(self, enabled):
        self.preview.show_formatting = enabled
        if self.project:
            preview = self.project.model_copy(deep=True)
            preview.settings = self.settings.model_dump()
            self.preview.show_project(preview)

    def refresh(self, fit=False, selected=None):
        if not self.project:
            return
        if self.grouping_undo and fingerprint(self.project) != self.grouping_undo[1]:
            self.grouping_undo = None
            self.undo_grouping_button.setEnabled(False)
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
        preview = self.project.model_copy(deep=True)
        preview.settings = self.settings.model_dump()
        self.preview.show_project(preview, fit=fit)
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
            mark_manual(candidate, new.id)
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
            mark_manual(self.project, obj.id)
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
            mark_manual(candidate, new.id)
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
            if table.model_dump(exclude={'needs_review','review_notes'}) != obj.model_dump(exclude={'needs_review','review_notes'}):
                mark_manual(candidate,table.id)
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

