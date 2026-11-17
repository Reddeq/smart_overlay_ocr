"""Experimental *fully local* PaddleOCR-VL document parsing adapter.

PaddleOCR-VL-1.6 generates text/table CONTENT and layout BLOCK boundaries.
Those boundaries are NOT guaranteed line-level OCR boxes. Never pretend they
are: retain the original image and mark all generated objects for review.
No call to ordinary PaddleOCR or any OpenRouter endpoint occurs here.
"""
from __future__ import annotations

from collections import Counter
from html import unescape
import json
import math
from pathlib import Path
import re
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .formatting import background, contrast_color
from .geometry import intersection
from .input import source_image
from .models import Box, Fragment, Project, Table, Text, uid
from .pipeline import format_project
from .structure import validate_project
from .tables import grid_lines, parse_html

VERSION = "v1.6"
TEXT_LABELS = {
    "text", "paragraph_title", "doc_title", "title", "abstract",
    "figure_title", "figure_caption", "table_caption", "vision_footnote",
    "header", "footer", "reference", "aside_text", "number", "footnote",
    "formula", "equation", "algorithm", "seal",
}
NON_TEXT_LABELS = {"image", "figure", "chart", "header_image", "footer_image"}
TABLE_LABELS = {"table"}


class LocalVLError(RuntimeError):
    """Local PaddleOCR-VL engine failed or returned unusable structure."""


def default_pipeline(*, device: str):
    """Import lazily: default Smart OCR works with its existing pinned stack."""
    if device not in ("cpu", "gpu:0"):
        raise ValueError("Устройство PaddleOCR-VL: cpu или gpu:0")
    try:
        from paddleocr import PaddleOCRVL
    except (ImportError, AttributeError) as exc:
        raise LocalVLError(
            "PaddleOCR-VL-1.6 недоступен. Установите дополнения по docs/local-vl.md "
            "в отдельном тестовом окружении или обновите PaddleOCR.") from exc
    try:
        return PaddleOCRVL(
            pipeline_version=VERSION,
            engine="paddle",
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=True,
            use_chart_recognition=False,
            use_queues=False,
            format_block_content=False,
        )
    except (ImportError, AttributeError, TypeError, RuntimeError, ValueError) as exc:
        raise LocalVLError("Не удалось инициализировать локальный PaddleOCR-VL-1.6 "
                           f"({type(exc).__name__}). Проверьте версии PaddleOCR/PaddlePaddle "
                           "и видеопамять; подробности в docs/local-vl.md") from exc


def _as_plain_dict(value):
    if hasattr(value, "json"):
        value = value.json
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise LocalVLError("PaddleOCR-VL вернул неструктурированный результат")
    return value.get("res", value)


def _native_json(obj):
    if isinstance(obj, dict):
        return {str(k): _native_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _native_json(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    raise TypeError(f"Unserializable VL result field: {type(obj).__name__}")


def block_box(value, width, height):
    """Accept only rect coordinates in original pixels, never guessed scale."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("Ожидалась прямоугольная геометрия блока [x1,y1,x2,y2]")
    try:
        x0, y0, x1, y1 = map(float, value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Координаты блока не являются числами") from exc
    if not all(math.isfinite(x) for x in (x0, y0, x1, y1)):
        raise ValueError("Нечисловые или бесконечные координаты")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Нулевая или отрицательная область")
    if x0 < -2 or y0 < -2 or x1 > width+2 or y1 > height+2:
        raise ValueError("Блок выходит за границы исходного изображения")
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    return Box(x=x0, y=y0, w=x1-x0, h=y1-y0)


def _fallback_text(content):
    text = str(content or "")
    text = re.sub(r"</(?:tr|p|div|br)\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return unescape(text).strip()


def _grid_edges(image, box, count, axis):
    """Use image-rule centers only with matching topology; otherwise mark unknown."""
    if count == 1:
        return [float(box.w if axis == "x" else box.h)], True
    total = box.w if axis == "x" else box.h
    try:
        vertical, horizontal = grid_lines(image, box)
        measurements = vertical if axis == "x" else horizontal
        internal = [float(v) for v in measurements if 2 < v < total-2]
        if len(internal) == count-1 and all(b > a+1 for a,b in zip(internal,internal[1:])):
            edges = [0, *internal, total]
            return [b-a for a,b in zip(edges,edges[1:])], True
    except (ValueError, cv2.error):
        pass
    return [total/count]*count, False


def _table_from_html(html, box, image):
    rows, cols, cells = parse_html(html)
    if rows > 100 or cols > 40 or len(cells) > 1000:
        raise ValueError("Слишком много ячеек для автоматического преобразования")
    widths, measured_x = _grid_edges(image, box, cols, "x")
    heights, measured_y = _grid_edges(image, box, rows, "y")
    notes = ["PaddleOCR-VL: проверьте строки, столбцы и объединённые ячейки"]
    if not measured_x or not measured_y:
        notes.append("Часть границ ячеек оценена равномерно: точной геометрии нет")
    table = Table(id=uid("table"), bbox=box, rows=rows, columns=cols,
                  col_widths=widths, row_heights=heights, cells=cells,
                  ocr_ids=[], needs_review=True, confidence=.5, review_notes=notes)
    return table


def _text_from_block(text, box, label):
    if not text.strip():
        return None, None
    identifier = uid("ocr")
    polygon = [(box.x, box.y), (box.right, box.y),
               (box.right, box.bottom), (box.x, box.bottom)]
    fragment = Fragment(id=identifier, text=text, bbox=box, polygon=polygon,
                        rotation=0, confidence=.5)
    obj = Text(id=uid("text"), text=text, bbox=box.model_copy(),
               ocr_ids=[identifier], confidence=.5, needs_review=True,
               review_notes=["VL определила границу блока, а не каждой строки: "
                             "проверьте положение фрейма и исходный текст",
                             f"Тип области: {label}"])
    return fragment, obj


def _font_for(text, default):
    if any("\u3040" <= c <= "\u30ff" for c in text):
        return "Yu Gothic"
    if any("\uac00" <= c <= "\ud7af" for c in text):
        return "Malgun Gothic"
    if any("\u3400" <= c <= "\u9fff" for c in text):
        return "Microsoft YaHei"
    return default


def parse_page(source: Project, settings, page: dict):
    """Create a new project *only* from VL blocks, with strict geometry checks."""
    width, height = source.source.width, source.source.height
    if page.get("width") is not None and int(page["width"]) != width:
        raise LocalVLError("VL изменила ширину изображения; геометрию нельзя перенести 1:1")
    if page.get("height") is not None and int(page["height"]) != height:
        raise LocalVLError("VL изменила высоту изображения; геометрию нельзя перенести 1:1")
    blocks = page.get("parsing_res_list")
    if not isinstance(blocks, list):
        raise LocalVLError("В ответе PaddleOCR-VL нет parsing_res_list")
    result = Project(source=source.source.model_copy(deep=True),
                     settings=settings.model_dump())
    original = source_image(result)
    issues, skipped = [], []
    counts = Counter()
    for index, entry in enumerate(blocks):
        if not isinstance(entry, dict):
            skipped.append({"index": index, "reason": "non-object block"})
            continue
        label = str(entry.get("block_label") or "").lower()
        counts[label] += 1
        if label in NON_TEXT_LABELS:
            # Keep diagrams as raster under editable text; never white them out.
            continue
        content = entry.get("block_content")
        if not isinstance(content, str) or not content.strip():
            skipped.append({"index": index, "label": label, "reason": "no text"})
            continue
        try:
            box = block_box(entry.get("block_bbox"), width, height)
        except ValueError as exc:
            skipped.append({"index": index, "label": label, "reason": str(exc)})
            continue
        if label not in TEXT_LABELS | TABLE_LABELS:
            # Unknown labels should not become untrusted text rectangles.
            skipped.append({"index": index, "label": label, "reason": "unhandled layout label"})
            continue
        if label in TABLE_LABELS:
            try:
                if "<table" not in content.lower():
                    raise ValueError("VL не вернула HTML таблицы")
                table = _table_from_html(content, box, original)
                if any(intersection(box, old.bbox) > 1 for old in result.objects):
                    raise ValueError("Таблица пересекает другую распознанную область")
                result.objects.append(table)
                continue
            except ValueError as exc:
                issues.append(f"Область {index}: таблица без проверенной сетки ({exc})")
                content = _fallback_text(content)
                # Fallback is editable text, NOT a fictitious 1-cell table.
        if label == "seal":
            issues.append(f"Область {index}: печать распознана как текст, проверьте вручную")
        if not content.strip():
            skipped.append({"index": index, "label": label, "reason": "empty after cleanup"})
            continue
        if any(intersection(box, obj.bbox) > box.w*box.h*.1 for obj in result.objects):
            # Do not cover other recognized labels/tables with a second overlay.
            skipped.append({"index": index, "label": label, "reason": "overlaps existing object"})
            continue
        fragment, obj = _text_from_block(content, box, label)
        if fragment is not None:
            result.fragments.append(fragment)
            result.objects.append(obj)
    if not result.objects:
        raise LocalVLError("PaddleOCR-VL не вернула ни одного пригодного текста/таблицы. "
                           "PPTX не создаётся, исходник сохранён.")
    format_project(result, settings)
    for obj in result.objects:
        obj.needs_review = True
        if isinstance(obj, Text):
            obj.style.font = _font_for(obj.text, settings.font)
            # Keep the measured VL rectangle: original-layout padding can
            # accidentally erase adjacent drawing lines.
            fragment = next(f for f in result.fragments if f.id == obj.ocr_ids[0])
            obj.bbox = fragment.bbox.model_copy()
            obj.style.fill = (background(original, obj.bbox) if settings.fill_mode == "auto"
                              else "#FFFFFF")
            obj.style.color = contrast_color(obj.style.fill)
        else:
            for cell in obj.cells:
                cell.style.font = _font_for(cell.text, settings.font)
    # Raw annotation is not a separate model confidence: report honestly.
    result.raw["local_vl"] = {
        "version": VERSION, "engine": "paddle", "source": "local",
        "block_labels": dict(counts), "skipped": skipped, "issues": issues,
        "converted_text_blocks": sum(isinstance(o, Text) for o in result.objects),
        "converted_tables": sum(isinstance(o, Table) for o in result.objects),
    }
    result.messages.append("PaddleOCR-VL-1.6: текст и таблицы распознаны локально; "
                           "границы объектов требуют визуальной проверки")
    if skipped:
        result.messages.append(f"VL: пропущено или отклонено областей: {len(skipped)}")
    result.messages.extend(issues)
    validate_project(result)
    return result


def save_diagnostics(directory, page, project):
    """Explicit opt-in; diagnostic JSON contains source text. Never commit it."""
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "paddle-vl-result.json").write_text(
        json.dumps(_native_json(page), ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "paddle-vl-summary.json").write_text(
        json.dumps(project.raw["local_vl"], ensure_ascii=False, indent=2), encoding="utf-8")
    preview = source_image(project)
    draw = ImageDraw.Draw(preview)
    for obj in project.objects:
        b = obj.bbox
        draw.rectangle((b.x, b.y, b.right, b.bottom),
                       outline="#0088ff" if isinstance(obj, Table) else "#ff6600", width=2)
    preview.save(directory / "paddle-vl-boxes.png")


def local_vl_ocr(source: Project, settings, *, device="cpu", pipeline_factory=None,
                 diagnostics_dir=None, progress=lambda _: None, check=lambda: None):
    """Run the full on-device VL pipeline, without reading any PaddleOCR text."""
    check()
    begin = time.monotonic()
    progress("PaddleOCR-VL-1.6: загрузка локальных моделей")
    factory = pipeline_factory or default_pipeline
    pipeline = factory(device=device)
    check()
    progress("PaddleOCR-VL-1.6: анализ схемы (локально)")
    # OpenCV BGR numpy; using source_image avoids inconsistent EXIF transforms.
    image = cv2.cvtColor(np.asarray(source_image(source)), cv2.COLOR_RGB2BGR)
    try:
        pages = list(pipeline.predict(input=image, layout_shape_mode="rect",
                                      format_block_content=False))
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        raise LocalVLError("Локальная обработка PaddleOCR-VL не завершена: "
                           f"{type(exc).__name__}; проверьте память, модели и зависимости") from exc
    check()
    if len(pages) != 1:
        raise LocalVLError(f"Ожидалась одна страница, PaddleOCR-VL вернула {len(pages)}")
    page = _as_plain_dict(pages[0])
    progress("PaddleOCR-VL: создание редактируемых фреймов и таблиц")
    project = parse_page(source, settings, page)
    project.raw["local_vl"]["device"] = device
    project.usage.append({"stage": "paddle_vl_local", "seconds": round(time.monotonic()-begin, 3)})
    if diagnostics_dir is not None:
        save_diagnostics(diagnostics_dir, page, project)
    return project
