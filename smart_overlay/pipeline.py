import logging
import time
from pathlib import Path

from .formatting import background, contrast_color, fitted_size, style_text
from .geometry import contains, intersection, union
from .input import source_image
from .models import Box, Project, Table, Text, uid
from .ocr import PaddleEngine
from .settings import Settings
from .structure import base_texts, validate_project
from .tables import assign_fragments, cell_box, geometric_tables, tables_from_result
from .vision import VisionClient, VisionError, VisionCancelled, analyze, refine

LOG = logging.getLogger(__name__)


class SmartFailure(RuntimeError):
    def __init__(self, project: Project, reason: str):
        super().__init__(reason)
        self.project = project


class SmartCancelled(SmartFailure):
    pass


def format_project(project: Project, settings: Settings, *, preserve_text_ids=()):
    image = source_image(project)
    fragments = {f.id: f for f in project.fragments}
    tables = [o for o in project.objects if isinstance(o, Table)]
    for obj in project.objects:
        if isinstance(obj, Text):
            line_height = min((min(fragments[i].bbox.h, fragments[i].bbox.w)
                               for i in obj.ocr_ids), default=obj.bbox.h/max(1, len(obj.text.splitlines())))
            before = obj.bbox.model_copy()
            style_text(obj, image, settings, line_height, add_padding=obj.id not in preserve_text_ids)
            if any(intersection(obj.bbox, t.bbox) > 0 for t in tables):
                obj.bbox = before
            if obj.confidence < settings.confidence_threshold:
                obj.needs_review = True
                obj.review_notes.append("Низкая уверенность OCR")
            project.raw.setdefault("automatic_text_formats", {})[obj.id] = {
                "bbox": obj.bbox.model_dump(), "style": obj.style.model_dump(), "rotation": obj.rotation}
        else:
            obj.style.font = settings.font
            for cell in obj.cells:
                box = cell_box(obj, cell)
                if settings.fill_mode == "auto":
                    cell.style.fill = background(image, box)
                    cell.style.color = contrast_color(cell.style.fill)
                cell.style.font = settings.font
                lines = [fragments[i] for i in cell.ocr_ids]
                if any(f.confidence < settings.confidence_threshold for f in lines):
                    obj.needs_review = True
                    obj.review_notes.append(f"Низкая уверенность OCR в ячейке {cell.row+1}:{cell.col+1}")
                initial = min((f.bbox.h for f in lines), default=box.h*0.55) * 72*settings.inches_per_pixel*0.8
                cell.style.font_size, overflow = fitted_size(cell.text, box, initial, settings)
                if overflow:
                    obj.needs_review = True
                    obj.review_notes.append(f"Возможное переполнение ячейки {cell.row+1}:{cell.col+1}")
    project.settings = settings.model_dump()


def local_ocr(project: Project, settings: Settings, engine=None, progress=lambda _: None) -> Project:
    start = time.monotonic()
    result = Project(source=project.source.model_copy(deep=True), settings=settings.model_dump())
    image = source_image(result)
    engine = engine or PaddleEngine(settings)
    progress("PaddleOCR: распознавание строк")
    result.fragments, raw = engine.recognize(image)
    result.raw["ocr"] = raw
    audit = getattr(engine, "last_language_audit", None)
    if audit:
        result.raw["language_routing"] = audit
        for warning in audit.get("warnings", []):
            result.messages.append("Языковые модели: " + warning)
    progress("PaddleOCR TableRecognitionPipelineV2: поиск таблиц")
    try:
        table_raw = engine.tables(image)
        result.raw["tables"] = table_raw
        result.objects = tables_from_result(table_raw, image, result.fragments)
        if audit and settings.ocr_language_mode != "single":
            # Paddle table detection uses the primary OCR run for layout. Its
            # HTML cell strings may therefore still contain the wrong script.
            # The actual line recognition variants own editable cell text.
            by_id = {f.id: f for f in result.fragments}
            for table in result.objects:
                for cell in table.cells:
                    if cell.ocr_ids:
                        cell.text = "\n".join(
                            f.text for f in sorted((by_id[i] for i in cell.ocr_ids),
                                                   key=lambda f: (f.bbox.y, f.bbox.x)))
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        # Text OCR remains available, but export requires acknowledgement of failure.
        result.messages.append(f"Обнаружение таблиц не завершено ({type(exc).__name__}). Проверьте и разметьте таблицы вручную.")
        result.raw["table_detection_failed"] = True
        result.raw["table_error"] = f"{type(exc).__name__}: {exc}"[:500]
    progress("OpenCV: проверка и поиск пропущенных сеток таблиц")
    geometric = geometric_tables(image, result.fragments, result.objects)
    result.objects.extend(geometric)
    result.raw["opencv_tables"] = [t.model_dump() for t in geometric]
    base_texts(result)
    if result.raw.get("table_detection_failed"):
        for obj in result.objects:
            obj.needs_review = True
            obj.review_notes.append("Проверка таблиц не выполнена")
    format_project(result, settings)
    if audit:
        uncertain = set(audit.get("uncertain_ids", []))
        failed_models = bool(audit.get("warnings"))
        for obj in result.objects:
            if failed_models or uncertain.intersection(obj.ocr_ids):
                obj.needs_review = True
                obj.review_notes.append("Многоязычное OCR: неоднозначное чтение или недоступная модель")
    result.usage.append({"stage": "local", "seconds": round(time.monotonic()-start, 3)})
    validate_project(result)
    return result


def reocr_text(project: Project, object_id: str, engine: PaddleEngine) -> Project:
    result = project.model_copy(deep=True)
    obj = next(o for o in result.objects if o.id == object_id)
    if not isinstance(obj, Text):
        raise ValueError("Повторное OCR области доступно для текстового блока")
    box = obj.bbox
    x0, y0 = int(box.x), int(box.y)
    image = source_image(result)
    crop = image.crop((x0, y0, min(image.width, int(box.right+1)), min(image.height, int(box.bottom+1))))
    fragments, raw = engine.recognize(crop, (x0, y0))
    if not fragments:
        obj.needs_review = True
        obj.review_notes.append("Повторное OCR не нашло текст; исходный результат сохранён")
        return result
    result.objects.remove(obj)
    result.ignored.update({i: f"Заменён повторным OCR области {object_id}" for i in obj.ocr_ids})
    result.fragments.extend(fragments)
    result.raw.setdefault("regions", []).append({"bbox": box.model_dump(), "results": raw})
    for f in fragments:
        result.objects.append(Text(id=f"text_{f.id}", bbox=f.bbox, text=f.text, ocr_ids=[f.id],
                                   rotation=f.rotation, confidence=f.confidence, style=obj.style.model_copy(),
                                   needs_review=f.confidence < engine.settings.confidence_threshold))
    validate_project(result)
    return result


def smart_ocr(project: Project, settings: Settings, api_key: str, cache: Path,
              consent: bool, engine=None, progress=lambda _: None, cancelled=lambda: False) -> Project:
    engine = engine or PaddleEngine(settings)
    client = VisionClient(settings, api_key, cache, consent, progress=progress, cancelled=cancelled)
    local = project.model_copy(deep=True)
    result = local
    start = time.monotonic()
    try:
        progress("VL: логические блоки и таблицы")
        result, analysis = analyze(local, client)
        if analysis.regions:
            image = source_image(result)
            replacements = {}
            unresolved = []
            for region in analysis.regions:
                if cancelled():
                    raise VisionCancelled("Smart OCR отменён пользователем")
                b = region.bbox
                x, y = int(b.x), int(b.y)
                crop = image.crop((x, y, min(image.width, int(b.right+1)), min(image.height, int(b.bottom+1))))
                fragments, raw = engine.recognize(crop, (x, y))
                result.raw.setdefault("regions", []).append({"bbox": b.model_dump(), "results": raw})
                if region.split_id:
                    replacements.setdefault(region.split_id, []).extend(fragments)
                else:
                    before_count = len(result.fragments)
                    for f in fragments:
                        if not any(intersection(f.bbox, old.bbox)/(f.bbox.w*f.bbox.h) > 0.4
                                   for old in result.fragments if old.id not in result.ignored):
                            result.fragments.append(f)
                    if len(result.fragments) == before_count:
                        unresolved.append(Text(id=uid("text"), bbox=b, text="", needs_review=True,
                                               review_notes=["VL предложила пропущенную надпись, но повторное OCR не дало новой геометрии: " + region.reason]))
            for parent, children in replacements.items():
                original = next(f for f in result.fragments if f.id == parent)
                unique = []
                for child in sorted(children, key=lambda f: f.confidence, reverse=True):
                    if not any(intersection(child.bbox, old.bbox)/min(child.bbox.w*child.bbox.h, old.bbox.w*old.bbox.h) > 0.6
                               for old in unique):
                        unique.append(child)
                children = unique
                measured = union([f.bbox for f in children]) if children else None
                if (len(children) < 2 or any(not contains(original.bbox, f.bbox, 2) for f in children)
                        or measured.w < original.bbox.w*0.8 or measured.h < original.bbox.h*0.8):
                    note = f"Повторное OCR не определило все части {parent}; задайте геометрию вручную"
                    result.messages.append(note)
                    for obj in result.objects:
                        if parent in obj.ocr_ids:
                            obj.needs_review = True
                            obj.review_notes.append(note)
                    continue
                result.ignored[parent] = "Заменён повторным OCR частей по запросу VL"
                for f in children:
                    f.parent_id = parent
                result.fragments.extend(children)
            tables = [o for o in result.objects if isinstance(o, Table)]
            active = [f for f in result.fragments if f.id not in result.ignored]
            for table in tables:
                previous = {c.id: (set(c.ocr_ids), c.text) for c in table.cells}
                assign_fragments(table, active)
                by_id = {f.id: f for f in active}
                for cell in table.cells:
                    old_ids, old_text = previous[cell.id]
                    added = set(cell.ocr_ids) - old_ids
                    if set(cell.ocr_ids) == old_ids:
                        continue
                    # Preserve edits to existing text. New geometry must not silently
                    # replace a manually corrected cell with raw OCR.
                    old_raw = "\n".join(f.text for f in sorted(
                        (f for f in result.fragments if f.id in old_ids),
                        key=lambda f: (f.bbox.y, f.bbox.x)))
                    if old_text == old_raw:
                        cell.text = "\n".join(f.text for f in sorted(
                            (by_id[i] for i in cell.ocr_ids), key=lambda f: (f.bbox.y, f.bbox.x)))
                    else:
                        extra = "\n".join(by_id[i].text for i in cell.ocr_ids if i in added)
                        cell.text = "\n".join(part for part in (old_text, extra) if part)
                    table.needs_review = True
                    table.review_notes.append("Добавлен текст повторного OCR: проверьте порядок строк в ячейке")
            result.objects = tables
            base_texts(result)
            validate_project(result)
            # One bounded second pass after newly measured geometry.
            result, second = analyze(result, client)
            for obj in unresolved:
                overlaps = [t for t in result.objects if isinstance(t, Table) and intersection(obj.bbox, t.bbox) > 0]
                if not overlaps:
                    result.objects.append(obj)
                for table in overlaps:
                    table.needs_review = True
                    table.review_notes.extend(obj.review_notes)
            if second.regions:
                result.messages.append("VL предлагает дополнительные области: выполните ручную проверку")
                for obj in result.objects:
                    obj.needs_review = True
        progress("VL: увеличенные фрагменты низкой уверенности")
        refine(result, client)
        if cancelled():
            raise VisionCancelled("Smart OCR отменён пользователем")
        format_project(result, settings, preserve_text_ids={o.id for o in local.objects if isinstance(o, Text)})
        validate_project(result)
    except (VisionError, ValueError, RuntimeError, OSError) as exc:
        # A failed geometric edit must not leak an invalid intermediate state.
        try:
            validate_project(result)
        except ValueError:
            result = local
        reason = str(exc) if isinstance(exc, VisionError) else f"Ошибка обработки Smart OCR ({type(exc).__name__})"
        if isinstance(exc, VisionCancelled):
            result.messages.append("Smart OCR отменён; локальные результаты сохранены")
            result.usage.extend(client.usage)
            raise SmartCancelled(result, reason) from exc
        result.messages.append("Smart OCR не завершён; локальные результаты сохранены: " + reason)
        result.usage.extend(client.usage)
        raise SmartFailure(result, reason) from exc
    finally:
        result.usage.append({"stage": "smart", "seconds": round(time.monotonic()-start, 3)})
    result.usage.extend(client.usage)
    return result

