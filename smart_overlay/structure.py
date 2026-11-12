from collections import Counter

from .geometry import contains, intersection, union
from .models import Box, Correction, Fragment, Project, Table, Text, uid


def validate_project(project: Project, for_export=False):
    # Revalidate mutable UI objects, including table topology and finite numbers.
    Project.model_validate(project.model_dump())
    bounds = Box(x=0, y=0, w=project.source.width, h=project.source.height)
    ids = [f.id for f in project.fragments]
    if len(ids) != len(set(ids)):
        raise ValueError("Повторяющиеся ID OCR")
    known = set(ids)
    object_ids = [o.id for o in project.objects]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("Повторяющиеся ID объектов")
    ownership = Counter()
    tables = [o for o in project.objects if isinstance(o, Table) and not o.excluded]
    for fragment in project.fragments:
        if not contains(bounds, fragment.bbox, 1):
            raise ValueError(f"OCR {fragment.id}: координаты вне изображения")
        if any(not (0 <= x <= bounds.w+1 and 0 <= y <= bounds.h+1) for x, y in fragment.polygon):
            raise ValueError(f"OCR {fragment.id}: многоугольник вне изображения")
    for obj in project.objects:
        if not contains(bounds, obj.bbox, 0.01):
            raise ValueError(f"{obj.id}: объект выходит за изображение")
        ownership.update(obj.ocr_ids)
        if isinstance(obj, Table):
            cell_ids = [c.id for c in obj.cells]
            if len(cell_ids) != len(set(cell_ids)):
                raise ValueError("Повторяющиеся ID ячеек")
            assigned = [i for c in obj.cells for i in c.ocr_ids]
            if Counter(assigned) != Counter(obj.ocr_ids):
                raise ValueError(f"{obj.id}: несогласованная принадлежность OCR ячейкам")
        elif not obj.excluded:
            for table in tables:
                if intersection(obj.bbox, table.bbox) > 1:
                    raise ValueError(f"{obj.id}: текстовый фрейм пересекает таблицу {table.id}")
        if for_export and not obj.excluded and obj.needs_review:
            raise ValueError(f"{obj.id}: требуется ручная проверка перед экспортом")
    for fragment_id, reason in project.ignored.items():
        if not reason.strip():
            raise ValueError("Для пропуска OCR требуется причина")
        ownership[fragment_id] += 1
    if set(ownership) - known:
        raise ValueError("Объект ссылается на неизвестный ID OCR")
    duplicates = [i for i, count in ownership.items() if count != 1]
    if duplicates:
        raise ValueError("OCR-фрагмент используется несколько раз: " + ", ".join(duplicates[:5]))
    missing = known - set(ownership)
    if missing:
        raise ValueError("Пропавшие OCR-фрагменты: " + ", ".join(sorted(missing)[:5]))


def base_texts(project: Project):
    table_ids = {i for o in project.objects if isinstance(o, Table) for i in o.ocr_ids}
    for f in project.fragments:
        if f.id in table_ids or f.id in project.ignored:
            continue
        project.objects.append(Text(id=uid("text"), bbox=f.bbox.model_copy(), ocr_ids=[f.id],
                                    text=f.text, rotation=f.rotation, confidence=f.confidence))


def merge_texts(project: Project, selected: list[str]) -> Text:
    objects = [o for o in project.objects if o.id in selected]
    if len(objects) < 2 or any(not isinstance(o, Text) or o.excluded for o in objects):
        raise ValueError("Выберите не менее двух обычных текстовых блоков")
    if max(o.rotation for o in objects) - min(o.rotation for o in objects) > 10:
        raise ValueError("Ориентация выбранных блоков различается")
    objects.sort(key=lambda o: (o.bbox.y, o.bbox.x))
    result = Text(id=uid("text"), bbox=union([o.bbox for o in objects]),
                  ocr_ids=[i for o in objects for i in o.ocr_ids], text="\n".join(o.text for o in objects),
                  rotation=objects[0].rotation, style=objects[0].style.model_copy(),
                  confidence=min(o.confidence for o in objects),
                  needs_review=any(o.needs_review for o in objects),
                  corrections=[c.model_copy() for o in objects for c in o.corrections],
                  review_notes=[n for o in objects for n in o.review_notes])
    candidate = project.model_copy(deep=True)
    candidate.objects = [o for o in candidate.objects if o.id not in selected] + [result]
    validate_project(candidate)
    project.objects = candidate.objects
    return result


def split_text(project: Project, object_id: str, boxes: list[Box], texts: list[str]):
    original = next(o for o in project.objects if o.id == object_id)
    if not isinstance(original, Text) or len(boxes) != len(texts) or len(boxes) < 2:
        raise ValueError("Для разделения нужны тексты и геометрия не менее двух частей")
    if any(not contains(original.bbox, b, 0.1) for b in boxes):
        raise ValueError("Части должны находиться внутри исходного блока")
    if any(intersection(a, b) > 0.1 for i, a in enumerate(boxes) for b in boxes[i+1:]):
        raise ValueError("Области частей пересекаются")
    candidate = project.model_copy(deep=True)
    candidate.objects = [o for o in candidate.objects if o.id != object_id]
    for fragment_id in original.ocr_ids:
        candidate.ignored[fragment_id] = f"Ручное разделение блока {object_id}; геометрия задана пользователем"
    for box, text in zip(boxes, texts):
        fragment = Fragment(id=uid("ocr"), bbox=box, text=text,
                            polygon=[(box.x, box.y), (box.right, box.y), (box.right, box.bottom), (box.x, box.bottom)],
                            confidence=1, origin="manual_split",
                            parent_id=original.ocr_ids[0] if original.ocr_ids else None)
        candidate.fragments.append(fragment)
        candidate.objects.append(Text(id=uid("text"), bbox=box, text=text, ocr_ids=[fragment.id],
                                      style=original.style.model_copy(), rotation=original.rotation,
                                      corrections=[c.model_copy() for c in original.corrections]))
    validate_project(candidate)
    project.objects, project.fragments, project.ignored = candidate.objects, candidate.fragments, candidate.ignored


def manual_text(obj: Text, text: str):
    if obj.text != text:
        obj.corrections.append(Correction(source="manual", before=obj.text, after=text,
                                          confidence=1, applied=True, reason="Редактирование пользователем"))
        obj.text = text

