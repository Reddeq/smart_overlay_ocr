"""One-click pipeline: local OCR -> conservative windowed VL -> native PPTX.

The UI is deliberately not responsible for grouping, table validation, or
export decisions. A network failure leaves a usable OCR-only presentation.
"""
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from uuid import uuid4
from typing import Callable

from .export import export_pptx
from .grouping import build_plan, candidate_project, run_grouping
from .input import load_image, save_project
from .models import Project, Table, Text
from .pipeline import local_ocr
from .settings import Settings, data_dir
from .structure import validate_project
from .table_review import build_table_plan, run_table_review, table_candidate_project


class ProcessingCancelled(RuntimeError):
    """The user cancelled before final export."""


@dataclass
class OneClickResult:
    project: Project
    pptx: Path
    project_file: Path
    warnings: list[str] = field(default_factory=list)
    grouping_requests: int = 0
    table_requests: int = 0

    @property
    def needs_review(self) -> int:
        return sum(not o.excluded and o.needs_review for o in self.project.objects)


def available_pptx(source: str | Path, destination: str | Path | None = None) -> Path:
    """Never overwrite an existing deliverable without an explicit destination."""
    original = Path(source)
    output = Path(destination) if destination else original.with_suffix(".pptx")
    if destination:
        return output
    number = 1
    while output.exists():
        output = original.with_name(f"{original.stem}-{number:02d}.pptx")
        number += 1
    return output


_NUMBER = re.compile(r"^[+−-]?\d+(?:[.,]\d+)*(?:\s*[%°])?$")
_CODE = re.compile(r"^(?=.*\d)[A-ZА-ЯЁ0-9_./+−:-]+$", re.IGNORECASE)


def is_standalone_designation(text: str) -> bool:
    """Conservative: leave numbers/codes in the original raster.

    A mixed phrase (e.g. 'Вилка D38999/26WA35PN') stays translatable.
    Ambiguous acronym-only words are left editable rather than silently lost.
    """
    text = text.strip()
    return bool(text and "\n" not in text and
                (_NUMBER.fullmatch(text) or (" " not in text and _CODE.fullmatch(text))))


def suppress_standalone_codes(project: Project) -> int:
    count = 0
    for obj in project.objects:
        if isinstance(obj, Text) and not obj.excluded and is_standalone_designation(obj.text):
            obj.excluded = True
            count += 1
    return count


def _cancelled(cancelled):
    if cancelled():
        raise ProcessingCancelled("Обработка отменена; готовый файл не заменён")


def _mark_uncertain(project: Project, reason: str):
    project.messages.append(reason)
    for obj in project.objects:
        if not obj.excluded:
            obj.needs_review = True
            if reason not in obj.review_notes:
                obj.review_notes.append(reason)


def _smart(project: Project, settings: Settings, api_key: str, proxy: str,
           progress: Callable[[str], None], cancelled: Callable[[], bool],
           warnings: list[str], max_requests: int) -> tuple[Project, int, int]:
    """Use experimentally validated windowed grouping, not legacy full-sheet Analysis."""
    grouping_requests = table_requests = 0
    remaining = max_requests
    try:
        plan = build_plan(project, settings, [], max_requests=remaining, review_ocr=False)
        grouping_requests = plan.requests
    except ValueError as exc:
        warnings.append(f"VL-группировка не запущена: {exc}")
        _mark_uncertain(project, "Не удалось подготовить VL-группировку: проверьте логические блоки")
        plan = None

    grouping_failed = False
    if plan and plan.grouping_requests:
        _cancelled(cancelled)
        progress(f"VL: группировка фрагментов ({plan.grouping_requests} областей)")
        result = run_grouping(plan, api_key, proxy, progress=progress, cancelled=cancelled)
        _cancelled(cancelled)
        proposals = [g["ocr_ids"] for g in result.consensus["accepted_merges"]]
        try:
            project = candidate_project(result, proposals)
        except ValueError as exc:
            warnings.append(f"Предложения VL не применены: {exc}")
            _mark_uncertain(project, "Ошибка проверки предложений VL")
        if result.status != "completed" or result.consensus["conflicts"] or result.consensus["blocked_ids"]:
            warnings.append("VL-группировка выполнена частично либо есть спорные области")
        grouping_failed = result.status != "completed"
        remaining -= grouping_requests

    _cancelled(cancelled)
    tables = [o.id for o in project.objects if isinstance(o, Table) and o.needs_review and not o.excluded]
    if tables and remaining and not grouping_failed:
        try:
            table_plan = build_table_plan(project, settings, tables, review_text=False,
                                          review_structure=True, max_requests=remaining)
            table_requests = table_plan.requests
            if table_requests:
                progress(f"VL: проверка сеток ({table_requests} таблиц)")
                reviewed = run_table_review(table_plan, api_key, proxy, progress=progress, cancelled=cancelled)
                _cancelled(cancelled)
                accepted = [record["table_id"] for record in reviewed.records
                            if record.get("kind") == "structure" and "reply" in record]
                if accepted:
                    # The table validator independently checks topology, OCR ownership,
                    # and geometry. Uncertain changes remain explicitly marked for review.
                    project = table_candidate_project(reviewed, selected_structures=accepted)
                if reviewed.status != "completed" or len(accepted) != table_requests:
                    warnings.append("Некоторые таблицы требуют ручной проверки")
        except (ValueError, RuntimeError, OSError) as exc:
            warnings.append(f"Автоматическая проверка таблиц недоступна: {type(exc).__name__}")
            _mark_uncertain(project, "Проверьте структуру таблиц")
    elif tables:
        warnings.append("VL-проверка таблиц пропущена: недостаточный бюджет или ошибка подключения")
    return project, grouping_requests, table_requests


def convert_drawing(source: str | Path, *, destination: str | Path | None = None,
                    settings: Settings | None = None, api_key: str = "",
                    consent: bool = False, proxy: str | None = None,
                    ocr_engine: str | None = None, vl_device: str | None = None,
                    vl_diagnostics: str | Path | None = None,
                    max_requests: int = 24,
                    progress: Callable[[str], None] = lambda _: None,
                    cancelled: Callable[[], bool] = lambda: False) -> OneClickResult:
    """Create editable PowerPoint with classic Smart OCR or independent local VL.

    Local VL never contacts OpenRouter, even with a key and consent.
    Sending data to the provider in Smart mode requires explicit consent.
    No API requests are made without both consent and a key. The complete
    source stays as the bottom PowerPoint raster; editing happens in overlays.
    """
    settings = (settings or Settings()).model_copy(deep=True)
    source = Path(source).resolve()
    output = available_pptx(source, destination)
    warnings: list[str] = []
    _cancelled(cancelled)
    progress("Загрузка чертежа")
    source_project = load_image(source)
    short_side = min(source_project.source.width, source_project.source.height)
    long_side = max(source_project.source.width, source_project.source.height)
    smallest_scale, largest_scale = 1 / short_side, 56 / long_side
    if smallest_scale > largest_scale:
        raise ValueError("Слишком вытянутый чертёж для одного слайда PowerPoint")
    # Keep the exact aspect ratio and OCR pixel geometry even for high-DPI images.
    settings.inches_per_pixel = min(largest_scale, max(smallest_scale, settings.inches_per_pixel))
    engine = ocr_engine or settings.ocr_engine
    if engine not in ("smart", "vl_local"):
        raise ValueError("Неизвестный движок OCR")
    settings.ocr_engine = engine
    if vl_device is not None:
        if vl_device not in ("cpu", "gpu:0"):
            raise ValueError("PaddleOCR-VL: допустимы устройства cpu и gpu:0")
        settings.vl_device = vl_device
    grouping_requests = table_requests = 0
    if engine == "vl_local":
        from .vl_local import local_vl_ocr
        run_dir = (Path(vl_diagnostics) if vl_diagnostics is not None else
                   data_dir() / "diagnostics" / "paddle-vl" /
                   f"{source.stem}-{uuid4().hex[:8]}")
        project = local_vl_ocr(
            source_project, settings, device=settings.vl_device,
            diagnostics_dir=run_dir, input_path=source,
            progress=progress, check=lambda: _cancelled(cancelled))
        warnings.append("Экспериментальный локальный PaddleOCR-VL: "
                        "проверьте геометрию фреймов и структуру таблиц")
        details = project.raw.get("local_vl", {})
        if details.get("skipped"):
            warnings.append(
                f"Не преобразовано областей: {len(details['skipped'])}. "
                "Причины и координаты — в локальной диагностике")
    elif engine == "smart":
        progress("Распознавание текста и таблиц (PaddleOCR / OpenCV)")
        project = local_ocr(source_project, settings, progress=progress)
    else:
        raise ValueError("Неизвестный движок OCR")
    _cancelled(cancelled)

    if engine == "vl_local":
        pass  # Never access the network/provider in local VL-only mode.
    elif consent and api_key:
        progress("Подготовка безопасной VL-группировки")
        try:
            project, grouping_requests, table_requests = _smart(
                project, settings, api_key, proxy or os.environ.get("SMART_OVERLAY_PROXY", "http://127.0.0.1:10809"),
                progress, cancelled, warnings, max_requests)
        except ProcessingCancelled:
            raise
        except (ValueError, RuntimeError, OSError) as exc:
            warnings.append(f"VL недоступна; сохранён локальный результат: {exc}")
            _mark_uncertain(project, "Smart OCR не завершён — проверьте OCR вручную")
    elif api_key and not consent:
        warnings.append("Облачный анализ отключён: нет согласия на отправку данных")
    else:
        warnings.append("Использован локальный OCR; для смысловой группировки настройте OpenRouter")

    _cancelled(cancelled)
    skipped = suppress_standalone_codes(project)
    if skipped:
        project.messages.append(f"Оставлено на оригинальном изображении обозначений: {skipped}")
    validate_project(project)
    progress("Создание редактируемого PowerPoint")
    # The output isn't written if export fails. Export to a temporary filename
    # in the same directory and rename only when fully generated.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".building-" + uuid4().hex[:8] + ".pptx")
    try:
        export_pptx(project, temporary, settings, allow_unreviewed=True, inline_text_fill=True)
        _cancelled(cancelled)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    # Keep a recoverable editable project in the per-user application directory,
    # not alongside client drawings; it includes the original raster and OCR text.
    saved = data_dir() / "projects" / f"{project.source.sha256}-{uuid4().hex[:8]}.socr.json"
    project.messages.extend(warnings)
    try:
        save_project(project, saved)
    except OSError as exc:
        warnings.append(f"PPTX создан, но промежуточный проект не сохранён: {type(exc).__name__}")
    progress(f"Готово: {output}")
    return OneClickResult(project=project, pptx=output, project_file=saved,
                          warnings=warnings, grouping_requests=grouping_requests,
                          table_requests=table_requests)
