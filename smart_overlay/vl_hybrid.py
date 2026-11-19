"""Bounded, fully local *selective* PaddleOCR-VL checking after regular OCR.

The classic OCR owns all source-pixel geometry and table topology. VL is only
given a small list of suspicious, independently cropped text fragments.
Ambiguous suggestions never silently replace technical labels.
"""
from __future__ import annotations

from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Callable

from .models import Project, Table, Text
from .settings import Settings
from .structure import validate_project

REPLACEMENTS = ("\ufffd", "\u25a1", "\u25af", "\u00a4")
_DIGITS = re.compile(r"\d+")
_CYR = re.compile(r"[А-Яа-яЁё]")
_LAT = re.compile(r"[A-Za-z]")


def suspicious(text: str) -> bool:
    """Very limited heuristics: ambiguous mixed writing remains reviewable."""
    return any(c in text for c in REPLACEMENTS) or bool(_CYR.search(text) and _LAT.search(text))


def candidates(project: Project, settings: Settings, *, max_checks: int | None = None):
    from .one_click import is_standalone_designation
    limit = settings.hybrid_max_checks if max_checks is None else max_checks
    used = {identifier for obj in project.objects if not obj.excluded
            for identifier in (obj.ocr_ids if isinstance(obj,Text)
                               else [i for cell in obj.cells for i in cell.ocr_ids])}
    rows = []
    for frag in project.fragments:
        line = frag.text.strip()
        if (frag.id not in used or not line or not any(ch.isalpha() for ch in line)
                or is_standalone_designation(line)):
            continue
        suspect = suspicious(line)
        if frag.confidence >= settings.hybrid_confidence and not suspect:
            continue
        rows.append((frag, suspect))
    # Prioritize clearly damaged strings, then low-confidence ones. Never
    # try every detected line just because there are hundreds of them.
    rows.sort(key=lambda entry: (0 if any(c in entry[0].text for c in REPLACEMENTS)
                                 else 1 if entry[1] else 2,
                                 entry[0].confidence, entry[0].bbox.y))
    return rows[:limit], len(rows)


def safe_change(old: str, suggestion: str, confidence: float) -> tuple[bool, str]:
    """Do not 'correct' codes, numbers, or different writing systems blindly."""
    old, suggestion = old.strip(), suggestion.strip()
    if not suggestion:
        return False, "empty_response"
    if old == suggestion:
        return False, "unchanged"
    if "\n" in suggestion and "\n" not in old:
        return False, "unexpected_multiline"
    if len(suggestion) > 240 or not (
        .65 <= len(suggestion)/max(1,len(old)) <= 1.6):
        return False, "length_mismatch"
    if _DIGITS.findall(old) != _DIGITS.findall(suggestion):
        return False, "technical_digits_changed"
    if suspicious(suggestion):
        return False, "mixed_script_or_replacement"
    if _CYR.search(old) and _LAT.search(suggestion) and not _CYR.search(suggestion):
        return False, "script_changed"
    if _LAT.search(old) and _CYR.search(suggestion) and not _LAT.search(suggestion):
        return False, "script_changed"
    if confidence >= .60 and not any(c in old for c in REPLACEMENTS):
        return False, "original_not_low_confidence"
    similarity = SequenceMatcher(None, old.casefold(), suggestion.casefold()).ratio()
    if similarity < .62 and not any(c in old for c in REPLACEMENTS):
        return False, "unrelated_text"
    return True, "safe_low_confidence_correction"


def _owner_locations(project: Project, frag_id: str):
    for obj in project.objects:
        if isinstance(obj, Text) and not obj.excluded and frag_id in obj.ocr_ids:
            yield ("text", obj, None)
        elif isinstance(obj, Table) and not obj.excluded:
            for cell in obj.cells:
                if frag_id in cell.ocr_ids:
                    yield ("cell", obj, cell)


def apply_suggestions(project: Project, checks, results: dict):
    """Apply only fully traceable low-risk corrections; audit all suggestions."""
    items = {fragment.id: fragment for fragment in project.fragments}
    audit = []
    for fragment, _ in checks:
        answer = results.get(fragment.id, {})
        suggestion = str(answer.get("text", "") or "").strip()
        base = {"ocr_id": fragment.id, "original": fragment.text,
                "suggested": suggestion, "ocr_confidence": fragment.confidence,
                "vl_seconds": answer.get("seconds"), "status": answer.get("status","missing")}
        if answer.get("status") != "ok":
            base["decision"] = "not_available"
            audit.append(base)
            continue
        valid, reason = safe_change(fragment.text, suggestion,fragment.confidence)
        owners = list(_owner_locations(project,fragment.id))
        base["reason"] = reason
        if not valid or not owners:
            base["decision"] = "unchanged" if reason == "unchanged" else "review_only"
            audit.append(base)
            for _, obj, _ in owners:
                obj.needs_review = True
                note = f"VL предлагает другое чтение для {fragment.id}: проверьте вручную"
                if note not in obj.review_notes:
                    obj.review_notes.append(note)
            continue
        # Only substitute when the editable object contains exactly this
        # recognizable string; never rewrite a merged multi-line label or table.
        replacements = []
        old = fragment.text
        for kind, obj, cell in owners:
            if kind == "text":
                if len(obj.ocr_ids) != 1 or obj.text != old:
                    replacements = []
                    break
                replacements.append((obj, "text", old))
            else:
                if len(cell.ocr_ids) != 1 or cell.text != old:
                    replacements = []
                    break
                replacements.append((cell, "text", old))
        if not replacements:
            base["decision"] = "review_only"
            base["reason"] = "merged_or_mismatched_owner"
            for _, obj, _ in owners:
                obj.needs_review = True
                obj.review_notes.append("Проверьте VL-вариант для " + fragment.id)
            audit.append(base)
            continue
        for obj, attr, _ in replacements:
            setattr(obj, attr, suggestion)
        items[fragment.id].text = suggestion
        # Do not promote VL output to calibrated confidence; keep it reviewable.
        for _, obj, _ in owners:
            obj.needs_review = True
            obj.review_notes.append("Текст уточнён локальной VL; проверьте технические обозначения")
        base["decision"] = "applied_requires_review"
        audit.append(base)
    validate_project(project)
    return audit


def selective_vl(project: Project, source_path: str | Path, settings: Settings,
                 *, diagnostics_dir: str | Path, progress: Callable[[str],None],
                 cancelled: Callable[[],bool]) -> tuple[Project,list[str]]:
    """One warm VLM process for <=N suspect OCR lines; local-only API."""
    from .one_click import ProcessingCancelled
    selected, total = candidates(project,settings)
    report = {"version":1, "mode":"hybrid_local", "suspected":total,
              "selected":len(selected), "max_checks":settings.hybrid_max_checks,
              "reason":"OCR confidence and damaged/mixed scripts",
              "items":[{"id":frag.id,"bbox":[frag.bbox.x,frag.bbox.y,
                       frag.bbox.right,frag.bbox.bottom],
                       "ocr_text":frag.text,"ocr_confidence":frag.confidence,
                       "reason":"script_or_garbage" if suspect else "low_confidence"}
                       for frag,suspect in selected]}
    folder=Path(diagnostics_dir)
    folder.mkdir(parents=True,exist_ok=True)
    plan=folder/"paddle-vl-hybrid-plan.json"
    plan.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    project.raw["hybrid_local"] = {"suspected":total, "selected":len(selected),
                                   "mode":"local", "diagnostics":str(folder)}
    warnings=[]
    if total > len(selected):
        warnings.append(f"Локальная VL проверит только {len(selected)} из {total} "
                        "сомнительных OCR-фрагментов (ограничение времени)")
    if not selected:
        project.messages.append("Гибридный OCR: сомнительных строк для локальной VL не найдено")
        return project,warnings
    progress(f"Локальная VL: проверка {len(selected)} из {len(project.fragments)} OCR-фрагментов")
    from .vl_tiled import run_isolated
    def check():
        if cancelled():
            raise ProcessingCancelled("Обработка отменена")
    try:
        run_isolated(source_path,settings,settings.vl_device,folder,
                     progress=progress,check=check,hybrid=True,
                     startup_timeout=settings.vl_startup_timeout_seconds)
    except ProcessingCancelled:
        raise
    except (OSError,ValueError,RuntimeError) as exc:
        warnings.append("VL-проверка не завершилась ("+type(exc).__name__+
                        "); используется исходный OCR, диагностика сохранена")
        project.raw["hybrid_local"]["error"]=type(exc).__name__
        for frag,_ in selected:
            for _,obj,_ in _owner_locations(project,frag.id):
                obj.needs_review=True
                obj.review_notes.append("Локальная VL не завершила проверку")
        return project,warnings
    outcomes={}
    for frag,_ in selected:
        path=folder/("hybrid-"+re.sub(r"[^A-Za-z0-9_-]","_",frag.id)+".json")
        if path.exists():
            try:
                outcomes[frag.id]=json.loads(path.read_text("utf-8"))
            except (ValueError,OSError):
                outcomes[frag.id]={"status":"bad_checkpoint"}
    audit=apply_suggestions(project,selected,outcomes)
    (folder/"paddle-vl-hybrid-audit.json").write_text(
        json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    project.raw["hybrid_local"].update(
        {"evaluated":len(audit),"applied":sum(x["decision"]=="applied_requires_review" for x in audit),
         "suggested_only":sum(x["decision"]=="review_only" for x in audit),
         "audit":audit})
    if any(x["decision"]=="review_only" for x in audit):
        warnings.append("Часть предложений VL оставлена на ручную проверку")
    return project,warnings
