"""Experimental all-line OCR ensemble for engineering drawings (fully local).

Baseline PP-OCRv5 is retained. PP-OCRv6_medium_rec (50 supported languages,
NO Cyrillic/Korean/Arabic) reads the SAME measured line crops using multiple
pixel variants; unsupported scripts use an appropriate PP-OCRv5 recognizer.
Inter-model confidence is NOT a calibrated quality measure. Differences and
unresolved cases are audited, not silently promoted to ground truth.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import re
import time

import cv2
import numpy as np

from .input import source_image
from .models import Text, Table
from .languages import run_recognizer
from .vl_hybrid import suspicious, safe_change, _owner_locations
from .structure import validate_project

V6_MODEL = "PP-OCRv6_medium_rec"
SCRIPT_V5 = {
    "cyrillic": "eslav_PP-OCRv5_mobile_rec",
    "arabic": "arabic_PP-OCRv5_mobile_rec",
    "korean": "korean_PP-OCRv5_mobile_rec",
}
_SCRIPT = {
    "cyrillic": re.compile(r"[\u0400-\u052f]"),
    "arabic": re.compile(r"[\u0600-\u06ff\u0750-\u077f]"),
    "korean": re.compile(r"[\uac00-\ud7af\u1100-\u11ff]"),
}
NUMBER = re.compile(r"\d+")


def model_for(text, settings):
    """Only route supported scripts to PP-OCRv6; unknown Latin-looking OCR
    can still be re-read by v6, allowing damaged Chinese to be discovered."""
    for script, pattern in _SCRIPT.items():
        if pattern.search(text):
            return SCRIPT_V5[script]
    if settings.ocr_language_mode == "manual":
        codes = set(settings.ocr_languages)
        if codes and codes.issubset({"ru","uk","be"}):
            return SCRIPT_V5["cyrillic"]
        if codes and codes.issubset({"ar"}):
            return SCRIPT_V5["arabic"]
        if codes and codes.issubset({"korean"}):
            return SCRIPT_V5["korean"]
    return V6_MODEL


def crops(image, fragments):
    """Original-pixel crops; image processing never changes project geometry."""
    out=[]
    for frag in fragments:
        b=frag.bbox
        x0=max(0,int(b.x)-3); y0=max(0,int(b.y)-3)
        x1=min(image.width,int(np.ceil(b.right))+3)
        y1=min(image.height,int(np.ceil(b.bottom))+3)
        if x1<=x0 or y1<=y0:
            raise ValueError("Неверная геометрия OCR: "+frag.id)
        out.append(np.asarray(image.crop((x0,y0,x1,y1)).convert("RGB")))
    return out


def enhance(crop, variant):
    """Upscaling/contrast for tiny technical labels, with bounded pixels."""
    h,w=crop.shape[:2]
    factor=min(2.0, max(1.0, 32/max(1,h)), 2048/max(1,w))
    if variant=="large":
        return cv2.resize(crop,None,fx=factor,fy=factor,
                          interpolation=cv2.INTER_CUBIC) if factor>1.01 else crop
    if variant=="contrast":
        gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
        enhanced=cv2.createCLAHE(clipLimit=1.7,tileGridSize=(4,4)).apply(gray)
        return cv2.cvtColor(enhanced,cv2.COLOR_GRAY2RGB)
    raise ValueError("Неизвестная обработка изображения")


def _predict(model, images):
    return run_recognizer(model, images)


def compare(base, variants):
    """Select a *candidate* by intra-model agreement, not score comparison."""
    nonempty=[(name,text.strip(),float(conf)) for name,text,conf in variants
              if text.strip() and 0<=float(conf)<=1]
    if len(nonempty)<2:
        return None,"insufficient_repeated_readings"
    by_text=Counter(text for _,text,_ in nonempty)
    top, count=by_text.most_common(1)[0]
    if count<2:
        return None,"recognizers_disagree"
    if top == base.text:
        return top,"agrees_with_original"
    return top,"repeated_candidate_differs"


def _substitute_if_safe(project, fragment, new_text, *, verified_by_vl=False):
    """Only single-fragment editable objects/cells may be changed."""
    from .one_click import is_standalone_designation
    if is_standalone_designation(fragment.text) or is_standalone_designation(new_text):
        return False,"technical_code_not_modified"
    safe,why=safe_change(fragment.text,new_text,
                         min(fragment.confidence,.59) if verified_by_vl
                         else fragment.confidence)
    if not safe:
        return False,why
    owners=list(_owner_locations(project,fragment.id))
    if not owners:
        return False,"no_editable_owner"
    for kind,obj,cell in owners:
        if kind=="text":
            if len(obj.ocr_ids)!=1 or obj.text!=fragment.text:
                return False,"multi_fragment_label"
        elif len(cell.ocr_ids)!=1 or cell.text!=fragment.text:
            return False,"multi_fragment_cell"
    for kind,obj,cell in owners:
        if kind=="text":
            obj.text=new_text
        else:
            cell.text=new_text
        obj.needs_review=True
        obj.review_notes.append("Уточнение OCR по повторному чтению; проверьте текст")
    fragment.text=new_text
    return True,"low_confidence_consensus_needs_review"


def evaluate_candidates(project, records, *, apply=False):
    """Attach disagreements to owning frames/cells, preserve original geometry."""
    by_id={f.id:f for f in project.fragments}
    counts=Counter()
    for entry in records:
        fragment=by_id[entry["id"]]
        candidate,reason=compare(fragment, entry.get("readings",[]))
        entry["consensus"]=candidate
        entry["comparison"]=reason
        differing=any(text.strip() and text.strip()!=fragment.text
                      for _,text,_ in entry.get("readings",[]))
        entry["disagreement"]=differing
        entry["decision"]="keep_original"
        if differing:
            counts["disagreements"]+=1
            for _,obj,_ in _owner_locations(project,fragment.id):
                obj.needs_review=True
                if "Разные модели OCR прочли эту надпись по-разному" not in obj.review_notes:
                    obj.review_notes.append("Разные модели OCR прочли эту надпись по-разному")
        if apply and reason=="repeated_candidate_differs" and candidate is not None:
            applied,why=_substitute_if_safe(project,fragment,candidate)
            entry["decision"]="applied_review_required" if applied else "review_only"
            entry["decision_reason"]=why
            counts["applied" if applied else "review_only"]+=1
        elif differing:
            entry["decision"]="review_only"
            counts["review_only"]+=1
        else:
            counts["agreed"]+=1
    validate_project(project)
    return dict(counts)


def run_ensemble(project, settings, directory, *,
                 progress=lambda _:None, cancelled=lambda:False,
                 model_factory=None):
    """Recognize every already-detected line with supported second reader.

    If one model is unavailable, retain baseline OCR and keep audit of failures.
    All input crops and output names are local; this never accesses OpenRouter.
    """
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    image=source_image(project)
    all_fragments=[f for f in project.fragments if f.text.strip()]
    overview={"mode":"dual_ocr", "baseline":"PP-OCRv5", "alternate":V6_MODEL,
              "total_fragments":len(all_fragments),"models":{},"warnings":[],
              "records":[], "elapsed_seconds":0}
    start=time.monotonic()
    factory=model_factory
    if factory is None:
        def factory(name):
            from paddleocr import TextRecognition
            return TextRecognition(model_name=name,device="cpu")
    grouped={}
    for fragment in all_fragments:
        grouped.setdefault(model_for(fragment.text,settings),[]).append(fragment)
    source_crops=crops(image,all_fragments)
    crop_by_id={f.id:crop for f,crop in zip(all_fragments,source_crops)}
    records={f.id:{"id":f.id,"original":f.text,
                   "confidence":f.confidence, "model":model_for(f.text,settings),
                   "bbox":[f.bbox.x,f.bbox.y,f.bbox.right,f.bbox.bottom],
                   "readings":[]} for f in all_fragments}
    try:
        for name,fragments in grouped.items():
            if cancelled():
                raise InterruptedError("OCR остановлен")
            progress("Второй OCR: "+name+f" ({len(fragments)} строк)")
            ts=time.monotonic()
            try:
                model=factory(name)
                for variant in ("large","contrast"):
                    for start_index in range(0,len(fragments),16):
                        if cancelled():
                            raise InterruptedError("OCR остановлен")
                        batch=fragments[start_index:start_index+16]
                        images=[enhance(crop_by_id[f.id],variant) for f in batch]
                        predictions=_predict(model,images)
                        for frag,(text,confidence) in zip(batch,predictions):
                            records[frag.id]["readings"].append(
                                [variant,text,float(confidence)])
                overview["models"][name]={"status":"ok","fragments":len(fragments),
                                         "seconds":round(time.monotonic()-ts,3)}
            except (ImportError,TypeError,ValueError,RuntimeError,OSError) as exc:
                overview["models"][name]={"status":"unavailable",
                                         "error_type":type(exc).__name__}
                overview["warnings"].append(
                    name+" недоступна ("+type(exc).__name__+"); исходный OCR сохранён")
    except InterruptedError:
        raise
    overview["records"]=list(records.values())
    overview["stats"]=evaluate_candidates(project,overview["records"],
                                           apply=settings.ensemble_apply_consensus)
    after={f.id:f.text for f in project.fragments}
    for entry in overview["records"]:
        entry["final"]=after[entry["id"]]
    overview["elapsed_seconds"]=round(time.monotonic()-start,3)
    (directory/"ocr-ensemble.json").write_text(
        json.dumps(overview,ensure_ascii=False,indent=2),encoding="utf-8")
    project.raw["ocr_ensemble"]={
        "models":overview["models"],"stats":overview["stats"],
        "total_fragments":len(all_fragments),
        "diagnostics":str(directory/"ocr-ensemble.json")}
    return project,overview


def disagreement_sample(project,report,limit=8):
    """Priority from actual disagreements, NOT only baseline confidence."""
    by_id={f.id:f for f in project.fragments}
    entries=[x for x in report["records"] if x.get("disagreement") and
             x["id"] in by_id and any(c.isalpha() for c in by_id[x["id"]].text)]
    entries.sort(key=lambda r: (0 if suspicious(r["original"]) else 1,
                                r["confidence"],r["bbox"][1]))
    return entries[:max(0,limit)]


def ensemble_hybrid(project, source_path, settings, *, directory,
                    progress=lambda _:None, cancelled=lambda:False,
                    model_factory=None):
    """Three-way *experimental* analysis: classic v5, compatible v6, batch VL.

    Never call OpenRouter. A failed optional recognizer does not block PPTX.
    High-confidence automatic changes require SAME answer from v6 and a
    completely ID-validated VL batch, plus non-numeric character safeguards.
    """
    from .one_click import ProcessingCancelled
    from .vl_hybrid import apply_suggestions
    from .vl_tiled import run_isolated
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    warnings=[]
    try:
        project,report=run_ensemble(project,settings,directory,
                                    progress=progress,cancelled=cancelled,
                                    model_factory=model_factory)
    except InterruptedError as exc:
        raise ProcessingCancelled(str(exc)) from exc
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        warnings.append("Второй OCR недоступен ("+type(exc).__name__+
                        "), сохранён результат PP-OCRv5")
        project.raw["ocr_ensemble"]={"status":"failed","error_type":type(exc).__name__}
        return project,warnings
    summary={
        "baseline_fragments":report["total_fragments"],
        "second_ocr_readings":sum(1 for x in report["records"] if x["readings"]),
        "ocr_disagreements":report["stats"].get("disagreements",0),
        "models":report["models"],"model_warnings":report["warnings"],
        "vl_batches_attempted":0,"vl_batches_verified":0,"vl_checked":0,
        "vl_applied":0,"vl_review_only":0,
        "baseline_requires_review":True,
    }
    warnings.extend(report["warnings"])
    if not settings.ensemble_batch_vl or not settings.ensemble_max_batches:
        project.raw["ocr_ensemble"].update(summary)
        (directory/"ocr-ensemble-summary.json").write_text(
            json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
        return project,warnings
    limit=settings.ensemble_batch_size*settings.ensemble_max_batches
    chosen=disagreement_sample(project,report,limit=limit)
    if not chosen:
        project.raw["ocr_ensemble"].update(summary)
        (directory/"ocr-ensemble-summary.json").write_text(
            json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
        return project,warnings
    items=[{"id":x["id"],"bbox":x["bbox"],
            "ocr_text":x["original"]} for x in chosen]
    plan={"mode":"batch","version":1,"items":items,
          "batch_size":settings.ensemble_batch_size,
          "max_batches":settings.ensemble_max_batches}
    (directory/"paddle-vl-hybrid-plan.json").write_text(
        json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    summary["vl_batches_attempted"]=(len(chosen)+settings.ensemble_batch_size-1)//settings.ensemble_batch_size
    def check():
        if cancelled():
            raise ProcessingCancelled("Остановлено пользователем")
    try:
        progress(f"Пакетная VL: {len(chosen)} расхождений в "
                 f"{summary['vl_batches_attempted']} изображениях")
        run_isolated(source_path,settings,settings.vl_device,directory,
                     progress=progress,check=check,hybrid=True,
                     startup_timeout=settings.vl_startup_timeout_seconds)
    except ProcessingCancelled:
        raise
    except (RuntimeError,ValueError,OSError) as exc:
        warnings.append("Пакетная VL не завершена ("+type(exc).__name__+
                        "); исходный OCR сохранён")
        summary["vl_error_type"]=type(exc).__name__
    # Only complete and fully verified batches have hybrid-<id>.json.
    outcomes={}
    batch_count=summary["vl_batches_attempted"]
    for batch in range(batch_count):
        path=directory/f"vl-batch-{batch:02d}.json"
        if not path.exists():
            continue
        try:
            batch_record=json.loads(path.read_text("utf-8"))
        except (OSError,ValueError):
            continue
        if batch_record.get("status")!="complete":
            continue
        ids=batch_record.get("ids",[])
        summary["vl_batches_verified"]+=1
        for identifier in ids:
            value=directory/f"hybrid-{identifier}.json"
            if not value.exists():
                continue
            try:
                response=json.loads(value.read_text("utf-8"))
            except (OSError,ValueError):
                continue
            if response.get("status")=="ok":
                outcomes[identifier]=response
    if summary["vl_batches_verified"]<summary["vl_batches_attempted"]:
        warnings.append("Часть пакетных ответов VL не содержит точного соответствия "
                        "всех ID; такие результаты не применялись")
    by_id={frag.id:frag for frag in project.fragments}
    checks=[(by_id[entry["id"]],True) for entry in chosen
            if entry["id"] in outcomes and entry["id"] in by_id]
    vl_audit=apply_suggestions(project,checks,outcomes) if checks else []
    report_by_id={x["id"]:x for x in report["records"]}
    for item in vl_audit:
        if item["decision"]!="review_only":
            continue
        record=report_by_id[item["ocr_id"]]
        proposed=item["suggested"]
        # Independent model consensus + full-ID VL agreement can overrule
        # an overly confident old OCR, but only on low-risk non-code strings.
        if (record.get("model") != V6_MODEL or
                proposed != record.get("consensus") or
                record.get("comparison") != "repeated_candidate_differs"):
            # A repeated PP-OCRv5 reading of Cyrillic is NOT independent
            # corroboration. Never auto-overwrite confident Russian labels.
            continue
        fragment=by_id[item["ocr_id"]]
        # hard floor for similarity; safe_change is reused with a capped
        # confidence because this case has additional independent evidence.
        from .vl_hybrid import safe_change
        from .one_click import is_standalone_designation
        if (is_standalone_designation(fragment.text) or
                is_standalone_designation(proposed)):
            continue
        from difflib import SequenceMatcher
        if SequenceMatcher(None,fragment.text.casefold(),proposed.casefold()).ratio()<.82:
            continue
        safe,why=safe_change(fragment.text,proposed,min(fragment.confidence,.59))
        if not safe:
            continue
        applied,details=_substitute_if_safe(project,fragment,proposed,
                                            verified_by_vl=True)
        if applied:
            item["decision"]="applied_by_v6_and_vl_review_required"
            item["reason"]="independent_ocr_and_complete_vl_batch_agree"
    summary["vl_checked"]=len(vl_audit)
    summary["vl_applied"]=sum(x["decision"].startswith("applied") for x in vl_audit)
    summary["vl_review_only"]=sum(x["decision"]=="review_only" for x in vl_audit)
    (directory/"ocr-ensemble-vl-audit.json").write_text(
        json.dumps(vl_audit,ensure_ascii=False,indent=2),encoding="utf-8")
    project.raw["ocr_ensemble"].update(summary)
    actual={fragment.id:fragment.text for fragment in project.fragments}
    for entry in report["records"]:
        entry["final"]=actual.get(entry["id"],entry["original"])
    (directory/"ocr-ensemble.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (directory/"ocr-ensemble-summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return project,warnings
