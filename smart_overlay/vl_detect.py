"""Geometry-only line detection and local VLM text extraction for drawing OCR.

TextDetection NEVER reads characters. PaddleOCR-VL reads the crop with document
layout disabled and the official prompt_label='ocr'. The detected polygon,
not a hallucinated VL bounding box, owns PowerPoint overlay geometry.
"""
from __future__ import annotations

from math import atan2, degrees, hypot, isfinite
from pathlib import Path
from typing import Callable

import numpy as np

from .vl_local import LocalVLError, _as_plain_dict
from .models import Box

DETECTOR_MODEL = "PP-OCRv5_mobile_det"


def default_detector(device: str):
    try:
        from paddleocr import TextDetection
    except ImportError as exc:
        raise LocalVLError("Не установлен модуль PaddleOCR TextDetection") from exc
    return TextDetection(model_name=DETECTOR_MODEL, device=device)


def detect_lines(image, detector, *, min_score=.35, max_regions=120):
    """Source-pixel quadrilaterals sorted by reading position, no OCR text."""
    bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    data = list(detector.predict(input=bgr))
    if len(data) != 1:
        raise LocalVLError("Детектор вернул неверное число страниц")
    page = _as_plain_dict(data[0])
    polys = page.get("dt_polys")
    if polys is None:
        raise LocalVLError("Ответ TextDetection не содержит dt_polys")
    scores = page.get("dt_scores")
    if scores is None:
        scores = [1.0] * len(polys)
    if len(scores) != len(polys):
        raise LocalVLError("dt_scores не согласованы с dt_polys")
    found, rejected = [], []
    w, h = image.size
    for index, (poly, score) in enumerate(zip(polys, scores)):
        try:
            pts = np.asarray(poly, dtype=float)
            confidence = float(score)
            if pts.shape != (4, 2) or not np.isfinite(pts).all() or not isfinite(confidence):
                raise ValueError("invalid quadrilateral")
            if confidence < min_score:
                rejected.append({"index": index, "reason": "low_detection_confidence"})
                continue
            x0, y0 = pts.min(axis=0).tolist()
            x1, y1 = pts.max(axis=0).tolist()
            if x0 < -2 or y0 < -2 or x1 > w + 2 or y1 > h + 2:
                raise ValueError("out of image")
            # Some detectors overshoot the page edge by 1-2 pixels; the
            # source polygon must still pass strict Project validation.
            pts[:,0] = np.clip(pts[:,0],0,w)
            pts[:,1] = np.clip(pts[:,1],0,h)
            x0, y0 = pts.min(axis=0).tolist()
            x1, y1 = pts.max(axis=0).tolist()
            if x1-x0 < 4 or y1-y0 < 4:
                raise ValueError("tiny line")
            box = Box(x=x0,y=y0,w=x1-x0,h=y1-y0)
            top = pts[1] - pts[0]
            angle = degrees(atan2(float(top[1]),float(top[0])))
            # 180-degree ambiguity is unresolved and explicitly reviewable.
            if angle > 90: angle -= 180
            if angle < -90: angle += 180
            found.append({"id": index, "bbox": [box.x,box.y,box.right,box.bottom],
                          "polygon": pts.tolist(), "rotation": angle,
                          "score": confidence})
        except (TypeError, ValueError) as exc:
            rejected.append({"index": index, "reason": str(exc)[:80]})
    found.sort(key=lambda x: (round(x["bbox"][1]/14),x["bbox"][0],x["bbox"][1]))
    # Overlapping predictions of the exact same line are common in technical drawings.
    accepted = []
    for region in found:
        a = region["bbox"]
        discard = False
        for previous in accepted:
            b = previous["bbox"]
            intersection = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
            area = (a[2]-a[0])*(a[3]-a[1])
            other = (b[2]-b[0])*(b[3]-b[1])
            if intersection/max(1,min(area,other)) > .88:
                discard=True
                break
        if discard:
            rejected.append({"index": region["id"], "reason":"near-duplicate"})
        else:
            accepted.append(region)
    if len(accepted) > max_regions:
        raise LocalVLError(
            f"Детектор нашёл {len(accepted)} строк, разрешено не более {max_regions}. "
            "Прервите эксперимент или увеличьте --vl-max-regions; "
            "данные детекции останутся в диагностике.")
    return accepted, rejected


def crop_region(image, region, padding=8):
    x0,y0,x1,y1=region["bbox"]
    left=max(0,int(x0)-padding)
    top=max(0,int(y0)-padding)
    right=min(image.width,int(np.ceil(x1))+padding)
    bottom=min(image.height,int(np.ceil(y1))+padding)
    return image.crop((left,top,right,bottom))


def extract_text(page):
    """Read only explicit recognized strings, never guess from model metadata."""
    raw = page.get("parsing_res_list")
    if isinstance(raw,list):
        contents = []
        for block in raw:
            if not isinstance(block,dict):
                continue
            value=block.get("block_content")
            if not isinstance(value,str) or not value.strip():
                continue
            label=str(block.get("block_label","text")).lower()
            if label in {"text","paragraph_title","title","doc_title","ocr","table_caption",
                         "figure_title","number","header","footer","reference","formula",
                         "abstract","aside_text","footnote","seal","image","figure"}:
                if label in {"image","figure"} and (
                    value.startswith("![") or value.startswith("<img") or "data:image" in value):
                    continue
                contents.append(value.strip())
        return "\n".join(contents).strip()
    # Versioned outputs may expose recognized text at the top level.
    rec=page.get("rec_texts")
    if isinstance(rec,list) and all(isinstance(x,str) for x in rec):
        return "\n".join(x.strip() for x in rec if x.strip())
    if isinstance(page.get("rec_text"),str):
        return page["rec_text"].strip()
    return ""


def detected_page(manifest, directory):
    """Assemble text objects with exact original geometry and a visible audit."""
    import json
    blocks, missing = [], []
    for region in manifest["regions"]:
        path=Path(directory)/f"region-{region['id']:04d}.json"
        if not path.is_file():
            missing.append({"id":region["id"],"reason":"missing_checkpoint"})
            continue
        record=json.loads(path.read_text("utf-8"))
        content=record.get("text","").strip()
        if record.get("status")!="ok" or not content:
            missing.append({"id":region["id"],"reason":record.get("status","empty_text")})
            continue
        blocks.append({
            "block_label":"text", "block_content":content,
            "block_bbox":region["bbox"], "block_polygon":region["polygon"],
            "block_rotation":region["rotation"], "_vl_region_id":region["id"],
        })
    return ({"width":manifest["width"],"height":manifest["height"],
             "parsing_res_list":blocks,"_vl_detector_geometry":True},
            {"regions":len(manifest["regions"]), "recognized":len(blocks),
             "unreadable":missing,"detector_rejections":manifest.get("rejected",[])})
