import json
import os

import cv2
import numpy as np
from PIL import Image

from .geometry import angle, polygon_box
from .models import Fragment, uid
from .settings import Settings


def result_dict(result) -> dict:
    """PaddleX Result.json is a property; unwrap its documented 'res' envelope."""
    value = result.json
    if isinstance(value, str):
        value = json.loads(value)
    return value.get("res", value)


def fragments_from_result(raw: dict, offset=(0, 0), origin="paddle") -> list[Fragment]:
    polygons, texts, scores = raw["rec_polys"], raw["rec_texts"], raw["rec_scores"]
    if not len(polygons) == len(texts) == len(scores):
        raise ValueError("PaddleOCR вернул несогласованные массивы результатов")
    fragments = []
    for polygon, text, score in zip(polygons, texts, scores):
        points = [(float(p[0]) + offset[0], float(p[1]) + offset[1]) for p in polygon]
        if not text.strip():
            continue
        fragments.append(Fragment(id=uid("ocr"), text=text, polygon=points,
                                  bbox=polygon_box(points), rotation=angle(points),
                                  confidence=float(score), origin=origin))
    return fragments


class PaddleEngine:
    def __init__(self, settings: Settings):
        # Cached local inference must not depend on model-host connectivity probes.
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        self.settings = settings
        self._ocr = None
        self._tables = None
        self._last_ocr = None
        self.last_language_audit = {}
        self._line_recognizers = {}

    def recognize(self, image: Image.Image, offset=(0, 0), *, detection=None) -> tuple[list[Fragment], list[dict]]:
        from paddleocr import PaddleOCR
        from .languages import MODELS, recognize_lines, resolve
        lang, additional = resolve(self.settings)
        if self._ocr is None:
            self._ocr = PaddleOCR(lang=lang, ocr_version=self.settings.ocr_version,
                                  use_doc_orientation_classify=False, use_doc_unwarping=False,
                                  use_textline_orientation=True, device="cpu",
                                  text_rec_score_thresh=0.0)
        detection = detection or {}
        if set(detection) - {"text_det_thresh", "text_det_box_thresh", "text_det_unclip_ratio"}:
            raise ValueError("Неизвестный параметр детектора OCR")
        results = list(self._ocr.predict(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), **detection))
        if offset == (0, 0):
            self._last_ocr = results[0] if results else None
        raw = [result_dict(r) for r in results]
        fragments = [f for page in raw for f in fragments_from_result(page, offset,
                     "region" if offset != (0, 0) else "paddle")]
        self.last_language_audit = {"active": [lang], "uncertain_ids": [], "warnings": []}
        if additional and fragments:
            # Reuse one detection result; recognition-only models see measured crops.
            def model_factory(model_name):
                if model_name not in self._line_recognizers:
                    from paddleocr import TextRecognition
                    self._line_recognizers[model_name] = TextRecognition(model_name=model_name, device="cpu")
                return self._line_recognizers[model_name]
            group_for_language = next((group for group, (code, _) in MODELS.items() if code == lang), "en")
            fragments, self.last_language_audit = recognize_lines(
                image, fragments, group_for_language, additional,
                auto=self.settings.ocr_language_mode == "auto",
                model_factory=model_factory, offset=offset)
            # Raw Paddle result stays intact for TableRecognitionPipelineV2.
            # Attach only compact final-language metadata for diagnostics.
        return fragments, raw

    def tables(self, image: Image.Image) -> list[dict]:
        from paddleocr import TableRecognitionPipelineV2
        if self._last_ocr is None:
            raise ValueError("Сначала выполните OCR всего изображения")
        if self._tables is None:
            # Same table subpipeline as PP-StructureV3, without its unconditional
            # ChartRecognition initialization in PaddleX 3.3.13.
            self._tables = TableRecognitionPipelineV2(
                layout_detection_model_name="PP-DocLayout_plus-L",
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_ocr_model=False, use_layout_detection=True, device="cpu")
        return [result_dict(r) for r in self._tables.predict(
            cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR),
            overall_ocr_res=self._last_ocr, use_table_orientation_classify=False,
            use_ocr_results_with_table_cells=False)]

