"""Language-aware OCR routing for mixed-script technical drawings.

One PaddleOCR pass provides line geometry and table-detection context. Additional
PP-OCRv5 text recognizers operate only on those measured line crops, preserving
OCR IDs/polygons. No VL API or language-detection network service is involved.
"""
from dataclasses import dataclass
import re
import unicodedata

import numpy as np

LANGUAGES = {
    "ru": ("Русский", "eslav"), "en": ("English", "en"),
    "ch": ("中文", "cjk"), "japan": ("日本語", "cjk"),
    "korean": ("한국어", "korean"), "ar": ("العربية", "arabic"),
    "fr": ("Français", "latin"), "de": ("Deutsch", "latin"),
    "es": ("Español", "latin"), "it": ("Italiano", "latin"),
    "pt": ("Português", "latin"), "uk": ("Українська", "eslav"),
    "be": ("Беларуская", "eslav"),
}
# The PaddleOCR v5 language resolver maps the displayed languages to these
# recognition models. Several languages can share the same recognizer.
MODELS = {
    "cjk": ("ch", "PP-OCRv5_server_rec"),
    "eslav": ("ru", "eslav_PP-OCRv5_mobile_rec"),
    "en": ("en", "en_PP-OCRv5_mobile_rec"),
    "latin": ("fr", "latin_PP-OCRv5_mobile_rec"),
    "korean": ("korean", "korean_PP-OCRv5_mobile_rec"),
    "arabic": ("ar", "arabic_PP-OCRv5_mobile_rec"),
}
AUTO_GROUPS = ("cjk", "eslav", "korean", "arabic", "latin")
# English is already supported in cjk/eslav/korean; Arabic needs English added.
EMBEDDED_ENGLISH = {"cjk", "eslav", "korean", "latin", "en"}
GROUP_SEQUENCE = ("cjk", "eslav", "korean", "arabic", "latin", "en")

SCRIPT_RANGES = {
    "cjk": ((0x3400, 0x9FFF), (0x3040, 0x30FF), (0xF900, 0xFAFF)),
    "eslav": ((0x0400, 0x052F),),
    "korean": ((0x1100, 0x11FF), (0xAC00, 0xD7AF)),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)),
}


def families(languages):
    unknown = set(languages) - set(LANGUAGES)
    if unknown:
        raise ValueError("Неизвестные языки OCR: " + ", ".join(sorted(unknown)))
    groups = {LANGUAGES[code][1] for code in languages}
    if "en" in groups and groups & (EMBEDDED_ENGLISH - {"en"}):
        groups.remove("en")
    return tuple(group for group in GROUP_SEQUENCE if group in groups)


def resolve(settings):
    """Choose primary PaddleOCR language and optional recognition-only passes."""
    mode = settings.ocr_language_mode
    if mode == "single":
        return settings.language, ()
    if settings.ocr_version != "PP-OCRv5":
        raise ValueError("Многоязычное распознавание поддерживается с PP-OCRv5")
    groups = AUTO_GROUPS if mode == "auto" else families(settings.ocr_languages)
    if not groups:
        raise ValueError("Выберите хотя бы один язык")
    # Keep a single model if all selected languages share its vocabulary.
    primary = groups[0]
    if mode == "manual" and "eslav" in groups and "cjk" not in groups:
        primary = "eslav"
    if mode == "manual" and groups == ("en",):
        primary = "en"
    return MODELS[primary][0], tuple(group for group in groups if group != primary)


def native_count(text, group):
    if group not in SCRIPT_RANGES:
        return 0
    ranges = SCRIPT_RANGES[group]
    return sum(any(left <= ord(char) <= right for left, right in ranges) for char in text)


def native_evidence(text, group, confidence):
    if group == "latin":
        # English is already covered by the primary CJK model. Probe the Latin
        # recognizer for diacritics that the primary model may not support.
        return confidence >= .68 and any(
            ord(c) > 127 and "LATIN" in unicodedata.name(c, "") for c in text)
    letters = sum(c.isalpha() for c in text)
    native = native_count(text, group)
    return bool(native >= 2 and native / max(letters, 1) >= .45 and confidence >= .62)


def group_score(text, confidence, group):
    """Cross-model confidences are not calibrated: script evidence is a hint."""
    native = native_count(text, group)
    letters = max(1, sum(c.isalpha() for c in text))
    rank = float(confidence)
    if group in SCRIPT_RANGES:
        rank += .17 * (native / letters) if native >= 2 else -.06
    elif group == "latin":
        rank += .09 if any(ord(c) > 127 and "LATIN" in unicodedata.name(c, "") for c in text) else 0
    elif group == "en":
        rank += .02 if re.search(r"[A-Za-z]{2,}", text) else 0
    if not any(c.isalnum() for c in text):
        rank -= .30
    return rank


@dataclass(frozen=True)
class Reading:
    group: str
    text: str
    confidence: float


def choose(readings, *, ambiguity=.10):
    """Return reading and whether conflicting plausible scripts need review."""
    if not readings:
        raise ValueError("Нет вариантов распознавания")
    ranked = sorted(readings, key=lambda x: group_score(x.text, x.confidence, x.group), reverse=True)
    best = ranked[0]
    conflicts = [x for x in ranked[1:] if x.text != best.text and
                 group_score(best.text, best.confidence, best.group) -
                 group_score(x.text, x.confidence, x.group) <= ambiguity]
    return best, bool(conflicts)


def line_crops(image, fragments):
    """Measured original-image line crops; no new OCR geometry is invented."""
    crops = []
    for frag in fragments:
        b = frag.bbox
        x0, y0 = max(0, int(b.x) - 2), max(0, int(b.y) - 2)
        x1, y1 = min(image.width, int(b.right + 3)), min(image.height, int(b.bottom + 3))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Некорректная область распознанной строки")
        crops.append(np.asarray(image.crop((x0, y0, x1, y1)).convert("RGB")))
    return crops


def representative_indices(fragments, limit=24):
    """Sample all page bands instead of only the first top-left OCR lines."""
    order = sorted(range(len(fragments)), key=lambda i: (
        fragments[i].bbox.y, fragments[i].bbox.x))
    if len(order) <= limit:
        return order
    return sorted({order[round(i * (len(order) - 1) / (limit - 1))] for i in range(limit)})


def run_recognizer(model, crops):
    """PaddleOCR TextRecognition accepts lists of numpy text-line images."""
    results = []
    for start in range(0, len(crops), 16):
        for output in model.predict(input=crops[start:start+16], batch_size=8):
            value = output.json
            if isinstance(value, str):
                import json
                value = json.loads(value)
            value = value.get("res", value)
            results.append((str(value.get("rec_text", "")), float(value.get("rec_score", 0))))
    if len(results) != len(crops):
        raise ValueError("Модель распознавания вернула не все строки")
    return results


def recognize_lines(image, fragments, primary_group, requested_groups, *,
                    auto, model_factory, progress=lambda _: None):
    """Return corrected fragments and audit data; never change measured geometry.

    Auto probes a bounded sample and enables extra script models only on positive
    native-script evidence. Manual always runs explicitly selected model groups.
    A failed secondary model retains original OCR and records an honest warning.
    """
    if not fragments or not requested_groups:
        return fragments, {"active": [primary_group], "uncertain_ids": [], "warnings": []}
    crops = line_crops(image, fragments)
    selected = list(requested_groups)
    probes = representative_indices(fragments) if auto else []
    options = [[Reading(primary_group, f.text, f.confidence)] for f in fragments]
    active, warnings = [primary_group], []

    for group in selected:
        progress("PaddleOCR: дополнительная модель " + group)
        try:
            model = model_factory(MODELS[group][1])
            if auto:
                predictions = run_recognizer(model, [crops[i] for i in probes])
                enabled = any(native_evidence(text, group, score) for text, score in predictions)
                if not enabled:
                    continue
                # Reuse the probe predictions; only remaining lines need inference.
                seen = dict(zip(probes, predictions))
                missing = [i for i in range(len(crops)) if i not in seen]
                seen.update(zip(missing, run_recognizer(model, [crops[i] for i in missing])))
                predictions = [seen[i] for i in range(len(crops))]
            else:
                predictions = run_recognizer(model, crops)
            active.append(group)
            for i, (text, confidence) in enumerate(predictions):
                if text.strip() and 0 <= confidence <= 1:
                    options[i].append(Reading(group, text, confidence))
        except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
            warnings.append("Модель " + group + " недоступна (" + type(exc).__name__ + ")")
            # Never make this model appear to have succeeded.
    chosen = []
    uncertain = []
    for fragment, readings in zip(fragments, options):
        best, conflict = choose(readings)
        updated = fragment.model_copy(deep=True)
        updated.text, updated.confidence = best.text, best.confidence
        chosen.append(updated)
        if conflict:
            uncertain.append(updated.id)
    return chosen, {"active": active, "uncertain_ids": uncertain, "warnings": warnings}
