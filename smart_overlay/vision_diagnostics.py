"""Local, bounded failure records for replaying VL validation without an API."""
import json
import logging
import re
from pathlib import Path

LOG = logging.getLogger(__name__)
MAX_CONTENT = 2*1024*1024
MAX_FILE = 4*1024*1024
PRIVATE_FIELDS = {"authorization", "api_key", "apikey", "headers", "access_token",
                  "original_b64", "raster_b64", "image_url"}


def sanitized(value, api_key):
    if isinstance(value, dict):
        return {sanitized(str(k), api_key): ("[REDACTED]" if str(k).lower() in PRIVATE_FIELDS else sanitized(v, api_key))
                for k, v in value.items()}
    if isinstance(value, list):
        return [sanitized(v, api_key) for v in value]
    if isinstance(value, str):
        if api_key:
            value = value.replace(api_key, "[API KEY]")
        value = re.sub(r"Bearer\s+[^\s\"']+", "Bearer [API KEY]", value, flags=re.IGNORECASE)
        value = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+", "[IMAGE REMOVED]", value)
        return value[:MAX_CONTENT]
    return value


def project_context(project):
    source = project.source.model_dump(exclude={"original_b64", "raster_b64"})
    # Empty raster fields allow model validation; replay never renders the image.
    source.update(original_b64="", raster_b64="")
    return {"version": project.version, "source": source,
            "fragments": [f.model_dump() for f in project.fragments],
            "objects": [o.model_dump() for o in project.objects], "ignored": project.ignored}


def save_failure(cache_dir, *, request_hash, attempt, model, response_model, stage, error, response, api_key, context=None):
    # Fixed attempt slots bound retained records to the configured retry limit.
    path = Path(cache_dir).parent / "diagnostics" / f"vl-failure-{attempt}.json"
    record = {"version": 1, "request_hash": request_hash, "attempt": attempt, "model": model,
              "response_model": response_model, "stage": stage, "error": error,
              "response": response, "context": context() if callable(context) else context}
    record = sanitized(record, api_key)
    # Remove these fields rather than including any image placeholders in context.
    if isinstance(record.get("context"), dict) and isinstance(record["context"].get("source"), dict):
        record["context"]["source"].update(original_b64="", raster_b64="")
    encoded = json.dumps(record, ensure_ascii=False, indent=2)
    if len(encoded.encode("utf-8")) > MAX_FILE:
        record["context"] = None
        record["context_omitted"] = "Context exceeds diagnostic size limit"
        encoded = json.dumps(record, ensure_ascii=False, indent=2)
    if len(encoded.encode("utf-8")) > MAX_FILE:
        record["response"] = None
        record["response_omitted"] = "Response exceeds diagnostic size limit"
        encoded = json.dumps(record, ensure_ascii=False, indent=2)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
    except OSError:
        LOG.warning("Could not save VL failure diagnostics for request %s", request_hash[:12])
        return None
    return path


def replay(path):
    # Imports are local to avoid a cycle with the request client.
    from .models import Project
    from .vision import Analysis, Reading, apply_analysis, response_content, response_json
    record = json.loads(Path(path).read_text("utf-8"))
    if record.get("version") != 1:
        raise ValueError("Неподдерживаемая версия диагностики VL")
    model = {"Analysis": Analysis, "Reading": Reading}.get(record.get("response_model"))
    if model is None:
        raise ValueError("Неизвестный тип ответа в диагностике VL")
    if record.get("response") is None:
        raise ValueError("Ответ не сохранён полностью: " + record.get("response_omitted", "no response"))
    if isinstance(record["response"], dict) and record["response"].get("_transport", {}).get("incomplete"):
        raise ValueError("HTTP-ответ не завершён. Сведения о полученных байтах и генерации сохранены в диагностике.")
    result = model.model_validate_json(response_json(response_content(record["response"])))
    if model is Analysis:
        if not record.get("context"):
            raise ValueError("Для проверки геометрии отсутствует контекст проекта")
        return apply_analysis(Project.model_validate(record["context"]), result)
    return result
