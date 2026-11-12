import json

import pytest

from smart_overlay.vision import Analysis, Reading, VisionClient, VisionError, apply_analysis, analysis_packet, response_schema
from smart_overlay.vision_diagnostics import project_context, replay, save_failure


def test_failure_record_has_replay_context_without_key_or_image(project, tmp_path, monkeypatch):
    from smart_overlay.settings import Settings
    from smart_overlay.structure import validate_project
    secret = "private-api-key"
    raw = {"id": "gen-test", "headers": {"Authorization": "Bearer " + secret},
           "choices": [{"message": {"content": json.dumps({"groups": [{"ocr_ids": ["invalid"], "confidence": 1}],
                                                          "tables": [], "ignored": {}, "regions": []})}}]}
    calls = []
    def post(*args, **kwargs):
        calls.append(True)
        return raw
    monkeypatch.setattr("smart_overlay.vision.post_json", post)
    client = VisionClient(Settings(model="test/model", retries=0), secret, tmp_path/"cache", True)
    with pytest.raises(VisionError, match="Диагностика"):
        client.request("test", [], Analysis, lambda result: apply_analysis(project, result),
                       context=lambda: project_context(project))
    assert len(calls) == 1
    record = json.loads(client.last_diagnostic.read_text("utf-8"))
    text = client.last_diagnostic.read_text("utf-8")
    assert secret not in text
    assert project.source.original_b64[:100] not in text
    assert project.source.raster_b64[:100] not in text
    assert record["context"]["source"]["original_b64"] == ""
    with pytest.raises(ValueError, match="неизвестный"):
        replay(client.last_diagnostic)
    # Replay after a local response correction does not make another paid call.
    record["response"]["choices"][0]["message"]["content"] = json.dumps({"groups": [], "tables": [], "ignored": {}, "regions": []})
    client.last_diagnostic.write_text(json.dumps(record), "utf-8")
    validate_project(replay(client.last_diagnostic))
    assert len(calls) == 1


def test_diagnostic_size_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr("smart_overlay.vision_diagnostics.MAX_FILE", 600)
    path = save_failure(tmp_path/"cache", request_hash="hash", attempt=1, model="test",
                        response_model="Reading", stage="JSON", error="invalid", api_key="secret",
                        response={"content": "x"*2000}, context={"text": "y"*2000})
    record = json.loads(path.read_text("utf-8"))
    assert record["context"] is None
    assert record["response"] is None
    assert path.stat().st_size <= 600
    with pytest.raises(ValueError, match="не сохранён"):
        replay(path)


def test_timeout_diagnostic_does_not_claim_to_replay_complete_response(tmp_path):
    path = save_failure(tmp_path/"cache", request_hash="hash", attempt=1, model="test",
                        response_model="Reading", stage="timeout", error="timeout", api_key="secret",
                        response={"_transport": {"incomplete": True, "http_status": 200, "received_bytes": 400}})
    with pytest.raises(ValueError, match="не завершён"):
        replay(path)


def test_compact_wire_packet_and_schema_omit_presentation_formatting(project):
    packet = analysis_packet(project)
    text = json.dumps(packet)
    assert '"style"' not in text and '"polygon"' not in text
    assert packet["tables"][0]["cells"][0]["ocr_ids"]
    schema = response_schema(Analysis)
    assert "Style" not in schema["$defs"]
    assert "style" not in schema["$defs"]["Cell"]["properties"]
    assert "id" not in schema["$defs"]["Cell"]["properties"]


def test_compact_reply_reuses_text_and_native_table_cells(project):
    reply = Analysis.model_validate({"groups": [{"ocr_ids": ["ocr_4"], "confidence": 1}],
            "tables": [{"id": project.objects[-1].id, "rows": 4, "columns": 3, "confidence": 1, "issues": []}],
            "ignored": {}, "regions": []})
    result = apply_analysis(project, reply)
    assert next(o for o in result.objects if o.ocr_ids == ["ocr_4"]).text == "Drain valve"
    table = next(o for o in result.objects if o.type == "table")
    assert table.cells == project.objects[-1].cells
