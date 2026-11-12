import argparse
import base64
import copy
import io
import json

import httpx
from PIL import Image
import pytest

from smart_overlay.diagnose_vl import (add_parser, audit_images, build_cases, case_plan, evaluate,
                                      main, partition, run_case, synthetic, wire)
from smart_overlay.input import save_project
from smart_overlay.settings import Settings
from smart_overlay.vision import analysis_images, analysis_prompt, request_payload, Analysis


def args(*options):
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["diagnose-vl", *options])


def cases(*options):
    return build_cases(args(*options), Settings(model="test/model", max_output_tokens=1024))


def response(content, **kwargs):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.001}, **kwargs}


def test_reproducible_examples_and_exact_wire_for_current_prompt():
    first, truth = synthetic(5)
    second, _ = synthetic(5)
    assert first == second
    assert truth == [["id_1", "id_2", "id_3"], ["id_4"], ["id_5"]]
    settings = Settings(model="test/model", max_output_tokens=1024, stream_responses=False)
    case = cases("--tests", "F")[0]
    assert case.payload == request_payload(settings, analysis_prompt(first), analysis_images(first, settings), Analysis)


def test_d_changes_only_response_format_and_schema_routing_requirement():
    experiments = cases("--tests", "D", "--formats", "none,json_object,json_schema")
    stripped = []
    for c in experiments:
        payload = copy.deepcopy(c.payload)
        payload.pop("response_format", None)
        payload.pop("provider", None)
        stripped.append(payload)
    assert stripped[0] == stripped[1] == stripped[2]
    assert experiments[2].payload["provider"]["require_parameters"]
    assert all(c.payload["stream"] is False for c in experiments)


def test_b_and_c_differ_only_by_one_image():
    b, c = cases("--tests", "B,C")
    content = c.payload["messages"][1]["content"]
    assert len(content) == 2 and content[1]["type"] == "image_url"
    without_image = copy.deepcopy(c.payload)
    without_image["messages"][1]["content"] = content[:1]
    assert without_image == b.payload


def test_visual_image_matches_mime_and_payload_and_does_not_leak_answer_in_prompt():
    case = cases("--tests", "V")[0]
    assert "TEST 123" not in wire(case).decode()
    audit = audit_images(case.payload)
    assert audit[0]["valid"] and (audit[0]["width"], audit[0]["height"]) == (360, 100)
    url = case.payload["messages"][1]["content"][1]["image_url"]["url"]
    raw = base64.b64decode(url.split(",", 1)[1], validate=True)
    assert Image.open(io.BytesIO(raw)).format == "PNG"
    assert case_plan(case)["images"] == 1
    assert evaluate(case, response('{"text":"TEST 123"}'))["content_valid"] is True
    assert evaluate(case, response('{"text":"TEST 12"}'))["content_valid"] is False
    case.payload["messages"][1]["content"][1]["image_url"]["url"] = url.replace("image/png", "image/jpeg")
    with pytest.raises(ValueError, match="MIME"):
        audit_images(case.payload)


@pytest.mark.parametrize("content,category,json_valid,schema_valid", [
    ("", "empty_message_content", False, False),
    ("not JSON", "invalid_json", False, False),
    ('{"merges":[]}', "valid_json_empty", True, True),
    ('{"merges":[["id_1","id_2","id_3"]]}', "valid_content", True, True),
    ('{"merges":[["id_1","id_4"]]}', "incorrect_grouping", True, True),
    ('{"merges":[["unknown","id_4"]]}', "incorrect_grouping", True, True),
    ('{"merges":[123]}', "schema_error", True, False),
])
def test_response_classification(content, category, json_valid, schema_valid):
    result = evaluate(cases("--tests", "B")[0], response(content))
    assert result["classification"] == category
    assert result["json_valid"] is json_valid and result["schema_valid"] is schema_valid
    assert result["full_text"] == content


def test_empty_analysis_passes_schema_but_never_counts_as_good_grouping():
    result = evaluate(cases("--tests", "F")[0], response('{"groups":[],"tables":[],"ignored":{},"regions":[]}'))
    assert result["classification"] == "valid_json_empty"
    assert result["schema_valid"] is True and result["content_valid"] is False


def test_full_analysis_measures_coverage_before_production_fallback():
    case = cases("--tests", "F")[0]
    payload = {"groups": [{"ocr_ids": ["id_1", "id_2", "id_3"], "confidence": 1}],
               "tables": [], "ignored": {}, "regions": []}
    result = evaluate(case, response(json.dumps(payload)))
    assert result["classification"] == "incorrect_grouping"
    assert result["content_check"] == "incomplete_source_coverage"
    payload["groups"].extend({"ocr_ids": [i], "confidence": 1} for i in ("id_4", "id_5"))
    assert evaluate(case, response(json.dumps(payload)))["classification"] == "valid_content"


def test_prototype_preserves_unmentioned_fragments_and_rejects_overlapping_merges():
    assert partition(["1", "2", "3"], [["1", "2"]]) == [["1", "2"], ["3"]]
    with pytest.raises(ValueError):
        partition(["1", "2", "3"], [["1", "2"], ["2", "3"]])
    case = cases("--tests", "X")[0]
    before = case.project.model_dump_json()
    result = evaluate(case, response('{"merges":[["id_1","id_2","id_3"]],"splits":[],"uncertain":[]}'))
    assert result["content_valid"] is True
    assert case.project.model_dump_json() == before


def test_scaling_and_pinned_provider():
    experiments = cases("--tests", "E", "--sizes", "5,20,80", "--provider", "provider-slug")
    assert len(experiments) == 6
    assert [len(c.project.fragments) for c in experiments] == [5, 5, 20, 20, 80, 80]
    assert all(c.payload["provider"] == {"order": ["provider-slug"], "allow_fallbacks": False} for c in experiments)


def test_plan_is_offline_no_project_mutation_and_no_images_in_reports(tmp_path, monkeypatch, capsys):
    project, _ = synthetic()
    path = tmp_path/"input.socr.json"
    save_project(project, path)
    before = path.read_bytes()
    monkeypatch.setattr("smart_overlay.diagnose_vl.post_json", lambda *a, **kw: pytest.fail("unexpected network"))
    output = tmp_path/"plan.json"
    assert main(args("--tests", "F-project,X-project", "--project", str(path), "--output", str(output))) == 0
    assert path.read_bytes() == before
    text = output.read_text("utf-8")
    assert "base64" not in text and "MAIN" not in text
    assert "MAIN" not in capsys.readouterr().out
    assert json.loads(text)["results"] == []
    assert output.with_suffix(".md").exists()


@pytest.mark.parametrize("options,match", [
    (["--execute"], "Confirm"),
    (["--execute", "--confirm-requests", "8", "--max-requests", "1"], "budget"),
    (["--execute", "--confirm-requests", "8", "--max-total-output-tokens", "512"], "budget"),
    (["--execute", "--confirm-requests", "8", "--max-request-bytes", "10"], "bytes"),
    (["--execute", "--confirm-requests", "8"], "API_KEY"),
])
def test_no_api_before_confirmation_budget_and_key_checks(options, match, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("smart_overlay.diagnose_vl.post_json", lambda *a, **kw: pytest.fail("unexpected network"))
    with pytest.raises(ValueError, match=match):
        main(args(*options, "--output", str(tmp_path/"report.json")))
    if match == "API_KEY":
        record = json.loads((tmp_path/"report.json").read_text("utf-8"))
        assert record["executed"] is False and record["results"] == []
        assert "API_KEY absent" in record["stopped"]


@pytest.mark.parametrize("body,category", [(b"", "empty_http_response"),
    (b"bad HTTP JSON", "invalid_http_json"),
    (json.dumps(response("")).encode(), "empty_message_content")])
def test_real_transport_distinguishes_http_and_message_emptiness(body, category, monkeypatch):
    original = httpx.AsyncClient
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr("smart_overlay.vision_transport.httpx.AsyncClient",
                        lambda **kw: original(transport=transport, **kw))
    result = run_case(cases("--tests", "A")[0], Settings(model="test"), "secret")
    assert result["classification"] == category
    assert result["http_status"] == 200
    assert result["received_bytes"] == len(body) and len(calls) == 1


def test_no_retries_on_failure_reports_redact_key_images_and_preserve_full_text(tmp_path, monkeypatch, capsys):
    secret = "my-private-secret"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    calls = []
    full_text = '{"merges":[],"note":"' + secret + ' data:image/png;base64,YWJj"}'
    def post(*a, **kw):
        calls.append(True)
        kw["observe"]({"http_status": 200, "received_bytes": 500})
        return response(full_text, provider="test-provider")
    monkeypatch.setattr("smart_overlay.diagnose_vl.post_json", post)
    output = tmp_path/"report.json"
    assert main(args("--tests", "B,C", "--execute", "--confirm-requests", "2", "--stop-on-failure",
                     "--output", str(output))) == 2
    text = output.read_text("utf-8")
    assert secret not in text and "YWJj" not in text and "base64" not in text
    assert secret not in capsys.readouterr().out
    result = json.loads(text)["results"][0]
    assert result["provider"] == "test-provider"
    assert "[API KEY]" in result["full_text"] and len(calls) == 1


def test_stop_scaling_on_first_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    calls = []
    def post(*a, **kw):
        calls.append(True)
        return response('{"groups":[]}')
    monkeypatch.setattr("smart_overlay.diagnose_vl.post_json", post)
    output = tmp_path/"report.json"
    assert main(args("--tests", "E", "--sizes", "5,10", "--execute", "--confirm-requests", "4",
                     "--output", str(output))) == 2
    assert len(calls) == 1
    assert json.loads(output.read_text("utf-8"))["stopped"].startswith("E-full-5")


def test_input_file_cannot_be_overwritten_by_report(tmp_path):
    path = tmp_path/"input.socr.json"
    project, _ = synthetic()
    save_project(project, path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="differ"):
        main(args("--project", str(path), "--output", str(path)))
    assert path.read_bytes() == before
