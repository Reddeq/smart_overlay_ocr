import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from smart_overlay.settings import Settings
from smart_overlay.vision import Reading, VisionClient, VisionError, VisionCancelled, refine


@pytest.fixture
def mock_http(monkeypatch):
    original = httpx.AsyncClient
    def install(handler):
        transport = httpx.MockTransport(handler)
        monkeypatch.setattr("smart_overlay.vision_transport.httpx.AsyncClient",
                            lambda **kwargs: original(transport=transport, **kwargs))
    return install


def response(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}],
                                    "usage": {"prompt_tokens": 10, "cost": 0.01, "secret": "no"}})


def test_no_network_without_consent(tmp_path):
    client = VisionClient(Settings(model="test/model"), "secret", tmp_path, False)
    with pytest.raises(VisionError, match="согласие"):
        client.request("test", [], Reading)
    assert list(tmp_path.iterdir()) == []


def test_retry_cache_and_usage(monkeypatch, mock_http, tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return response({"text": "Valve", "confidence": 0.99, "reason": "visible"}) if len(calls) > 1 else response({"bad": True})
    mock_http(handler)
    monkeypatch.setattr("smart_overlay.vision.time.sleep", lambda _: None)
    client = VisionClient(Settings(model="test/model"), "secret", tmp_path, True)
    assert client.request("test", [], Reading).text == "Valve"
    assert len(calls) == 2
    assert client.request("test", [], Reading).text == "Valve"
    assert len(calls) == 2
    assert client.usage[-1]["cached"]
    assert all("secret" not in p.read_text() for p in tmp_path.iterdir())
    assert client.usage[0] == {"prompt_tokens": 10, "cost": 0.01}


def test_auth_error_is_not_retried(mock_http, tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(401)
    mock_http(handler)
    client = VisionClient(Settings(model="test/model"), "secret", tmp_path, True)
    with pytest.raises(VisionError, match="401"):
        client.request("test", [], Reading)
    assert len(calls) == 1


def test_low_confidence_table_cell_gets_enlarged_review(project):
    table = project.objects[-1]
    cell = table.cells[0]
    fragment = next(f for f in project.fragments if f.id == cell.ocr_ids[0])
    fragment.confidence = 0.3
    calls = []
    class Client:
        settings = Settings()
        def request(self, prompt, images, model):
            calls.append(images)
            return Reading(text="Unclear", confidence=0.5, reason="blurred")
    original = cell.text
    refine(project, Client())
    assert len(calls) == 1 and len(calls[0]) == 2
    assert cell.text == original
    assert table.needs_review
    assert not table.corrections[-1].applied


def test_retry_includes_failed_answer_and_precise_geometry_feedback(monkeypatch, mock_http, tmp_path, caplog):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return response({"text": "Valve", "confidence": 0.99, "reason": "visible"})
    checks = []
    def check(result):
        checks.append(result)
        if len(checks) == 1:
            raise ValueError("Пропавшие OCR-фрагменты: ocr_123")
    mock_http(handler)
    monkeypatch.setattr("smart_overlay.vision.time.sleep", lambda _: None)
    client = VisionClient(Settings(model="test/model"), "secret", tmp_path, True)
    assert client.request("test", [], Reading, check).text == "Valve"
    assert len(calls) == 2
    retry = calls[1]["messages"]
    assert retry[-2]["role"] == "assistant"
    assert json.loads(retry[-2]["content"])["text"] == "Valve"
    assert "ocr_123" in retry[-1]["content"]
    assert "Геометрия/связи OCR" in caplog.text
    assert "ocr_123" in caplog.text


def test_schema_error_reports_field_without_logging_input_or_key(mock_http, tmp_path, caplog):
    secret = "private-api-key"
    mock_http(lambda request: response({"text": "Valve", "confidence": secret, "reason": "visible"}))
    client = VisionClient(Settings(model="test/model", retries=0), secret, tmp_path, True)
    with pytest.raises(VisionError, match="confidence") as exc:
        client.request("test", [], Reading)
    assert "Схема JSON" in str(exc.value)
    assert secret not in caplog.text
    assert secret not in str(exc.value)


def test_complete_json_fence_is_accepted_without_retry(mock_http, tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        content = '```json\n{"text":"Valve","confidence":0.99,"reason":"visible"}\n```'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
    mock_http(handler)
    client = VisionClient(Settings(model="test/model"), "secret", tmp_path, True)
    assert client.request("test", [], Reading).text == "Valve"
    assert len(calls) == 1


@pytest.mark.parametrize("raw, expected", [
    ({"choices": [{"finish_reason": "length"}]}, "finish_reason=length"),
    ({"choices": [{"message": {"content": None}}]}, "нетекстовый content"),
    ({"choices": []}, "choices"),
    ([], "JSON-объектом"),
    ({"error": {"code": 503}}, "503"),
])
def test_invalid_api_envelope_is_explained(mock_http, tmp_path, raw, expected):
    mock_http(lambda request: httpx.Response(200, json=raw))
    client = VisionClient(Settings(model="test/model", retries=0), "secret", tmp_path, True)
    with pytest.raises(VisionError, match=expected):
        client.request("test", [], Reading)


def test_cache_write_failure_does_not_repeat_paid_request(mock_http, tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return response({"text": "Valve", "confidence": 0.99, "reason": "visible"})
    mock_http(handler)
    cache_file = tmp_path / "not_a_directory"
    cache_file.write_text("occupied")
    client = VisionClient(Settings(model="test/model"), "secret", cache_file, True)
    assert client.request("test", [], Reading).text == "Valve"
    assert len(calls) == 1


def test_redirect_does_not_forward_key_or_image(mock_http, tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(307, headers={"Location": "https://another-host.invalid/"})
    mock_http(handler)
    client = VisionClient(Settings(model="test/model"), "secret", tmp_path, True)
    with pytest.raises(VisionError, match="307"):
        client.request("test", [], Reading)
    assert len(calls) == 1


def test_connect_timeout_before_post_has_bounded_safe_retries(mock_http, tmp_path, monkeypatch):
    calls = []
    def handler(request):
        calls.append(request)
        raise httpx.ConnectTimeout("private-key-should-not-appear", request=request)
    mock_http(handler)
    monkeypatch.setattr("smart_overlay.vision.time.sleep", lambda _: None)
    client = VisionClient(Settings(model="test/model"), "secret", tmp_path, True)
    with pytest.raises(VisionError, match="Тайм-аут подключения") as exc:
        client.request("test", [], Reading)
    assert "private-key" not in str(exc.value)
    assert len(calls) == 3


def test_timeout_after_post_started_is_not_retried(mock_http, tmp_path):
    calls = []
    async def handler(request):
        calls.append(request)
        await request.extensions["trace"]("http11.send_request_headers.started",
                                           {"request": SimpleNamespace(method=b"POST")})
        raise httpx.ReadTimeout("private", request=request)
    mock_http(handler)
    client = VisionClient(Settings(model="test/model"), "secret", tmp_path/"cache", True)
    with pytest.raises(VisionError, match="Тайм-аут"):
        client.request("test", [], Reading)
    assert len(calls) == 1
    record = json.loads(client.last_diagnostic.read_text("utf-8"))
    assert record["response"]["_transport"]["request_started"] is True


def test_empty_analysis_is_not_cached_and_old_empty_cache_is_bypassed(mock_http, tmp_path):
    from smart_overlay.vision import Analysis
    calls = []
    empty = {"groups": [], "tables": [], "ignored": {}, "regions": []}
    useful = dict(empty, groups=[{"ocr_ids": ["ocr_1"], "confidence": 1}])
    def handler(request):
        calls.append(request)
        return response(empty if len(calls) == 1 else useful)
    mock_http(handler)
    client = VisionClient(Settings(model="test/model"), "secret", tmp_path/"cache", True)
    cache_if = lambda r: bool(r.groups or r.tables or r.ignored or r.regions)
    assert not client.request("test", [], Analysis, cache_if=cache_if).groups
    assert not list((tmp_path/"cache").glob("*.json"))
    diagnostic = json.loads(client.last_diagnostic.read_text("utf-8"))
    (tmp_path/"cache").mkdir()
    (tmp_path/"cache"/(diagnostic["request_hash"]+".json")).write_text(json.dumps(empty), "utf-8")
    assert client.request("test", [], Analysis, cache_if=cache_if).groups
    assert client.request("test", [], Analysis, cache_if=cache_if).groups
    assert len(calls) == 2


def test_network_phases_are_reported_without_private_trace_data(mock_http, tmp_path, caplog):
    messages = []
    async def handler(request):
        for event in ("connection.connect_tcp.started", "connection.start_tls.started",
                      "http11.send_request_body.started", "http11.send_request_body.complete"):
            info = {"return_value": None} if event.endswith("complete") else {"private": "private-key", "request": SimpleNamespace(method=b"POST")}
            await request.extensions["trace"](event, info)
        return response({"text": "Valve", "confidence": 0.99, "reason": "visible"})
    mock_http(handler)
    client = VisionClient(Settings(model="test/model"), "secret", tmp_path, True, progress=messages.append)
    client.request("test", [], Reading)
    for phase in ("DNS/TCP", "TLS", "отправка запроса", "запрос отправлен", "HTTP 200", "проверка JSON"):
        assert any(phase in message for message in messages)
    assert "private-key" not in caplog.text
    assert not any("private-key" in message for message in messages)


def test_cancellation_interrupts_wait_and_does_not_retry(mock_http, tmp_path):
    calls = []
    state = {"cancelled": False, "closed": False}
    async def handler(request):
        calls.append(request)
        state["cancelled"] = True
        try:
            await asyncio.sleep(60)
        finally:
            state["closed"] = True
    mock_http(handler)
    client = VisionClient(Settings(model="test/model"), "secret", tmp_path, True,
                          cancelled=lambda: state["cancelled"])
    with pytest.raises(VisionCancelled, match="отменён"):
        client.request("test", [], Reading)
    assert len(calls) == 1 and state["closed"]


def test_total_timeout_stops_keepalive_response(mock_http):
    from smart_overlay.vision_transport import post_json, RequestTimedOut
    closed = []
    class KeepAlive(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                await asyncio.sleep(0.01)
                yield b" "
        async def aclose(self):
            closed.append(True)
    mock_http(lambda request: httpx.Response(200, stream=KeepAlive()))
    with pytest.raises(RequestTimedOut, match="Тайм-аут запроса"):
        post_json("https://test.invalid", "secret", b"{}", connect_timeout=1, request_timeout=0.1,
                  progress=lambda _: None, cancelled=lambda: False, label="test")
    assert closed


def test_partial_model_response_does_not_trigger_three_attempts(project, mock_http, tmp_path):
    from smart_overlay.pipeline import smart_ocr
    from smart_overlay.structure import validate_project
    calls = []
    partial = {"groups": [{"ocr_ids": ["ocr_4"], "text": "Drain valve", "confidence": 1, "reason": "visible"}],
               "tables": [], "ignored": {}, "regions": []}
    def handler(request):
        calls.append(request)
        return response(partial)
    mock_http(handler)
    result = smart_ocr(project, Settings(model="test/model"), "secret", tmp_path, True)
    validate_project(result)
    assert len(calls) == 1
    assert any("VL пропустила OCR-фрагменты: 4" in message for message in result.messages)
    assert len(result.objects) == len(project.objects)


def test_sse_text_usage_and_comments_are_processed(mock_http, tmp_path):
    messages = []
    text = json.dumps({"text": "Клапан", "confidence": 1, "reason": "visible"}, ensure_ascii=False)
    events = [": OPENROUTER PROCESSING\n\n"]
    for part in [text[:17], text[17:]]:
        events.append("data: " + json.dumps({"id": "gen-stream", "choices": [{"index": 0, "delta": {"content": part}}]}, ensure_ascii=False) + "\n\n")
    events.extend(["data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 20}}) + "\n\n", "data: [DONE]\n\n"])
    wire = "".join(events).encode("utf-8")
    class Pieces(httpx.AsyncByteStream):
        async def __aiter__(self):
            for i in range(0, len(wire), 3):
                yield wire[i:i+3]  # includes UTF-8 and SSE boundary splits
    mock_http(lambda request: httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=Pieces()))
    client = VisionClient(Settings(model="test/model"), "secret", tmp_path/"cache", True, progress=messages.append)
    assert client.request("test", [], Reading).text == "Клапан"
    assert client.usage[-1]["completion_tokens"] == 20
    assert any("генерация:" in m for m in messages)
    assert not any("Клапан" in m for m in messages)


def test_sse_error_body_is_reported(mock_http, tmp_path):
    event = 'data: {"error":{"code":503},"choices":[]}\n\n'
    mock_http(lambda request: httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=event))
    client = VisionClient(Settings(model="test/model", retries=0), "secret", tmp_path/"cache", True)
    with pytest.raises(VisionError, match="503"):
        client.request("test", [], Reading)
    assert client.last_diagnostic.exists()


def test_sse_truncation_does_not_repeat_same_token_budget(mock_http, tmp_path):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"},
            content='data: {"choices":[{"delta":{"content":"{"},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n')
    mock_http(handler)
    client = VisionClient(Settings(model="test/model", retries=2), "secret", tmp_path/"cache", True)
    with pytest.raises(VisionError, match="finish_reason=length"):
        client.request("test", [], Reading)
    assert len(calls) == 1
    assert calls[0]["stream"] is True and calls[0]["max_tokens"] == 8192


def test_unfinished_sse_is_not_accepted_as_complete(mock_http, tmp_path):
    wire = 'data: {"choices":[{"delta":{"content":"{}"}}]}\n\n'
    mock_http(lambda request: httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=wire))
    client = VisionClient(Settings(model="test/model", retries=0), "secret", tmp_path/"cache", True)
    with pytest.raises(VisionError, match="оборвался"):
        client.request("test", [], Reading)
    record = json.loads(client.last_diagnostic.read_text("utf-8"))
    assert record["response"]["_transport"]["incomplete"]
