import copy
import json

import pytest

from smart_overlay.diagnose_real import (Manifest, Region, assess, check_proxy, normalize,
                                        prepare_case, preview, validate_manifest, main)
from smart_overlay.diagnose_vl import Merges, GROUPING_EXAMPLES_PROMPT, synthetic
from smart_overlay.geometry import union
from smart_overlay.models import Box
from smart_overlay.settings import Settings


@pytest.fixture
def sample():
    p, expected = synthetic()
    r = Region(name="R1", bbox=Box(x=0, y=0, w=600, h=180),
               ocr_ids=[f.id for f in p.fragments], expected_merges=[expected[0]],
               code_only_ids=["id_5"], uncertain_ids=[], rationale="Visual reference")
    return p, r


def test_singletons_removed_without_losing_fragment_or_modifying_raw(sample):
    p, r = sample
    raw = '{"merges":[["id_1","id_2","id_3"],["id_4"]]}'
    reply = Merges.model_validate_json(raw)
    before = reply.model_dump_json()
    normalized = normalize(p, r, reply, [])
    assert reply.model_dump_json() == before
    assert normalized["contract_violations"] == 1
    assert normalized["applied_merges"] == [["id_1", "id_2", "id_3"]]
    assert ["id_4"] in normalized["final_partition"] and ["id_5"] in normalized["final_partition"]
    assert assess(r, normalized)["known_partition_correct"] is True


@pytest.mark.parametrize("groups", [[["unknown"]], [["id_1"], ["id_1", "id_2"]],
                                    [["id_1", "id_1"]], [[]]])
def test_invalid_ids_are_rejected_before_singleton_normalization(sample, groups):
    p, r = sample
    with pytest.raises(ValueError, match="unknown_duplicate"):
        normalize(p, r, Merges(merges=groups), [])


def test_geometric_and_ambiguous_merges_require_review(sample):
    p, r = sample
    reply = Merges(merges=[["id_1", "id_4"]])
    normalized = normalize(p, r, reply, [])
    assert not normalized["applied_merges"] and normalized["review_groups"]
    r.uncertain_ids = ["id_2"]
    normalized = normalize(p, r, Merges(merges=[["id_1", "id_2", "id_3"]]), [])
    assert normalized["review_groups"][0]["reason"] == "ambiguous_membership"


def test_table_crossing_is_never_repaired(sample):
    p, r = sample
    mask = Box(x=20, y=42, w=150, h=7)
    normalized = normalize(p, r, Merges(merges=[["id_1", "id_2", "id_3"]]), [mask])
    assert normalized["review_groups"][0]["reason"] == "merge_crosses_table"
    assert not normalized["applied_merges"]


def test_scores_false_and_missed_merges_and_full_partition(sample):
    p, r = sample
    # Syntactically valid, geometrically plausible partial label; it is still semantically wrong.
    normalized = normalize(p, r, Merges(merges=[["id_1", "id_2"]]), [])
    m = assess(r, normalized)
    assert m["incorrect_merges"] == 1 and m["missed_merges"] == 1
    assert m["known_partition_correct"] is False
    r.expected_merges = []
    assert assess(r, normalize(p, r, Merges(merges=[]), []))["known_partition_correct"] is True


def test_packet_keeps_global_geometry_ids_and_excludes_reference(sample, tmp_path):
    p, r = sample
    case, packet, crop = prepare_case(p, r, Settings(model="qwen/qwen3-vl-8b-instruct", max_output_tokens=1024))
    assert packet["fragments"][0]["bbox"] == p.fragments[0].bbox.model_dump()
    assert packet["fragments"][0]["id"] == p.fragments[0].id
    assert "expected_merges" not in json.dumps(packet) and "uncertain_ids" not in json.dumps(packet)
    text = case.payload["messages"][1]["content"][0]["text"]
    assert text == GROUPING_EXAMPLES_PROMPT.replace("{{OCR_DATA}}", json.dumps(packet, ensure_ascii=False, separators=(",", ":")))
    assert case.payload["provider"] == {"order": ["parasail"], "allow_fallbacks": False}
    assert case.payload["stream"] is False and case.payload["temperature"] == 0
    preview(p, r, tmp_path/"reference.png")
    assert (tmp_path/"reference.png").is_file()


def test_manifest_rejects_clipped_or_table_members(sample):
    p, r = sample
    regions = [r.model_copy(update={"name": name}) for name in ("R1", "R2", "R3")]
    m = Manifest(source_sha256=p.source.sha256, provenance="visual", table_exclusions=[], regions=regions)
    validate_manifest(p, m)
    m.regions[0].bbox.w = 100
    with pytest.raises(ValueError, match="clips"):
        validate_manifest(p, m)
    m.regions[0].bbox.w = 600
    m.table_exclusions = [p.fragments[0].bbox]
    with pytest.raises(ValueError, match="Table OCR"):
        validate_manifest(p, m)


@pytest.mark.parametrize("value", ["http://example.com:8080", "http://user:password@127.0.0.1:10809", "", "socks5://127.0.0.1:10808"])
def test_no_direct_or_private_proxy_credentials(value):
    with pytest.raises(ValueError):
        check_proxy(value)


def test_existing_local_http_proxy_is_allowed():
    assert check_proxy("http://127.0.0.1:10809") == "http://127.0.0.1:10809"


def runner_fixture(tmp_path, monkeypatch, sample):
    from smart_overlay.input import save_project
    import sys
    p, r = sample
    monkeypatch.chdir(tmp_path)
    save_project(p, tmp_path/"source.socr.json")
    manifest = Manifest(source_sha256=p.source.sha256, provenance="visual test fixture", table_exclusions=[],
                        regions=[r.model_copy(update={"name": name}) for name in ("R1", "R2", "R3")])
    (tmp_path/"regions.json").write_text(manifest.model_dump_json(), "utf-8")
    argv = ["diagnose_real", "--project", "source.socr.json", "--regions", "regions.json", "--output-dir",
            "outputs/run", "--proxy", "http://127.0.0.1:10809", "--execute", "--confirm-requests", "3", "--max-requests", "3"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.setenv(name, "")
    return argv


def test_runner_proxy_preflight_before_exactly_one_request_per_region(tmp_path, monkeypatch, sample):
    import os
    runner_fixture(tmp_path, monkeypatch, sample)
    events = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["proxy"] == "http://127.0.0.1:10809" and kwargs.get("verify") is not False
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url):
            import httpx
            events.append("public VPN GET")
            return httpx.Response(200, json={"data": {"endpoints": [{"provider_name": "Parasail"}]}}, request=httpx.Request("GET", url))
    monkeypatch.setattr("smart_overlay.diagnose_real.httpx.Client", Client)
    def request(case, settings, key):
        assert events[0] == "public VPN GET"
        assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:10809" and not os.environ["NO_PROXY"]
        events.append(case.name)
        return {"test": case.name, "classification": "valid_json_empty", "http_status": 200,
                "full_text": '{"merges":[]}', "finish_reason": "stop", "content_valid": False}
    monkeypatch.setattr("smart_overlay.diagnose_real.run_case", request)
    assert main() == 0
    assert events == ["public VPN GET", "R1", "R2", "R3"]
    report = json.loads((tmp_path/"outputs/run/results.json").read_text("utf-8"))
    assert all(r["metrics"]["missed_merges"] == 1 for r in report["results"])
    with pytest.raises(SystemExit):
        main()  # Cannot accidentally repeat an already executed series.
    assert events == ["public VPN GET", "R1", "R2", "R3"]


def test_confirmation_mismatch_prevents_any_network(tmp_path, monkeypatch, sample):
    argv = runner_fixture(tmp_path, monkeypatch, sample)
    argv[argv.index("--confirm-requests")+1] = "2"
    monkeypatch.setattr("smart_overlay.diagnose_real.httpx.Client", lambda **kw: pytest.fail("Unexpected network"))
    with pytest.raises(SystemExit):
        main()


def test_failed_vpn_preflight_never_falls_back_to_direct_model_request(tmp_path, monkeypatch, sample):
    runner_fixture(tmp_path, monkeypatch, sample)
    import httpx
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url):
            raise httpx.ConnectError("VPN unavailable")
    monkeypatch.setattr("smart_overlay.diagnose_real.httpx.Client", Client)
    monkeypatch.setattr("smart_overlay.diagnose_real.run_case", lambda *a: pytest.fail("Unexpected generation"))
    with pytest.raises(httpx.ConnectError):
        main()
