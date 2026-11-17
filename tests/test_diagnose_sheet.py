import json

import pytest

from smart_overlay.diagnose_vl import synthetic
from smart_overlay.diagnose_real import Manifest, Region
from smart_overlay.diagnose_sheet import (AutoRegion, automatic_regions, reconcile, logical_blocks,
                                         evaluate_controls, fragment_flags, save_artifacts, main)
from smart_overlay.geometry import contains


def region(name, ids):
    from smart_overlay.models import Box
    return AutoRegion(name=name, bbox=Box(x=0, y=0, w=600, h=180), ocr_ids=ids)


def result(name, groups, review=None):
    return {"test": name, "normalized": {"applied_merges": groups, "review_groups": review or []}}


def test_automatic_windows_cover_all_ids_and_protect_multiline_label():
    p, _ = synthetic()
    original = p.model_dump_json()
    windows, selection = automatic_regions(p, [], width=250, height=65, step_x=180, step_y=50)
    assert {i for r in windows for i in r.ocr_ids} == {f.id for f in p.fragments}
    for r in windows:
        assert all(contains(r.bbox, f.bbox) for f in p.fragments if f.id in r.ocr_ids)
        if set(r.ocr_ids) & {"id_1", "id_2", "id_3"}:
            assert {"id_1", "id_2", "id_3"} <= set(r.ocr_ids)
    assert p.model_dump_json() == original
    assert selection["protected_components"] and not selection["oversized_regions"]


def test_manual_table_masks_are_respected_and_no_reference_is_required():
    p, _ = synthetic()
    windows, selection = automatic_regions(p, [p.fragments[4].bbox])
    assert "id_5" in selection["excluded_ids"]
    assert all("id_5" not in r.ocr_ids for r in windows)
    assert all(not r.uncertain_ids for r in windows)


def test_agreement_deduplicates_exact_group():
    regions = [region("A", ["a", "b", "c"]), region("B", ["a", "b", "c"])]
    merged = reconcile(regions, [result("A", [["a", "b"]]), result("B", [["a", "b"]])])
    assert merged["accepted_merges"] == [{"ocr_ids": ["a", "b"], "support_regions": ["A", "B"], "needs_review": False}]
    assert not merged["conflicts"]


def test_complete_view_disagreement_never_forces_a_merge():
    regions = [region("A", ["a", "b", "c"]), region("B", ["a", "b", "c"])]
    merged = reconcile(regions, [result("A", [["a", "b"]]), result("B", [])])
    assert not merged["accepted_merges"] and merged["conflicts"]
    assert "complete_view_disagrees:B" in merged["conflicts"][0]["reasons"]


def test_overlapping_proposals_are_not_transitively_unioned():
    regions = [region("A", ["a", "b"]), region("B", ["b", "c"])]
    merged = reconcile(regions, [result("A", [["a", "b"]]), result("B", [["b", "c"]])])
    assert not merged["accepted_merges"] and len(merged["conflicts"]) == 2
    assert not any(g["ocr_ids"] == ["a", "b", "c"] for g in merged["conflicts"])


def test_partial_visibility_is_not_a_negative_vote():
    regions = [region("A", ["a", "b", "c"]), region("B", ["a", "b"])]
    merged = reconcile(regions, [result("A", [["a", "b", "c"]]), result("B", [])])
    assert merged["accepted_merges"][0]["ocr_ids"] == ["a", "b", "c"]
    assert merged["accepted_merges"][0]["needs_review"] is True


def test_pending_failed_and_abstaining_views_mark_review():
    regions = [region("A", ["a", "b"]), region("B", ["a", "b"])]
    merged = reconcile(regions, [result("A", [["a", "b"]])])
    assert merged["pending_regions"] == ["B"]
    assert merged["accepted_merges"][0]["needs_review"]
    merged = reconcile(regions, [result("A", [["a", "b"]]), {"test": "B"}])
    assert merged["failed_regions"] == ["B"] and merged["accepted_merges"][0]["needs_review"]
    merged = reconcile(regions, [result("A", [["a", "b"]]), result("B", [], [{"ocr_ids": ["a", "b"]}])])
    assert not merged["conflicts"] and merged["accepted_merges"][0]["needs_review"]


def test_full_partition_preserves_table_ids_and_codes_in_a_text_label():
    p, _ = synthetic()
    p.fragments[2].text = "UNIT-27"
    windows, selection = automatic_regions(p, [p.fragments[4].bbox])
    consensus = {"accepted_merges": [{"ocr_ids": ["id_1", "id_2", "id_3"], "needs_review": False}],
                 "conflicts": [], "blocked_ids": []}
    blocks = logical_blocks(p, selection, consensus)
    assert blocks[0]["kind"] == "text"
    assert next(b for b in blocks if b["ocr_ids"] == ["id_5"])["kind"] == "excluded_table_or_source"
    flat = [i for b in blocks for i in b["ocr_ids"]]
    assert len(flat) == len(set(flat)) == len(p.fragments)


def controls(p):
    return Manifest(source_sha256=p.source.sha256, provenance="visual fixture", table_exclusions=[], regions=[
        Region(name=name, bbox=region('x', ['id_1']).bbox, ocr_ids=[f.id for f in p.fragments],
               expected_merges=[["id_1", "id_2", "id_3"]], uncertain_ids=[], code_only_ids=[], rationale="fixture")
        for name in ("R1", "R2", "R3")])


def test_control_score_separates_selection_and_model_errors(tmp_path):
    p, _ = synthetic()
    reference = controls(p)
    windows, selection = automatic_regions(p, [])
    consensus = {"accepted_merges": [], "conflicts": [], "blocked_ids": []}
    blocks = logical_blocks(p, selection, consensus)
    scores = evaluate_controls(reference, windows, selection, consensus, blocks)
    assert all(not s['selection_missing_ids'] and not s['selection_split_merges'] for s in scores)
    assert all(s['missed_merges'] == 1 and not s['known_partition_correct'] for s in scores)
    consensus['accepted_merges'] = [{"ocr_ids": ["id_1", "id_2", "id_3"], "needs_review": False}]
    scores = evaluate_controls(reference, windows, selection, consensus, logical_blocks(p, selection, consensus))
    assert all(s['known_partition_correct'] for s in scores)
    save_artifacts(tmp_path, p, [], windows, selection, [], reference)
    assert (tmp_path/'sheet.png').is_file() and (tmp_path/'index.html').is_file()


def test_budget_confirmation_before_network(tmp_path, monkeypatch):
    import sys
    from smart_overlay.input import save_project
    p, _ = synthetic()
    monkeypatch.chdir(tmp_path)
    save_project(p, 'source.socr.json')
    (tmp_path/'controls.json').write_text(controls(p).model_dump_json(), 'utf-8')
    monkeypatch.setattr(sys, 'argv', ['diagnose_sheet', '--project', 'source.socr.json', '--controls', 'controls.json',
                                    '--output-dir', 'outputs/run', '--proxy', 'http://127.0.0.1:10809',
                                    '--execute', '--confirm-requests', '99'])
    monkeypatch.setattr('smart_overlay.diagnose_sheet.httpx.Client', lambda **kw: pytest.fail('Unexpected network'))
    with pytest.raises(SystemExit):
        main()


def runner_setup(tmp_path, monkeypatch):
    import sys
    from smart_overlay.input import save_project
    p, _ = synthetic()
    monkeypatch.chdir(tmp_path)
    save_project(p, 'source.socr.json')
    (tmp_path/'controls.json').write_text(controls(p).model_dump_json(), 'utf-8')
    windows, _ = automatic_regions(p, [])
    count = sum(len(r.ocr_ids) > 1 for r in windows)
    monkeypatch.setattr(sys, 'argv', ['diagnose_sheet', '--project', 'source.socr.json', '--controls', 'controls.json',
                                    '--output-dir', 'outputs/run', '--proxy', 'http://127.0.0.1:10809',
                                    '--execute', '--confirm-requests', str(count), '--max-requests', str(count)])
    monkeypatch.setenv('OPENROUTER_API_KEY', 'secret')
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY'):
        monkeypatch.setenv(name, '')
    return count


@pytest.mark.parametrize('api_status', [200, 403])
def test_runner_exact_budget_vpn_first_and_no_retries(tmp_path, monkeypatch, api_status):
    import httpx
    import os
    count = runner_setup(tmp_path, monkeypatch)
    events = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs['proxy'] == 'http://127.0.0.1:10809' and kwargs.get('verify') is not False
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url):
            events.append('public VPN GET')
            return httpx.Response(200, json={'data': {'endpoints': [{'provider_name': 'Parasail'}]}},
                                  request=httpx.Request('GET',url))
    monkeypatch.setattr('smart_overlay.diagnose_sheet.httpx.Client', Client)
    def request(case, settings, key):
        assert events[0] == 'public VPN GET'
        assert os.environ['HTTPS_PROXY'] == 'http://127.0.0.1:10809' and os.environ['NO_PROXY'] == ''
        assert settings.retries == 0
        events.append(case.name)
        return {'test': case.name, 'http_status': api_status, 'finish_reason': 'stop',
                'full_text': '{"merges":[]}' if api_status == 200 else None}
    monkeypatch.setattr('smart_overlay.diagnose_sheet.run_case', request)
    assert main() == 0
    assert len(events)-1 == (count if api_status == 200 else 1)
    report = json.loads((tmp_path/'outputs/run/results.json').read_text('utf-8'))
    assert report['planned_requests'] == count and report['vpn_preflight']['http_status'] == 200
    if api_status == 403:
        assert report['stopped']
    with pytest.raises(SystemExit):
        main()
    assert len(events)-1 == (count if api_status == 200 else 1)


def test_runner_never_sends_model_request_if_vpn_preflight_fails(tmp_path, monkeypatch):
    import httpx
    runner_setup(tmp_path, monkeypatch)
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url): raise httpx.ConnectError('VPN unavailable')
    monkeypatch.setattr('smart_overlay.diagnose_sheet.httpx.Client', Client)
    monkeypatch.setattr('smart_overlay.diagnose_sheet.run_case', lambda *a: pytest.fail('Unexpected generation'))
    with pytest.raises(httpx.ConnectError):
        main()
