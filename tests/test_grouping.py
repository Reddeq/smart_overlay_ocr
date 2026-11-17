import json
import zipfile

import httpx
import pytest

from smart_overlay.diagnose_vl import synthetic
from smart_overlay.grouping import (GroupingResult, MASKS_KEY, build_plan, candidate_project, fingerprint,
                                    finish, mark_manual, preflight, run_grouping, saved_masks)
from smart_overlay.input import open_project, save_project
from smart_overlay.models import Box
from smart_overlay.settings import Settings
from smart_overlay.structure import manual_text, validate_project
from smart_overlay.vision_transport import RequestCancelled


@pytest.fixture
def plan():
    p, _ = synthetic()
    return build_plan(p, Settings(), [])


def reply(case):
    groups = [["id_1", "id_2", "id_3"]] if {"id_1", "id_2", "id_3"} <= set(f.id for f in case.project.fragments) else []
    return {"test": case.name, "http_status": 200, "finish_reason": "stop", "full_text": json.dumps({"merges": groups}),
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}}


def completed(plan, monkeypatch):
    monkeypatch.setattr("smart_overlay.grouping.preflight", lambda *a: None)
    monkeypatch.setattr("smart_overlay.grouping.run_case", lambda case, *a, **kw: reply(case))
    return run_grouping(plan, "secret", "http://127.0.0.1:10809")


def test_explicit_apply_preserves_source_and_existing_ids(plan, monkeypatch):
    before = fingerprint(plan.project)
    result = completed(plan, monkeypatch)
    groups = [g['ocr_ids'] for g in result.consensus['accepted_merges']]
    assert groups == [["id_1", "id_2", "id_3"]]
    candidate = candidate_project(result, groups)
    validate_project(candidate)
    assert len(candidate.objects) == 3
    assert candidate.source == plan.project.source and candidate.fragments == plan.project.fragments
    assert candidate.objects[0].id == plan.project.objects[3].id
    assert next(o for o in candidate.objects if len(o.ocr_ids) == 3).needs_review
    assert fingerprint(plan.project) == before


@pytest.mark.parametrize('change', ['text', 'geometry', 'style', 'marker', 'excluded', 'origin'])
def test_manual_and_legacy_edits_are_protected(change):
    p, _ = synthetic()
    obj = p.objects[0]
    if change == 'text': manual_text(obj, "Edited by user")
    elif change == 'geometry': obj.bbox = obj.bbox.model_copy(update={'x': obj.bbox.x+2})
    elif change == 'style': obj.style.color = '#FF0000'
    elif change == 'marker': mark_manual(p, obj.id)
    elif change == 'excluded': obj.excluded = True
    else: p.fragments[0].origin = 'manual_split'
    plan = build_plan(p, Settings(), [])
    assert obj.id in plan.protected_objects
    assert 'id_1' not in plan.selection['eligible_ids']


def test_table_masks_exclude_members_and_persist_only_after_apply(plan, monkeypatch, tmp_path):
    mask = plan.project.fragments[-1].bbox.model_copy()
    source = plan.project
    masked = build_plan(source, Settings(), [mask])
    assert 'id_5' not in masked.selection['eligible_ids']
    assert MASKS_KEY not in source.raw
    result = completed(masked, monkeypatch)
    candidate = candidate_project(result, [])
    path = tmp_path/'masked.socr.json'
    save_project(candidate, path)
    assert saved_masks(open_project(path)) == [mask]
    assert source.objects == plan.project.objects and MASKS_KEY not in source.raw


def test_existing_table_and_composite_object_never_rebuilt(project):
    plan = build_plan(project, Settings(), [])
    assert project.objects[-1].id in plan.protected_objects
    assert next(o.id for o in project.objects if len(o.ocr_ids) == 3) in plan.protected_objects
    result = finish(GroupingResult(plan))
    candidate = candidate_project(result, [])
    assert candidate.objects[-1] == project.objects[-1]


def test_cancel_before_network_keeps_original(plan, monkeypatch):
    monkeypatch.setattr('smart_overlay.grouping.preflight', lambda *a: pytest.fail('Unexpected network'))
    original = fingerprint(plan.project)
    result = run_grouping(plan, 'secret', 'http://127.0.0.1:10809', cancelled=lambda: True)
    assert result.status == 'cancelled' and not result.results
    assert fingerprint(plan.project) == original


def test_partial_cancellation_retains_previous_proposals_without_retry(plan, monkeypatch):
    calls = []
    monkeypatch.setattr('smart_overlay.grouping.preflight', lambda *a: None)
    def request(case, settings, key, **kw):
        assert kw['proxy'] == 'http://127.0.0.1:10809'
        assert settings.retries == 0 and callable(kw['cancelled'])
        calls.append(case.name)
        if len(calls) == 2:
            raise RequestCancelled('cancelled')
        return reply(case)
    monkeypatch.setattr('smart_overlay.grouping.run_case', request)
    result = run_grouping(plan, 'secret', 'http://127.0.0.1:10809')
    assert result.status == 'cancelled' and len(calls) == 2 and len(result.results) == 1
    assert result.consensus['accepted_merges'][0]['ocr_ids'] == ['id_1', 'id_2', 'id_3']
    validate_project(candidate_project(result, [['id_1', 'id_2', 'id_3']]))


def test_vpn_failure_never_falls_back_to_generation(plan, monkeypatch):
    def fail(*a): raise httpx.ConnectError('VPN unavailable')
    monkeypatch.setattr('smart_overlay.grouping.preflight', fail)
    monkeypatch.setattr('smart_overlay.grouping.run_case', lambda *a, **kw: pytest.fail('Unexpected generation'))
    result = run_grouping(plan, 'secret', 'http://127.0.0.1:10809')
    assert result.status == 'partial' and not result.results


def test_api_failure_stops_after_one_attempt(plan, monkeypatch):
    calls = []
    monkeypatch.setattr('smart_overlay.grouping.preflight', lambda *a: None)
    def request(case, *a, **kw):
        calls.append(case.name)
        return {'test': case.name, 'http_status': 403}
    monkeypatch.setattr('smart_overlay.grouping.run_case', request)
    result = run_grouping(plan, 'secret', 'http://127.0.0.1:10809')
    assert len(calls) == 1 and result.status == 'partial'


def test_preflight_uses_explicit_proxy_and_is_cancellable(monkeypatch):
    import asyncio
    original = httpx.AsyncClient
    seen = []
    async def handler(request):
        await asyncio.sleep(60)
    def client(**kwargs):
        seen.append(kwargs)
        assert kwargs.pop('proxy') == 'http://127.0.0.1:10809'
        return original(transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr('smart_overlay.grouping.httpx.AsyncClient', client)
    with pytest.raises(RequestCancelled):
        preflight(Settings(), 'http://127.0.0.1:10809', lambda: True)
    assert seen[0]['trust_env'] is False and seen[0].get('verify') is not False


def test_grouped_project_roundtrip_and_existing_real_pptx_export(plan, monkeypatch, tmp_path):
    from smart_overlay.export import export_pptx
    result = completed(plan, monkeypatch)
    candidate = candidate_project(result, [['id_1', 'id_2', 'id_3']])
    save_project(candidate, tmp_path/'grouped.socr.json')
    reopened = open_project(tmp_path/'grouped.socr.json')
    assert reopened == candidate
    export_pptx(reopened, tmp_path/'grouped.pptx', Settings(), allow_unreviewed=True)
    with zipfile.ZipFile(tmp_path/'grouped.pptx') as deck:
        xml = deck.read('ppt/slides/slide1.xml').decode()
        assert all(text in xml for text in ('MAIN', 'CONTROL', 'PANEL 1', 'VALVE 1', 'PUMP 1'))


def test_real_transport_cancellation_through_explicit_proxy_keeps_previous_response(plan, monkeypatch):
    import asyncio
    import os
    import threading
    original = httpx.AsyncClient
    stop = threading.Event()
    requests = []
    environment = {k: os.environ.get(k) for k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY')}
    async def handler(request):
        requests.append(request)
        if request.method == 'GET':
            assert 'authorization' not in request.headers
            return httpx.Response(200, json={'data': {'endpoints': [{'provider_name': 'Parasail'}]}})
        posts = [r for r in requests if r.method == 'POST']
        if len(posts) == 2:
            stop.set()
            await asyncio.sleep(60)
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': '{"merges":[["id_1","id_2","id_3"]]}'}}]})
    def client(**kwargs):
        assert kwargs.pop('proxy') == 'http://127.0.0.1:10809'
        assert kwargs['trust_env'] is False and kwargs.get('verify') is not False
        return original(transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr('smart_overlay.grouping.httpx.AsyncClient', client)
    result = run_grouping(plan, 'secret', 'http://127.0.0.1:10809', cancelled=stop.is_set)
    assert result.status == 'cancelled' and len(result.results) == 1
    assert [r.method for r in requests] == ['GET', 'POST', 'POST']
    assert {k: os.environ.get(k) for k in environment} == environment
    assert result.consensus['accepted_merges'][0]['ocr_ids'] == ['id_1','id_2','id_3']


def test_unknown_selection_and_wrong_budget_are_rejected(plan, monkeypatch):
    with pytest.raises(ValueError, match='неизвестное'):
        candidate_project(completed(plan, monkeypatch), [['id_1','id_5']])
    with pytest.raises(ValueError, match='предел'):
        build_plan(plan.project, Settings(), [], max_requests=1)
