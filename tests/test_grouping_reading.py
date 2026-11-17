import json

import pytest

from smart_overlay.diagnose_vl import synthetic, evaluate
from smart_overlay.grouping import (build_plan, candidate_project, reading_case, run_grouping)
from smart_overlay.settings import Settings
from smart_overlay.vision_transport import RequestCancelled


def make_plan():
    p, _ = synthetic()
    p.fragments[0].confidence = .2
    p.fragments[1].confidence = .7
    return build_plan(p, Settings(), [], review_ocr=True, max_requests=100)


def test_low_confidence_budget_is_unique_and_respects_masks_and_threshold():
    plan = make_plan()
    assert plan.reading_ids == ['id_1','id_2']
    assert plan.requests == plan.grouping_requests+2
    filtered = build_plan(plan.project, Settings(confidence_threshold=.5), [plan.project.fragments[0].bbox],
                          review_ocr=True, max_requests=100)
    assert filtered.reading_ids == []
    with pytest.raises(ValueError, match='уточнение OCR'):
        build_plan(plan.project, Settings(), [], review_ocr=True, max_requests=plan.grouping_requests)


def test_reading_packet_has_two_images_and_validates_without_confidence_auto_acceptance():
    case = reading_case(make_plan(),'id_1')
    parts = case.payload['messages'][1]['content']
    assert sum(p['type']=='image_url' for p in parts) == 2
    assert case.payload['stream'] is False
    assert case.payload['provider']['order'] == ['parasail']
    r=evaluate(case, {'choices':[{'finish_reason':'stop','message':{'content':
        '{"text":"CORRECTED","confidence":1,"reason":"visible"}'}}]})
    assert r['classification']=='valid_content_unverified' and r['content_valid'] is None


def test_replacements_are_explicit_preserve_raw_ocr_and_feed_merged_text(monkeypatch):
    plan=make_plan()
    monkeypatch.setattr('smart_overlay.grouping.preflight',lambda *a:None)
    calls=[]
    def request(case,*a,**kw):
        calls.append(case.name)
        data=({'text':'CORRECTED '+case.project.fragments[0].id,'confidence':.5,'reason':'visible'}
              if case.response_model.__name__=='Reading' else {'merges':[['id_1','id_2','id_3']]}
              if {'id_1','id_2','id_3'} <= {f.id for f in case.project.fragments} else {'merges':[]})
        return {'test':case.name,'http_status':200,'finish_reason':'stop','full_text':json.dumps(data)}
    monkeypatch.setattr('smart_overlay.grouping.run_case',request)
    original=plan.project.model_dump_json()
    result=run_grouping(plan,'secret','http://127.0.0.1:10809')
    assert len(calls)==plan.requests and len(result.readings)==2
    unchanged=candidate_project(result,[])
    assert unchanged.objects[0].text=='MAIN'
    fixed=candidate_project(result,[['id_1','id_2','id_3']],['id_1'])
    merged=next(o for o in fixed.objects if len(o.ocr_ids)==3)
    assert merged.text.startswith('CORRECTED id_1\nCONTROL')
    assert merged.corrections[0].before=='MAIN' and merged.needs_review
    assert fixed.fragments[0].text=='MAIN' and plan.project.model_dump_json()==original


def test_cancel_mid_reading_retains_completed_reading_no_retries(monkeypatch):
    plan=make_plan()
    monkeypatch.setattr('smart_overlay.grouping.preflight',lambda *a:None)
    reads=[]
    def request(case,*a,**kw):
        if case.response_model.__name__=='Reading':
            reads.append(case.name)
            if len(reads)==2:raise RequestCancelled('cancelled')
            content={'text':'CHECKED','confidence':.9,'reason':'visible'}
        else:content={'merges':[]}
        return {'test':case.name,'http_status':200,'finish_reason':'stop','full_text':json.dumps(content)}
    monkeypatch.setattr('smart_overlay.grouping.run_case',request)
    result=run_grouping(plan,'secret','http://127.0.0.1:10809')
    assert result.status=='cancelled' and len(result.readings)==1 and len(reads)==2
    candidate=candidate_project(result,[],['id_1'])
    assert candidate.objects[0].text=='CHECKED'
