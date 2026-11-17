import json
import time

import httpx
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication, QDialog

from smart_overlay.app import MainWindow, TableReviewResultDialog, TableReviewSetupDialog
from smart_overlay.export import export_pptx
from smart_overlay.grouping import fingerprint, mark_manual
from smart_overlay.input import open_project, save_project
from smart_overlay.models import Table
from smart_overlay.settings import Settings
from smart_overlay.structure import validate_project
from smart_overlay.table_review import (GridReview, TableReviewResult, build_table_plan, ordered_text,
    run_table_review, structure_candidate, structure_payload, table_candidate_project)
from smart_overlay.tables import new_table
from smart_overlay.vision_transport import RequestCancelled, RequestTimedOut


@pytest.fixture
def case(project):
    original=next(o for o in project.objects if isinstance(o,Table))
    grid=dict(x_lines=[sum(original.col_widths[:i])/original.bbox.w for i in range(original.columns+1)],
              y_lines=[sum(original.row_heights[:i])/original.bbox.h for i in range(original.rows+1)],
              cells=[{k:v for k,v in c.model_dump().items() if k in {'row','col','rowspan','colspan','ocr_ids'}} for c in original.cells],
              reason='Visible ruled lines')
    # Deliberately wrong local topology: all real text is in one merged cell.
    wrong=new_table(original.bbox,1,1)
    wrong.id=original.id; wrong.ocr_ids=original.ocr_ids[:]
    wrong.cells[0].ocr_ids=wrong.ocr_ids[:]
    wrong.cells[0].text=ordered_text(project,wrong.ocr_ids)
    wrong.needs_review=True
    project.objects=[wrong if o.id==wrong.id else o for o in project.objects]
    for f in project.fragments:
        if f.id in wrong.ocr_ids[:2]:f.confidence=.3
    plan=build_table_plan(project,Settings(),[wrong.id])
    return plan,wrong.id,grid


def reply(content):
    return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(content)}}],
            'usage':{'prompt_tokens':10,'completion_tokens':20,'cost':.01}}


def result_for(case):
    plan,identifier,grid=case
    target=plan.reading_targets[0]
    return TableReviewResult(plan,records=[
        {'kind':'structure','table_id':identifier,'ocr_id':None,'reply':grid},
        {'kind':'reading','table_id':identifier,'ocr_id':target[1],
         'reply':{'text':'CORRECTED','confidence':1,'reason':'visible'}}])


def test_plan_fixed_budget_and_threshold(case):
    plan,identifier,_=case
    assert plan.requests==3 and plan.output_tokens==6144
    assert len(set(plan.reading_targets))==2
    assert build_table_plan(plan.project,Settings(confidence_threshold=.1),[identifier]).requests==1
    assert build_table_plan(plan.project,Settings(),[identifier],review_structure=False).requests==2
    assert build_table_plan(plan.project,Settings(),[identifier],review_text=False).requests==1
    with pytest.raises(ValueError,match='Нужно 3'):
        build_table_plan(plan.project,Settings(),[identifier],max_requests=2)
    with pytest.raises(ValueError):
        build_table_plan(plan.project,Settings(),[identifier,identifier])


@pytest.mark.parametrize('change',['manual','excluded','text','origin','correction'])
def test_manual_changes_are_protected(case,change):
    plan,identifier,_=case
    project=plan.project.model_copy(deep=True)
    table=next(o for o in project.objects if o.id==identifier)
    if change=='manual':mark_manual(project,identifier)
    elif change=='excluded':table.excluded=True
    elif change=='text':table.cells[0].text='manual text'
    elif change=='origin':project.fragments[0].origin='region'
    else:
        from smart_overlay.models import Correction
        table.corrections.append(Correction(source='manual',before='a',after='b',confidence=1,applied=True,reason='edit'))
    result=build_table_plan(project,Settings(),[identifier])
    assert result.requests==0 and identifier in result.skipped


def test_structure_packet_is_local_and_has_no_text_corrections(case):
    plan,identifier,_=case
    payload=structure_payload(plan,identifier)
    assert payload['max_tokens']==4096 and payload['stream'] is False
    assert payload['provider']=={'order':['parasail'],'allow_fallbacks':False}
    content=payload['messages'][1]['content']
    assert len(content)==2 and content[1]['type']=='image_url'
    assert 'Do not translate or correct text' in content[0]['text']


def test_structure_repairs_topology_preserving_source_and_ids(case):
    plan,identifier,grid=case
    before=fingerprint(plan.project)
    table=structure_candidate(plan,identifier,grid)
    assert (table.rows,table.columns)==(4,3)
    assert table.bbox==next(o for o in plan.project.objects if o.id==identifier).bbox
    assert set(table.ocr_ids)==set(i for c in table.cells for i in c.ocr_ids)
    assert len({c.id for c in table.cells})==len(table.cells)
    assert any(c.colspan==3 for c in table.cells) and any(c.rowspan==2 for c in table.cells)
    assert fingerprint(plan.project)==before


@pytest.mark.parametrize('damage',['missing','duplicate','unknown','unsorted','extent','overlap','hole','wrong_cell','fractional'])
def test_bad_structure_cannot_be_applied(case,damage):
    plan,identifier,grid=case
    grid=json.loads(json.dumps(grid))
    if damage=='missing':grid['cells'][0]['ocr_ids']=[]
    elif damage=='duplicate':grid['cells'][1]['ocr_ids'].append(grid['cells'][0]['ocr_ids'][0])
    elif damage=='unknown':grid['cells'][0]['ocr_ids']=['unknown']
    elif damage=='unsorted':grid['x_lines']=[0,.8,.2,1]
    elif damage=='extent':grid['y_lines'][-1]=.9
    elif damage=='overlap':grid['cells'][0]['rowspan']=2
    elif damage=='hole':grid['cells'][0]['colspan']=2
    elif damage=='wrong_cell':grid['cells'][1]['ocr_ids'],grid['cells'][2]['ocr_ids']=grid['cells'][2]['ocr_ids'],grid['cells'][1]['ocr_ids']
    else:grid['cells'][0]['row']=.5
    with pytest.raises(ValueError):structure_candidate(plan,identifier,grid)


def test_reading_and_structure_independently_apply_and_export(case,tmp_path):
    result=result_for(case); plan,identifier,_=case
    target=plan.reading_targets[0]
    before=fingerprint(plan.project)
    unchanged=table_candidate_project(result)
    assert unchanged.objects==plan.project.objects
    text_only=table_candidate_project(result,[],[target])
    table=next(o for o in text_only.objects if o.id==identifier)
    assert table.rows==1 and 'CORRECTED' in table.cells[0].text
    fixed=table_candidate_project(result,[identifier],[target])
    table=next(o for o in fixed.objects if o.id==identifier)
    assert table.rows==4 and any(c.text=='CORRECTED' for c in table.cells)
    assert table.needs_review and len(table.corrections)==2
    assert fixed.fragments==plan.project.fragments and fixed.source==plan.project.source
    assert fingerprint(plan.project)==before
    assert fixed.objects[:-1]==plan.project.objects[:-1]
    validate_project(fixed)
    save_project(fixed,tmp_path/'review.socr.json')
    assert open_project(tmp_path/'review.socr.json')==fixed
    export_pptx(fixed,tmp_path/'review.pptx',Settings(),allow_unreviewed=True)
    assert (tmp_path/'review.pptx').is_file()
    assert build_table_plan(fixed,Settings(),[identifier]).requests==0
    with pytest.raises(ValueError):table_candidate_project(result,[identifier,identifier])


def test_transport_calls_once_per_prepared_job_and_uses_proxy(case,monkeypatch):
    plan,identifier,grid=case
    monkeypatch.setattr('smart_overlay.table_review.preflight',lambda *a:None)
    calls=[]
    def post(url,key,payload,**kwargs):
        assert isinstance(payload,bytes)
        payload=json.loads(payload.decode('utf-8'))
        calls.append((payload,kwargs))
        return reply(grid if payload['max_tokens']==4096 else {'text':'READ','confidence':.7,'reason':'visible'})
    monkeypatch.setattr('smart_overlay.table_review.post_json',post)
    result=run_table_review(plan,'secret','http://127.0.0.1:10809')
    assert result.status=='completed' and len(calls)==plan.requests
    assert all(c[1]['proxy']=='http://127.0.0.1:10809' for c in calls)
    assert all('reply' in r for r in result.records)
    assert table_candidate_project(result).usage[-1]['cost']==.01


@pytest.mark.parametrize('review_text,review_structure',[(False,True),(True,False),(True,True)])
def test_real_http_transport_serializes_table_requests(case,monkeypatch,review_text,review_structure):
    original_plan,identifier,grid=case
    project=original_plan.project.model_copy(deep=True)
    target=original_plan.reading_targets[0][1]
    next(f for f in project.fragments if f.id==target).text='Давление'
    table=next(o for o in project.objects if o.id==identifier)
    table.cells[0].text=ordered_text(project,table.ocr_ids)
    plan=build_table_plan(project,Settings(),[identifier],review_text=review_text,review_structure=review_structure)
    requests=[]

    def handler(request):
        requests.append(request)
        if request.method=='GET':
            assert 'authorization' not in request.headers
            return httpx.Response(200,json={'data':{'endpoints':[{'provider_name':'Parasail'}]}})
        assert request.headers['content-type']=='application/json'
        assert request.headers['authorization']=='Bearer test-key'
        payload=json.loads(request.content.decode('utf-8'))
        assert payload['model']=='qwen/qwen3-vl-8b-instruct'
        assert payload['provider']=={'order':['parasail'],'allow_fallbacks':False}
        assert payload['response_format']=={'type':'json_object'} and payload['stream'] is False
        data=grid if payload['max_tokens']==4096 else {'text':'Уточнено','confidence':.7,'reason':'visible'}
        return httpx.Response(200,json=reply(data))

    original_client=httpx.AsyncClient
    def client(**kwargs):
        assert kwargs.pop('proxy')=='http://127.0.0.1:10809'
        assert kwargs['trust_env'] is False and kwargs.get('verify') is not False
        return original_client(transport=httpx.MockTransport(handler),**kwargs)

    monkeypatch.setattr('smart_overlay.vision_transport.httpx.AsyncClient',client)
    result=run_table_review(plan,'test-key','http://127.0.0.1:10809')
    assert result.status=='completed' and all('reply' in r for r in result.records)
    assert [r.method for r in requests]==['GET']+['POST']*plan.requests
    assert any('Давление' in r.content.decode('utf-8') for r in requests[1:])


@pytest.mark.parametrize('error',[RequestCancelled('stop'),RequestTimedOut('stop'),httpx.ConnectError('offline')])
def test_partial_failure_retains_first_answer_without_retry(case,monkeypatch,error):
    plan,_,grid=case
    monkeypatch.setattr('smart_overlay.table_review.preflight',lambda *a:None)
    calls=[]
    def post(*a,**k):
        calls.append(True)
        if len(calls)==2:raise error
        return reply(grid)
    monkeypatch.setattr('smart_overlay.table_review.post_json',post)
    result=run_table_review(plan,'secret','http://127.0.0.1:10809')
    assert len(calls)==2 and result.records[0]['reply']==grid
    assert result.status in ('cancelled','partial')


def test_invalid_and_truncated_reply_not_selected_or_retried(case,monkeypatch):
    plan,_,_=case
    monkeypatch.setattr('smart_overlay.table_review.preflight',lambda *a:None)
    calls=[]
    def post(*a,**k):
        calls.append(True)
        value=reply({'text':'CUT','confidence':1,'reason':'x'})
        value['choices'][0]['finish_reason']='length'
        return value
    monkeypatch.setattr('smart_overlay.table_review.post_json',post)
    result=run_table_review(plan,'secret','http://127.0.0.1:10809')
    assert len(calls)==plan.requests and all('reply' not in r for r in result.records)
    assert all('error' in r for r in result.records)


def test_preflight_failure_or_already_cancelled_sends_no_paid_request(case,monkeypatch):
    plan,_,_=case
    monkeypatch.setattr('smart_overlay.table_review.post_json',lambda *a,**k:pytest.fail('Paid request'))
    monkeypatch.setattr('smart_overlay.table_review.preflight',lambda *a:(_ for _ in ()).throw(httpx.ConnectError('offline')))
    assert run_table_review(plan,'secret','http://127.0.0.1:10809').status=='partial'
    assert run_table_review(plan,'secret','http://127.0.0.1:10809',cancelled=lambda:True).status=='cancelled'


def test_api_envelope_error_stops_remaining_jobs(case,monkeypatch):
    plan,_,_=case
    calls=[]
    monkeypatch.setattr('smart_overlay.table_review.preflight',lambda *a:None)
    def post(*a,**k):
        calls.append(True)
        return {'error':{'code':429,'message':'secret'}}
    monkeypatch.setattr('smart_overlay.table_review.post_json',post)
    result=run_table_review(plan,'secret','http://127.0.0.1:10809')
    assert len(calls)==1 and result.status=='partial' and 'secret' not in result.message


def test_invalid_structure_reports_reason_without_sensitive_schema_inputs(case,monkeypatch):
    plan,_,grid=case
    monkeypatch.setattr('smart_overlay.table_review.preflight',lambda *a:None)
    wrong=json.loads(json.dumps(grid)); wrong['cells'][0]['ocr_ids']=['unknown']
    monkeypatch.setattr('smart_overlay.table_review.post_json',lambda *a,**k:reply(wrong))
    result=run_table_review(plan,'secret','http://127.0.0.1:10809')
    assert 'OCR ID' in result.records[0]['error']
    assert all('secret' not in r['error'] for r in result.records)


def test_setup_budget_and_preview_require_explicit_selection(gui_app,case):
    plan,identifier,_=case
    setup=TableReviewSetupDialog(plan.project,Settings())
    setup.prepare()
    assert setup.plan.requests==3 and setup.send_button.isEnabled()
    setup.review_structure.setChecked(False)
    assert setup.plan is None and not setup.send_button.isEnabled()
    setup.reject()
    result=result_for(case)
    dialog=TableReviewResultDialog(result)
    assert not dialog.apply_button.isEnabled() and dialog.selected()==([],[])
    assert dialog.candidate.objects==plan.project.objects
    dialog.proposals.item(0,0).setCheckState(Qt.CheckState.Checked)
    dialog.proposals.item(1,0).setCheckState(Qt.CheckState.Checked)
    assert dialog.apply_button.isEnabled() and next(o for o in dialog.candidate.objects if o.id==identifier).rows==4
    dialog.reject()


def test_apply_discard_stale_and_undo(gui_app,case,monkeypatch):
    result=result_for(case); plan,identifier,_=case
    window=MainWindow(); window.project=plan.project.model_copy(deep=True)
    before=fingerprint(window.project)
    monkeypatch.setattr(TableReviewResultDialog,'exec',lambda self:QDialog.DialogCode.Rejected)
    window.show_table_review_result(result)
    assert fingerprint(window.project)==before and not window.dirty
    def accept(dialog):
        dialog.proposals.item(0,0).setCheckState(Qt.CheckState.Checked)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(TableReviewResultDialog,'exec',accept)
    window.show_table_review_result(result)
    assert window.dirty and window.undo_grouping_button.isEnabled()
    window.undo_grouping()
    assert fingerprint(window.project)==before and not window.dirty
    window.project.messages.append('later change')
    errors=[]; monkeypatch.setattr(window,'error',errors.append)
    window.show_table_review_result(result)
    assert errors
    window.close()


def test_worker_preview_is_opened_after_worker_completion(gui_app,case,monkeypatch):
    result=result_for(case)
    window=MainWindow(); viewed=[]
    def preview(value):
        assert window.worker is None and window.table_review_button.isEnabled()
        viewed.append(value)
    monkeypatch.setattr(window,'show_table_review_result',preview)
    window.start(lambda progress:result,cancellable=True)
    deadline=time.monotonic()+3
    while window.worker and time.monotonic()<deadline:
        QApplication.processEvents(); time.sleep(.01)
    assert viewed==[result]
    QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
    window.close()
