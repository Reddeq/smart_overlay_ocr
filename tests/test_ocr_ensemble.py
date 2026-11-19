"""Offline tests for two-reader OCR and strict VL batch QA (no model downloads)."""
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from smart_overlay.diagnose_vl import synthetic
from smart_overlay.ocr_ensemble import (model_for, enhance, compare, run_ensemble,
                                        disagreement_sample, ensemble_hybrid)
from smart_overlay.vl_batch import contact_sheet, parse_numbered, parse_vl_result
from smart_overlay.vl_worker import process_hybrid_batch
from smart_overlay.settings import Settings
from smart_overlay.structure import validate_project


def baseline():
    project,_=synthetic()
    f=project.fragments[0]
    f.text="MAlN"
    f.confidence=.32
    project.objects[0].text="MAlN"
    return project


class FakeRecognition:
    def __init__(self, answer="MAIN"):
        self.answer=answer
        self.calls=[]
    def predict(self, *, input, batch_size=8):
        self.calls.append([x.shape for x in input])
        return [SimpleNamespace(json={"res":{"rec_text":self.answer,"rec_score":.89}})
                for _ in input]


def test_v6_is_never_used_to_recognize_cyrillic_as_russian():
    settings=Settings(ocr_language_mode="manual",ocr_languages=["ru","en"])
    assert model_for("Кассета 2",settings)=="eslav_PP-OCRv5_mobile_rec"
    assert model_for("CONTROL",settings)=="PP-OCRv6_medium_rec"
    assert model_for("测量",settings)=="PP-OCRv6_medium_rec"
    assert model_for("العربية",settings)=="arabic_PP-OCRv5_mobile_rec"
    assert model_for("한글",settings)=="korean_PP-OCRv5_mobile_rec"
    ru_only=Settings(ocr_language_mode="manual",ocr_languages=["ru"])
    assert model_for("MP-5",ru_only)=="eslav_PP-OCRv5_mobile_rec"


def test_preprocess_does_not_change_source_geometry():
    import numpy as np
    source=np.full((15,70,3),255,dtype=np.uint8)
    source[4:8,10:30]=50
    big=enhance(source,"large")
    cont=enhance(source,"contrast")
    assert big.shape[0] > source.shape[0]
    assert cont.shape == source.shape
    assert source[5,15,0] == 50


def test_two_repeated_readings_give_candidate_not_absolute_truth():
    project=baseline()
    f=project.fragments[0]
    assert compare(f,[("large","MAIN",.99),("contrast","MAIN",.97)]) == (
        "MAIN","repeated_candidate_differs")
    assert compare(f,[("large","MAIN",.99),("contrast","MAlN",.97)])[0] is None
    assert compare(f,[("large","MAlN",.99),("contrast","MAlN",.97)])[1]=="agrees_with_original"


def test_ensemble_reads_all_fragments_in_batches_and_writes_diagnostic(tmp_path):
    project=baseline()
    model=FakeRecognition()
    result, report=run_ensemble(
        project,Settings(ensemble_apply_consensus=False),tmp_path,
        model_factory=lambda name:model)
    assert report["total_fragments"]==len(project.fragments)
    assert len(report["records"])==len(project.fragments)
    assert model.calls
    assert len(model.calls)==2
    assert result.fragments[0].text=="MAlN"
    assert result.objects[0].needs_review
    assert result.objects[0].bbox==project.objects[0].bbox
    assert (tmp_path/"ocr-ensemble.json").exists()
    assert disagreement_sample(result,report,limit=3)


def test_safe_consensus_changes_only_low_confidence_single_line(tmp_path):
    project=baseline()
    initial=project.objects[0].bbox.model_copy()
    result,report=run_ensemble(project,Settings(ensemble_apply_consensus=True),
                               tmp_path,model_factory=lambda name:FakeRecognition())
    assert result.objects[0].text=="MAIN"
    assert result.objects[0].bbox==initial
    assert result.objects[0].needs_review
    assert report["stats"]["applied"]>=1
    validate_project(result)


def test_numbered_vl_contact_sheet_and_strict_all_id_validation():
    source=Image.new("RGB",(700,200),"white")
    items=[
        {"id":"ocr_a","bbox":[20,20,180,60]},
        {"id":"ocr_b","bbox":[20,100,250,140]},
    ]
    collage=contact_sheet(source,items)
    assert collage.height>180
    correct={"parsing_res_list":[{"block_label":"text",
                                  "block_content":"01: POWER SUPPLY\n02: CONTROL"}]}
    assert parse_vl_result(correct,2)==({1:"POWER SUPPLY",2:"CONTROL"},"complete")
    assert parse_numbered("01: POWER",2)[0]=={}
    assert parse_numbered("02: CONTROL\n01: POWER",2)[0]=={}
    assert parse_numbered("01: POWER\n03: CONTROL",2)[0]=={}
    assert parse_numbered("01: POWER\n02: CONTROL\nEXTRA",2)[0]=={}


class FakeBatchVL:
    def __init__(self,text):
        self.text=text
        self.calls=[]
    def predict(self,*,input,**kwargs):
        self.calls.append((input.shape,kwargs))
        return [SimpleNamespace(json={"res":{"parsing_res_list":[
            {"block_label":"text","block_content":self.text}
        ]}})]


def test_batch_worker_one_model_call_validates_mapping(tmp_path):
    folder=tmp_path/"batch"
    folder.mkdir()
    plan={"mode":"batch","items":[
        {"id":"ocr_a","bbox":[10,20,150,45],"ocr_text":"P0WER"},
        {"id":"ocr_b","bbox":[10,80,150,105],"ocr_text":"C0NTROL"}
    ]}
    (folder/"paddle-vl-hybrid-plan.json").write_text(json.dumps(plan))
    fake=FakeBatchVL("01: POWER\n02: CONTROL")
    process_hybrid_batch(
        Image.new("RGB",(400,180),"white"),folder,"cpu",
        Settings(ensemble_batch_size=8,ensemble_max_batches=1),
        pipeline_factory=lambda *,device,layout:fake)
    assert len(fake.calls)==1
    assert fake.calls[0][1]["prompt_label"]=="ocr"
    assert (folder/"vl-batch-00.png").exists()
    assert json.loads((folder/"vl-batch-00.json").read_text())["status"]=="complete"
    assert json.loads((folder/"hybrid-ocr_a.json").read_text())["text"]=="POWER"


def test_batch_worker_invalid_numbering_never_writes_individual_suggestions(tmp_path):
    folder=tmp_path/"bad"
    folder.mkdir()
    (folder/"paddle-vl-hybrid-plan.json").write_text(json.dumps({
        "mode":"batch","items":[{"id":"ocr_a","bbox":[0,0,100,30]},
                                {"id":"ocr_b","bbox":[0,40,100,70]}]}))
    process_hybrid_batch(Image.new("RGB",(200,100),"white"),folder,"cpu",
                         Settings(),
                         pipeline_factory=lambda *,device,layout:FakeBatchVL(
                             "01: guessed\n01: repeated"))
    assert json.loads((folder/"vl-batch-00.json").read_text())["status"]=="unverified"
    assert not (folder/"hybrid-ocr_a.json").exists()


def test_ensemble_hybrid_without_vl_preserves_report(tmp_path):
    project=baseline()
    result,warnings=ensemble_hybrid(
        project,"unused.png",Settings(ensemble_batch_vl=False),
        directory=tmp_path,model_factory=lambda name:FakeRecognition())
    assert result.raw["ocr_ensemble"]["total_fragments"]==len(project.fragments)
    assert (tmp_path/"ocr-ensemble-summary.json").exists()
    assert not (tmp_path/"paddle-vl-hybrid-plan.json").exists()


def test_ensemble_hybrid_uses_only_disagreements_for_vl(tmp_path,monkeypatch):
    project=baseline()
    called=[]
    def fake_isolated(source,settings,device,folder,**kwargs):
        called.append(kwargs)
        plan=json.loads((Path(folder)/"paddle-vl-hybrid-plan.json").read_text())
        items=plan["items"]
        mapped={str(i):"MAIN" for i in range(1,len(items)+1)}
        (Path(folder)/"vl-batch-00.json").write_text(json.dumps({
            "status":"complete","ids":[x["id"] for x in items],
            "parsed":mapped}))
        for item in items:
            (Path(folder)/f"hybrid-{item['id']}.json").write_text(json.dumps({
                "status":"ok","text":"MAIN","seconds":1.5}))
        return {"hybrid":True},{"hybrid":True}
    monkeypatch.setattr("smart_overlay.vl_tiled.run_isolated",fake_isolated)
    result,warnings=ensemble_hybrid(
        project,"drawing.png",
        Settings(ensemble_batch_vl=True,ensemble_batch_size=8),
        directory=tmp_path,model_factory=lambda name:FakeRecognition())
    assert called and called[0]["hybrid"] is True
    assert (tmp_path/"ocr-ensemble-vl-audit.json").exists()
    assert result.raw["ocr_ensemble"]["vl_batches_verified"]==1
    assert result.objects[0].needs_review


def test_human_truth_cer_eval_scores_actual_errors_not_pptx_exist(tmp_path):
    import csv
    from smart_overlay.ocr_eval import (edit_distance,truth_template,
                                        evaluate_csv)
    assert edit_distance("MAlN","MAIN")==1
    report={"records":[
        {"id":"ocr_1","original":"MAlN","consensus":"MAIN","final":"MAIN"},
        {"id":"ocr_2","original":"Касета","consensus":None,"final":"Касета"},
    ]}
    path=tmp_path/"ocr-ensemble.json"
    path.write_text(json.dumps(report,ensure_ascii=False),encoding="utf-8")
    template=truth_template(path,tmp_path/"reference.csv")
    assert template.exists()
    with template.open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.writer(stream)
        writer.writerow(["id","original","truth"])
        writer.writerow(["ocr_1","MAlN","MAIN"])
        writer.writerow(["ocr_2","Касета","Кассета"])
    quality=evaluate_csv(path,template,tmp_path/"comparison.json")
    assert quality["baseline_ppocrv5"]["exact_matches"]==0
    assert quality["alternate_consensus"]["measured"]==1
    assert quality["final_pptx"]["exact_matches"]==1
    assert quality["baseline_ppocrv5"]["character_error_rate"] > (
        quality["final_pptx"]["character_error_rate"])
