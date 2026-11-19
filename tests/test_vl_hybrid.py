"""Hybrid local OCR tests: only suspect fragments are VL-reviewed, not all OCR.

Fake VL predictions and fake export: never downloads weights or sends data.
"""
from pathlib import Path
import json
from types import SimpleNamespace

from PIL import Image
import pytest

from smart_overlay.diagnose_vl import synthetic
from smart_overlay.models import Cell, Table
from smart_overlay.one_click import convert_drawing
from smart_overlay.settings import Settings
from smart_overlay.structure import validate_project
from smart_overlay.vl_hybrid import candidates, safe_change, apply_suggestions, selective_vl
from smart_overlay.vl_worker import process_hybrid


def sample():
    project, _ = synthetic()
    project.fragments[0].text = "MAlN"
    project.fragments[0].confidence = .31
    project.objects[0].text = "MAlN"
    project.objects[0].confidence = .31
    project.fragments[1].confidence = .68
    return project


def test_only_selected_suspicious_labels_are_checked():
    project = sample()
    settings=Settings(hybrid_max_checks=1, hybrid_confidence=.74)
    chosen,total=candidates(project,settings)
    assert total == 2
    assert len(chosen)==1
    assert chosen[0][0].id==project.fragments[0].id
    assert all(f.text!="VALVE 1" for f,_ in chosen)


def test_technical_numbers_scripts_and_unrelated_predictions_are_protected():
    assert safe_change("MAlN","MAIN",.31)[0]
    assert not safe_change("Кассета 2","Кассета 3",.30)[0]
    assert not safe_change("24V DC","48V DC",.30)[0]
    assert not safe_change("POWER","Выход",.30)[0]
    assert not safe_change("CONTROL","PURPLE VIOLET",.30)[0]
    assert not safe_change("MAIN","MAIN",.30)[0]


def test_only_single_owner_updated_with_no_geometry_change():
    project=sample()
    old_box=project.objects[0].bbox.model_copy(deep=True)
    original_conf=project.fragments[0].confidence
    audit=apply_suggestions(project,[(project.fragments[0],False)],
                            {project.fragments[0].id:{"status":"ok","text":"MAIN","seconds":.8}})
    assert audit[0]["decision"]=="applied_requires_review"
    assert project.fragments[0].text==project.objects[0].text=="MAIN"
    assert project.objects[0].bbox==old_box
    assert project.fragments[0].confidence==original_conf
    assert project.objects[0].needs_review
    validate_project(project)


def test_table_cell_corrected_only_when_ocr_ownership_is_unambiguous():
    project=sample()
    first=project.objects.pop(0)
    frag=project.fragments[0]
    table=Table(id="new_table",bbox=first.bbox,rows=1,columns=1,
                row_heights=[first.bbox.h],col_widths=[first.bbox.w],
                cells=[Cell(row=0,col=0,text=frag.text,ocr_ids=[frag.id])])
    project.objects.append(table)
    result=apply_suggestions(project,[(frag,False)],
                             {frag.id:{"status":"ok","text":"MAIN","seconds":1.2}})
    assert result[0]["decision"]=="applied_requires_review"
    assert table.cells[0].text=="MAIN"
    validate_project(project)


class FakeVL:
    def __init__(self):
        self.calls=[]
    def predict(self, *, input, **kwargs):
        self.calls.append((input.shape,kwargs))
        return [SimpleNamespace(json={"res":{
            "parsing_res_list":[{"block_label":"text","block_content":"MAIN"}]}})]


def test_hybrid_worker_reads_only_selected_crops(tmp_path):
    image=Image.new("RGB",(600,200),"white")
    folder=tmp_path/"run"
    folder.mkdir()
    plan={"items":[{"id":"ocr_123","bbox":[30,20,120,45],"ocr_text":"MAlN"}]}
    (folder/"paddle-vl-hybrid-plan.json").write_text(json.dumps(plan),encoding="utf-8")
    model=FakeVL()
    process_hybrid(image,folder,"cpu",Settings(hybrid_max_checks=3),
                   pipeline_factory=lambda *,device,layout:model)
    saved=json.loads((folder/"hybrid-ocr_123.json").read_text("utf-8"))
    assert saved["status"]=="ok" and saved["text"]=="MAIN"
    assert len(model.calls)==1
    assert model.calls[0][0][0]<100
    assert model.calls[0][1]["use_layout_detection"] is False
    assert model.calls[0][1]["prompt_label"]=="ocr"


def test_hybrid_exposes_audit_and_never_uses_cloud(monkeypatch,tmp_path):
    project=sample()
    settings=Settings(ocr_engine="hybrid_local",hybrid_max_checks=1)
    def fake_isolated(source_path,settings,device,folder,**kwargs):
        plan=json.loads((Path(folder)/"paddle-vl-hybrid-plan.json").read_text("utf-8"))
        identifier=plan["items"][0]["id"]
        (Path(folder)/f"hybrid-{identifier}.json").write_text(
            json.dumps({"status":"ok","text":"MAIN","seconds":.5}),encoding="utf-8")
        return {"hybrid":True},{"hybrid":True}
    monkeypatch.setattr("smart_overlay.vl_tiled.run_isolated",fake_isolated)
    updated,warnings=selective_vl(project,"some.png",settings,
                                  diagnostics_dir=tmp_path/"diagnostics",
                                  progress=lambda _:None,cancelled=lambda:False)
    assert updated.objects[0].text=="MAIN"
    assert updated.raw["hybrid_local"]["selected"]==1
    assert updated.raw["hybrid_local"]["applied"]==1
    assert (tmp_path/"diagnostics"/"paddle-vl-hybrid-audit.json").exists()


def test_failed_vl_still_exports_original_ocr_pptx(monkeypatch,tmp_path):
    source=tmp_path/"input.png"
    source.write_bytes(b"mock-image")
    project=sample()
    monkeypatch.setattr("smart_overlay.one_click.load_image",
                        lambda path: project.model_copy(deep=True))
    monkeypatch.setattr("smart_overlay.one_click.local_ocr",
                        lambda *a,**k: project.model_copy(deep=True))
    monkeypatch.setattr("smart_overlay.one_click._smart",
                        lambda *a,**k: pytest.fail("OpenRouter must never run"))
    monkeypatch.setattr("smart_overlay.vl_hybrid.selective_vl",
                        lambda *a,**k: (_ for _ in ()).throw(RuntimeError("local VL timeout")))
    monkeypatch.setattr("smart_overlay.one_click.data_dir",lambda:tmp_path/"app")
    exported=[]
    def export(project,output,settings,*,allow_unreviewed,inline_text_fill):
        exported.append(project.model_copy(deep=True))
        Path(output).write_bytes(b"pptx fixture")
    monkeypatch.setattr("smart_overlay.one_click.export_pptx",export)
    converted=convert_drawing(source,settings=Settings(ocr_engine="hybrid_local"),
                              api_key="some_cloud_key",consent=True)
    assert converted.pptx.read_bytes()==b"pptx fixture"
    assert exported[0].objects[0].text=="MAlN"
    assert any("сохранён обычный OCR" in x for x in converted.warnings)


def test_hybrid_gui_option_available(gui_app,monkeypatch,tmp_path):
    from smart_overlay.quick_app import QuickWindow
    monkeypatch.setattr("smart_overlay.quick_app.load_settings",Settings)
    monkeypatch.setattr("smart_overlay.quick_app.data_dir",lambda:tmp_path)
    app=QuickWindow()
    try:
        index=app.mode_combo.findData("hybrid_local")
        assert index>=0
        app.mode_combo.setCurrentIndex(index)
        assert app.settings.ocr_engine=="hybrid_local"
        assert app.language_button.isEnabled()
    finally:
        app.close()
        app.deleteLater()
