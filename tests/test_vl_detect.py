"""Detector -> local VL OCR tests: geometry, checkpoints, short benchmark.

No actual models or network in these tests.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import pytest

from smart_overlay.settings import Settings
from smart_overlay.input import load_image
from smart_overlay.vl_detect import detect_lines, extract_text, detected_page, crop_region
from smart_overlay.vl_local import parse_page
from smart_overlay.vl_worker import process


class FakeDetector:
    def __init__(self, polygons=None, scores=None):
        self.polygons = polygons if polygons is not None else [
            [[15,20],[130,20],[130,44],[15,44]],
            [[15,70],[170,70],[170,94],[15,94]]
        ]
        self.scores = scores if scores is not None else [.95, .93]
        self.calls = 0

    def predict(self, *, input):
        self.calls += 1
        return [SimpleNamespace(json={"res":{
            "dt_polys": np.array(self.polygons,dtype=np.int16),
            "dt_scores": self.scores,
        }})]


class FakeVL:
    def __init__(self, text="POWER SUPPLY"):
        self.calls=[]
        self.text=text

    def predict(self, *, input, **kwargs):
        self.calls.append((input.shape,kwargs))
        return [SimpleNamespace(json={"res":{
            "parsing_res_list":[
                {"block_label":"text", "block_content":self.text,
                 "block_bbox":[0,0,input.shape[1],input.shape[0]]}
            ]
        }})]


@pytest.fixture
def drawing(tmp_path):
    path=tmp_path/"drawing.png"
    image=Image.new("RGB",(320,180),"white")
    draw=ImageDraw.Draw(image)
    draw.text((15,20),"POWER SUPPLY",fill="black")
    draw.text((15,70),"CONTROL",fill="black")
    image.save(path)
    return path


def test_detectors_return_original_polygons_without_ocr(drawing):
    image=Image.open(drawing)
    detector=FakeDetector()
    lines,rejected=detect_lines(image,detector)
    assert detector.calls==1
    assert len(lines)==2
    assert not rejected
    assert lines[0]["polygon"]==[[15.,20.],[130.,20.],[130.,44.],[15.,44.]]
    assert lines[0]["bbox"]==[15.,20.,130.,44.]
    assert crop_region(image,lines[0]).size == (131,40)
    assert lines[0]["score"]==pytest.approx(.95)


def test_low_confidence_dedup_and_excess_are_recorded(drawing):
    image=Image.open(drawing)
    detector=FakeDetector(polygons=[
        [[15,20],[130,20],[130,44],[15,44]],
        [[16,20],[129,20],[129,44],[16,44]],
        [[20,70],[80,70],[80,94],[20,94]]
    ],scores=[.95,.9,.1])
    lines,rejected=detect_lines(image,detector)
    assert len(lines)==1
    assert sorted(x["reason"] for x in rejected)==[
        "low_detection_confidence","near-duplicate"]
    with pytest.raises(Exception,match="лимит|разрешено"):
        detect_lines(image,FakeDetector(),max_regions=1)


def test_explicit_vl_text_only_no_hallucinated_coordinates():
    assert extract_text({"parsing_res_list":[{"block_label":"text",
                                               "block_content":"Питание 24V"}]})=="Питание 24V"
    assert extract_text({"rec_text":"RESET"})=="RESET"
    assert extract_text({"parsing_res_list":[]})==""
    assert extract_text({"metadata":{"content":"inferred text"}})==""


def test_detector_worker_recognizes_crops_without_layout_and_creates_editable_project(drawing,tmp_path):
    detector,model=FakeDetector(),FakeVL()
    dest=tmp_path/"run"
    settings=Settings(vl_strategy="detect",vl_max_regions=4)
    process(drawing,dest,"cpu",settings,
            detector_factory=lambda *,device:detector,
            pipeline_factory=lambda *,device,layout:model)
    assert detector.calls==1
    assert len(model.calls)==2
    for shape,kwargs in model.calls:
        assert shape[0]<100 and shape[1]<200
        assert kwargs["use_layout_detection"] is False
        assert kwargs["prompt_label"]=="ocr"
        assert kwargs["max_new_tokens"]==128
    manifest=json.loads((dest/"paddle-vl-regions.json").read_text("utf-8"))
    assert len(manifest["regions"])==2
    assert (dest/"region-0000.json").exists()
    assert (dest/"region-0001.json").exists()
    parsed,audit=detected_page(manifest,dest)
    assert audit["recognized"]==2
    assert parsed["_vl_detector_geometry"]
    assert parsed["parsing_res_list"][0]["block_bbox"]==[15.,20.,130.,44.]
    result=parse_page(load_image(drawing),settings,parsed)
    assert len(result.fragments)==2
    assert result.fragments[0].polygon == [(15.,20.),(130.,20.),(130.,44.),(15.,44.)]
    assert all(o.needs_review for o in result.objects)


def test_benchmark_only_one_line_and_reports_model_vs_inference_timings(drawing,tmp_path):
    model=FakeVL()
    directory=tmp_path/"bench"
    process(drawing,directory,"cpu",Settings(),
            detector_factory=lambda *,device:FakeDetector(),
            pipeline_factory=lambda *,device,layout:model,
            benchmark=True)
    assert len(model.calls)==1
    assert model.calls[0][1]["max_new_tokens"]==96
    report=json.loads((directory/"paddle-vl-benchmark.json").read_text("utf-8"))
    assert report["detected_regions"]==2
    assert report["vl_inference_seconds"]>=0
    assert report["vl_model_load_seconds"]>=0
    assert report["max_new_tokens"]==96
    assert len(list(directory.glob("region-*.json")))==1


def test_unreadable_vl_text_keeps_region_visible_in_diagnostics(drawing,tmp_path):
    model=FakeVL(text="")
    directory=tmp_path/"bad"
    process(drawing,directory,"cpu",Settings(),
            detector_factory=lambda *,device:FakeDetector(),
            pipeline_factory=lambda *,device,layout:model)
    plan=json.loads((directory/"paddle-vl-regions.json").read_text("utf-8"))
    page,audit=detected_page(plan,directory)
    assert page["parsing_res_list"]==[]
    assert len(audit["unreadable"])==2
