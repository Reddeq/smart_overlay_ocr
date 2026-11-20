"""Cloud-only OCR integration tests: NO paid API calls, NO PaddleOCR downloads."""
import json
from pathlib import Path

from PIL import Image
import pytest

from smart_overlay.cloud_ocr import (
    CloudBlock, CloudReply, CloudOCRError, merge_blocks,
    prompt, recognize_cloud, payload,
)
from smart_overlay.input import load_image
from smart_overlay.one_click import convert_drawing
from smart_overlay.settings import Settings
from smart_overlay.vl_tiled import tile_bounds
from smart_overlay.models import Table, Text


@pytest.fixture
def image_path(tmp_path):
    path=tmp_path/"drawing.png"
    Image.new("RGB",(640,480),"white").save(path)
    return path


def answer(blocks):
    return {"choices":[{"finish_reason":"stop","message":{
        "content":json.dumps({"blocks":blocks},ensure_ascii=False)}}],
        "usage":{"prompt_tokens":150,"completion_tokens":250,"total_tokens":400}}


def test_cloud_payload_uses_raw_image_not_ocr_transcript(image_path):
    settings=Settings(cloud_ocr_model="google/gemini-2.5-pro")
    p=payload(settings,{"image":Image.open(image_path),"tile":{
        "x0":0,"y0":0,"x1":640,"y1":480}})
    assert p["model"] == "google/gemini-2.5-pro"
    assert p["response_format"] == {"type":"json_object"}
    parts=p["messages"][1]["content"]
    assert parts[1]["type"]=="image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "PRIMARY recognition directly from the IMAGE" in parts[0]["text"]


def test_single_cloud_request_produces_editable_text_and_merged_native_table(image_path,tmp_path):
    settings=Settings(ocr_engine="cloud_full")
    calls=[]
    def fake(settings,api_key,proxy,body,progress,cancelled,index,total):
        calls.append((settings.cloud_ocr_model,api_key,proxy,index,total))
        return answer([
            {"kind":"text","bbox":[20,20,150,45],"text":"Кассета 2"},
            {"kind":"table","bbox":[300,100,520,300],
             "html":"<table><tr><th colspan='2'>Конт.</th></tr>"
                    "<tr><td>0а.0в.0с</td><td>Цепь</td></tr></table>"}
        ])
    project=recognize_cloud(
        load_image(image_path),settings,api_key="test-key",consent=True,
        proxy="http://127.0.0.1:10809",diagnostics_dir=tmp_path/"diag",
        requester=fake)
    assert len(calls)==1
    assert project.source.width==640 and project.source.height==480
    assert any(isinstance(o,Text) and o.text=="Кассета 2" for o in project.objects)
    table=next(o for o in project.objects if isinstance(o,Table))
    assert table.cells[0].colspan==2
    assert table.cells[0].text=="Конт."
    assert project.raw["cloud_ocr"]["requests"]==1
    assert (tmp_path/"diag"/"cloud-ocr-merged.json").exists()
    assert project.objects[0].needs_review


def test_unconsented_cloud_never_contacts_provider(image_path,tmp_path):
    calls=[]
    def fake(*args):
        calls.append(1)
        return answer([])
    for key,consent in [("test-key",False),("",True)]:
        with pytest.raises(CloudOCRError):
            recognize_cloud(load_image(image_path),Settings(),api_key=key,
                            consent=consent,proxy=None,
                            diagnostics_dir=tmp_path/"diagnostics",
                            requester=fake)
    assert calls==[]


def test_bounded_tiles_offset_and_overlap_suppression():
    tiles=tile_bounds(1408,1024,size=1024,overlap=96,max_tiles=4)
    assert len(tiles)==2
    first,second=tiles
    assert first["x0"]==0 and second["x0"]>0
    replies=[
        CloudReply(blocks=[
            CloudBlock(kind="text",bbox=[100,50,260,80],text="Главный блок"),
            CloudBlock(kind="text",bbox=[850,80,920,103],text="XP1"),
        ]),
        CloudReply(blocks=[
            CloudBlock(kind="text",bbox=[850-second["x0"],80,
                                        920-second["x0"],103],text="XP1"),
            CloudBlock(kind="text",bbox=[700,400,790,430],text="Кассета 2")
        ])
    ]
    page,skipped=merge_blocks(tiles,replies,1408,1024)
    labels=[b["block_content"] for b in page["parsing_res_list"]]
    assert labels.count("XP1")==1
    assert labels.count("Кассета 2")==1
    assert any(s["reason"]=="duplicate_overlap" for s in skipped)
    assert any(b["block_bbox"][0]>1000 for b in page["parsing_res_list"])


def test_empty_or_truncated_cloud_response_does_not_create_pptx(image_path,monkeypatch,tmp_path):
    from smart_overlay.cloud_ocr import _request
    monkeypatch.setattr("smart_overlay.one_click.local_ocr",
                        lambda *a,**k: pytest.fail("PaddleOCR must NOT run"))
    monkeypatch.setattr("smart_overlay.one_click._smart",
                        lambda *a,**k: pytest.fail("Old cloud grouping must NOT run"))
    monkeypatch.setattr("smart_overlay.one_click.export_pptx",
                        lambda *a,**k: pytest.fail("Do not export bad cloud response"))
    monkeypatch.setattr("smart_overlay.cloud_ocr._request",
                        lambda *args,**kwargs: answer([]))
    with pytest.raises(CloudOCRError,match="не вернуло"):
        convert_drawing(image_path,settings=Settings(ocr_engine="cloud_full"),
                        api_key="test-key",consent=True,
                        vl_diagnostics=tmp_path/"diag-empty")
    assert not image_path.with_suffix(".pptx").exists()


def test_full_cloud_one_click_never_invokes_local_paddle_or_old_cloud(image_path,monkeypatch,tmp_path):
    monkeypatch.setattr("smart_overlay.one_click.local_ocr",
                        lambda *a,**kw: pytest.fail("Local OCR must not run"))
    monkeypatch.setattr("smart_overlay.one_click._smart",
                        lambda *a,**kw: pytest.fail("Legacy cloud/grouping must not run"))
    monkeypatch.setattr("smart_overlay.cloud_ocr._request",
                        lambda *args,**kwargs: answer([
                            {"kind":"text","bbox":[20,20,200,45],
                             "text":"Основной блок"}
                        ]))
    monkeypatch.setattr("smart_overlay.one_click.data_dir",
                        lambda:tmp_path/"app")
    exported=[]
    def export(p,output,settings,*,allow_unreviewed,inline_text_fill):
        exported.append(p)
        Path(output).write_bytes(b"mock pptx")
    monkeypatch.setattr("smart_overlay.one_click.export_pptx",export)
    result=convert_drawing(
        image_path,settings=Settings(ocr_engine="cloud_full"),api_key="test-key",
        consent=True,vl_diagnostics=tmp_path/"diag")
    assert result.pptx.read_bytes()==b"mock pptx"
    assert any(o.text=="Основной блок" for o in exported[0].objects if isinstance(o,Text))
    assert exported[0].raw["cloud_ocr"]["model"]=="google/gemini-2.5-pro"


def test_cloud_cancel_prevents_any_pptx(image_path,monkeypatch,tmp_path):
    from smart_overlay.one_click import ProcessingCancelled
    monkeypatch.setattr("smart_overlay.cloud_ocr._request",
                        lambda *a,**k: pytest.fail("No request after cancellation"))
    with pytest.raises(ProcessingCancelled):
        convert_drawing(image_path,settings=Settings(ocr_engine="cloud_full"),
                        consent=True,api_key="test-key",cancelled=lambda:True)
    assert not image_path.with_suffix(".pptx").exists()


def test_unknown_or_out_of_bounds_blocks_get_rejected():
    tiles=tile_bounds(640,480,size=1024,overlap=96)
    reply=CloudReply(blocks=[
        CloudBlock(kind="text",bbox=[-5,10,120,30],text="BAD"),
        CloudBlock(kind="image",bbox=[10,20,50,50],text="GRAPHIC"),
        CloudBlock(kind="text",bbox=[10,10,100,35],text="OK"),
    ])
    page,skipped=merge_blocks(tiles,[reply],640,480)
    assert [p["block_content"] for p in page["parsing_res_list"]]==["OK"]
    assert len(skipped)==2
