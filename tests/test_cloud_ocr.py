"""Cloud-only OCR integration tests: NO paid API calls, NO PaddleOCR downloads."""
import json
from pathlib import Path

from PIL import Image
import pytest

from smart_overlay.cloud_ocr import (
    CloudBlock, CloudReply, CloudOCRError, merge_blocks,
    prompt, recognize_cloud, payload, balanced_cloud_tiles, decode_reply,
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
        "content":json.dumps({"coordinate_system":"pixel_xyxy","blocks":blocks},ensure_ascii=False)}}],
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
    assert "NORMALIZED 0-1000 XYXY" in parts[0]["text"]
    assert p["reasoning"] == {"max_tokens": 1024}


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


def test_regression_1408x1024_cloud_no_640px_duplicate_overlap():
    """The exact failed user's drawing formerly produced 2 giant 1024px tiles."""
    tiles=balanced_cloud_tiles(1408,1024,size=640,overlap=48,max_tiles=8)
    assert len(tiles)==6
    assert {(t["x1"]-t["x0"],t["y1"]-t["y0"]) for t in tiles}=={(502,536)}
    row=sorted((t for t in tiles if t["y0"]==0),key=lambda t:t["x0"])
    col=sorted((t for t in tiles if t["x0"]==0),key=lambda t:t["y0"])
    assert 47<=row[0]["x1"]-row[1]["x0"]<=49
    assert col[0]["y1"]-col[1]["y0"]==48
    assert row[0]["core_x1"]==row[1]["core_x0"]
    assert col[0]["core_y1"]==col[1]["core_y0"]==512
    assert all(max(t["x1"]-t["x0"],t["y1"]-t["y0"])<=640 for t in tiles)
    with pytest.raises(CloudOCRError,match="6 облачных"):
        balanced_cloud_tiles(1408,1024,size=640,overlap=48,max_tiles=2)


def test_balanced_tiling_covers_all_pixels_with_small_overlap():
    for width,height in [(600,400),(1408,1024),(2048,2048),(2476,1100),(1920,1080)]:
        tiles=balanced_cloud_tiles(width,height,size=768,overlap=64,max_tiles=32)
        assert all((t["x1"]-t["x0"]<=768 and t["y1"]-t["y0"]<=768) for t in tiles)
        assert max(t["x1"] for t in tiles)==width
        assert max(t["y1"] for t in tiles)==height
        for x,y in [(0,0),(width-1,height-1),(width//3,height//2)]:
            owners=[t for t in tiles if
                    t["core_x0"]<=x<t["core_x1"] and
                    t["core_y0"]<=y<t["core_y1"]]
            assert len(owners)==1


def test_accept_compact_six_element_cloud_ocr_json():
    body=json.dumps({"blocks":[
        ["t",20,30,130,55,"Кассета 2"],
        ["h",200,110,420,300,"<table><tr><td>Конт.</td></tr></table>"]
    ]},ensure_ascii=False)
    parsed=decode_reply(body)
    assert parsed.blocks[0].kind=="text"
    assert parsed.blocks[0].text=="Кассета 2"
    assert parsed.blocks[1].kind=="table"
    assert parsed.blocks[1].html.startswith("<table>")
    with pytest.raises(ValueError,match="неверный компактный блок"):
        decode_reply('{"blocks":[["t",12,24]]}')


def test_reasoning_budget_only_for_supported_gemini_25(image_path):
    img=Image.open(image_path)
    tile={"x0":0,"y0":0,"x1":640,"y1":480}
    settings=Settings(cloud_ocr_model="anthropic/claude-sonnet-4")
    result=payload(settings,{"image":img,"tile":tile})
    assert "reasoning" not in result
    with pytest.raises(ValueError,match="Бюджет рассуждений"):
        Settings(cloud_ocr_max_tokens=1024,cloud_ocr_reasoning_tokens=1024)


def test_truncated_cloud_json_does_not_retry_and_records_usage(image_path,tmp_path):
    called=[]
    def truncated(*args):
        called.append(True)
        return {"choices":[{"finish_reason":"length",
                            "message":{"content":""}}],
                "usage":{"prompt_tokens":1020,"completion_tokens":8192,
                         "total_tokens":9212,
                         "completion_tokens_details":{"reasoning_tokens":7000},
                         "cost":0.05}}
    folder=tmp_path/"failed"
    with pytest.raises(CloudOCRError,match="finish_reason=length"):
        recognize_cloud(load_image(image_path),Settings(),
                        api_key="test-key",consent=True,proxy=None,
                        diagnostics_dir=folder,requester=truncated)
    assert len(called)==1
    report=json.loads((folder/"cloud-ocr-error.json").read_text("utf-8"))
    assert report["kind"]=="TruncatedResponse"
    assert report["finish_reason"]=="length"
    assert report["usage"]["completion_tokens_details"]["reasoning_tokens"]==7000
    assert report["automatic_retry"] is False
    assert not (folder/"cloud-ocr-merged.json").exists()


def test_upgrade_previous_saved_cloud_crop_defaults(tmp_path,monkeypatch):
    from smart_overlay.settings import load_settings
    monkeypatch.setenv("LOCALAPPDATA",str(tmp_path))
    base=Settings().model_dump()
    base.pop("cloud_ocr_profile_version")
    base["cloud_ocr_tile_size"]=1024
    base["cloud_ocr_overlap"]=96
    folder=tmp_path/"SmartOverlayOCR"
    folder.mkdir()
    (folder/"settings.json").write_text(json.dumps(base),encoding="utf-8")
    actual=load_settings()
    assert actual.cloud_ocr_tile_size==640
    assert actual.cloud_ocr_overlap==48
    assert actual.cloud_ocr_profile_version==3


def test_gemini_normalized_coordinates_are_scaled_before_pptx():
    """Synthetic version of the reported out-of-image XYXY error."""
    tiles=balanced_cloud_tiles(736,544,size=640,overlap=48,max_tiles=4)
    # Keep this fixture single-tile to verify exact recovered positions.
    tile={"index":0,"x0":0,"y0":0,"x1":736,"y1":544,
          "core_x0":0,"core_x1":736,"core_y0":0,"core_y1":544}
    reply=CloudReply(
        coordinate_system="normalized_1000_xyxy",
        blocks=[
            CloudBlock(kind="text",bbox=[40,280,103,296],text="МПРК-3"),
            CloudBlock(kind="text",bbox=[136,724,304,828],text="Преобразователь")
        ])
    page,skipped=merge_blocks([tile],[reply],736,544)
    assert not skipped
    assert len(page["parsing_res_list"])==2
    first=page["parsing_res_list"][0]["block_bbox"]
    second=page["parsing_res_list"][1]["block_bbox"]
    assert first==pytest.approx([29.44,152.32,75.808,161.024])
    assert second==pytest.approx([100.096,393.856,223.744,450.432])


def test_legacy_gemini_reply_with_many_out_of_range_pixels_infers_normalized_scale():
    """22 of 32 uploaded user blocks had y>544, but all fit 0..1000."""
    from smart_overlay.cloud_ocr import _reply_coordinates
    tile={"x0":0,"y0":0,"x1":736,"y1":544}
    bbox=[[40,280,103,296],[136,724,304,828],[60,624,100,640],
          [484,844,524,860]]
    reply=CloudReply(blocks=[
        CloudBlock(kind="text",bbox=b,text="TEST") for b in bbox
    ])
    assert _reply_coordinates(reply,tile,"pixel_xyxy")=="normalized_1000_xyxy"
    page,skipped=merge_blocks(
        [{**tile,"index":0,"core_x0":0,"core_y0":0,
          "core_x1":736,"core_y1":544}], [reply],736,544)
    assert len(page["parsing_res_list"])==4
    assert not skipped


def test_pixel_coordinate_response_remains_unchanged_with_explicit_metadata():
    tile={"x0":0,"y0":0,"x1":736,"y1":544,"index":0,
          "core_x0":0,"core_y0":0,"core_x1":736,"core_y1":544}
    reply=decode_reply('{"coordinate_system":"pixel_xyxy",'
                       '"blocks":[["t",10,20,100,35,"Кассета"]]}')
    page,skipped=merge_blocks([tile],[reply],736,544,
                              fallback_coordinate_system="normalized_1000_xyxy")
    assert not skipped
    assert page["parsing_res_list"][0]["block_bbox"]==[10,20,100,35]


def test_saved_cloud_defaults_v2_migrate_to_smaller_tiles(tmp_path,monkeypatch):
    from smart_overlay.settings import load_settings
    monkeypatch.setenv("LOCALAPPDATA",str(tmp_path))
    base=Settings().model_dump()
    base["cloud_ocr_profile_version"]=2
    base["cloud_ocr_tile_size"]=768
    base["cloud_ocr_overlap"]=64
    folder=tmp_path/"SmartOverlayOCR"
    folder.mkdir()
    (folder/"settings.json").write_text(json.dumps(base),encoding="utf-8")
    actual=load_settings()
    assert actual.cloud_ocr_tile_size==640
    assert actual.cloud_ocr_overlap==48
    assert actual.cloud_ocr_profile_version==3


def test_manual_nondefault_cloud_tiles_are_never_overridden(tmp_path,monkeypatch):
    from smart_overlay.settings import load_settings
    monkeypatch.setenv("LOCALAPPDATA",str(tmp_path))
    base=Settings().model_dump()
    base["cloud_ocr_profile_version"]=2
    base["cloud_ocr_tile_size"]=1300
    base["cloud_ocr_overlap"]=50
    folder=tmp_path/"SmartOverlayOCR"
    folder.mkdir()
    (folder/"settings.json").write_text(json.dumps(base),encoding="utf-8")
    actual=load_settings()
    assert (actual.cloud_ocr_tile_size,actual.cloud_ocr_overlap)==(1300,50)
