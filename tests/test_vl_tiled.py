"""Offline tests for bounded local VL tiling and checkpoint protocol."""
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pytest

from smart_overlay.settings import Settings
from smart_overlay.vl_tiled import tile_bounds, tile_has_ink, merge_tile_pages
from smart_overlay.vl_worker import process


def test_tile_bounds_cover_exact_source_without_duplicate_cores():
    tiles = tile_bounds(2000, 1150, size=896, overlap=96)
    assert len(tiles) > 1
    assert tiles[0]["x0"] == tiles[0]["core_x0"] == 0
    assert max(t["x1"] for t in tiles) == 2000
    assert max(t["y1"] for t in tiles) == 1150
    # At least one tile accepts any image-space coordinate, at most one accepts
    # an interior coordinate not lying exactly on a shared partition boundary.
    for x,y in ((10,10), (850,50), (1200,500), (1999,1149)):
        owners = [t for t in tiles if (t["core_x0"] < x < t["core_x1"] or
                       (x == 0 and t["core_x0"] == 0)) and
                       (t["core_y0"] < y < t["core_y1"] or
                       (y == 0 and t["core_y0"] == 0))]
        assert len(owners) == 1
    with pytest.raises(ValueError, match="Недопустимые"):
        tile_bounds(1000,1000,size=384,overlap=200)
    with pytest.raises(Exception, match="лимит"):
        tile_bounds(10000,10000,size=384,overlap=32,max_tiles=5)


def test_blank_tile_skipped_but_sparse_text_tile_kept():
    image = Image.new("RGB", (640, 480), "white")
    box = {"x0":0, "y0":0, "x1":640, "y1":480}
    assert not tile_has_ink(image, box)
    draw = ImageDraw.Draw(image)
    draw.rectangle((100,100,220,120), fill="black")
    assert tile_has_ink(image, box)


def test_tile_merger_retains_original_pixel_coordinates_and_deduplicates(tmp_path):
    tiles = tile_bounds(1500, 800, size=896, overlap=96)
    def fake_record(t, entries):
        return {"status": "ok", "page": {
            "width": t["x1"]-t["x0"], "height": t["y1"]-t["y0"],
            "parsing_res_list": entries}}
    # Two copies of a label inside a shared overlap. Its center belongs to
    # one core; only that copy should survive.
    for tile in tiles:
        copies = []
        if tile["x0"] <= 850 < tile["x1"]:
            copies.append({
                "block_label":"text",
                "block_bbox":[850-tile["x0"],100,890-tile["x0"],128],
                "block_content":"CONTROL"})
        path=tmp_path/f"tile-{tile['index']:04d}.json"
        path.write_text(json.dumps(fake_record(tile,copies)),encoding="utf-8")
    page, stats = merge_tile_pages(tiles,tmp_path,1500,800)
    assert len(page["parsing_res_list"]) == 1
    block = page["parsing_res_list"][0]
    assert block["block_bbox"] == [850,100,890,128]
    assert stats["finished"] == len(tiles)
    assert not stats["missing_or_failed"]


class FakePipeline:
    def __init__(self):
        self.calls=[]

    def predict(self, *, input, **kwargs):
        self.calls.append((input.shape,kwargs))
        return [SimpleNamespace(json={"res":{
            "width":input.shape[1],
            "height":input.shape[0],
            "parsing_res_list":[
                {"block_label":"text","block_bbox":[40,40,190,80],
                 "block_content":"POWER SUPPLY"}
            ]}})]


def test_worker_saves_manifest_and_each_tile_before_completion(tmp_path):
    source=tmp_path/"drawing.png"
    image=Image.new("RGB",(950,600),"white")
    ImageDraw.Draw(image).text((50,50),"POWER SUPPLY",fill="black")
    ImageDraw.Draw(image).rectangle((650,300,800,320),fill="black")
    image.save(source)
    fake=FakePipeline()
    settings=Settings(vl_tile_size=640,vl_tile_overlap=64)
    out=tmp_path/"diagnostics"
    process(source,out,"cpu",settings,pipeline_factory=lambda *,device:fake)
    assert (out/"paddle-vl-tiles.json").exists()
    manifest=json.loads((out/"paddle-vl-tiles.json").read_text("utf-8"))
    assert len(manifest["tiles"])>1
    assert len(list(out.glob("tile-*.json")))==len(manifest["tiles"])
    assert len(fake.calls)>=1
    assert all(c[1]["max_new_tokens"]==settings.vl_max_new_tokens for c in fake.calls)
    for file in out.glob("tile-*.json"):
        assert json.loads(file.read_text("utf-8"))["status"] in ("ok","skipped_blank")
