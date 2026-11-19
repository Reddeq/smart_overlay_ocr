"""Child process for local PaddleOCR-VL: one warm model, bounded tiles.

Run only through vl_tiled.run_isolated. stdout is a line-oriented JSON event
stream; individual results are atomically saved immediately after predict().
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np

from .input import load_image, source_image
from .settings import Settings
from .vl_local import VERSION, _as_plain_dict, _native_json, default_pipeline
from .vl_tiled import tile_bounds, tile_has_ink


def emit(event, **data):
    print(json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)


def write_json(path, data):
    file = Path(path)
    temp = file.with_name(file.name + ".part")
    temp.write_text(json.dumps(_native_json(data), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    temp.replace(file)


def process(source_path, directory, device, settings, *, pipeline_factory=default_pipeline):
    project = load_image(source_path)
    source = source_image(project)
    tiles = tile_bounds(source.width, source.height,
                        settings.vl_tile_size, settings.vl_tile_overlap,
                        max_tiles=settings.vl_max_tiles)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"version": VERSION, "width": source.width, "height": source.height,
                "tiles": tiles, "device": device,
                "limits": {"max_new_tokens": settings.vl_max_new_tokens}}
    write_json(directory / "paddle-vl-tiles.json", manifest)
    emit("loading", total=len(tiles))
    # One model instance per job, not per tile, to avoid repeated heavy loads.
    pipeline = None
    for tile in tiles:
        index = tile["index"]
        destination = directory / f"tile-{index:04d}.json"
        started = time.monotonic()
        emit("tile", index=index, total=len(tiles))
        if not tile_has_ink(source, tile):
            write_json(destination, {"status": "skipped_blank", "tile": index})
        else:
            if pipeline is None:
                pipeline = pipeline_factory(device=device)
                emit("started", total=len(tiles))
                emit("tile", index=index, total=len(tiles))
            crop = source.crop((tile["x0"], tile["y0"], tile["x1"], tile["y1"]))
            image = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)
            pages = list(pipeline.predict(
                input=image, layout_shape_mode="rect",
                format_block_content=False,
                max_new_tokens=settings.vl_max_new_tokens))
            if len(pages) != 1:
                raise RuntimeError(f"Expected one VL page in tile {index}, got {len(pages)}")
            page = _native_json(_as_plain_dict(pages[0]))
            write_json(destination, {"status": "ok", "tile": index, "page": page})
        emit("checkpoint", done=index+1, total=len(tiles),
             seconds=round(time.monotonic()-started, 2))
    emit("complete", total=len(tiles))


def main():
    if len(sys.argv) != 5:
        sys.stderr.write("Usage: python -m smart_overlay.vl_worker image dir device settings.json\n")
        return 2
    image, folder, device, settings_file = sys.argv[1:]
    try:
        settings = Settings.model_validate_json(Path(settings_file).read_text("utf-8"))
        process(image, folder, device, settings)
        return 0
    except BaseException as exc:
        # Detailed local traceback is private; UI only receives the type.
        traceback.print_exc(file=sys.stderr)
        emit("error", message=f"{type(exc).__name__}: {str(exc)[:160]}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
