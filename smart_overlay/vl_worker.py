"""Isolated fully local PaddleOCR-VL worker.

Default: geometry-only TextDetection -> read small measured line crops with
PaddleOCR-VL (no page layout) -> checkpoint every region. Benchmark mode reads
one representative region only. The legacy tiled layout mode remains opt-in.
"""
from __future__ import annotations

import json
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


def process_tiled(source, directory, device, settings, *, pipeline_factory):
    """Existing page-layout based fallback; no new behavior for legacy tests."""
    tiles = tile_bounds(source.width, source.height,
                        settings.vl_tile_size, settings.vl_tile_overlap,
                        max_tiles=settings.vl_max_tiles)
    manifest = {"version": VERSION, "width": source.width, "height": source.height,
                "tiles": tiles, "device": device, "strategy": "tiles",
                "limits": {"max_new_tokens": settings.vl_max_new_tokens}}
    write_json(directory / "paddle-vl-tiles.json", manifest)
    emit("loading", total=len(tiles))
    pipeline = None
    for tile in tiles:
        index = tile["index"]
        destination = directory / f"tile-{index:04d}.json"
        started = time.monotonic()
        has_ink = tile_has_ink(source, tile)
        if has_ink and pipeline is None:
            pipeline = pipeline_factory(device=device)
            emit("started", total=len(tiles))
        emit("tile", index=index, total=len(tiles))
        if not has_ink:
            write_json(destination, {"status": "skipped_blank", "tile": index})
        else:
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


def process_detected(source, directory, device, settings, *,
                     detector_factory=None, pipeline_factory=None, benchmark=False):
    """Detect all text geometry once; no conventional text recognizer is used."""
    from .vl_detect import (default_detector, detect_lines, crop_region,
                            extract_text, DETECTOR_MODEL)
    detector_factory = detector_factory or default_detector
    pipeline_factory = pipeline_factory or default_pipeline
    start = time.monotonic()
    emit("detecting")
    detector = detector_factory(device=device)
    detection_loaded = time.monotonic()
    # Keep a large safety ceiling during detection so that the manifest can
    # record counts even when the selected maximum is exceeded.
    regions, rejected = detect_lines(source, detector, max_regions=5000)
    detected = time.monotonic()
    manifest = {"version": VERSION, "strategy": "detect", "width": source.width,
                "height": source.height, "device": device, "regions": regions,
                "rejected": rejected,
                "limits": {"max_regions": settings.vl_max_regions,
                           "max_new_tokens": min(128, settings.vl_max_new_tokens)}}
    write_json(directory / "paddle-vl-regions.json", manifest)
    emit("detected", total=len(regions), detector_seconds=round(detected-start, 2))
    if not regions:
        raise RuntimeError("TextDetection не обнаружил ни одной текстовой области")
    if len(regions) > settings.vl_max_regions and not benchmark:
        raise RuntimeError(
            f"Обнаружено {len(regions)} строк при лимите {settings.vl_max_regions}. "
            "См. paddle-vl-regions.json; нельзя молча пропускать надписи")
    if benchmark:
        # Keep benchmark representative but bounded: one confident, visible line.
        regions = [max(regions, key=lambda x: (x["score"],
                   min(7000,(x["bbox"][2]-x["bbox"][0])*
                            (x["bbox"][3]-x["bbox"][1]))))]
    count = len(regions)
    emit("loading", total=count)
    load_start = time.monotonic()
    pipeline = pipeline_factory(device=device, layout=False)
    load_end = time.monotonic()
    emit("started", total=count, model_load_seconds=round(load_end-load_start,2))
    token_cap = min(96 if benchmark else 128, settings.vl_max_new_tokens)
    recognized = unreadable = 0
    for i, region in enumerate(regions):
        stamp = time.monotonic()
        emit("region", index=i, total=count, region_id=region["id"])
        crop = crop_region(source, region)
        bgr = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)
        # No PP-DocLayoutV3 on a text line. 'ocr' is the documented
        # prompt_label for no-layout PaddleOCR-VL.
        pages = list(pipeline.predict(
            input=bgr, use_layout_detection=False,
            prompt_label="ocr", format_block_content=False,
            max_new_tokens=token_cap))
        if len(pages) != 1:
            raise RuntimeError("PaddleOCR-VL не вернул одну страницу для строки")
        raw = _native_json(_as_plain_dict(pages[0]))
        text = extract_text(raw)
        seconds = round(time.monotonic()-stamp,3)
        record = {"status":"ok" if text else "unreadable",
                  "id":region["id"],"text":text,"seconds":seconds,
                  "page":raw,"region":region}
        write_json(directory/f"region-{region['id']:04d}.json", record)
        if text:
            recognized += 1
        else:
            unreadable += 1
        emit("checkpoint", done=i+1,total=count,seconds=seconds,
             readable=recognized)
        if benchmark:
            stats = {
                "strategy":"detect", "device":device,
                "detector_model":DETECTOR_MODEL,
                "detector_load_seconds":round(detection_loaded-start,3),
                "detection_seconds":round(detected-detection_loaded,3),
                "vl_model_load_seconds":round(load_end-load_start,3),
                "vl_inference_seconds":seconds,
                "benchmark_region_id":region["id"],
                "region_pixels":[crop.width,crop.height],
                "detected_regions":len(manifest["regions"]),
                "recognized":bool(text),
                "max_new_tokens":token_cap
            }
            write_json(directory/"paddle-vl-benchmark.json",stats)
            emit("benchmark_complete",report=stats)
            return
        if i == 0 and seconds > settings.vl_preflight_seconds:
            raise RuntimeError(
                f"Первый VL-фрагмент занял {seconds:.1f} с: "
                f"превышен порог {settings.vl_preflight_seconds} с. "
                "CPU слишком медленный для полного распознавания, "
                "первый результат сохранён")
    emit("complete", total=count, recognized=recognized, unreadable=unreadable)


def process(source_path, directory, device, settings, *,
            pipeline_factory=None, detector_factory=None, benchmark=False):
    project = load_image(source_path)
    source = source_image(project)
    directory=Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if settings.vl_strategy == "detect" or benchmark:
        process_detected(source, directory, device, settings, benchmark=benchmark,
                         detector_factory=detector_factory,
                         pipeline_factory=pipeline_factory)
    else:
        process_tiled(source,directory,device,settings,
                      pipeline_factory=pipeline_factory or default_pipeline)


def main():
    if len(sys.argv) not in (5,6):
        sys.stderr.write("Usage: python -m smart_overlay.vl_worker image dir device settings.json [benchmark]\n")
        return 2
    image,folder,device,settings_file=sys.argv[1:5]
    benchmark = len(sys.argv) == 6 and sys.argv[5] == "benchmark"
    try:
        settings=Settings.model_validate_json(Path(settings_file).read_text("utf-8"))
        process(image,folder,device,settings,benchmark=benchmark)
        return 0
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        emit("error",message=f"{type(exc).__name__}: {str(exc)[:200]}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
