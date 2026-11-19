"""Bounded on-device VL tiling and interruptible worker protocol.

Workers write per-tile JSON checkpoints immediately, so no document-wide
predict() can silently consume 15+ minutes without visible progress.
The worker is a *local* Python child process; it never invokes a web API.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Thread
import time

from .vl_local import LocalVLError


def tile_bounds(width: int, height: int, size=896, overlap=96, *, max_tiles=48):
    if size < 256 or overlap < 0 or overlap * 2 >= size:
        raise ValueError("Недопустимые параметры сетки PaddleOCR-VL")
    if max_tiles < 1:
        raise ValueError("Недопустимый лимит областей")
    def starts(total):
        if total <= size:
            return [0]
        step = size - overlap
        result = list(range(0, total - size + 1, step))
        if not result or result[-1] != total - size:
            result.append(total-size)
        return result
    xs, ys = starts(width), starts(height)
    if len(xs)*len(ys) > max_tiles:
        raise LocalVLError(
            f"Требуется {len(xs)*len(ys)} участков (лимит {max_tiles}). "
            "Увеличьте размер участка или разделите лист вручную.")
    # Partition overlapping stripes at the true midpoint between adjacent
    # tile edges. The last tile can overlap more than the requested overlap.
    xcores = [(0 if i == 0 else (xs[i-1]+size+xs[i])/2,
               width if i == len(xs)-1 else (xs[i]+size+xs[i+1])/2)
              for i in range(len(xs))]
    ycores = [(0 if i == 0 else (ys[i-1]+size+ys[i])/2,
               height if i == len(ys)-1 else (ys[i]+size+ys[i+1])/2)
              for i in range(len(ys))]
    tiles = []
    for yi,y in enumerate(ys):
        for xi,x in enumerate(xs):
            tiles.append({
                "index": len(tiles), "total": len(xs)*len(ys),
                "x0": x, "y0": y, "x1": min(width,x+size),
                "y1": min(height,y+size),
                "core_x0": xcores[xi][0], "core_x1": xcores[xi][1],
                "core_y0": ycores[yi][0], "core_y1": ycores[yi][1],
            })
    return tiles


def tile_has_ink(image, box):
    """Skip truly blank white regions; graphics are *not* assumed to be text."""
    import numpy as np
    patch = np.asarray(image.crop((box["x0"],box["y0"],box["x1"],box["y1"])).convert("L"))
    return bool(patch.size and (patch < 235).sum() > max(12, patch.size * .0002))


def merge_tile_pages(tiles, checkpoint_dir, width, height):
    """Put original-pixel bboxes on one page, reject overlap duplicates.

    A box's center must fall in exactly one tile's non-overlap core, preventing
    two presentations of the same label from becoming duplicate PPTX shapes.
    Geometry near seams remains a review concern, not an invented correction.
    """
    skipped, errors, accepted = [], [], []
    finished = 0
    for tile in tiles:
        filename = Path(checkpoint_dir) / f"tile-{tile['index']:04d}.json"
        if not filename.is_file():
            errors.append({"tile": tile["index"], "reason": "missing checkpoint"})
            continue
        record = json.loads(filename.read_text("utf-8"))
        if record.get("status") == "skipped_blank":
            finished += 1
            continue
        if record.get("status") != "ok":
            errors.append({"tile": tile["index"], "reason": str(record.get("error", "failed"))})
            continue
        page = record.get("page", {})
        if page.get("width") is not None and int(page["width"]) != tile["x1"]-tile["x0"]:
            errors.append({"tile": tile["index"], "reason": "width mismatch"})
            continue
        if page.get("height") is not None and int(page["height"]) != tile["y1"]-tile["y0"]:
            errors.append({"tile": tile["index"], "reason": "height mismatch"})
            continue
        finished += 1
        for index, block in enumerate(page.get("parsing_res_list", [])):
            if not isinstance(block, dict):
                continue
            raw = block.get("block_bbox")
            if not isinstance(raw, list) or len(raw) != 4:
                skipped.append({"tile": tile["index"], "index": index, "reason": "no rectangular bbox"})
                continue
            try:
                x0,y0,x1,y1 = (float(v) for v in raw)
            except (ValueError, TypeError):
                skipped.append({"tile": tile["index"], "index": index, "reason": "non-numeric bbox"})
                continue
            cx,cy = (x0+x1)/2+tile["x0"],(y0+y1)/2+tile["y0"]
            if not (tile["core_x0"] <= cx <= tile["core_x1"] and
                    tile["core_y0"] <= cy <= tile["core_y1"]):
                skipped.append({"tile": tile["index"], "index": index, "reason": "overlap duplicate or seam"})
                continue
            updated = dict(block)
            updated["block_bbox"] = [x0+tile["x0"],y0+tile["y0"],
                                     x1+tile["x0"],y1+tile["y0"]]
            # Prevent a full-tile image's content becoming one huge overlay.
            updated["_vl_tile_index"] = tile["index"]
            accepted.append(updated)
    results = {"width": width, "height": height, "parsing_res_list": accepted}
    return results, {"tiles": len(tiles), "finished": finished,
                     "missing_or_failed": errors, "seam_skips": skipped,
                     "blocks": len(accepted)}


def _monitor_lines(pipe, output):
    try:
        for line in pipe:
            try:
                output.put(json.loads(line))
            except ValueError:
                output.put({"event": "log", "message": line.strip()[:100]})
    finally:
        output.put({"event": "eof"})


def run_isolated(source_path, settings, device, diagnostics_dir, *,
                 progress, check, tile_timeout=180, startup_timeout=240,
                 benchmark=False, hybrid=False):
    """Interruptible, fully local process: bounded detection, model load, OCR.

    The supervisor kills the worker on timeout/cancel. Each recognized region
    or legacy tile is committed to disk independently. No partial PPTX export.
    """
    diagnostics_dir = Path(diagnostics_dir).resolve()
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    settings_path = diagnostics_dir / "vl-worker-settings.json"
    settings_path.write_text(settings.model_dump_json(), encoding="utf-8")
    cmd = [sys.executable, "-u", "-m", "smart_overlay.vl_worker",
           str(Path(source_path).resolve()), str(diagnostics_dir),
           device, str(settings_path)]
    if benchmark:
        cmd.append("benchmark")
    elif hybrid:
        cmd.append("hybrid")
    stderr_path = diagnostics_dir / "paddle-vl-worker.log"
    state_file = diagnostics_dir / "paddle-vl-progress.json"
    events = Queue()
    active_tile = None
    completed_count = 0
    strategy = "hybrid" if hybrid else ("detect" if benchmark else settings.vl_strategy)
    stage = "starting"
    def checkpoint_status(status, **details):
        payload = {"status": status, "strategy": strategy,
                   "completed_tiles": completed_count, "completed_regions": completed_count,
                   "active_tile": active_tile, "active_region": active_tile,
                   **details}
        temp = state_file.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        temp.replace(state_file)
    checkpoint_status("initializing")
    began = time.monotonic()
    deadline = began + startup_timeout
    overall_deadline = None
    completed_event = None
    proc = None
    try:
        with stderr_path.open("w", encoding="utf-8") as errlog:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=errlog,
                stdin=subprocess.DEVNULL, text=True, bufsize=1,
                encoding="utf-8", errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8",
                     "PYTHONUTF8": "1"})
            reader = Thread(target=_monitor_lines, args=(proc.stdout,events), daemon=True)
            reader.start()
            while True:
                check()
                now = time.monotonic()
                if now > deadline or (overall_deadline is not None and now > overall_deadline):
                    if active_tile is not None:
                        where = f"{'строка' if strategy in ('detect', 'hybrid') else 'участок'} {active_tile+1}"
                    else:
                        where = stage
                    raise LocalVLError(
                        f"Истекло время обработки ({where}). "
                        f"Журнал: {stderr_path}. Промежуточные данные сохранены.")
                try:
                    event = events.get(timeout=.25)
                except Empty:
                    if proc.poll() is not None and not reader.is_alive():
                        break
                    continue
                name = event.get("event")
                if name == "detecting":
                    stage = "поиск надписей"
                    deadline = time.monotonic() + startup_timeout
                    checkpoint_status("detecting")
                    progress("VL: быстрый поиск текстовых строк")
                elif name == "detected":
                    stage = "детекция завершена"
                    checkpoint_status("detected", total_regions=event["total"],
                                      detector_seconds=event.get("detector_seconds"))
                    progress(f"VL: найдено {event['total']} строк; "
                             "начинаю локальное распознавание")
                elif name == "loading":
                    stage = "загрузка модели"
                    deadline = time.monotonic() + startup_timeout
                    checkpoint_status("loading_model", total_regions=event["total"])
                    progress(f"VL: подготовлено {event['total']} областей; загружаю модель…")
                elif name == "started":
                    stage = "модель загружена"
                    overall_deadline = time.monotonic() + settings.vl_total_timeout_seconds
                    deadline = time.monotonic() + (settings.vl_region_timeout_seconds
                              if strategy in ("detect", "hybrid") else tile_timeout)
                    checkpoint_status("model_ready", total_regions=event["total"],
                                      model_load_seconds=event.get("model_load_seconds"))
                    progress(f"VL: модель загружена, ожидается {event['total']} распознаваний")
                elif name in ("tile","region"):
                    active_tile = int(event["index"])
                    stage = "распознавание"
                    if strategy in ("detect", "hybrid"):
                        budget = (settings.vl_preflight_seconds if
                                  completed_count == 0 else settings.vl_region_timeout_seconds)
                        deadline = time.monotonic() + min(budget,settings.vl_region_timeout_seconds)
                    else:
                        deadline = time.monotonic() + tile_timeout
                    checkpoint_status("running_region" if strategy in ("detect", "hybrid") else "running_tile",
                                      total_regions=event["total"])
                    progress(f"VL: {'строка' if strategy in ('detect', 'hybrid') else 'участок'} "
                             f"{active_tile+1}/{event['total']}…")
                elif name == "checkpoint":
                    completed_count = int(event["done"])
                    active_tile = None
                    deadline = time.monotonic() + (settings.vl_region_timeout_seconds
                              if strategy in ("detect", "hybrid") else tile_timeout)
                    checkpoint_status("region_completed" if strategy in ("detect","hybrid") else "tile_completed",
                                      total_regions=event["total"],
                                      inference_seconds=event.get("seconds"))
                    progress(f"VL: сохранено {event['done']}/{event['total']} "
                             f"({event.get('seconds', 0):.1f} с)")
                elif name == "error":
                    raise LocalVLError(
                        f"Локальный VL: {event.get('message','ошибка')}. "
                        f"См. {stderr_path}")
                elif name in ("complete","benchmark_complete"):
                    completed_event = name
                    checkpoint_status("benchmark_done" if benchmark else "worker_complete")
                    break
                elif name == "eof":
                    break
            exit_code = proc.wait(timeout=5)
            if not completed_event or exit_code != 0:
                raise LocalVLError(
                    f"Процесс PaddleOCR-VL завершился без полного результата "
                    f"(код {exit_code}). Журнал: {stderr_path}")
    except BaseException as exc:
        checkpoint_status("cancelled" if type(exc).__name__ == "ProcessingCancelled"
                          else "failed", error_type=type(exc).__name__,
                          message=str(exc)[:400])
        raise
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    if benchmark:
        report_file = diagnostics_dir / "paddle-vl-benchmark.json"
        if not report_file.exists():
            raise LocalVLError("Бенчмарк завершился без отчёта: " + str(stderr_path))
        checkpoint_status("completed", benchmark=True)
        return json.loads(report_file.read_text("utf-8")), {"benchmark": True}
    if hybrid:
        checkpoint_status("completed", hybrid=True)
        return {"hybrid": True}, {"hybrid": True}
    if strategy == "detect":
        from .vl_detect import detected_page
        manifest = diagnostics_dir / "paddle-vl-regions.json"
        if not manifest.exists():
            raise LocalVLError("Детектор не сохранил области: " + str(stderr_path))
        result, summary = detected_page(
            json.loads(manifest.read_text("utf-8")), diagnostics_dir)
    else:
        manifest = diagnostics_dir / "paddle-vl-tiles.json"
        if not manifest.exists():
            raise LocalVLError("Нет манифеста VL: " + str(stderr_path))
        report = json.loads(manifest.read_text("utf-8"))
        result, summary = merge_tile_pages(
            report["tiles"], diagnostics_dir, report["width"],report["height"])
        if summary["missing_or_failed"]:
            checkpoint_status("failed", missing_or_failed=summary["missing_or_failed"])
            raise LocalVLError("Не все участки обработаны, PPTX не создаётся")
    checkpoint_status("completed", summary=summary)
    return result, summary
