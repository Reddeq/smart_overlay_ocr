"""Local real-drawing experiments; no production OCR, editor or export changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field

from .diagnose_vl import (Case, GROUPING_EXAMPLES_PROMPT, Merges, partition, run_case, simple_payload)
from .geometry import contains, intersection, union
from .input import open_project, source_image
from .models import Box, Model, Table
from .settings import Settings
from .vision import image_part, response_json
from .vision_diagnostics import sanitized


class Region(Model):
    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    bbox: Box
    ocr_ids: list[str] = Field(min_length=5, max_length=15)
    expected_merges: list[list[str]]
    code_only_ids: list[str]
    uncertain_ids: list[str]
    rationale: str


class Manifest(Model):
    version: int = 1
    source_sha256: str
    provenance: str
    table_exclusions: list[Box]
    regions: list[Region] = Field(min_length=3, max_length=5)


def validate_manifest(project, manifest):
    if manifest.version != 1 or manifest.source_sha256 != project.source.sha256:
        raise ValueError("Source hash/version does not match the reference manifest")
    if len({r.name for r in manifest.regions}) != len(manifest.regions):
        raise ValueError("Duplicate region name")
    fragments = {f.id: f for f in project.fragments}
    table_ids = {i for o in project.objects if isinstance(o, Table) for i in o.ocr_ids}
    masks = manifest.table_exclusions+[o.bbox for o in project.objects if isinstance(o, Table)]
    page = Box(x=0, y=0, w=project.source.width, h=project.source.height)
    for mask in masks:
        if not contains(page, mask):
            raise ValueError("Table mask outside source image")
    for region in manifest.regions:
        ids = set(region.ocr_ids)
        if len(ids) != len(region.ocr_ids) or ids-set(fragments):
            raise ValueError("Duplicate/unknown source IDs in reference")
        if not contains(page, region.bbox):
            raise ValueError("Crop outside source image")
        for identifier in ids:
            box = fragments[identifier].bbox
            if not contains(region.bbox, box):
                raise ValueError("Crop clips a selected OCR fragment")
            if identifier in table_ids or any(intersection(box, mask) > 0 for mask in masks):
                raise ValueError("Table OCR cannot enter ordinary grouping")
        # A non-table fragment intersecting the crop must be included completely.
        for f in project.fragments:
            excluded = f.id in table_ids or any(intersection(f.bbox, mask) > 0 for mask in masks)
            if not excluded and intersection(f.bbox, region.bbox) > 0 and f.id not in ids:
                raise ValueError("Crop clips or omits a non-table OCR fragment")
        partition(region.ocr_ids, region.expected_merges)
        merged = {i for group in region.expected_merges for i in group}
        if (set(region.uncertain_ids)-ids or set(region.code_only_ids)-ids
                or merged & set(region.uncertain_ids) or merged & set(region.code_only_ids)
                or set(region.code_only_ids) & set(region.uncertain_ids)):
            raise ValueError("Ambiguous/code-only reference membership is inconsistent")


def prepare_case(project, region, settings):
    by_id = {f.id: f for f in project.fragments}
    selected = [by_id[i] for i in region.ocr_ids]
    box = region.bbox
    crop = source_image(project).crop((int(box.x), int(box.y), int(box.right), int(box.bottom)))
    display = crop.resize((crop.width*2, crop.height*2))
    packet = {"image_size": [project.source.width, project.source.height],
              "crop_bbox": box.model_dump(), "image_scale": 2,
              "coordinates": "All OCR boxes and crop_bbox use original image pixels; the supplied image is this crop at 2x scale.",
              "fragments": [f.model_dump(include={"id", "text", "bbox", "rotation", "confidence"}) for f in selected],
              "tables": []}
    prompt = GROUPING_EXAMPLES_PROMPT.replace("{{OCR_DATA}}", json.dumps(packet, ensure_ascii=False, separators=(",", ":")))
    payload = simple_payload(settings, prompt, [image_part(display)], Merges, "json_object")
    payload["provider"] = {"order": ["parasail"], "allow_fallbacks": False}
    subset = project.model_copy(deep=True)
    subset.fragments, subset.objects, subset.ignored = selected, [], {}
    return Case(region.name, payload, Merges, subset, prompt_variant="examples-real"), packet, crop


def geometry_error(members, masks):
    boxes = [f.bbox for f in members]
    box = union(boxes)
    if any(intersection(box, m) > 0 for m in masks):
        return "merge_crosses_table"
    if max(f.rotation for f in members)-min(f.rotation for f in members) > 10:
        return "orientation_mismatch"
    if members != sorted(members, key=lambda f: (f.bbox.y, f.bbox.x)):
        return "reading_order_changed"
    height = max(b.h for b in boxes)
    for a, b in zip(boxes, boxes[1:]):
        if b.y-a.bottom > height*3:
            return "vertical_gap"
        if max(0, max(a.x, b.x)-min(a.right, b.right)) > height*3:
            return "horizontal_gap"
    return None


def normalize(project, region, reply, masks):
    """Remove only known unique singleton entries; unsafe proposals are never repaired."""
    ids, used = set(region.ocr_ids), set()
    for group in reply.merges:
        if not group or len(set(group)) != len(group) or set(group)-ids or set(group) & used:
            raise ValueError("unknown_duplicate_or_empty_merge")
        used.update(group)
    by_id = {f.id: f for f in project.fragments}
    singletons = [group[0] for group in reply.merges if len(group) == 1]
    normalized = [list(group) for group in reply.merges if len(group) >= 2]
    applied, review = [], []
    for group in normalized:
        reason = ("ambiguous_membership" if set(group) & set(region.uncertain_ids) else
                  geometry_error([by_id[i] for i in group], masks))
        if reason:
            review.append({"ocr_ids": group, "reason": reason})
        else:
            applied.append(group)
    return {"normalized_merges": normalized, "applied_merges": applied, "review_groups": review,
            "removed_singletons": singletons, "contract_violations": len(singletons),
            "normalization_log": [{"kind": "singleton_removed", "ocr_id": i} for i in singletons],
            "final_partition": partition(region.ocr_ids, applied)}


def assess(region, normalized):
    expected = {tuple(g) for g in region.expected_merges}
    uncertain = set(region.uncertain_ids)
    proposed = {tuple(g) for g in normalized["normalized_merges"] if not set(g) & uncertain}
    applied = {tuple(g) for g in normalized["applied_merges"]}
    known = [i for i in region.ocr_ids if i not in uncertain]
    actual_partition = {tuple(g) for g in normalized["final_partition"] if not set(g) & uncertain}
    reference_partition = {tuple(g) for g in partition(known, region.expected_merges)}
    return {"fragments": len(region.ocr_ids), "expected_merges": len(expected),
            "correct_merges": len(proposed & expected), "incorrect_merges": len(proposed-expected),
            "missed_merges": len(expected-proposed), "unapplied_expected_merges": len(expected-applied),
            "contract_violations": normalized["contract_violations"],
            "review_groups": len(normalized["review_groups"]), "uncertain_fragments": len(uncertain),
            "known_partition_correct": actual_partition == reference_partition,
            "fully_verified": not uncertain and not normalized["review_groups"]}


def preview(project, region, path, normalized=None):
    """Deterministic diagnostic overlays; source files are never edited."""
    scale = 3
    b = region.bbox
    crop = source_image(project).crop((int(b.x), int(b.y), int(b.right), int(b.bottom)))
    crop = crop.resize((crop.width*scale, crop.height*scale))
    canvas = Image.new("RGB", (crop.width+335, max(crop.height+55, 430)), "white")
    canvas.paste(crop, (0, 55))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("arial.ttf", 15)
    by_id = {f.id: f for f in project.fragments}

    def rectangle(ids, color, width, label=""):
        box = union([by_id[i].bbox for i in ids])
        xy = ((box.x-b.x)*scale, (box.y-b.y)*scale+55,
              (box.right-b.x)*scale, (box.bottom-b.y)*scale+55)
        draw.rectangle(xy, outline=color, width=width)
        if label:
            draw.text((xy[0], max(55, xy[1]-16)), label, fill=color, font=font)

    draw.text((8, 6), region.name+": blue OCR; green correct; red wrong; orange missed/reference; gray uncertain", font=font, fill="black")
    for index, identifier in enumerate(region.ocr_ids, 1):
        rectangle([identifier], "#397DCF", 1, str(index))
        draw.text((crop.width+6, 55+index*22), f"{index:02}: {identifier}", font=font, fill="black")
    expected = {tuple(g) for g in region.expected_merges}
    applied = normalized["applied_merges"] if normalized else []
    for group in region.expected_merges:
        if group not in applied:
            rectangle(group, "#D07400", 5, "missed" if normalized else "reference")
    if normalized:
        for group in normalized["normalized_merges"]:
            rectangle(group, "#15853B" if tuple(group) in expected else "#D51F30", 3, "merge")
        for group in normalized["final_partition"]:
            if len(group) == 1 and group[0] not in region.uncertain_ids:
                rectangle(group, "#15853B", 2)
    for identifier in region.uncertain_ids:
        rectangle([identifier], "#777777", 3, "review")
    canvas.save(path)


def check_proxy(value):
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or not parsed.port or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Use the existing local HTTP VPN proxy, without credentials in the URL")
    return value


def save_report(path, report, key):
    path.write_text(json.dumps(sanitized(report, key), ensure_ascii=False, indent=2), "utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proxy", type=check_proxy, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-requests", type=int)
    parser.add_argument("--max-requests", type=int, default=5)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(Path("outputs").resolve()):
        parser.error("Private reports must stay inside outputs/")
    project = open_project(args.project)
    manifest = Manifest.model_validate_json(args.regions.read_text("utf-8"))
    validate_manifest(project, manifest)
    settings = Settings(model="qwen/qwen3-vl-8b-instruct", stream_responses=False, max_output_tokens=1024, retries=0)
    cases = [prepare_case(project, r, settings) for r in manifest.regions]
    count = len(cases)
    if not 1 <= args.max_requests <= 5 or count > args.max_requests:
        parser.error("Request budget exceeded")
    if args.execute and args.confirm_requests != count:
        parser.error(f"Confirm exactly {count} requests with --confirm-requests {count}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir/"results.json"
    if report_path.exists() and json.loads(report_path.read_text("utf-8")).get("executed"):
        parser.error("A live report already exists; refusing to repeat requests")
    report = {"version": 1, "executed": False, "planned_requests": count,
              "source_sha256": project.source.sha256,
              "reference_sha256": hashlib.sha256(args.regions.read_bytes()).hexdigest(),
              "vpn_preflight": None, "results": []}
    for region, (case, packet, crop) in zip(manifest.regions, cases):
        crop.save(args.output_dir/(region.name+"-crop.png"))
        (args.output_dir/(region.name+"-packet.json")).write_text(json.dumps(packet, ensure_ascii=False, indent=2), "utf-8")
        preview(project, region, args.output_dir/(region.name+"-reference.png"))
    save_report(report_path, report, "")
    print(f"Prepared {count} regions; {count} requests, at most {count*1024} output tokens; no retries.")
    if not args.execute:
        return 0
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        parser.error("OPENROUTER_API_KEY is absent")
    # Proxy applies only to this diagnostic process. No direct route/fallback is used.
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        os.environ[name] = args.proxy
    os.environ["NO_PROXY"] = ""
    with httpx.Client(proxy=args.proxy, timeout=10, follow_redirects=False) as client:
        response = client.get(settings.endpoint+"/models/"+settings.model+"/endpoints")
        response.raise_for_status()
        if not any(e.get("provider_name") == "Parasail" for e in response.json().get("data", {}).get("endpoints", [])):
            raise ValueError("Parasail unavailable in VPN preflight")
    report.update(executed=True, vpn_preflight={"http_status": 200, "protocol": "HTTP proxy with TLS verification"})
    save_report(report_path, report, key)
    masks = manifest.table_exclusions+[o.bbox for o in project.objects if isinstance(o, Table)]
    for region, (case, packet, crop) in zip(manifest.regions, cases):
        result = run_case(case, settings, key)  # Exactly one attempt; no repair/validation retries.
        if isinstance(result.get("full_text"), str):
            (args.output_dir/(region.name+"-raw-response.txt")).write_text(result["full_text"], "utf-8")
        normalized = None
        if result.get("full_text") and result.get("finish_reason") != "length" and result.get("http_status") == 200:
            try:
                reply = Merges.model_validate_json(response_json(result["full_text"]))
                normalized = normalize(project, region, reply, masks)
                result["normalized"] = normalized
                result["metrics"] = assess(region, normalized)
                result["envelope_classification"] = result["classification"]
                match = result["metrics"]["known_partition_correct"]
                result["classification"] = ("reference_match_with_uncertain_ocr" if match and region.uncertain_ids else
                                            "reference_match" if match else "reference_mismatch")
                result["content_valid"] = match if result["metrics"]["fully_verified"] else None
                by_id = {f.id: f for f in project.fragments}
                result["logical_blocks"] = [{"ocr_ids": group,
                    "bbox": union([by_id[i].bbox for i in group]).model_dump(),
                    "revive": not all(i in region.code_only_ids for i in group),
                    "needs_review": bool(set(group) & set(region.uncertain_ids))}
                    for group in normalized["final_partition"]]
                (args.output_dir/(region.name+"-normalized.json")).write_text(
                    json.dumps(sanitized({"normalized": normalized, "logical_blocks": result["logical_blocks"]}, key),
                               ensure_ascii=False, indent=2), "utf-8")
            except ValueError as exc:
                result["validation_error"] = str(exc).splitlines()[0][:150]
        preview(project, region, args.output_dir/(region.name+"-result.png"), normalized)
        report["results"].append(result)
        save_report(report_path, report, key)
        print(region.name+": "+("assessed; singleton violations="+str(normalized["contract_violations"])
                                  if normalized else "not_assessable"))
        if result.get("http_status") != 200 or result.get("classification") in {"timeout", "transport_error"}:
            report["stopped"] = "Network/API failure; diagnose VPN before any further requests"
            save_report(report_path, report, key)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
