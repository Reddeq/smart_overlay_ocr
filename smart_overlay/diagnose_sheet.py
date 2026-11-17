"""Experimental full-sheet grouping, independent of the production Smart OCR."""
import argparse
import hashlib
import html
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field

from .diagnose_real import Manifest, check_proxy, normalize, prepare_case, save_report
from .diagnose_vl import Merges, run_case
from .geometry import contains, intersection, union
from .input import open_project, source_image
from .models import Box, Model, Table
from .settings import Settings
from .vision import response_json


class AutoRegion(Model):
    name: str
    bbox: Box
    ocr_ids: list[str] = Field(min_length=1)
    # Reference decisions are deliberately absent from selection and normalization.
    uncertain_ids: list[str] = Field(default_factory=list)


def eligible_fragments(project, masks):
    table_ids = {i for o in project.objects if isinstance(o, Table) for i in o.ocr_ids}
    return [f for f in project.fragments if f.id not in project.ignored and f.id not in table_ids
            and not any(intersection(f.bbox, mask) > 0 for mask in masks)]


def line_components(fragments):
    """Protect plausible vertical continuations from crop boundaries, without merging objects."""
    neighbors = {f.id: set() for f in fragments}
    for n, a in enumerate(fragments):
        for b in fragments[n+1:]:
            top, bottom = sorted([a, b], key=lambda f: f.bbox.y)
            x_overlap = max(0, min(a.bbox.right, b.bbox.right)-max(a.bbox.x, b.bbox.x))
            center_gap = abs((a.bbox.y+a.bbox.h/2)-(b.bbox.y+b.bbox.h/2))
            if (abs(a.rotation-b.rotation) <= 10 and x_overlap >= min(a.bbox.w, b.bbox.w)*0.5
                    and center_gap >= min(a.bbox.h, b.bbox.h)*0.65
                    and bottom.bbox.y-top.bbox.bottom <= max(a.bbox.h, b.bbox.h)*1.25):
                neighbors[a.id].add(b.id)
                neighbors[b.id].add(a.id)
    groups, seen = [], set()
    for identifier in neighbors:
        if identifier in seen:
            continue
        pending, group = [identifier], set()
        while pending:
            current = pending.pop()
            if current in group:
                continue
            group.add(current)
            pending.extend(neighbors[current]-group)
        seen.update(group)
        groups.append(group)
    return groups


def automatic_regions(project, masks, *, width=420, height=300, step_x=300, step_y=210, max_fragments=24):
    if min(width, height, step_x, step_y, max_fragments) <= 0 or step_x > width or step_y > height:
        raise ValueError("Invalid window size/step/budget")
    eligible = eligible_fragments(project, masks)
    by_id = {f.id: f for f in eligible}
    components = line_components(eligible)
    component_of = {i: group for group in components for i in group}
    page_w, page_h = project.source.width, project.source.height
    candidates = {}

    def expand(seed):
        box, ids = seed, set()
        while True:
            touched = {f.id for f in eligible if intersection(f.bbox, box) > 0}
            full = set().union(*(component_of[i] for i in touched)) if touched else set()
            if full == ids:
                return box, ids
            ids = full
            # Retain a small context margin around every protected fragment, including frame edges.
            padded = [Box(x=by_id[i].bbox.x-12, y=by_id[i].bbox.y-12,
                          w=by_id[i].bbox.w+24, h=by_id[i].bbox.h+24) for i in ids]
            joined = union([box]+padded)
            left, top = max(0, math.floor(joined.x)), max(0, math.floor(joined.y))
            right, bottom = min(page_w, math.ceil(joined.right)), min(page_h, math.ceil(joined.bottom))
            box = Box(x=left, y=top, w=right-left, h=bottom-top)

    def add(seed, depth=0):
        box, ids = expand(seed)
        if not ids:
            return
        # Subdivide seed windows, never their protected continuation components.
        if len(ids) > max_fragments and depth < 5 and min(seed.w, seed.h) > 85:
            if seed.w >= seed.h:
                add(Box(x=seed.x, y=seed.y, w=seed.w/2, h=seed.h), depth+1)
                add(Box(x=seed.x+seed.w/2, y=seed.y, w=seed.w/2, h=seed.h), depth+1)
            else:
                add(Box(x=seed.x, y=seed.y, w=seed.w, h=seed.h/2), depth+1)
                add(Box(x=seed.x, y=seed.y+seed.h/2, w=seed.w, h=seed.h/2), depth+1)
            return
        key = frozenset(ids)
        if key not in candidates or box.w*box.h < candidates[key].w*candidates[key].h:
            candidates[key] = box

    for y in range(0, page_h, step_y):
        for x in range(0, page_w, step_x):
            add(Box(x=x, y=y, w=min(width, page_w-x), h=min(height, page_h-y)))
    # Remove redundant ID-subset views; retain overlap between distinct remaining views.
    keys = sorted(candidates, key=lambda ids: (-len(ids), sorted(ids)))
    kept = []
    for ids in keys:
        if not any(ids <= other for other in kept):
            kept.append(ids)
    regions = []
    for ids in sorted(kept, key=lambda ids: (candidates[ids].y, candidates[ids].x)):
        regions.append(AutoRegion(name=f"A{len(regions)+1:02}", bbox=candidates[ids],
                                  ocr_ids=[f.id for f in eligible if f.id in ids]))
    covered = {i for r in regions for i in r.ocr_ids}
    if covered != set(by_id):
        raise ValueError("Automatic selection did not cover all eligible OCR")
    for r in regions:
        if any(not contains(r.bbox, by_id[i].bbox) for i in r.ocr_ids):
            raise ValueError("Automatic crop clips an OCR fragment")
    return regions, {"eligible_ids": [f.id for f in eligible],
                     "excluded_ids": [f.id for f in project.fragments if f.id not in covered],
                     "protected_components": [sorted(group) for group in components if len(group) > 1],
                     "oversized_regions": [r.name for r in regions if len(r.ocr_ids) > max_fragments],
                     "overlap_ids": [i for i in by_id if sum(i in r.ocr_ids for r in regions) > 1]}


def reconcile(regions, results):
    """Use exact proposals only; never take the transitive union of competing merges."""
    region_by_name = {r.name: r for r in regions}
    supports, usable, blocked = defaultdict(list), {}, set()
    failures = []
    completed = {r['test'] for r in results}
    pending = [r.name for r in regions if r.name not in completed]
    blocked.update(i for r in regions if r.name in pending for i in r.ocr_ids)
    for result in results:
        name = result["test"]
        region = region_by_name[name]
        normalized = result.get("normalized")
        if normalized is None:
            blocked.update(region.ocr_ids)
            failures.append(name)
            continue
        abstain = {i for g in normalized["review_groups"] for i in g["ocr_ids"]}
        blocked.update(abstain)
        groups = {tuple(g) for g in normalized["applied_merges"]}
        usable[name] = (set(region.ocr_ids), groups, abstain)
        for group in groups:
            supports[group].append(name)
    conflicts = defaultdict(set)
    groups = list(supports)
    for n, group in enumerate(groups):
        for other in groups[n+1:]:
            if set(group) & set(other):
                conflicts[group].add("overlapping_incompatible_proposal")
                conflicts[other].add("overlapping_incompatible_proposal")
        for name, (ids, proposals, abstain) in usable.items():
            if set(group) <= ids and not set(group) & abstain and group not in proposals:
                conflicts[group].add("complete_view_disagrees:"+name)
    accepted = [{"ocr_ids": list(g), "support_regions": supports[g], "needs_review":
                 len(supports[g]) < 2 or bool(set(g) & blocked)} for g in groups if g not in conflicts]
    rejected = [{"ocr_ids": list(g), "support_regions": supports[g], "reasons": sorted(reasons)}
                for g, reasons in conflicts.items()]
    return {"accepted_merges": accepted, "conflicts": rejected, "blocked_ids": sorted(blocked),
            "failed_regions": failures, "pending_regions": pending}


def fragment_flags(fragment):
    text = fragment.text.strip()
    graphic = (fragment.bbox.h < 12 or fragment.bbox.w < 12 or
               not any(c.isalnum() for c in text))
    code = bool(text and any(c.isdigit() for c in text) and not re.search(r"\s", text)
                and re.fullmatch(r"[\w()+*/.,~≈=−+\-]+", text))
    return {"code_only_heuristic": code and not graphic,
            "needs_review": graphic or fragment.confidence < 0.85}


def logical_blocks(project, selection, consensus):
    by_id = {f.id: f for f in project.fragments}
    excluded = set(selection["excluded_ids"])
    conflict_ids = {i for g in consensus["conflicts"] for i in g["ocr_ids"]}
    assigned, blocks = set(), []
    for merged in consensus["accepted_merges"]:
        ids = merged["ocr_ids"]
        flags = [fragment_flags(by_id[i]) for i in ids]
        blocks.append({**merged, "bbox": union([by_id[i].bbox for i in ids]).model_dump(),
                       "kind": "code" if all(f["code_only_heuristic"] for f in flags) else "text",
                       "needs_review": merged["needs_review"] or any(f["needs_review"] for f in flags)})
        assigned.update(ids)
    for f in project.fragments:
        if f.id in assigned:
            continue
        flag = fragment_flags(f)
        kind = ("excluded_table_or_source" if f.id in excluded else
                "code" if flag["code_only_heuristic"] else "text")
        blocks.append({"ocr_ids": [f.id], "bbox": f.bbox.model_dump(), "kind": kind,
                       "needs_review": f.id not in excluded and (flag["needs_review"] or f.id in conflict_ids
                                                                 or f.id in consensus["blocked_ids"])})
    flat = [i for block in blocks for i in block["ocr_ids"]]
    if len(flat) != len(set(flat)) or set(flat) != set(by_id):
        raise ValueError("Full-sheet partition lost or duplicated OCR IDs")
    return blocks


def evaluate_controls(reference, regions, selection, consensus, blocks):
    covered = {i for r in regions for i in r.ocr_ids}
    result = []
    for control in reference.regions:
        known = set(control.ocr_ids)-set(control.uncertain_ids)
        expected = {tuple(g) for g in control.expected_merges}
        expected_partition = expected | {(i,) for i in known if not any(i in g for g in expected)}
        actual = {tuple(b["ocr_ids"]) for b in blocks if set(b["ocr_ids"]) & known}
        proposed = {tuple(g["ocr_ids"]) for g in consensus["accepted_merges"] if set(g["ocr_ids"]) & known}
        result.append({"name": control.name, "known_fragments": len(known),
                       "selection_missing_ids": sorted(known-covered),
                       "selection_split_merges": [list(g) for g in expected
                           if not any(set(g) <= set(r.ocr_ids) for r in regions)],
                       "correct_merges": len(expected & proposed), "incorrect_merges": len(proposed-expected),
                       "missed_merges": len(expected-proposed), "known_partition_correct": actual == expected_partition,
                       "uncertain_fragments": len(control.uncertain_ids)})
    return result


def overview(project, masks, regions, blocks, consensus, path, *, draw_regions=False):
    scale = 2
    im = source_image(project).resize((project.source.width*scale, project.source.height*scale))
    draw, font = ImageDraw.Draw(im), ImageFont.truetype("arial.ttf", 18)
    def rect(box, color, width=2, label=""):
        xy = (box.x*scale, box.y*scale, box.right*scale, box.bottom*scale)
        draw.rectangle(xy, outline=color, width=width)
        if label:
            draw.text((xy[0], max(0, xy[1]-19)), label, font=font, fill=color)
    for mask in masks:
        rect(mask, "#888888", 3, "table excluded")
    if draw_regions:
        for r in regions:
            rect(r.bbox, "#427EC4", 1, r.name)
    for block in blocks:
        if block["kind"] == "excluded_table_or_source":
            continue
        color = "#D08A00" if block["needs_review"] else "#555555" if block["kind"] == "code" else "#12813C"
        rect(Box.model_validate(block["bbox"]), color, 4 if len(block["ocr_ids"]) > 1 else 2)
    by_id = {f.id: f for f in project.fragments}
    for group in consensus["conflicts"]:
        rect(union([by_id[i].bbox for i in group["ocr_ids"]]), "#D41F31", 5, "conflict")
    im.save(path)


def save_artifacts(root, project, masks, regions, selection, results, reference):
    consensus = reconcile(regions, results)
    blocks = logical_blocks(project, selection, consensus)
    controls = evaluate_controls(reference, regions, selection, consensus, blocks)
    combined = {"consensus": consensus, "logical_blocks": blocks, "controls": controls,
                "reference_scope_only": True, "code_classification_is_heuristic": True}
    save_report(root/"combined.json", combined, "")
    overview(project, masks, regions, blocks, consensus, root/"sheet.png")
    overview(project, masks, regions, blocks, consensus, root/"selection.png", draw_regions=True)
    answered = {item['test'] for item in results if isinstance(item.get('full_text'), str)}
    links = ''.join(f'<li>{html.escape(r.name)}: {len(r.ocr_ids)} OCR · '
                    f'<a href="{html.escape(r.name)}-crop.png">crop</a> · '
                    f'<a href="{html.escape(r.name)}-packet.json">OCR packet</a>'
                    +(f' · <a href="{html.escape(r.name)}-raw-response.txt">model reply</a>'
                      if r.name in answered else '')+'</li>' for r in regions)
    page = ('<!doctype html><meta charset="utf-8"><title>Full-sheet diagnostic</title>'
            '<style>body{font:16px sans-serif;margin:24px}img{max-width:100%}pre{white-space:pre-wrap}</style>'
            '<h1>Full-sheet diagnostic</h1><p>Blue: automatic windows; green: text; gray: code/table; '
            'orange: review; red: conflicting proposals. Green does not mean independently verified.</p>'
            '<p>Table masks are manual. Numeric/code flags are heuristics. Unlabelled areas have no quality score.</p>'
            '<a href="combined.json">Partition, conflicts and control scores</a> · '
            '<a href="sheet.png">Full-resolution result</a> · '
            '<a href="selection.png">Automatic window map</a><br><img src="sheet.png">'
            '<h2>Control regions</h2><pre>'+html.escape(json.dumps(controls, indent=2))+'</pre><h2>Windows</h2><ul>'+links+'</ul>')
    (root/"index.html").write_text(page, "utf-8")
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True, help="Previous controls; only table masks used in selection")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proxy", type=check_proxy, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-requests", type=int)
    parser.add_argument("--max-requests", type=int, default=20)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(Path('outputs').resolve()):
        parser.error("Private artifacts must stay in outputs/")
    p = open_project(args.project)
    reference = Manifest.model_validate_json(args.controls.read_text('utf-8'))
    if reference.source_sha256 != p.source.sha256:
        parser.error("Control source hash differs")
    masks = reference.table_exclusions+[o.bbox for o in p.objects if isinstance(o, Table)]
    regions, selection = automatic_regions(p, masks)
    if selection['oversized_regions']:
        parser.error("Protected continuation component exceeds region budget; review selection before generation")
    settings = Settings(model='qwen/qwen3-vl-8b-instruct', stream_responses=False, max_output_tokens=1024, retries=0)
    cases = [prepare_case(p, r, settings) for r in regions]
    paid = [(r, case) for r, (case, packet, crop) in zip(regions, cases) if len(r.ocr_ids) > 1]
    count = len(paid)
    if not 1 <= args.max_requests <= 30 or count > args.max_requests:
        parser.error(f"Plan needs {count} requests, budget exceeded")
    if args.execute and args.confirm_requests != count:
        parser.error(f"Confirm exactly {count} requests")
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    report_path = root/'results.json'
    if report_path.exists() and json.loads(report_path.read_text('utf-8')).get('executed'):
        parser.error('Live report exists; no automatic repeat')
    report = {'executed': False, 'planned_requests': count, 'maximum_output_tokens': count*1024,
              'source_sha256': p.source.sha256, 'controls_sha256': hashlib.sha256(args.controls.read_bytes()).hexdigest(),
              'regions': [r.model_dump() for r in regions], 'selection': selection, 'results': []}
    for r, (case, packet, crop) in zip(regions, cases):
        crop.save(root/(r.name+'-crop.png'))
        (root/(r.name+'-packet.json')).write_text(json.dumps(packet,ensure_ascii=False,indent=2),'utf-8')
        if len(r.ocr_ids) == 1:
            report['results'].append({'test': r.name, 'local_only': True,
                'normalized': normalize(p, r, Merges(merges=[]), masks)})
    save_report(report_path, report, '')
    save_artifacts(root, p, masks, regions, selection, report['results'], reference)
    print(f'Automatic windows: {len(regions)}; covered OCR: {len(selection["eligible_ids"])}; '
          f'requests: {count}; maximum output tokens: {count*1024}; no retries.')
    if not args.execute:
        return 0
    key = os.environ.get('OPENROUTER_API_KEY','')
    if not key:
        parser.error('OPENROUTER_API_KEY absent')
    for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY'):
        os.environ[name] = args.proxy
    os.environ['NO_PROXY'] = ''
    with httpx.Client(proxy=args.proxy, timeout=10, follow_redirects=False) as client:
        response = client.get(settings.endpoint+'/models/'+settings.model+'/endpoints')
        response.raise_for_status()
        if not any(e.get('provider_name') == 'Parasail' for e in response.json().get('data',{}).get('endpoints',[])):
            raise ValueError('Parasail not available through VPN')
    report.update(executed=True, vpn_preflight={'http_status':200,'tls_verified':True})
    save_report(report_path, report, key)
    for r, case in paid:
        item = run_case(case, settings, key)
        if isinstance(item.get('full_text'),str):
            (root/(r.name+'-raw-response.txt')).write_text(item['full_text'],'utf-8')
        if item.get('http_status') == 200 and item.get('finish_reason') != 'length' and item.get('full_text'):
            try:
                reply = Merges.model_validate_json(response_json(item['full_text']))
                item['normalized'] = normalize(p,r,reply,masks)
                item['generic_classification'] = item.get('classification')
                item['classification'] = ('geometry_review_required' if item['normalized']['review_groups']
                                          else 'merge_proposals_validated')
                # There is no per-window semantic ground truth. Score only the final control partition.
                item['content_valid'] = None
            except ValueError:
                item['validation_error'] = 'Invalid schema, IDs or merge contract; no repair applied'
        report['results'].append(item)
        save_report(report_path,report,key)
        save_artifacts(root,p,masks,regions,selection,report['results'],reference)
        print(r.name+': '+('validated' if 'normalized' in item else 'review required'))
        if item.get('http_status') != 200:
            report['stopped'] = 'Network/API failure; diagnose VPN before more requests'
            save_report(report_path,report,key)
            break
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
