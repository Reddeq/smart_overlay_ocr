"""Local OCR comparison. Never applies experimental results to the source project."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .geometry import angle, intersection, polygon_box
from .input import open_project, source_image
from .models import Box
from .ocr import PaddleEngine
from .settings import Settings


def restore_points(points, rotation, width, height, scale, x, y):
    restored = []
    for px, py in points:
        px, py = px/scale, py/scale
        if rotation == 90:
            px, py = width-py, px
        elif rotation == 180:
            px, py = width-px, height-py
        elif rotation == 270:
            px, py = py, height-px
        restored.append((px+x, py+y))
    return restored


def deduplicate(fragments):
    result = []
    conflicts = []
    for f in sorted(fragments, key=lambda f: f.confidence, reverse=True):
        duplicate = next((old for old in result if
            intersection(f.bbox, old.bbox)/max(f.bbox.w*f.bbox.h, old.bbox.w*old.bbox.h) > .65
            and abs((f.rotation-old.rotation+90) % 180-90) < 15), None)
        if duplicate is None:
            result.append(f)
        elif f.text != duplicate.text:
            conflicts.append({'kept': duplicate.model_dump(), 'alternative': f.model_dump()})
    return result, conflicts


def distance(a, b):
    row = list(range(len(b)+1))
    for i, left in enumerate(a, 1):
        next_row = [i]
        for j, right in enumerate(b, 1):
            next_row.append(min(row[j]+1, next_row[-1]+1, row[j-1]+(left != right)))
        row = next_row
    return row[-1]


def evaluate(fragments, reference):
    """Reference entries are manually annotated single lines: {bbox, text}."""
    edges = []
    for i, entry in enumerate(reference):
        box = Box.model_validate(entry['bbox'])
        for j, f in enumerate(fragments):
            area = intersection(box, f.bbox)
            iou = area/(box.w*box.h+f.bbox.w*f.bbox.h-area)
            if iou >= .3:
                edges.append((iou, i, j))
    matched_ref, matched_pred = set(), set()
    exact = errors = 0
    for _, i, j in sorted(edges, reverse=True):
        if i in matched_ref or j in matched_pred:
            continue
        matched_ref.add(i)
        matched_pred.add(j)
        expected, actual = reference[i]['text'], fragments[j].text
        exact += expected == actual
        errors += distance(expected, actual)
    errors += sum(len(entry['text']) for i, entry in enumerate(reference) if i not in matched_ref)
    return {'reference_lines': len(reference), 'detected_lines': len(matched_ref),
            'recall': len(matched_ref)/max(1,len(reference)), 'exact_lines': exact,
            'character_error_rate_including_misses': errors/max(1,sum(len(e['text']) for e in reference)),
            'unmatched_predictions': len(fragments)-len(matched_pred),
            'note': 'Unmatched predictions are false positives only when the reference covers the entire image.'}


def preprocess(image, mode):
    if mode == 'contrast':
        return ImageOps.autocontrast(ImageOps.grayscale(image)).convert('RGB')
    if mode == 'lines':
        gray = np.asarray(ImageOps.grayscale(image))
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((1, max(40,image.width//8)),np.uint8))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((max(40,image.height//8),1),np.uint8))
        # An experimental variant: long character strokes may also be removed.
        return Image.fromarray(cv2.inpaint(np.asarray(image), horizontal|vertical, 2, cv2.INPAINT_TELEA))
    return image


def run(project, engine, *, tile_size=700, scale=2, rotations=(0,), mode='original', detection=None, progress=print):
    image = source_image(project)
    start = time.monotonic()
    baseline, _ = engine.recognize(image)
    baseline_seconds = time.monotonic()-start
    candidates = list(baseline)
    start = time.monotonic()
    stride = max(1, tile_size-100)
    for y in range(0,image.height,stride):
        for x in range(0,image.width,stride):
            crop = preprocess(image.crop((x,y,min(image.width,x+tile_size),min(image.height,y+tile_size))), mode)
            for rotation in rotations:
                progress(f'OCR crop {x},{y}; rotation {rotation}; scale {scale}')
                rotated = crop.rotate(rotation,expand=True)
                enlarged = rotated.resize((rotated.width*scale,rotated.height*scale),Image.Resampling.LANCZOS)
                fragments, _ = engine.recognize(enlarged, detection=detection)
                for f in fragments:
                    f.polygon = restore_points(f.polygon,rotation,crop.width,crop.height,scale,x,y)
                    f.bbox = polygon_box(f.polygon)
                    f.rotation = angle(f.polygon)
                    f.origin = 'region'
                    candidates.append(f)
    merged, conflicts = deduplicate(candidates)
    return baseline, merged, {'baseline_seconds': baseline_seconds,
                             'additional_seconds': time.monotonic()-start,'conflicts':conflicts}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--reference',type=Path,help='JSON list of manually annotated {bbox,text} lines')
    parser.add_argument('--tile-size',type=int,default=700)
    parser.add_argument('--scale',type=int,choices=[1,2,3],default=2)
    parser.add_argument('--rotations',type=int,nargs='+',choices=[0,90,180,270],default=[0])
    parser.add_argument('--mode',choices=['original','contrast','lines'],default='original')
    parser.add_argument('--language',help='Override OCR language for this experiment only')
    parser.add_argument('--box-thresh',type=float)
    parser.add_argument('--thresh',type=float)
    parser.add_argument('--unclip-ratio',type=float)
    args=parser.parse_args()
    if not 200 <= args.tile_size <= 1200:
        parser.error('tile-size must be between 200 and 1200')
    for value in (args.box_thresh,args.thresh):
        if value is not None and not 0 <= value <= 1:
            parser.error('thresholds must be between 0 and 1')
    if args.unclip_ratio is not None and not 0 < args.unclip_ratio <= 5:
        parser.error('unclip-ratio must be in (0,5]')
    project=open_project(args.project)
    settings=Settings.model_validate(project.settings) if project.settings else Settings()
    if args.language:
        settings = settings.model_copy(update={'language':args.language})
    detection={k:v for k,v in [('text_det_box_thresh',args.box_thresh),('text_det_thresh',args.thresh),
                               ('text_det_unclip_ratio',args.unclip_ratio)] if v is not None}
    baseline,merged,report=run(project,PaddleEngine(settings),tile_size=args.tile_size,scale=args.scale,
                              rotations=args.rotations,mode=args.mode,detection=detection)
    report.update({'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                   'baseline_count':len(baseline),'experimental_count':len(merged),
                   'baseline':[f.model_dump() for f in baseline], 'experimental':[f.model_dump() for f in merged]})
    if args.reference:
        reference=json.loads(args.reference.read_text('utf-8'))
        report['metrics']={name:evaluate(fragments,reference) for name,fragments in [('baseline',baseline),('experimental',merged)]}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
    for name,fragments in [('baseline',baseline),('experimental',merged)]:
        image=source_image(project)
        draw=ImageDraw.Draw(image)
        for f in fragments:
            draw.rectangle((f.bbox.x,f.bbox.y,f.bbox.right,f.bbox.bottom),outline='red',width=1)
        image.save(args.output_dir/f'{name}.png')
    print(json.dumps({k:v for k,v in report.items() if k in {'baseline_count','experimental_count','metrics','baseline_seconds','additional_seconds'}},ensure_ascii=False))


if __name__ == '__main__':
    main()
