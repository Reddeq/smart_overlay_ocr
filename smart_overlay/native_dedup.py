"""Conservative normalization of repeated PDF text drawing operations.

Only an entire source line whose glyphs are already present is discarded.
Comparing glyph geometry, not span boundaries, also handles fragmented copies.
The original order, formatting and geometry of retained lines are unchanged.
"""
from collections import defaultdict
from dataclasses import dataclass
import math


# PDF points, independent of raster DPI. Allow round-off, not displaced shadows.
_TOLERANCE=0.01


@dataclass
class _Glyph:
    text: str
    origin: tuple
    bbox: tuple
    direction: tuple
    style: tuple
    line: dict


def _glyph(char,span,line):
    origin=tuple(char.get('origin',()))
    bbox=tuple(char.get('bbox',()))
    if (len(origin)!=2 or len(bbox)!=4
            or not all(math.isfinite(v) for v in (*origin,*bbox))):
        return None
    return _Glyph(char.get('c',''),origin,bbox,
                  tuple(round(v,6) for v in line.get('dir',(1,0))),
                  (span.get('font'),round(span.get('size',0),4),span.get('flags'),
                   span.get('color'),span.get('alpha',255),span.get('char_flags'),
                   span.get('bidi',0),char.get('synthetic',False)),line)


def _same(a,b):
    return (a.text==b.text and a.style==b.style
            and all(abs(x-y)<=_TOLERANCE for x,y in zip(a.bbox,b.bbox)))


def deduplicate_text(data):
    """Return canonical rawdict plus bounded diagnostics; never mutate input.

    A spatial index limits comparison to neighboring glyph origins. Prefer the
    most complete source lines, so a short earlier fragment cannot truncate a
    later full copy. Repeated characters inside one source line are never
    deleted. Conflicting styles/text and partial overlaps retain both lines.
    """
    result=dict(data)
    blocks=[]
    lines=[]
    for block in data.get('blocks',[]):
        if block.get('type')!=0:
            blocks.append(block)
            continue
        clean_block=dict(block)
        clean_block['lines']=[]
        blocks.append(clean_block)
        for line in block.get('lines',[]):
            clean_line=dict(line)
            clean_line['spans']=[dict(span,text=''.join(c.get('c','') for c in span.get('chars',[])))
                                 for span in line.get('spans',[])]
            clean_block['lines'].append(clean_line)
            chars=[(char,span) for span in clean_line['spans'] for char in span.get('chars',[])]
            lines.append((clean_line,chars))

    buckets=defaultdict(list)
    removed=set()
    for line,chars in sorted(lines,key=lambda item:-len(item[1])):
        entries=[_glyph(char,span,line) for char,span in chars]
        matched=[]
        conflicts=[]
        covered=[]
        for entry in entries:
            if entry is None:
                matched.append(False)
                continue
            x,y=(math.floor(v/_TOLERANCE) for v in entry.origin)
            nearby=[candidate for dx in (-1,0,1) for dy in (-1,0,1)
                    for candidate in buckets.get((x+dx,y+dy,entry.direction),())
                    if all(abs(a-b)<=_TOLERANCE for a,b in zip(entry.origin,candidate.origin))]
            identical=[candidate for candidate in nearby if _same(entry,candidate)]
            matched.append(bool(identical))
            if entry.text.strip():
                covered.extend(identical)
                conflicts.extend(candidate for candidate in nearby if candidate.text.strip()
                                 and (not _same(entry,candidate)
                                      or candidate.line.get('_dedup_ambiguous')))
        ink=[match for (char,_),match in zip(chars,matched) if char.get('c','').strip()]
        partial=any(ink) and not all(ink)
        if conflicts or partial:
            line['_dedup_ambiguous']=True
            for candidate in conflicts+(covered if partial else []):
                candidate.line['_dedup_ambiguous']=True
        elif ink and all(ink):
            removed.add(id(line))
            continue
        # Insert only after inspecting the whole line. Zero-advance repeated
        # characters in one original text sequence must retain their semantics.
        for entry in entries:
            if entry is not None:
                x,y=(math.floor(v/_TOLERANCE) for v in entry.origin)
                buckets[x,y,entry.direction].append(entry)

    for block in blocks:
        if block.get('type')==0:
            block['lines']=[line for line in block['lines'] if id(line) not in removed]
    result['blocks']=[b for b in blocks if b.get('type')!=0 or b['lines']]
    ambiguous=[line for line,_ in lines if line.get('_dedup_ambiguous')]
    diagnostics={
        'coordinate_tolerance_pt':_TOLERANCE,
        'source_lines':len(lines),
        'source_glyphs':sum(len(chars) for _,chars in lines),
        'removed_lines':len(removed),
        'removed_glyphs':sum(len(chars) for line,chars in lines if id(line) in removed),
        'ambiguous_lines':len(ambiguous),
        'ambiguous_examples':[''.join(s['text'] for s in line['spans'])[:160]
                              for line in ambiguous[:8]],
    }
    return result,diagnostics
