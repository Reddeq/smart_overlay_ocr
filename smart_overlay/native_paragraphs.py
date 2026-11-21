"""Logical editable paragraphs; physical lines remain immutable geometry."""
import re
from collections import Counter
from .models import NativeParagraph

_EXPLICIT_LIST = re.compile(r'^\s*(?:\d+[.)]|[A-Za-zА-Яа-я][.)]|[•●▪‣–—-])\s+')
_BARE_LIST = re.compile(r'^\s*(\d{1,3})\s+(?=(?:[*†‡]+\s*)?[^\W\d_])')
LIST_START = re.compile(_EXPLICIT_LIST.pattern+'|'+_BARE_LIST.pattern)


def _size(line):
    weights=Counter()
    for run in line.runs:
        weights[run.font_size]+=len(run.text.strip())
    return weights.most_common(1)[0][0] if weights else 12.


def normalize_layout(layout):
    # Bare numbering needs corroboration: do not turn a wrapped measurement
    # such as "20 mm minimum" into a fresh list item merely for its digits.
    numbers=[int(match.group(1)) for line in layout.lines
             if (match:=_BARE_LIST.match(''.join(run.text for run in line.runs)))]
    bare_list=any(b==a+1 for a,b in zip(numbers,numbers[1:]))
    def starts_item(text):
        return bool(_EXPLICIT_LIST.match(text) or (bare_list and _BARE_LIST.match(text)))
    paragraphs = []
    previous = None
    for line in layout.lines:
        runs = [run.model_copy(deep=True) for run in line.runs]
        while runs and not runs[0].text.strip():
            runs.pop(0)
        while runs and not runs[-1].text.strip():
            runs.pop()
        if not runs:
            continue
        runs[0].text = runs[0].text.lstrip()
        runs[-1].text = runs[-1].text.rstrip()
        text = ''.join(run.text for run in runs)
        size = _size(line)
        boundary = previous is None or starts_item(text)
        if previous is not None:
            before=_size(previous)
            boundary |= max(size,before)>min(size,before)*1.25
            boundary |= bool(previous.line_spacing and previous.line_spacing > size * 1.8)
            paragraph_text = ''.join(run.text for run in paragraphs[-1].runs)
            if not starts_item(paragraph_text):
                boundary |= line.indent > previous.indent + size * .6
        if boundary:
            paragraphs.append(NativeParagraph(runs=runs))
        else:
            last = paragraphs[-1].runs[-1]
            # Only explicit soft hyphens prove discretionary hyphenation.
            # Keep ordinary hyphens in ambiguous compounds and identifiers.
            if last.text.endswith('\u00ad'):
                last.text = last.text[:-1]
            elif not (last.text.endswith('-') and len(last.text) > 1
                      and last.text[-2].isalnum() and text[0].isalnum()):
                last.text += ' '
            paragraphs[-1].runs.extend(runs)
        previous = line
    layout.paragraphs = paragraphs
    return '\n'.join(''.join(run.text for run in p.runs) for p in paragraphs)
