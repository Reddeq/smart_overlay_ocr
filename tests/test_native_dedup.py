"""Pure canonicalization contracts using actual PyMuPDF extraction."""
from copy import deepcopy

import pytest

pymupdf=pytest.importorskip('pymupdf')

from smart_overlay.native_dedup import deduplicate_text


def _lines(data):
    return [line for block in data['blocks'] if block.get('type')==0 for line in block['lines']]


def _texts(data):
    return [''.join(span['text'] for span in line['spans']) for line in _lines(data)]


def test_canonicalization_preserves_input_reading_order_and_is_idempotent():
    with pymupdf.open() as doc:
        page=doc.new_page(width=400,height=250)
        for x,y,text in ((40,50,'B'),(40,50,'B'),(40,150,'LONGER TEXT'),(40,150,'LONGER TEXT')):
            page.insert_text((x,y),text,fontsize=12)
        original=page.get_text('rawdict',sort=True)
    snapshot=deepcopy(original)
    clean,diagnostics=deduplicate_text(original)
    assert original==snapshot
    assert _texts(clean)==['B','LONGER TEXT']
    # MuPDF can already coalesce an immediately overprinted single glyph.
    assert diagnostics['removed_lines']==len(_lines(original))-len(_lines(clean))
    assert diagnostics['removed_lines']>0
    twice,again=deduplicate_text(clean)
    assert twice==clean
    assert again['removed_lines']==again['removed_glyphs']==0


def test_full_copy_is_not_truncated_by_an_earlier_short_fragment():
    with pymupdf.open() as doc:
        page=doc.new_page(width=400,height=200)
        page.insert_text((40,60),'SHORT',fontsize=12)
        page.insert_text((40,60),'SHORT AND COMPLETE',fontsize=12)
        original=page.get_text('rawdict',sort=True)
    clean,diagnostics=deduplicate_text(original)
    assert _texts(clean)==['SHORT AND COMPLETE']
    assert diagnostics['removed_glyphs']==len('SHORT')


def test_coincident_repeated_characters_inside_one_extracted_sequence_survive():
    with pymupdf.open() as doc:
        page=doc.new_page(width=400,height=200)
        page.insert_text((40,60),'AA',fontsize=12)
        data=page.get_text('rawdict',sort=True)
    # Exercise ambiguous zero-advance geometry inside a single original line;
    # a duplicate-source-line filter must not rewrite that original sequence.
    span=_lines(data)[0]['spans'][0]
    span['chars'][1]['origin']=span['chars'][0]['origin']
    span['chars'][1]['bbox']=span['chars'][0]['bbox']
    clean,diagnostics=deduplicate_text(data)
    assert _texts(clean)==['AA']
    assert diagnostics['removed_glyphs']==0
