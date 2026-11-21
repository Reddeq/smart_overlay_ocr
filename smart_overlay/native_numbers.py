"""Conservative whole-label numeric classification, never a text substitution."""
import re
import unicodedata

_SYMBOLS = frozenset('.,+-−–—±×÷*/⁄:;%‰°′″\'"()[]{}=<>≤≥≈~⌀Øø∅√²³')


def is_numeric_label(text):
    text = unicodedata.normalize('NFKC', text).strip()
    # A bare enumeration marker may belong to prose in a separate PDF line.
    if re.fullmatch(r'\d+[.)]', text):
        return False
    return (any(ch.isdecimal() for ch in text)
            and all(ch.isdecimal() or ch.isspace() or ch in _SYMBOLS or ch in '^·⋅'
                    or unicodedata.category(ch)=='Sm' for ch in text))
