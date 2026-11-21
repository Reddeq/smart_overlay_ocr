"""Strict one-based PDF page selections and stable filename labels."""
import re
from collections.abc import Iterable


def parse_pdf_pages(spec: str, total: int) -> list[int]:
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ValueError("PDF должен содержать хотя бы одну страницу")
    selection = spec.strip()
    if not selection or selection.lower() == 'all':
        return list(range(1, total + 1))
    pages = set()
    for term in selection.split(','):
        match = re.fullmatch(r'\s*([0-9]+)\s*(?:-\s*([0-9]+)\s*)?', term)
        if not match:
            raise ValueError(f"Некорректное указание страниц: {term!r}. Например: 1, 2, 6-9")
        first = int(match[1])
        last = int(match[2]) if match[2] else first
        if not 1 <= first <= last <= total:
            raise ValueError(f"Неверный диапазон {term!r}: страницы должны идти по возрастанию от 1 до {total}")
        pages.update(range(first, last + 1))
    return sorted(pages)


def format_pdf_pages(pages: Iterable[int]) -> str:
    ordered = sorted(set(pages))
    if any(isinstance(p, bool) or not isinstance(p, int) or p < 1 for p in ordered):
        raise ValueError("Page numbers must be positive integers")
    runs = []
    for page in ordered:
        if runs and page == runs[-1][1] + 1:
            runs[-1][1] = page
        else:
            runs.append([page, page])
    return ','.join(f'{a:02d}' if a == b else f'{a:02d}-{b:02d}' for a, b in runs)
