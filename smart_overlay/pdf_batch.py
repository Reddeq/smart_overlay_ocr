"""Transactional, disk-spooled native PDF batch conversion."""
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable

from . import export
from .input import open_project, save_project
from .native_pdf import PDF_ZOOM, native_pdf_project, pdf_info
from .one_click import ProcessingCancelled
from .pdf_pages import format_pdf_pages, parse_pdf_pages
from .settings import Settings, data_dir


@dataclass
class PdfPageGroup:
    pptx: Path
    pages: list[int]
    project_files: list[Path]
    warnings: list[str] = field(default_factory=list)
    needs_review: int = 0


@dataclass
class PdfBatchResult:
    groups: list[PdfPageGroup]
    selected_pages: list[int]
    warnings: list[str] = field(default_factory=list)
    needs_review: int = 0


def export_pptx_pages(projects, output, settings, **kwargs):
    """Resolve the shared exporter lazily; no per-page PPTX intermediates."""
    return export.export_pptx_pages(projects, output, settings, **kwargs)


def convert_pdf_pages(source, *, pages='', destination=None, settings=None,
                      progress: Callable[[str], None] = lambda _: None,
                      cancelled: Callable[[], bool] = lambda: False) -> PdfBatchResult:
    def check():
        if cancelled():
            raise ProcessingCancelled('PDF conversion cancelled; no final files published')

    check()
    source = Path(source).resolve()
    if source.suffix.lower() != '.pdf':
        raise ValueError('Native PDF conversion requires a PDF source')
    info = pdf_info(source)
    selected = parse_pdf_pages(pages, info['pages'])
    base = Path(destination).resolve() if destination is not None else source.with_suffix('.pptx')
    if base.suffix.lower() != '.pptx':
        raise ValueError('Destination must be a .pptx base path')
    settings = (settings or Settings()).model_copy(deep=True)
    settings.ocr_engine = 'native_pdf'
    formats = []
    for page in selected:
        width, height = info['page_sizes'][page - 1]
        if not all(math.isfinite(v) and v > 0 for v in (width, height)):
            raise ValueError(f'Invalid PDF page size: {page}')
        if max(width, height) / min(width, height) > 56:
            raise ValueError(f'Page {page} is too elongated for PowerPoint')
        for representative, members in formats:
            same_orientation = (representative[0] > representative[1]) == (width > height)
            if same_orientation and all(abs(a-b) <= .5 for a,b in zip(representative, (width,height))):
                members.append(page)
                break
        else:
            formats.append(((width,height), [page]))
    check()
    base.parent.mkdir(parents=True, exist_ok=True)
    project_root = data_dir() / 'projects'
    project_root.mkdir(parents=True, exist_ok=True)
    spool = Path(tempfile.mkdtemp(prefix='pdf-batch-', dir=project_root))
    stage = None
    published = []
    complete = False
    try:
        stage = Path(tempfile.mkdtemp(prefix='.pdf-batch-', dir=base.parent))
        groups = []
        staged_outputs = []
        for index, (_, members) in enumerate(formats):
            check()
            paths = []
            warnings = []
            review = 0
            # One common scale for the group; use raster-rounded dimensions so
            # every slide dimension remains within PowerPoint's 1..56in bounds.
            sizes = [(math.ceil(info['page_sizes'][p-1][0] * PDF_ZOOM),
                      math.ceil(info['page_sizes'][p-1][1] * PDF_ZOOM)) for p in members]
            lower = max(1/min(size) for size in sizes)
            upper = min(56/max(size) for size in sizes)
            if lower > upper:
                raise ValueError('No common PowerPoint scale for PDF group')
            group_settings = settings.model_copy(deep=True)
            group_settings.inches_per_pixel = min(upper, max(lower, 1/(72*PDF_ZOOM)))
            for page in members:
                check()
                progress(f'PDF: page {page}/{info["pages"]}')
                def page_progress(message, page=page):
                    progress(f'PDF: page {page}/{info["pages"]}: {message}')
                try:
                    project = native_pdf_project(
                        source, page, detect_tables=settings.native_pdf_detect_tables,
                        ignore_numbers=settings.native_pdf_ignore_numbers, progress=page_progress)
                except (ValueError,OSError) as exc:
                    raise ValueError(f'Страница {page}: {exc}') from exc
                check()
                project.settings = group_settings.model_dump()
                meta = project.raw.get('native_pdf', {})
                page_warnings = [str(issue) for issue in meta.get('unresolved', [])]
                page_warnings.extend(str(issue) for issue in meta.get('font_warnings', []))
                warnings.extend(f'Page {page}: {issue}' for issue in page_warnings)
                review += sum(not obj.excluded and obj.needs_review for obj in project.objects)
                project.messages.extend(page_warnings)
                path = spool / f'page-{page:02d}.socr.json'
                save_project(project, path)
                paths.append(path)
                del project
            check()
            temporary = stage / f'group-{index}.pptx'
            progress(f'PDF: export pages {format_pdf_pages(members)}')
            def projects(paths=paths):
                for path in paths:
                    check()
                    yield open_project(path)
            export_pptx_pages(projects(), temporary, group_settings,
                              allow_unreviewed=True, inline_text_fill=True)
            check()
            filename = f'{base.stem}_{format_pdf_pages(members)}.pptx'
            groups.append(PdfPageGroup(base.with_name(filename), members, paths, warnings, review))
            staged_outputs.append(temporary)
        check()
        # Exclusive hard-link creation never replaces a competing writer's file.
        # All exports have succeeded before any final PPTX becomes visible.
        for group, temporary in zip(groups, staged_outputs):
            check()
            candidate = group.pptx
            number = 1
            while True:
                check()
                try:
                    os.link(temporary, candidate)
                    break
                except FileExistsError:
                    candidate = group.pptx.with_name(f'{group.pptx.stem}-{number:02d}.pptx')
                    number += 1
            published.append(candidate)
            group.pptx = candidate
        check()
        result = PdfBatchResult(groups, selected,
                                [w for group in groups for w in group.warnings],
                                sum(group.needs_review for group in groups))
        complete = True
        return result
    finally:
        if not complete:
            for path in published:
                path.unlink(missing_ok=True)
            shutil.rmtree(spool)
        if stage is not None:
            shutil.rmtree(stage)
