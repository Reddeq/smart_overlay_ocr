"""Born-digital (vector) PDF -> editable text/table overlays, without OCR.

Use PDF text spans and ruled cell geometry, not inferred visual OCR.
The displayed PDF background is rendered with *text removed* but all vector
paths and images preserved, preventing duplicate text and opaque masks.
Work is performed on an in-memory PDF; the original input is never modified.
Currently one PDF page per PowerPoint slide/file.
"""
from __future__ import annotations

import base64
from collections import defaultdict
import hashlib
import math
from pathlib import Path
from statistics import median

from .models import Box, Cell, Fragment, Project, Source, Style, Table, Text, uid
from .structure import validate_project

PDF_ZOOM = 2.0
MAX_PDF_BYTES = 64 * 1024 * 1024
MAX_PREVIEW_PIXELS = 32_000_000


class NativePDFError(ValueError):
    """A digital PDF cannot safely be converted without guessing its structure."""


def _pdf_lib():
    try:
        import pymupdf
        return pymupdf
    except ImportError as exc:
        raise NativePDFError(
            "Для цифровых PDF необходим PyMuPDF: python -m pip install 'PyMuPDF>=1.26,<2'"
        ) from exc


def _read_pdf(path):
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        raise NativePDFError("Режим цифровых чертежей пока поддерживает PDF")
    raw = path.read_bytes()
    if not raw.startswith(b"%PDF-"):
        raise NativePDFError("Файл не содержит заголовка PDF")
    if len(raw) > MAX_PDF_BYTES:
        raise NativePDFError("PDF больше 64 МБ; сначала разделите его на отдельные листы")
    fitz = _pdf_lib()
    try:
        document = fitz.open(stream=raw, filetype="pdf")
    except (RuntimeError, ValueError) as exc:
        raise NativePDFError(f"Не удалось открыть цифровой PDF: {exc}") from exc
    if document.needs_pass:
        document.close()
        raise NativePDFError("PDF защищён паролем; сначала получите разрешённую копию")
    return raw, document


def _page(document, number, zoom):
    if not 1 <= number <= len(document):
        raise NativePDFError(f"Страница {number} отсутствует (всего {len(document)})")
    page = document[number-1]
    if page.rotation:
        raise NativePDFError(
            "Пока не поддерживается PDF с Rotate=90/180/270; "
            "перед конвертацией нормализуйте поворот страницы.")
    if page.rect.width * zoom * page.rect.height * zoom > MAX_PREVIEW_PIXELS:
        raise NativePDFError("Страница слишком велика для безопасного рендеринга")
    return page


def pdf_info(path):
    raw, doc = _read_pdf(path)
    try:
        return {"pages": len(doc), "page_sizes": [
            [round(doc[i].rect.width, 2),round(doc[i].rect.height,2)]
            for i in range(len(doc))]}
    finally:
        doc.close()


def _render_png(page, zoom):
    fitz = _pdf_lib()
    return page.get_pixmap(matrix=fitz.Matrix(zoom,zoom),
                           colorspace=fitz.csRGB,alpha=False,
                           annots=False).tobytes("png")


def _make_source(raw,path,png,page,zoom):
    # PyMuPDF raster rounds fractional PDF point dimensions to actual pixels.
    # Use exact PNG dimensions rather than approximating with round().
    import io
    from PIL import Image
    with Image.open(io.BytesIO(png)) as rendered:
        width,height=rendered.size
    return Source(
        filename=Path(path).name,
        width=width,
        height=height,
        mode="PDF",
        orientation=1,
        sha256=hashlib.sha256(raw).hexdigest(),
        original_b64=base64.b64encode(raw).decode("ascii"),
        raster_b64=base64.b64encode(png).decode("ascii"),
        raster_mime="image/png",
    )


def preview_pdf(path, page_number=1, *, zoom=PDF_ZOOM):
    """Preview actual PDF page (with text); extraction is deferred to conversion."""
    raw,document=_read_pdf(path)
    try:
        page=_page(document,page_number,zoom)
        source=_make_source(raw,path,_render_png(page,zoom),page,zoom)
        project=Project(source=source)
        project.raw["native_pdf_preview"]={
            "page":page_number,"page_count":len(document),"zoom":zoom}
        return project
    finally:
        document.close()


def _safe_rect(rect,width,height,zoom):
    """Source PDF points -> positive pixel box contained in rendered page."""
    x0,y0,x1,y1=map(float,rect)
    left=max(0.,min(float(width),x0*zoom))
    top=max(0.,min(float(height),y0*zoom))
    right=max(0.,min(float(width),x1*zoom))
    bottom=max(0.,min(float(height),y1*zoom))
    if right-left<=.15 or bottom-top<=.15:
        return None
    return Box(x=left,y=top,w=right-left,h=bottom-top)


def _polygon(box):
    return [(box.x,box.y),(box.right,box.y),
            (box.right,box.bottom),(box.x,box.bottom)]


def _pdf_color(value):
    return f"#{(int(value) & 0xffffff):06X}"


def _font_name(value):
    name=(value or "").split("+")[-1]
    return name if name in ("Arial","Helvetica","Times-Roman","Courier",
                            "Times New Roman","Calibri") else "Arial"


def _line_rotation(line):
    vector=line.get("dir",(1,0))
    if len(vector)!=2:
        return 0.
    angle=math.degrees(math.atan2(float(vector[1]),float(vector[0])))
    return round(angle,2)


def _read_lines(page, source, zoom):
    """Preserve whole PDF text lines as fragments, and block grouping for CAT."""
    fragments=[]
    groups=defaultdict(list)
    samples={}
    skipped=[]
    for block_index,block in enumerate(page.get_text("dict",sort=True).get("blocks",[])):
        if block.get("type")!=0:
            continue
        for line_index,line in enumerate(block.get("lines",[])):
            spans=line.get("spans",[])
            text="".join(str(span.get("text","")) for span in spans)
            if not text.strip():
                continue
            bbox=_safe_rect(line["bbox"],source.width,source.height,zoom)
            if bbox is None:
                skipped.append({"block":block_index,"line":line_index,
                                "reason":"invalid_bounds"})
                continue
            fid=uid("pdf_text")
            frag=Fragment(id=fid,text=text,bbox=bbox,polygon=_polygon(bbox),
                          confidence=1.0,origin="region",rotation=_line_rotation(line))
            fragments.append(frag)
            groups[block_index].append(fid)
            samples[fid]=spans[0] if spans else {}
    return fragments,groups,samples,skipped


def _edges(values,tolerance=.65):
    result=[]
    for value in sorted(map(float,values)):
        if not result or abs(value-result[-1])>tolerance:
            result.append(value)
        else:
            result[-1]=(result[-1]+value)/2
    return result


def _edge_index(edges,value):
    return min(range(len(edges)),key=lambda i:abs(edges[i]-value))


def _rect_overlap(rect,box):
    # Both boxes must be in the SAME PDF-point coordinate space.
    x0,y0,x1,y1 = (box.x0,box.y0,box.x1,box.y1) if hasattr(box,"x0") else (
        box.x,box.y,box.right,box.bottom)
    left=max(rect.x0,x0); top=max(rect.y0,y0)
    right=min(rect.x1,x1); bottom=min(rect.y1,y1)
    return max(0,right-left)*max(0,bottom-top)


def _fragment_pdf_bbox(frag,zoom):
    """Fragments are stored in 2× raster pixels; tables use PDF points."""
    fitz=_pdf_lib()
    box=frag.bbox
    return fitz.Rect(box.x/zoom,box.y/zoom,box.right/zoom,box.bottom/zoom)


def _native_tables(page, fragments, zoom):
    """Rebuild merged native PowerPoint cells from actual PDF line geometry.

    Reject inaccurate/ambiguous proposals, retaining text as ordinary editable
    objects instead of guessing merges. Uses direct PDF cell rectangles.
    """
    fitz=_pdf_lib()
    try:
        detections=page.find_tables(strategy="lines").tables
    except (RuntimeError,ValueError,AttributeError) as exc:
        return [],set(),[f"PDF find_tables: {type(exc).__name__}"]
    found=[]
    assigned=set()
    rejected=[]
    for index,table in enumerate(detections):
        if table.row_count<2 or table.col_count<1:
            continue
        raw_cells=[fitz.Rect(cell) for row in table.rows
                   for cell in row.cells if cell is not None]
        if not raw_cells:
            continue
        bbox=fitz.Rect(table.bbox)
        xedges=_edges([bbox.x0,bbox.x1]+[v for c in raw_cells for v in (c.x0,c.x1)])
        yedges=_edges([bbox.y0,bbox.y1]+[v for c in raw_cells for v in (c.y0,c.y1)])
        cols,rows=len(xedges)-1,len(yedges)-1
        if not (1<=cols<=100 and 2<=rows<=500):
            rejected.append(f"Таблица #{index}: слишком сложная сетка")
            continue
        possible=[]
        occupied=set()
        for rect in raw_cells:
            c0=_edge_index(xedges,rect.x0); c1=_edge_index(xedges,rect.x1)
            r0=_edge_index(yedges,rect.y0); r1=_edge_index(yedges,rect.y1)
            if c1<=c0 or r1<=r0:
                continue
            footprint={(r,c) for r in range(r0,r1) for c in range(c0,c1)}
            if footprint & occupied:
                possible=[]
                break
            occupied.update(footprint)
            possible.append((rect,r0,c0,r1-r0,c1-c0))
        if not possible or len(occupied)!=rows*cols:
            rejected.append(f"Таблица #{index}: неоднозначные объединения")
            continue
        table_fragments=[frag for frag in fragments
                         if frag.id not in assigned and
                         _rect_overlap(bbox,_fragment_pdf_bbox(frag,zoom))>
                         .48*frag.bbox.w*frag.bbox.h/(zoom*zoom)]
        if len(table_fragments)<2:
            # Diagram wires can accidentally be identified as empty tables.
            continue
        box=_safe_rect(bbox,max(1,round(page.rect.width*zoom)),
                       max(1,round(page.rect.height*zoom)),zoom)
        if box is None:
            continue
        if any(_rect_overlap(bbox,other_bbox)>.25*bbox.get_area()
               for other_bbox in [fitz.Rect(t["pdf_bbox"]) for t in found]):
            continue
        cells=[]
        table_ids=[]
        seen=set()
        ambiguous=False
        for cellrect,r,c,rs,cs in possible:
            owned=[frag for frag in table_fragments
                   if frag.id not in seen and
                   _rect_overlap(cellrect,_fragment_pdf_bbox(frag,zoom))>
                   .48*frag.bbox.w*frag.bbox.h/(zoom*zoom)]
            for frag in owned:
                seen.add(frag.id)
            owned.sort(key=lambda f:(f.bbox.y,f.bbox.x))
            ids=[f.id for f in owned]
            table_ids.extend(ids)
            cells.append(Cell(row=r,col=c,rowspan=rs,colspan=cs,
                              text="\n".join(f.text for f in owned),
                              ocr_ids=ids,
                              style=Style(fill="#FFFFFF",font="Arial",font_size=9,
                                          border_pt=0)))
        # Fragment centers lying between cells need manual review, not duplication.
        if any(f.id not in seen for f in table_fragments):
            rejected.append(f"Таблица #{index}: текст на границе ячейки, оставлен как обычный блок")
            continue
        if not table_ids:
            continue
        pixel_x=[max(0,min(page.rect.width*zoom,v*zoom)) for v in xedges]
        pixel_y=[max(0,min(page.rect.height*zoom,v*zoom)) for v in yedges]
        col_widths=[b-a for a,b in zip(pixel_x,pixel_x[1:])]
        row_heights=[b-a for a,b in zip(pixel_y,pixel_y[1:])]
        # Match the model geometry exactly to the sum of its grid sizes.
        box=Box(x=pixel_x[0],y=pixel_y[0],w=sum(col_widths),h=sum(row_heights))
        try:
            native=Table(
                id=uid("pdf_table"),bbox=box,rows=rows,columns=cols,
                row_heights=row_heights,col_widths=col_widths,cells=cells,
                ocr_ids=table_ids,style=Style(border_pt=0),
                confidence=1.0,needs_review=ambiguous,
                review_notes=(["Часть PDF-текста на границах ячеек: проверьте вручную"]
                              if ambiguous else []),
            )
        except ValueError as exc:
            rejected.append(f"Таблица #{index}: структура невалидна ({type(exc).__name__})")
            continue
        found.append({"table":native,"pdf_bbox":tuple(bbox)})
        assigned.update(table_ids)
    return [item["table"] for item in found],assigned,rejected


def _styles_for_group(lines,samples):
    sample=samples.get(lines[0].id,{})
    return Style(
        font=_font_name(sample.get("font","")),
        font_size=max(3.,min(256.,float(sample.get("size",10) or 10))),
        color=_pdf_color(sample.get("color",0)),
        fill="#FFFFFF",
        border_pt=0,
    )


def _pdf_text_objects(fragments, groups, table_ids, samples):
    by_id={f.id:f for f in fragments}
    objects=[]
    for _,ids in groups.items():
        free=[by_id[i] for i in ids if i not in table_ids]
        if not free:
            continue
        # A text block can partially overlap a table: only join remaining
        # lines belonging to the same PDF text block.
        x0=min(f.bbox.x for f in free)
        y0=min(f.bbox.y for f in free)
        x1=max(f.bbox.right for f in free)
        y1=max(f.bbox.bottom for f in free)
        bbox=Box(x=x0,y=y0,w=x1-x0,h=y1-y0)
        bad_encoding=any("\ufffd" in f.text or "\x00" in f.text for f in free)
        objects.append(Text(
            id=uid("pdf_block"),bbox=bbox,text="\n".join(f.text for f in free),
            ocr_ids=[f.id for f in free],
            style=_styles_for_group(free,samples),
            confidence=1.0,rotation=free[0].rotation,
            needs_review=bad_encoding or
                any(abs(f.rotation-free[0].rotation)>1 for f in free),
            review_notes=(["PDF: подозрительная кодировка либо поворот текста"]
                          if bad_encoding or any(
                             abs(f.rotation-free[0].rotation)>1 for f in free)
                          else []),
        ))
    return objects


def _clean_background(raw,page_number,zoom):
    """Preserve PDF graphics & images while removing PDF text in memory.

    Transparent redaction avoids white boxes obliterating wires/table lines.
    The untouched original PDF bytes remain stored as the project's source.
    """
    fitz=_pdf_lib()
    temporary=fitz.open(stream=raw,filetype="pdf")
    try:
        page=temporary[page_number-1]
        if page.get_text().strip():
            page.add_redact_annot(page.rect,fill=False)
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                                  graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                                  text=fitz.PDF_REDACT_TEXT_REMOVE)
        return _render_png(page,zoom)
    finally:
        temporary.close()


def native_pdf_project(path,page_number=1,*,zoom=PDF_ZOOM,progress=lambda _:None):
    """Single born-digital PDF page to validated editable text/table project."""
    raw,doc=_read_pdf(path)
    try:
        page=_page(doc,page_number,zoom)
        progress("PDF: извлечение текстовых блоков")
        # PNG dimensions use PyMuPDF's pixel rounding convention.
        original_png=_render_png(page,zoom)
        source=_make_source(raw,path,original_png,page,zoom)
        fragments,groups,samples,skipped=_read_lines(page,source,zoom)
        if not fragments:
            raise NativePDFError(
                "В PDF нет извлекаемого текста. Вероятно, это скан/контуры: "
                "используйте отдельный OCR-режим.")
        progress("PDF: восстановление таблиц по векторным линиям")
        tables,table_ids,table_warnings=_native_tables(page,fragments,zoom)
        texts=_pdf_text_objects(fragments,groups,table_ids,samples)
        progress("PDF: создание фона без исходного текста")
        clean_png=_clean_background(raw,page_number,zoom)
        source.raster_b64=base64.b64encode(clean_png).decode("ascii")
        result=Project(source=source,fragments=fragments,objects=[*tables,*texts])
        result.raw["native_pdf"]={
            "page":page_number,"pages":len(doc),"zoom":zoom,
            "page_size_pt":[page.rect.width,page.rect.height],
            "extracted_lines":len(fragments),
            "grouped_text_blocks":len(texts),
            "native_tables":len(tables),
            "vector_paths":len(page.get_drawings()),
            "text_redacted_background":True,
            "unresolved":skipped+table_warnings,
        }
        result.messages.append(
            f"Цифровой PDF: {len(texts)} текстовых блоков, "
            f"{len(tables)} редактируемых таблиц, без OCR")
        if table_warnings:
            result.messages.extend(table_warnings)
        validate_project(result)
        return result
    finally:
        doc.close()
