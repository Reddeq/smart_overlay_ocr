"""Born-digital (vector) PDF -> editable text/table overlays, without OCR.

Use PDF text spans and ruled cell geometry, not inferred visual OCR.
The displayed PDF background is rendered with *text removed* but all vector
paths and images preserved, preventing duplicate text and opaque masks.
Work is performed on an in-memory PDF; the original input is never modified.
Currently one PDF page per PowerPoint slide/file.
"""
from __future__ import annotations

import base64
from collections import Counter, defaultdict
from bisect import bisect_right
import hashlib
import math
from pathlib import Path
from statistics import median

from .models import Box, Cell, Fragment, Project, Source, Style, Table, Text, uid
from .structure import validate_project
from .vector_grid import GridCell, VectorGrid, discover_vector_grids

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
    """Keep the actual embedded PDF font family instead of replacing with Arial.

    An installed matching font on the PowerPoint machine is needed for an exact
    visual match. The PDF font program itself is not bundled or redistributed.
    """
    name=(value or "").split("+")[-1].split(",")[0].strip()
    return name or "Arial"


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


def _fallback_grids(page):
    """Small/simple tables PyMuPDF detects without CAD-style repeated strokes."""
    fitz=_pdf_lib()
    try:
        detections=page.find_tables(strategy="lines").tables
    except (RuntimeError, ValueError, AttributeError):
        return []
    proposals=[]
    for table in detections:
        bbox=tuple(float(v) for v in table.bbox)
        if ((bbox[2]-bbox[0])*(bbox[3]-bbox[1]) >
                .24*page.rect.get_area() or
                table.row_count<2 or table.col_count<1):
            continue  # The outside page frame is NOT one giant table.
        raw=[fitz.Rect(rect) for row in table.rows
             for rect in row.cells if rect]
        if not raw:
            continue
        xedges=_cluster_pdf_edges([bbox[0],bbox[2]]+
            [v for rect in raw for v in (rect.x0,rect.x1)])
        yedges=_cluster_pdf_edges([bbox[1],bbox[3]]+
            [v for rect in raw for v in (rect.y0,rect.y1)])
        nr,nc=len(yedges)-1,len(xedges)-1
        if nr<2 or nc<1 or nr*nc>1600:
            continue
        occupied=set()
        cells=[]
        for rect in raw:
            c0=_edge_index(xedges,rect.x0)
            c1=_edge_index(xedges,rect.x1)
            r0=_edge_index(yedges,rect.y0)
            r1=_edge_index(yedges,rect.y1)
            if c0>=c1 or r0>=r1:
                break
            footprint={(r,c) for r in range(r0,r1) for c in range(c0,c1)}
            if footprint & occupied:
                break
            occupied|=footprint
            cells.append(GridCell(r0,c0,r1-r0,c1-c0,tuple(rect)))
        if len(occupied)!=nr*nc:
            continue
        proposals.append(VectorGrid(
            (xedges[0],yedges[0],xedges[-1],yedges[-1]),
            tuple(xedges),tuple(yedges),tuple(cells),"pymupdf_lines"))
    return proposals


def _cluster_pdf_edges(values):
    from .vector_grid import _cluster
    return _cluster(values,.7)


def _raw_pdf_characters(page):
    """Direct PDF glyphs, retaining the font and text order of each line."""
    raw=[]
    for bi,block in enumerate(page.get_text("rawdict",sort=True)["blocks"]):
        if block.get("type")!=0:
            continue
        for li,line in enumerate(block.get("lines",[])):
            for span in line.get("spans",[]):
                style={"font":span.get("font","Arial"),
                       "size":span.get("size",10),
                       "color":span.get("color",0)}
                for ch in span.get("chars",[]):
                    char=ch.get("c","")
                    if not char:
                        continue
                    box=ch.get("bbox")
                    if box is None:
                        continue
                    raw.append({"line":(bi,li),"char":char,
                                "bbox":box,"style":style,
                                "rotation":_line_rotation(line)})
    return raw


def _cell_content(characters,zoom,width,height):
    """Return printable text and conservative style from actual PDF characters."""
    if not characters:
        return "",None,None,False
    by_line=defaultdict(list)
    for ch in characters:
        by_line[ch["line"]].append(ch)
    rows=sorted(by_line.values(),key=lambda group:(
        min(t["bbox"][1] for t in group),
        min(t["bbox"][0] for t in group)))
    text="\n".join("".join(t["char"] for t in row).strip()
                   for row in rows).strip()
    if not text:
        return "",None,None,False
    weighted=Counter((ch["style"]["font"],round(float(ch["style"]["size"]),2),
                      ch["style"]["color"]) for ch in characters
                     if not ch["char"].isspace())
    font,size,color=weighted.most_common(1)[0][0]
    style=Style(font=_font_name(font),font_size=max(3.,min(256.,size)),
                color=_pdf_color(color),fill="#FFFFFF",border_pt=0)
    xs=[float(ch["bbox"][0]) for ch in characters]
    ys=[float(ch["bbox"][1]) for ch in characters]
    xe=[float(ch["bbox"][2]) for ch in characters]
    ye=[float(ch["bbox"][3]) for ch in characters]
    bbox=_safe_rect((min(xs),min(ys),max(xe),max(ye)),width,height,zoom)
    caution=len(weighted)>2 or any("\ufffd" in ch["char"] for ch in characters)
    return text,style,bbox,caution


def _table_from_grid(grid,original_fragments,characters,source,zoom):
    """Assign every *character* to its real vector cell, splitting PDF lines
    which cross column borders (frequent in engineering specifications).
    """
    x0,y0,x1,y1=grid.bbox
    xs,ys=grid.xedges,grid.yedges
    nr,nc=len(ys)-1,len(xs)-1
    owner={}
    for index,cell in enumerate(grid.cells):
        for r in range(cell.row,cell.row+cell.rowspan):
            for c in range(cell.col,cell.col+cell.colspan):
                owner[(r,c)]=index
    if len(owner)!=nr*nc:
        return None
    groups=defaultdict(list)
    text_count=0
    for char in characters:
        a,b,c,d=char["bbox"]
        cx=(float(a)+float(c))/2
        cy=(float(b)+float(d))/2
        if not (x0<=cx<x1 and y0<=cy<y1):
            continue
        col=bisect_right(xs,cx)-1
        row=bisect_right(ys,cy)-1
        cell=owner.get((row,col))
        if cell is None:
            continue
        groups[cell].append(char)
        if char["char"].strip():
            text_count+=1
    if text_count<3:
        return None
    cells=[]
    extra=[]
    caution=False
    for index,cell in enumerate(grid.cells):
        content,style,fragment_box,bad=_cell_content(
            groups[index],zoom,source.width,source.height)
        caution|=bad
        ids=[]
        if content and fragment_box:
            fid=uid("pdf_cell")
            extra.append(Fragment(id=fid,text=content,
                bbox=fragment_box,polygon=_polygon(fragment_box),
                confidence=1,origin="region"))
            ids=[fid]
        cells.append(Cell(row=cell.row,col=cell.col,
            rowspan=cell.rowspan,colspan=cell.colspan,
            text=content,ocr_ids=ids,
            style=style or Style(border_pt=0)))
    original_ids=set()
    for frag in original_fragments:
        rect=_fragment_pdf_bbox(frag,zoom)
        area=rect.get_area()
        if (area and
                _rect_overlap(_pdf_lib().Rect(grid.bbox),rect)>=.48*area):
            original_ids.add(frag.id)
    if not original_ids:
        return None
    box=_safe_rect(grid.bbox,source.width,source.height,zoom)
    if box is None:
        return None
    xpixel=[x*zoom for x in xs]
    ypixel=[y*zoom for y in ys]
    box=Box(x=xpixel[0],y=ypixel[0],
            w=xpixel[-1]-xpixel[0],h=ypixel[-1]-ypixel[0])
    try:
        table=Table(
            id=uid("pdf_table"),bbox=box,rows=nr,columns=nc,
            row_heights=[b-a for a,b in zip(ypixel,ypixel[1:])],
            col_widths=[b-a for a,b in zip(xpixel,xpixel[1:])],
            cells=cells,ocr_ids=[x.id for x in extra],
            style=Style(font=next((c.style.font for c in cells if c.text),"Arial"),
                        border_pt=0),
            needs_review=caution,
            review_notes=(["Смешанные PDF-шрифты/нечитаемые символы в ячейках"]
                          if caution else []))
    except ValueError:
        return None
    return table,original_ids,extra


def _native_tables(page, original_fragments, source, zoom):
    """Prioritize actual CAD strokes over generic table heuristics."""
    vector=discover_vector_grids(page)
    fallback=_fallback_grids(page)
    chars=_raw_pdf_characters(page)
    assigned=set()
    extra=[]
    tables=[]
    warnings=[]
    for grid in [*vector,*fallback]:
        bbox=_pdf_lib().Rect(grid.bbox)
        if any(_rect_overlap(bbox,_pdf_lib().Rect(t.bbox.x/zoom,
                t.bbox.y/zoom,t.bbox.right/zoom,t.bbox.bottom/zoom)) >
                .18*bbox.get_area() for t in tables):
            continue
        proposal=_table_from_grid(grid,original_fragments,chars,source,zoom)
        if proposal is None:
            continue
        table,ids,fragments=proposal
        if ids & assigned:
            warnings.append("Перекрывающиеся таблицы PDF: оставлена первая сетка")
            continue
        tables.append(table)
        assigned|=ids
        extra.extend(fragments)
    return tables,assigned,extra,warnings,{"vector_candidates":len(vector),
            "fallback_candidates":len(fallback),"reconstructed":len(tables)}


def _styles_for_group(lines,samples):
    sample=samples.get(lines[0].id,{})
    return Style(
        font=_font_name(sample.get("font","")),
        font_size=max(3.,min(256.,float(sample.get("size",10) or 10))),
        color=_pdf_color(sample.get("color",0)),
        fill="#FFFFFF",
        border_pt=0,
    )


def _separate_logical_blocks(lines, samples):
    """One PDF block may include dozens of adjacent drawing columns.

    A logical CAT paragraph is vertically consecutive, left-aligned,
    single-font and single-direction. Same-baseline column headers are NOT
    concatenated with newlines merely because a PDF producer grouped them.
    """
    buckets=[]
    for frag in sorted(lines,key=lambda f:(f.bbox.y,f.bbox.x)):
        sample=samples.get(frag.id,{})
        best=None
        best_gap=None
        for bucket in buckets:
            last=bucket[-1]
            previous=samples.get(last.id,{})
            if (_font_name(sample.get("font"))!=_font_name(previous.get("font"))
                    or abs(frag.rotation-last.rotation)>1):
                continue
            dx=abs(frag.bbox.x-last.bbox.x)
            if dx>max(3.,min(frag.bbox.h,last.bbox.h)*.65):
                continue
            vertical_gap=frag.bbox.y-last.bbox.bottom
            # A real multiline label has consecutive baseline positions.
            # Equal-baseline fragments are probably neighboring table columns.
            if vertical_gap<(-.20*min(frag.bbox.h,last.bbox.h)):
                continue
            if vertical_gap>max(5.,last.bbox.h*.45):
                continue
            if best_gap is None or abs(vertical_gap)<best_gap:
                best=bucket
                best_gap=abs(vertical_gap)
        if best is None:
            buckets.append([frag])
        else:
            best.append(frag)
    return buckets


def _pdf_text_objects(fragments, groups, table_ids, samples):
    by_id={f.id:f for f in fragments}
    objects=[]
    for _,ids in groups.items():
        free=[by_id[i] for i in ids if i not in table_ids]
        if not free:
            continue
        for members in _separate_logical_blocks(free,samples):
            x0=min(f.bbox.x for f in members)
            y0=min(f.bbox.y for f in members)
            x1=max(f.bbox.right for f in members)
            y1=max(f.bbox.bottom for f in members)
            bbox=Box(x=x0,y=y0,w=x1-x0,h=y1-y0)
            bad_encoding=any(
                "\ufffd" in f.text or "\x00" in f.text or
                any("\uac00" <= ch <= "\ud7af" for ch in f.text)
                for f in members)
            uneven=any(abs(f.rotation-members[0].rotation)>1 for f in members)
            objects.append(Text(
                id=uid("pdf_block"),bbox=bbox,
                text="\n".join(f.text for f in members),
                ocr_ids=[f.id for f in members],
                style=_styles_for_group(members,samples),
                confidence=1.0,rotation=members[0].rotation,
                needs_review=bad_encoding or uneven,
                review_notes=(
                    ["PDF: проверьте шрифт/кодировку или поворот текста"]
                    if bad_encoding or uneven else []),
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
        tables,table_ids,cell_fragments,table_warnings,table_diagnostics=(
            _native_tables(page,fragments,source,zoom))
        texts=_pdf_text_objects(fragments,groups,table_ids,samples)
        extracted_lines=len(fragments)
        fragments=[f for f in fragments if f.id not in table_ids]+cell_fragments
        progress("PDF: создание фона без исходного текста")
        clean_png=_clean_background(raw,page_number,zoom)
        source.raster_b64=base64.b64encode(clean_png).decode("ascii")
        result=Project(source=source,fragments=fragments,objects=[*tables,*texts])
        detected_fonts=sorted({
            _font_name(sample.get("font","")) for sample in samples.values()
            if sample.get("font")})
        if any(name not in ("Arial","Calibri","Times New Roman",
                            "Helvetica","Courier") for name in detected_fonts):
            result.messages.append(
                "PDF использует специальный шрифт: "+
                ", ".join(detected_fonts[:10])+
                ". Для совпадения в PowerPoint он должен быть установлен "
                "на компьютере; шрифты из PDF не распространяются.")
        result.raw["native_pdf"]={
            "page":page_number,"pages":len(doc),"zoom":zoom,
            "source_fonts":detected_fonts,
            "page_size_pt":[page.rect.width,page.rect.height],
            "extracted_lines":extracted_lines,
            "cell_fragments":len(cell_fragments),
            "table_detection":table_diagnostics,
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
