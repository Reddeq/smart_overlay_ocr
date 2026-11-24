"""Born-digital (vector) PDF -> editable text/table overlays, without OCR.

Use PDF text spans and ruled cell geometry, not inferred visual OCR.
The background preserves the ENTIRE original page, including authentic PDF
glyphs, paths and images. Visible editable frames and cells cover the source
using their OWN fills, without separate mask shapes. Legacy source-exact
and glyph-mask projects retain their saved export behavior.
The original input is never modified.
Currently one PDF page per PowerPoint slide/file.
"""
from __future__ import annotations

import base64
from collections import Counter, defaultdict
from bisect import bisect_right
import hashlib
import math
import os
from pathlib import Path
import shlex
from statistics import median
import sys

from .models import (Box, Cell, Fragment, NativeLayout, NativeLine, NativeRun,
                     Project, Source, Style, Table, Text, uid)
from .native_dedup import deduplicate_text
from .native_numbers import is_numeric_label
from .native_paragraphs import LIST_START, normalize_layout
from .structure import validate_project
from .vector_grid import GridCell, LabelRowIndex, VectorGrid, discover_vector_grids

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
        requirement="PyMuPDF>=1.26.4,<2"
        if os.name=="nt":
            executable=sys.executable.replace("'","''")
            command=f"& '{executable}' -m pip install '{requirement}'"
        else:
            command=shlex.join([sys.executable,"-m","pip","install",requirement])
        raise NativePDFError(
            f"Не удалось загрузить PyMuPDF в Python приложения: {sys.executable}\n"
            f"Причина: {exc}\nУстановите зависимость в это окружение:\n{command}"
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


def _page(document, number):
    if not 1 <= number <= len(document):
        raise NativePDFError(f"Страница {number} отсутствует (всего {len(document)})")
    page = document[number-1]
    if page.rotation:
        raise NativePDFError(
            "Пока не поддерживается PDF с Rotate=90/180/270; "
            "перед конвертацией нормализуйте поворот страницы.")
    return page


def pdf_raster_size(width, height, zoom):
    """Use MuPDF's integer pixel bounds, including fractional page dimensions."""
    fitz = _pdf_lib()
    rect = (fitz.Rect(0, 0, width, height) * fitz.Matrix(zoom, zoom)).irect
    return rect.width, rect.height


def pdf_render_zoom(width, height, zoom=PDF_ZOOM):
    """Keep the requested zoom for normal sheets; fit large sheets to the pixel budget."""
    if not all(math.isfinite(v) and v > 0 for v in (width, height, zoom)):
        raise NativePDFError("Некорректные размеры или масштаб страницы PDF")
    # Start with the continuous area limit, then account for integer pixel bounds.
    zoom = min(zoom, math.sqrt(MAX_PREVIEW_PIXELS / width / height))
    low, high = 0.0, zoom
    if math.prod(pdf_raster_size(width, height, zoom)) > MAX_PREVIEW_PIXELS:
        for _ in range(24):
            mid = (low + high) / 2
            if math.prod(pdf_raster_size(width, height, mid)) <= MAX_PREVIEW_PIXELS:
                low = mid
            else:
                high = mid
        zoom = low
    if min(pdf_raster_size(width, height, zoom)) < 1:
        raise NativePDFError("Страница слишком вытянута для безопасного рендеринга")
    return zoom


def pdf_info(path):
    raw, doc = _read_pdf(path)
    try:
        return {"pages": len(doc), "page_sizes": [
            [page.rect.width, page.rect.height]
            for page in doc]}
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
        page=_page(document,page_number)
        zoom=pdf_render_zoom(page.rect.width,page.rect.height,zoom)
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


def _font_family(value):
    name=_font_name(value)
    standard={"Helvetica":"Arial","Helvetica-Bold":"Arial",
              "Helvetica-Oblique":"Arial","Helvetica-BoldOblique":"Arial",
              "Times-Roman":"Times New Roman","Times-Bold":"Times New Roman",
              "Times-Italic":"Times New Roman","Times-BoldItalic":"Times New Roman",
              "Courier":"Courier New","Courier-Bold":"Courier New",
              "Courier-Oblique":"Courier New","Courier-BoldOblique":"Courier New"}
    return standard.get(name,name) or "Arial"


def _line_rotation(line):
    vector=line.get("dir",(1,0))
    if len(vector)!=2:
        return 0.
    angle=math.degrees(math.atan2(float(vector[1]),float(vector[0])))
    return round(angle,2)


def _read_lines(page, source, zoom, text_data=None):
    """Preserve whole PDF text lines as fragments, and block grouping for CAT."""
    fragments=[]
    groups=defaultdict(list)
    samples={}
    skipped=[]
    data=page.get_text("dict",sort=True) if text_data is None else text_data
    for block_index,block in enumerate(data.get("blocks",[])):
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
            quad=_pdf_lib().recover_line_quad(line)
            polygon=[(max(0.,min(source.width,p.x*zoom)),
                      max(0.,min(source.height,p.y*zoom)))
                     for p in (quad.ul,quad.ur,quad.lr,quad.ll)]
            frag=Fragment(id=fid,text=text,bbox=bbox,polygon=polygon,
                          confidence=1.0,origin="region",rotation=_line_rotation(line))
            fragments.append(frag)
            groups[block_index].append(fid)
            samples[fid]={**(spans[0] if spans else {}),"_spans":spans,
                          "_block":block_index,
                          "_dedup_ambiguous":line.get("_dedup_ambiguous",False)}
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


def _raw_pdf_characters(page, text_data=None):
    """Direct PDF glyphs, retaining the font and text order of each line."""
    raw=[]
    data=page.get_text("rawdict",sort=True) if text_data is None else text_data
    for bi,block in enumerate(data["blocks"]):
        if block.get("type")!=0:
            continue
        for li,line in enumerate(block.get("lines",[])):
            for span in line.get("spans",[]):
                style={"font":span.get("font","Arial"),
                       "size":span.get("size",10),
                       "color":span.get("color",0),
                       "flags":span.get("flags",0)}
                for ch in span.get("chars",[]):
                    char=ch.get("c","")
                    if not char:
                        continue
                    box=ch.get("bbox")
                    if box is None:
                        continue
                    raw.append({"line":(bi,li),"char":char,
                                "bbox":box,"style":style,
                                "origin":ch.get("origin",(box[0],box[3])),
                                "rotation":_line_rotation(line),
                                "dedup_ambiguous":line.get("_dedup_ambiguous",False)})
    return raw


def _cell_rows(characters):
    by_line=defaultdict(list)
    for ch in characters:
        by_line[ch["line"]].append(ch)
    rotation=characters[0]["rotation"] if characters else 0
    cs,sn=math.cos(math.radians(rotation)),math.sin(math.radians(rotation))
    rows=sorted(by_line.values(),key=lambda group:(
        min(-t["origin"][0]*sn+t["origin"][1]*cs for t in group),
        min(t["origin"][0]*cs+t["origin"][1]*sn for t in group)))
    for row in rows:
        while row and row[0]["char"].isspace():
            row.pop(0)
        while row and row[-1]["char"].isspace():
            row.pop()
    return [row for row in rows if row]


def _cell_layout(characters,bbox,style):
    if bbox is None:
        return None
    rows=_cell_rows(characters)
    lines=[]
    rotation=characters[0]["rotation"]
    cs,sn=math.cos(math.radians(rotation)),math.sin(math.radians(rotation))
    baselines=[-row[0]["origin"][0]*sn+row[0]["origin"][1]*cs for row in rows]
    for i,row in enumerate(rows):
        runs=[]
        previous=None
        run_start=0.
        for ch in row:
            s=ch["style"]
            origin=ch["origin"][0]*cs+ch["origin"][1]*sn
            if s==previous:
                runs[-1].text+=ch["char"]
            else:
                runs.append(NativeRun(text=ch["char"],font=_font_name(s["font"]),
                    font_size=max(3.,min(256.,float(s["size"]))),color=_pdf_color(s["color"]),
                    bold=bool(int(s.get("flags",0)) & 16),
                    italic=bool(int(s.get("flags",0)) & 2)))
                previous=s
                run_start=origin
            x0,y0,x1,y1=ch["bbox"]
            end=max(x*cs+y*sn for x,y in ((x0,y0),(x1,y0),(x1,y1),(x0,y1)))
            runs[-1].advance=max(0.,end-run_start)
        spacing=baselines[i+1]-baselines[i] if i+1<len(rows) else None
        lines.append(NativeLine(runs=runs,line_spacing=spacing if spacing and spacing>0 else None))
    return NativeLayout(frame=bbox,rotation=rotation,lines=lines,
                        reference_style=style.model_copy()) if lines else None


def _cell_anchors(layout,style,bounds,zoom):
    """Alignment is not padding: large merged-cell insets can grow PPT rows."""
    cs,sn=math.cos(math.radians(layout.rotation)),math.sin(math.radians(layout.rotation))
    def local(rect):
        x0,y0,x1,y1=rect
        points=[(x*cs+y*sn,-x*sn+y*cs)
                for x,y in ((x0,y0),(x1,y0),(x1,y1),(x0,y1))]
        return (min(p[0] for p in points),min(p[1] for p in points),
                max(p[0] for p in points),max(p[1] for p in points))
    a,b,c,d=local(bounds)
    frame=layout.frame
    x0,y0,x1,y1=local((frame.x/zoom,frame.y/zoom,frame.right/zoom,frame.bottom/zoom))
    left,right=max(0.,x0-a),max(0.,c-x1)
    top,bottom=max(0.,y0-b),max(0.,d-y1)
    if abs(left-right)<=max(.75,style.font_size*.2):
        style.align="center"
        left=right=0.
    elif right<left and right<style.font_size*.75:
        style.align="right"
        left=0.
    else:
        right=0.
    if abs(top-bottom)<=max(1.5,style.font_size*.35):
        layout.valign="middle"
        top=bottom=0.
    elif bottom<top and bottom<style.font_size*.75:
        layout.valign="bottom"
        top=0.
    else:
        bottom=0.
    layout.margin=(top,right,bottom,left)
    layout.reference_style=style.model_copy()


def _cell_content(characters,zoom,width,height):
    """Return printable text and conservative style from actual PDF characters."""
    if not characters:
        return "",None,None,False
    rows=_cell_rows(characters)
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
    style.bold=bool(int(rows[0][0]["style"].get("flags",0)) & 16)
    style.italic=bool(int(rows[0][0]["style"].get("flags",0)) & 2)
    xs=[float(ch["bbox"][0]) for ch in characters]
    ys=[float(ch["bbox"][1]) for ch in characters]
    xe=[float(ch["bbox"][2]) for ch in characters]
    ye=[float(ch["bbox"][3]) for ch in characters]
    bbox=_safe_rect((min(xs),min(ys),max(xe),max(ye)),width,height,zoom)
    caution=(len(weighted)>2 or
             any("\ufffd" in ch["char"] or
                 any("\uac00"<=letter<="\ud7af" for letter in ch["char"])
                 for ch in characters))
    return text,style,bbox,caution


def _table_from_grid(grid,original_fragments,characters,source,zoom,border_pt=.5):
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
        layout=_cell_layout(groups[index],fragment_box,style)
        if layout is not None:
            rotations={ch["rotation"] for ch in groups[index]}
            if len(rotations)>1 or layout.rotation not in (0,90,-90):
                return None  # Native table cells cannot reproduce arbitrary rotations.
            _cell_anchors(layout,style,cell.bbox,zoom)
        if style is not None:
            style.border_pt=border_pt
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
            style=style or Style(border_pt=border_pt),native_layout=layout))
    original_ids=set()
    for frag in original_fragments:
        rect=_fragment_pdf_bbox(frag,zoom)
        area=rect.get_area()
        if (area and
                _rect_overlap(_pdf_lib().Rect(grid.bbox),rect)>=.48*area):
            original_ids.add(frag.id)
    if not original_ids:
        return None
    # PDF glyphs removed from the background MUST have an editable replacement.
    # If cell extraction lost a substantial part, reject this proposal rather
    # than silently deleting the original text from the final PowerPoint.
    original_count=sum(len("".join(f.text.split()))
                       for f in original_fragments if f.id in original_ids)
    recovered_count=sum(len("".join(cell.text.split())) for cell in cells)
    if original_count and recovered_count < original_count*.90:
        return None
    box=_safe_rect(grid.bbox,source.width,source.height,zoom)
    if box is None:
        return None
    xpixel=[x*zoom for x in xs]
    ypixel=[y*zoom for y in ys]
    box=Box(x=xpixel[0],y=ypixel[0],
            w=xpixel[-1]-xpixel[0],h=ypixel[-1]-ypixel[0])
    try:
        ambiguous=any(ch.get("dedup_ambiguous") for group in groups.values() for ch in group)
        table=Table(
            id=uid("pdf_table"),bbox=box,rows=nr,columns=nc,
            row_heights=[b-a for a,b in zip(ypixel,ypixel[1:])],
            col_widths=[b-a for a,b in zip(xpixel,xpixel[1:])],
            cells=cells,ocr_ids=[x.id for x in extra],
            style=Style(font=next((c.style.font for c in cells if c.text),"Arial"),
                        border_pt=0),
            needs_review=caution or ambiguous,
            review_notes=(["Смешанные PDF-шрифты/нечитаемые символы в ячейках"]
                          if caution else [])+(
                         ["PDF: неоднозначное наложение текста ячеек — обе версии сохранены"]
                          if ambiguous else []))
    except ValueError:
        return None
    return table,original_ids,extra


def _raster_grids(page,source,fragments,existing,zoom):
    """Image-grid fallback: locate cells visually, never recognize their text."""
    if len(fragments)<2:
        return []
    regions=set()
    for info in page.get_image_info():
        box=_safe_rect(info["bbox"],source.width,source.height,zoom)
        if box is not None and box.w>=50 and box.h>=35:
            regions.add((int(box.x),int(box.y),math.ceil(box.right),math.ceil(box.bottom)))
    if not regions:
        return []  # CAD/vector pages do not pay for image analysis or OCR models.
    from .geometry import intersection
    from .input import source_image
    from .tables import geometric_tables
    from PIL import Image
    image=None
    grids=[]
    for x0,y0,x1,y1 in sorted(regions):
        region=Box(x=x0,y=y0,w=x1-x0,h=y1-y0)
        members=[f for f in fragments if intersection(f.bbox,region)>.5*f.bbox.w*f.bbox.h]
        if len(members)<2:
            continue
        if image is None:
            image=source_image(Project(source=source))
        crop=image.crop((x0,y0,x1,y1))
        factor=min(1.,2800/max(crop.size))
        if factor<1:
            crop=crop.resize((max(1,round(crop.width*factor)),max(1,round(crop.height*factor))),
                             Image.Resampling.LANCZOS)
        sx,sy=crop.width/region.w,crop.height/region.h
        def local_box(box):
            return Box(x=(box.x-x0)*sx,y=(box.y-y0)*sy,
                       w=box.w*sx,h=box.h*sy)
        local_fragments=[f.model_copy(update={"bbox":local_box(f.bbox),
            "polygon":[((x-x0)*sx,(y-y0)*sy) for x,y in f.polygon]}) for f in members]
        local_existing=[t.model_copy(update={"bbox":local_box(t.bbox)}) for t in existing]
        for table in geometric_tables(crop,local_fragments,local_existing):
            xs=[table.bbox.x]
            ys=[table.bbox.y]
            for width in table.col_widths:
                xs.append(xs[-1]+width)
            for height in table.row_heights:
                ys.append(ys[-1]+height)
            xs=[(x/sx+x0)/zoom for x in xs]
            ys=[(y/sy+y0)/zoom for y in ys]
            bounds=(xs[0],ys[0],xs[-1],ys[-1])
            if (xs[-1]-xs[0])*(ys[-1]-ys[0])>.24*page.rect.get_area():
                continue
            cells=tuple(GridCell(c.row,c.col,c.rowspan,c.colspan,
                (xs[c.col],ys[c.row],xs[c.col+c.colspan],ys[c.row+c.rowspan]))
                for c in table.cells)
            grids.append(VectorGrid(bounds,tuple(xs),tuple(ys),cells,"raster_grid_native_text"))
    return grids


def _native_tables(page, original_fragments, source, zoom, drawings=None, text_data=None):
    """Prioritize actual CAD strokes over generic table heuristics."""
    drawings=page.get_drawings() if drawings is None else drawings
    vector=discover_vector_grids(page,drawings=drawings)
    fallback=[]
    raster=[]
    chars=_raw_pdf_characters(page,text_data)
    assigned=set()
    extra=[]
    tables=[]
    warnings=[]
    stroke_styles=[(tuple(path["rect"]),float(path["width"])) for path in drawings
                   if path.get("type") in ("s","fs") and path.get("width",0)>0
                   and any(item[0]=="l" and abs(item[1]-item[2])>5
                           for item in path.get("items",[]))]
    def candidates():
        yield from vector
        remaining=[f for f in original_fragments if f.id not in assigned]
        raster.extend(_raster_grids(page,source,remaining,tables,zoom))
        yield from raster
        # Generic full-page detection is expensive and can swallow title blocks
        # into the sheet frame. Use it only if no vector proposal was accepted.
        if not tables:
            fallback.extend(_fallback_grids(page))
            yield from fallback
    for grid in candidates():
        bbox=_pdf_lib().Rect(grid.bbox)
        if any(_rect_overlap(bbox,_pdf_lib().Rect(t.bbox.x/zoom,
                t.bbox.y/zoom,t.bbox.right/zoom,t.bbox.bottom/zoom)) >
                .18*bbox.get_area() for t in tables):
            continue
        widths=[width for (a,b,c,d),width in stroke_styles
                if a-.1<bbox.x1 and c+.1>bbox.x0 and b-.1<bbox.y1 and d+.1>bbox.y0]
        border_pt=max(.1,min(20.,median(widths))) if widths else .5
        proposal=_table_from_grid(grid,original_fragments,chars,source,zoom,border_pt)
        if proposal is None:
            if len(warnings)<10:
                warnings.append("Таблица PDF не восстановлена безопасно: текст сохранён надписями")
            continue
        table,ids,fragments=proposal
        if ids & assigned:
            warnings.append("Перекрывающиеся таблицы PDF: оставлена первая сетка")
            continue
        if grid.discovery=="raster_grid_native_text":
            table.confidence=.7
            table.needs_review=True
            table.review_notes.append("Сетка найдена по изображению; текст и оформление взяты из PDF")
        tables.append(table)
        assigned|=ids
        extra.extend(fragments)
    return tables,assigned,extra,warnings,{"vector_candidates":len(vector),
            "fallback_candidates":len(fallback),"raster_candidates":len(raster),
            "reconstructed":len(tables)}


def _styles_for_group(lines,samples):
    sample=samples.get(lines[0].id,{})
    style=Style(
        font=_font_name(sample.get("font","")),
        font_size=max(3.,min(256.,float(sample.get("size",10) or 10))),
        color=_pdf_color(sample.get("color",0)),
        fill="#FFFFFF",
        border_pt=0,
        bold=bool(int(sample.get("flags",0)) & 16),
        italic=bool(int(sample.get("flags",0)) & 2),
    )
    if len(lines)>1:
        boxes=[_local_bounds(f,lines[0].rotation) for f in lines]
        positions={"left":[b[0] for b in boxes],
                   "center":[(b[0]+b[2])/2 for b in boxes],
                   "right":[b[2] for b in boxes]}
        spread={name:max(values)-min(values) for name,values in positions.items()}
        if spread["left"]>2:
            style.align="center" if spread["center"]<=spread["right"] else "right"
    return style


def _local_bounds(frag, rotation):
    radians=math.radians(rotation)
    cs,sn=math.cos(radians),math.sin(radians)
    us=[x*cs+y*sn for x,y in frag.polygon]
    vs=[-x*sn+y*cs for x,y in frag.polygon]
    return min(us),min(vs),max(us),max(vs)


def _native_layout(members,samples,zoom):
    rotation=members[0].rotation
    boxes=[_local_bounds(f,rotation) for f in members]
    left,top=min(b[0] for b in boxes),min(b[1] for b in boxes)
    right,bottom=max(b[2] for b in boxes),max(b[3] for b in boxes)
    cs,sn=math.cos(math.radians(rotation)),math.sin(math.radians(rotation))
    u,v=(left+right)/2,(top+bottom)/2
    w,h=right-left,bottom-top
    frame=Box(x=u*cs-v*sn-w/2,y=u*sn+v*cs-h/2,w=w,h=h)
    baselines=[]
    for frag in members:
        origin=samples[frag.id]["_spans"][0]["origin"]
        baselines.append((-origin[0]*sn+origin[1]*cs)*zoom)
    lines=[]
    for i,frag in enumerate(members):
        runs=[]
        for span in samples[frag.id]["_spans"]:
            quad=_pdf_lib().recover_quad((cs,sn),span)
            runs.append(NativeRun(text=span.get("text",""),
                font=_font_name(span.get("font")),
                font_size=max(3.,min(256.,float(span.get("size",10)))),
                color=_pdf_color(span.get("color",0)),
                bold=bool(int(span.get("flags",0)) & 16),
                italic=bool(int(span.get("flags",0)) & 2),
                advance=math.hypot(quad.ur.x-quad.ul.x,quad.ur.y-quad.ul.y)))
        distance=(baselines[i+1]-baselines[i])/zoom if i+1<len(members) else None
        lines.append(NativeLine(runs=runs,line_spacing=distance if distance and distance>0 else None,
                                indent=max(0.,(boxes[i][0]-left)/zoom)))
    return NativeLayout(frame=frame,rotation=rotation,lines=lines,
                        reference_style=_styles_for_group(members,samples))


def _separate_logical_blocks(lines, samples, separators=(), zoom=PDF_ZOOM):
    """One PDF block may include dozens of adjacent drawing columns.

    A logical CAT paragraph is consecutive and aligned in local coordinates,
    with a shared font family/direction. Same-baseline column headers are NOT
    concatenated with newlines merely because a PDF producer grouped them.
    """
    buckets=[]
    for frag in sorted(lines,key=lambda f:(round(f.rotation),
                       _local_bounds(f,f.rotation)[1],_local_bounds(f,f.rotation)[0])):
        sample=samples.get(frag.id,{})
        best=None
        best_gap=None
        for bucket in buckets:
            last=bucket[-1]
            previous=samples.get(last.id,{})
            if (_font_family(sample.get("font"))!=_font_family(previous.get("font"))
                    or abs(frag.rotation-last.rotation)>1):
                continue
            a,b=_local_bounds(frag,last.rotation),_local_bounds(last,last.rotation)
            height=min(a[3]-a[1],b[3]-b[1])
            dx=min(abs(a[0]-b[0]),abs(a[2]-b[2]),abs((a[0]+a[2]-b[0]-b[2])/2))
            if dx>max(3.,height*.35):
                first=_local_bounds(bucket[0],last.rotation)
                overlap=min(a[2],b[2])-max(a[0],b[0])
                hanging=(LIST_START.match(bucket[0].text)
                         and first[0]-.5*zoom<=a[0]<=first[0]+height*1.5
                         and overlap>.5*min(a[2]-a[0],b[2]-b[0]))
                if not hanging:
                    continue
            vertical_gap=a[1]-b[3]
            # A real multiline label has consecutive baseline positions.
            # Equal-baseline fragments are probably neighboring table columns.
            if vertical_gap<(-.20*height):
                continue
            if vertical_gap>max(5.,height*.45):
                continue
            cs,sn=math.cos(math.radians(last.rotation)),math.sin(math.radians(last.rotation))
            def local(point):
                x,y=point
                return (x*cs+y*sn)*zoom,(-x*sn+y*cs)*zoom
            before=local(previous["_spans"][0]["origin"])[1]
            after=local(sample["_spans"][0]["origin"])[1]
            common_left,common_right=max(a[0],b[0]),min(a[2],b[2])
            blocked=False
            for start,stop in separators:
                u0,v0=local(start); u1,v1=local(stop)
                overlap=min(max(u0,u1),common_right)-max(min(u0,u1),common_left)
                if (abs(v0-v1)<.4*zoom and before<v0<after
                        and overlap>.6*max(1.,common_right-common_left)):
                    blocked=True
                    break
            if blocked:
                continue
            if best_gap is None or abs(vertical_gap)<best_gap:
                best=bucket
                best_gap=abs(vertical_gap)
        if best is None:
            buckets.append([frag])
        else:
            best.append(frag)
    return buckets


def _split_wired_labels(members, row_index, samples, zoom):
    if len(members) < 2 or abs(members[0].rotation) > 1:
        yield members
        return
    matches = [row_index.attachments(
        (f.bbox.x / zoom, f.bbox.y / zoom, f.bbox.right / zoom, f.bbox.bottom / zoom),
        float(samples[f.id]['size'])) for f in members]
    separate = set()
    for i, (before, after) in enumerate(zip(matches, matches[1:])):
        if any((edge, other, row + 1) in after for edge, other, row in before):
            separate.update((i, i + 1))
    start = 0
    for i in sorted(separate):
        if start < i:
            yield members[start:i]
        yield members[i:i+1]
        start = i + 1
    if start < len(members):
        yield members[start:]


def _pdf_text_objects(fragments, groups, table_ids, samples, zoom=PDF_ZOOM, separators=(),
                      *, split_lines=False, row_index=None):
    by_id={f.id:f for f in fragments}
    objects=[]
    # PDF producers can split one paragraph into several independent blocks.
    free=[by_id[i] for ids in groups.values() for i in ids if i not in table_ids]
    if split_lines:
        blocks = ([line] for line in free)
    else:
        blocks = _separate_logical_blocks(free, samples, separators, zoom)
        if row_index is not None:
            blocks = (block for members in blocks
                      for block in _split_wired_labels(members, row_index, samples, zoom))
    for members in blocks:
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
        ambiguous=any(samples.get(f.id,{}).get("_dedup_ambiguous") for f in members)
        objects.append(Text(
            id=uid("pdf_block"),bbox=bbox,
            text="\n".join(f.text for f in members),
            ocr_ids=[f.id for f in members],
            style=_styles_for_group(members,samples),
            confidence=1.0,rotation=members[0].rotation,
            native_layout=_native_layout(members,samples,zoom),
            needs_review=bad_encoding or uneven or ambiguous,
            review_notes=(["PDF: проверьте шрифт/кодировку или поворот текста"]
                          if bad_encoding or uneven else [])+(
                         ["PDF: неоднозначное наложение надписей — обе версии сохранены"]
                          if ambiguous else []),
        ))
    return objects


def _clean_background(raw,page_number,zoom):
    """Legacy-only PDF background cleanup for already saved old projects.

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


def native_pdf_project(path,page_number=1,*,zoom=PDF_ZOOM,detect_tables=True,
                       ignore_numbers=False,split_lines=False,progress=lambda _:None):
    """Single born-digital PDF page to validated editable text/table project."""
    raw,doc=_read_pdf(path)
    try:
        page=_page(doc,page_number)
        zoom=pdf_render_zoom(page.rect.width,page.rect.height,zoom)
        progress("PDF: извлечение текстовых блоков")
        # PNG dimensions use PyMuPDF's pixel rounding convention.
        original_png=_render_png(page,zoom)
        source=_make_source(raw,path,original_png,page,zoom)
        text_data,deduplication=deduplicate_text(page.get_text("rawdict",sort=True))
        fragments,groups,samples,skipped=_read_lines(page,source,zoom,text_data)
        if not fragments:
            raise NativePDFError(
                "В PDF нет извлекаемого текста. Вероятно, это скан/контуры: "
                "используйте отдельный OCR-режим.")
        tables,table_ids,cell_fragments,table_warnings=[],set(),[],[]
        table_diagnostics={"vector_candidates":0,"fallback_candidates":0,
                           "raster_candidates":0,"reconstructed":0}
        drawings=page.get_drawings()
        if detect_tables:
            progress("PDF: восстановление таблиц по векторным линиям")
            tables,table_ids,cell_fragments,table_warnings,table_diagnostics=(
                _native_tables(page,fragments,source,zoom,drawings=drawings,text_data=text_data))
        # Ruled separators still delimit labels in text-only mode; this does
        # not infer any table topology or run the table detectors.
        separators=[(tuple(item[1]),tuple(item[2])) for path in drawings
                    if path.get("type") in ("s","fs") for item in path.get("items",[])
                    if item[0]=="l" and abs(item[1]-item[2])>10]
        row_index = None if split_lines else LabelRowIndex(page, drawings)
        texts=_pdf_text_objects(fragments,groups,table_ids,samples,zoom,separators,
                                split_lines=split_lines,row_index=row_index)
        extracted_lines=len(fragments)
        for obj in [*texts, *(cell for table in tables for cell in table.cells)]:
            if obj.native_layout is not None:
                obj.text=normalize_layout(obj.native_layout)
        ignored={}
        if ignore_numbers:
            kept=[]
            for obj in texts:
                if is_numeric_label(obj.text):
                    ignored.update({fid:"numeric_standalone_label" for fid in obj.ocr_ids})
                else:
                    kept.append(obj)
            texts=kept
        fragments=[f for f in fragments if f.id not in table_ids]+cell_fragments
        # Export overlays use filled text frames/cells, never a redacted raster.
        progress("PDF: сохранение полной страницы с исходным текстом")
        source.raster_b64=base64.b64encode(original_png).decode("ascii")
        result=Project(source=source,fragments=fragments,objects=[*tables,*texts],ignored=ignored)
        detected_fonts=sorted({_font_name(span["font"]) for sample in samples.values()
                               for span in sample["_spans"] if span.get("font")})
        font_warnings=[]
        if any(_font_family(name) not in ("Arial","Calibri","Times New Roman",
                                         "Courier New") for name in detected_fonts):
            font_warnings.append(
                "PDF использует специальный шрифт: "+
                ", ".join(detected_fonts[:10])+
                ". Для совпадения в PowerPoint он должен быть установлен "
                "на компьютере; шрифты из PDF не распространяются.")
        result.messages.extend(font_warnings)
        result.raw["native_pdf"]={
            "page":page_number,"pages":len(doc),"zoom":zoom,
            "source_fonts":detected_fonts,
            "font_warnings":font_warnings,
            "page_size_pt":[page.rect.width,page.rect.height],
            "extracted_lines":extracted_lines,
            "cell_fragments":len(cell_fragments),
            "table_detection":table_diagnostics,
            "grouped_text_blocks":len(texts),
            "native_tables":len(tables),
            "vector_paths":len(drawings),
            "text_redacted_background":False,
            "background_mode":"full_original",
            "overlay_mode":"editable",
            "detect_tables":detect_tables,
            "ignore_numbers":ignore_numbers,
            "split_lines":split_lines,
            "deduplication":deduplication,
            "unresolved":skipped+table_warnings,
        }
        result.messages.append(
            f"Цифровой PDF: {len(texts)} текстовых блоков, "
            f"{len(tables)} редактируемых таблиц, без OCR")
        if deduplication["removed_lines"]:
            result.messages.append(
                f"PDF: исключено наложенных копий строк: {deduplication['removed_lines']} "
                f"({deduplication['removed_glyphs']} символов); исходная PNG не изменена")
        if deduplication["ambiguous_lines"]:
            result.messages.append(
                f"PDF: неоднозначных наложенных строк: {deduplication['ambiguous_lines']}; "
                "сохранены и помечены для проверки")
        if table_warnings:
            result.messages.extend(table_warnings)
        validate_project(result)
        return result
    finally:
        doc.close()
