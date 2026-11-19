"""Numbered contact sheets for experimental local PaddleOCR-VL batch QA.

The VL OCR pipeline has no contractual promise to preserve label IDs. Thus
ONLY a fully mapped, one-to-one, correctly ordered set of responses is usable.
No partial results are trusted or automatically merged into the presentation.
"""
from __future__ import annotations

from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont

from .vl_detect import extract_text


def contact_sheet(source: Image.Image, items: list[dict], *, width=840):
    if not 1 <= len(items) <= 16:
        raise ValueError("Пакет VL должен содержать 1–16 областей")
    sheet=Image.new("RGB",(width, 14 + len(items)*95),"white")
    pen=ImageDraw.Draw(sheet)
    font=None
    for face in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            font=ImageFont.truetype(face,28)
            break
        except OSError:
            continue
    if font is None:
        try:
            font=ImageFont.load_default(size=24)
        except TypeError:
            font=ImageFont.load_default()
    for i,entry in enumerate(items,1):
        b=entry["bbox"]
        x0,y0,x1,y1=map(float,b)
        x0,y0=max(0,int(x0)-8),max(0,int(y0)-8)
        x1,y1=min(source.width,int(x1)+8),min(source.height,int(y1)+8)
        if x1<=x0 or y1<=y0:
            raise ValueError("Некорректная область для пакетного VL")
        crop=source.crop((x0,y0,x1,y1)).convert("RGB")
        crop.thumbnail((width-115,70),Image.Resampling.LANCZOS)
        top=9+(i-1)*95
        pen.text((12,top+25),f"{i:02d}:",fill="black",font=font)
        sheet.paste(crop,(75,top+2))
        pen.rectangle((74,top+1,width-18,top+77),outline="#aaaaaa",width=1)
        pen.line((0,top+84,width,top+84),fill="#dddddd",width=1)
    return sheet


def parse_numbered(text: str, expected: int):
    """Strict ID coverage; extra or continuation lines mean no verified result."""
    if not isinstance(text,str) or not text.strip():
        return {}, "empty"
    found={}
    rows=text.strip().splitlines()
    if len(rows)!=expected:
        return {}, "line_count_mismatch"
    for i,line in enumerate(rows,1):
        hit=re.fullmatch(r"\s*(?:\[)?(\d{2})(?:\])?\s*[:.)|\-]\s*(.+?)\s*",line)
        if hit is None:
            return {}, "invalid_numbering"
        number=int(hit.group(1))
        if number!=i or number in found or number<1 or number>expected:
            return {}, "wrong_order_or_duplicate"
        value=hit.group(2).strip()
        if not value or len(value)>240:
            return {}, "invalid_text"
        found[number]=value
    return found,"complete"


def parse_vl_result(page: dict, count: int):
    return parse_numbered(extract_text(page),count)


def save_collage(source, records, directory, index=0):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    output=directory/f"vl-batch-{index:02d}.png"
    contact_sheet(source,records).save(output)
    return output
