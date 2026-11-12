"""Deterministic synthetic fixture; ground truth is NOT an OCR accuracy claim."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smart_overlay.input import load_image, save_project
from smart_overlay.models import Box, Fragment, Style, Text
from smart_overlay.settings import Settings
from smart_overlay.tables import merge_cells, new_table, cell_box


def create(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 24) if font_path.exists() else ImageFont.load_default(size=24)
    draw.text((50, 20), "SMART OVERLAY OCR / SYNTHETIC TECHNICAL DRAWING", font=font, fill="#20354a")
    labels = [("ocr_1", "Maximum permissible", Box(x=60, y=90, w=260, h=30)),
              ("ocr_2", "operating pressure", Box(x=60, y=124, w=250, h=30)),
              ("ocr_3", "0.25 MPa", Box(x=60, y=158, w=180, h=30)),
              ("ocr_4", "Drain valve", Box(x=60, y=280, w=180, h=32)),
              ("ocr_5", "Outlet", Box(x=290, y=280, w=100, h=32))]
    for _, text, box in labels:
        draw.text((box.x, box.y), text, fill="black", font=font)
    draw.rectangle((450, 95, 1090, 260), outline="#263b4a", width=3)
    draw.line((450, 180, 1090, 180), fill="#263b4a", width=3)
    draw.line((150, 320, 150, 410, 400, 410, 400, 200, 450, 200), fill="#263b4a", width=3)
    draw.polygon([(450, 200), (430, 190), (430, 210)], fill="#263b4a")
    table = new_table(Box(x=500, y=360, w=620, h=280), 4, 3, Style(font_size=11))
    table.row_heights = [55, 75, 75, 75]
    table.col_widths = [250, 190, 180]
    table = merge_cells(table, 0, 0, 0, 2)
    table = merge_cells(table, 2, 0, 3, 0)
    values = {(0, 0): "Design parameters", (1, 0): "Parameter", (1, 1): "Value", (1, 2): "Unit",
              (2, 0): "Operating limits", (2, 1): "0.25", (2, 2): "MPa", (3, 1): "80", (3, 2): "C"}
    fragments = []
    for i, cell in enumerate(table.cells):
        cell.text = values[(cell.row, cell.col)]
        b = cell_box(table, cell)
        draw.rectangle((b.x, b.y, b.right, b.bottom), outline="black", width=2)
        draw.text((b.x+8, b.y+8), cell.text, font=font, fill="black")
        fid = f"table_ocr_{i}"
        fb = Box(x=b.x+8, y=b.y+8, w=min(b.w-16, draw.textlength(cell.text, font=font)), h=30)
        fragments.append(Fragment(id=fid, bbox=fb, text=cell.text, confidence=1,
                                   polygon=[(fb.x, fb.y), (fb.right, fb.y), (fb.right, fb.bottom), (fb.x, fb.bottom)]))
        cell.ocr_ids = [fid]
    table.ocr_ids = [f.id for f in fragments]
    draw.text((60, 710), "Original raster + editable text and a native merged-cell table", fill="#435b6a", font=font)
    image_path = directory / "technical-demo.png"
    image.save(image_path)
    project = load_image(image_path)
    for fid, text, b in labels:
        fragments.append(Fragment(id=fid, bbox=b, text=text, confidence=1,
                         polygon=[(b.x, b.y), (b.right, b.y), (b.right, b.bottom), (b.x, b.bottom)]))
    project.fragments = fragments
    project.objects = [Text(id="pressure", bbox=Box(x=57, y=87, w=270, h=108),
                            text="Maximum permissible\noperating pressure\n0.25 MPa",
                            ocr_ids=["ocr_1", "ocr_2", "ocr_3"], style=Style(font_size=11)),
                       Text(id="drain", bbox=Box(x=57, y=277, w=186, h=38), text="Drain valve",
                            ocr_ids=["ocr_4"], style=Style(font_size=11)),
                       Text(id="outlet", bbox=Box(x=287, y=277, w=106, h=38), text="Outlet",
                            ocr_ids=["ocr_5"], style=Style(font_size=11)), table]
    project.settings = Settings().model_dump()
    project.messages = ["Синтетический пример с вручную заданной эталонной разметкой; не результат PaddleOCR/VL."]
    save_project(project, directory / "technical-demo.socr.json")
    return project


if __name__ == "__main__":
    create(Path(__file__).parent)
