import math

import cv2
import numpy as np
from PIL import Image, ImageFont

from .geometry import padded, rotated_frame
from .models import Box, Style, Text
from .settings import Settings


def background(image: Image.Image, box: Box) -> str:
    area = padded(box, 6, 6, image.width, image.height)
    pixels = np.array(image.crop((int(area.x), int(area.y), math.ceil(area.right), math.ceil(area.bottom))))
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    # Exclude edges and dark ink; take the dominant quantized background colour.
    edges = cv2.dilate(cv2.Canny(gray, 40, 100), np.ones((3, 3), np.uint8))
    sample = pixels[(edges == 0) & (gray >= np.percentile(gray, 40))]
    if not len(sample):
        return "#FFFFFF"
    bins = sample // 16
    colors, counts = np.unique(bins, axis=0, return_counts=True)
    dominant = colors[counts.argmax()]
    rgb = np.median(sample[np.all(bins == dominant, axis=1)], axis=0).astype(int)
    return "#" + "".join(f"{v:02X}" for v in rgb)


def contrast_color(fill: str) -> str:
    red, green, blue = (int(fill[i:i+2], 16) for i in (1, 3, 5))
    return "#FFFFFF" if red*0.299+green*0.587+blue*0.114 < 128 else "#000000"


def fitted_size(text: str, box: Box, initial: float, settings: Settings) -> tuple[float, bool]:
    def fits(size):
        # Pillow's bundled scalable font is used for a conservative estimate.
        font = ImageFont.load_default(size=max(1, round(size / (72 * settings.inches_per_pixel))))
        line_count = 0
        for line in text.split("\n"):
            line_count += max(1, math.ceil(font.getlength(line) / max(1, box.w)))
        return line_count * font.size * 1.25 <= box.h

    size = max(settings.min_font, min(initial, settings.max_font))
    floor = max(settings.min_font, size * 0.75)
    while size > floor and not fits(size):
        size = max(floor, size - 0.5)
    return size, not fits(size)


def style_text(obj: Text, image: Image.Image, settings: Settings, line_height: float, *, add_padding=True):
    # Rotated boxes are already axis-aligned bounds: rotate the inner frame later.
    if add_padding:
        obj.bbox = padded(obj.bbox, settings.padding_x, settings.padding_y, image.width, image.height)
    initial = line_height * 72 * settings.inches_per_pixel * 0.8
    size, overflow = fitted_size(obj.text, rotated_frame(obj.bbox, obj.rotation), initial, settings)
    fill = background(image, obj.bbox) if settings.fill_mode == "auto" else "#FFFFFF"
    obj.style = Style(font=settings.font, font_size=size, fill=fill, color=contrast_color(fill))
    if overflow:
        obj.needs_review = True
        obj.review_notes.append("Возможное переполнение текста: проверьте размер шрифта")
