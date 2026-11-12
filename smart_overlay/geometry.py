import math

from .models import Box


def union(boxes: list[Box]) -> Box:
    x, y = min(b.x for b in boxes), min(b.y for b in boxes)
    return Box(x=x, y=y, w=max(b.right for b in boxes) - x, h=max(b.bottom for b in boxes) - y)


def intersection(a: Box, b: Box) -> float:
    return max(0, min(a.right, b.right) - max(a.x, b.x)) * max(0, min(a.bottom, b.bottom) - max(a.y, b.y))


def contains(outer: Box, inner: Box, tolerance: float = 0) -> bool:
    return (inner.x >= outer.x - tolerance and inner.y >= outer.y - tolerance
            and inner.right <= outer.right + tolerance and inner.bottom <= outer.bottom + tolerance)


def polygon_box(points) -> Box:
    xs, ys = zip(*points)
    return Box(x=min(xs), y=min(ys), w=max(xs) - min(xs), h=max(ys) - min(ys))


def angle(points) -> float:
    a, b = points[:2]
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def padded(box: Box, px: float, py: float, width: int, height: int) -> Box:
    x, y = max(0, box.x - px), max(0, box.y - py)
    return Box(x=x, y=y, w=min(width, box.right + px) - x, h=min(height, box.bottom + py) - y)


def rotated_frame(box: Box, rotation: float, polygons=()) -> Box:
    """Unrotated rectangle centred on an axis-aligned OCR bounding box."""
    radians = math.radians(rotation)
    a, b = abs(math.cos(radians)), abs(math.sin(radians))
    determinant = a*a-b*b
    w = (a*box.w-b*box.h)/determinant if abs(determinant) > 1e-5 else -1
    h = (a*box.h-b*box.w)/determinant if abs(determinant) > 1e-5 else -1
    if w <= 0 or h <= 0:
        points = [point for polygon in polygons for point in polygon]
        if points:
            cs, sn = math.cos(radians), math.sin(radians)
            us = [x*cs+y*sn for x, y in points]
            vs = [-x*sn+y*cs for x, y in points]
            w, h = max(us)-min(us), max(vs)-min(vs)
        else:
            w, h = box.w, box.h
        factor = max(box.w/max(0.001, a*w+b*h), box.h/max(0.001, b*w+a*h))
        w, h = w*factor, h*factor
    return Box(x=box.x+(box.w-w)/2, y=box.y+(box.h-h)/2, w=w, h=h)

