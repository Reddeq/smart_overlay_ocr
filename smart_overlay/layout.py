"""Geometry shared by the editable overlay preview and PPTX export."""
from .geometry import intersection, rotated_frame
from .models import Box, Table


def subtract_box(area, obstacle):
    if intersection(area, obstacle) <= 0:
        return [area]
    left, right = max(area.x, obstacle.x), min(area.right, obstacle.right)
    top, bottom = max(area.y, obstacle.y), min(area.bottom, obstacle.bottom)
    candidates = [(area.x, area.y, area.w, top-area.y),
                  (area.x, bottom, area.w, area.bottom-bottom),
                  (area.x, top, left-area.x, bottom-top),
                  (right, top, area.right-right, bottom-top)]
    return [Box(x=x, y=y, w=w, h=h) for x, y, w, h in candidates if w > 0 and h > 0]


def text_layout(project, obj):
    fragments = {f.id: f for f in project.fragments}
    members = [fragments[i] for i in obj.ocr_ids]
    frame = rotated_frame(obj.bbox, obj.rotation, [f.polygon for f in members])
    masks = []
    if not obj.text.strip():
        return frame, masks
    # Mask each measured line, preserving the drawing in the gaps of merged labels.
    areas = [f.bbox for f in members] or [obj.bbox]
    for area in areas:
        pieces = [area]
        for table in project.objects:
            if isinstance(table, Table) and not table.excluded:
                pieces = [part for piece in pieces for part in subtract_box(piece, table.bbox)]
        masks.extend(pieces)
    return frame, masks
