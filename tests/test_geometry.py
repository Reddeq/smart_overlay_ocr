import math

import pytest

from smart_overlay.geometry import rotated_frame
from smart_overlay.models import Box


@pytest.mark.parametrize("rotation", [0, 15, 45, 90, -90, 180])
def test_rotated_frame_covers_original_polygon(rotation):
    a = math.radians(rotation)
    def rotate(x, y):
        return (300+x*math.cos(a)-y*math.sin(a), 200+x*math.sin(a)+y*math.cos(a))
    points = [rotate(x, y) for x,y in [(-100,-20), (100,-20), (100,20), (-100,20)]]
    xs, ys = zip(*points)
    box = Box(x=min(xs), y=min(ys), w=max(xs)-min(xs), h=max(ys)-min(ys))
    frame = rotated_frame(box, rotation, [points])
    assert frame.w == pytest.approx(200)
    assert frame.h == pytest.approx(40)
    assert frame.x+frame.w/2 == pytest.approx(300)
    assert frame.y+frame.h/2 == pytest.approx(200)
