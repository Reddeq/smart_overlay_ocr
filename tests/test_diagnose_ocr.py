import pytest

from smart_overlay.diagnose_ocr import deduplicate, evaluate, restore_points, run
from smart_overlay.models import Box, Fragment


@pytest.mark.parametrize('rotation,point',[(0,(10,20)),(90,(20,90)),(180,(90,30)),(270,(30,10))])
def test_coordinate_inverse(rotation,point):
    # A point at original (10,20) in a 100x50 crop, rotated counterclockwise.
    restored=restore_points([(point[0]*2,point[1]*2)],rotation,100,50,2,300,400)
    assert restored==[(310,420)]


def f(identifier,text='ABC',confidence=.9):
    return Fragment(id=identifier,text=text,confidence=confidence,bbox=Box(x=10,y=20,w=30,h=10),
                    polygon=[(10,20),(40,20),(40,30),(10,30)])


def test_reference_matching_is_one_to_one_and_counts_missed_characters():
    reference=[{'text':'ABC','bbox':f('ref').bbox.model_dump()},
               {'text':'XYZ','bbox':Box(x=100,y=100,w=30,h=10).model_dump()}]
    result=evaluate([f('one'),f('two')],reference)
    assert result['recall']==.5 and result['exact_lines']==1
    assert result['unmatched_predictions']==1
    assert result['character_error_rate_including_misses']==.5


def test_conflicting_readings_remain_in_report():
    merged,conflicts=deduplicate([f('low','ABD',.8),f('high')])
    assert len(merged)==1 and merged[0].text=='ABC'
    assert conflicts[0]['alternative']['text']=='ABD'


def test_experiment_does_not_modify_project(project):
    before=project.model_dump_json()
    class Engine:
        def recognize(self,image,**kwargs):
            return [f('item')],[]
    baseline,merged,report=run(project,Engine(),tile_size=700,scale=2,rotations=(0,90),progress=lambda _:None)
    assert baseline and merged
    assert project.model_dump_json()==before
