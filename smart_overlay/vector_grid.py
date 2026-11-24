"""Recover CAD table grids from the actual PDF strokes, without visual OCR.

PyMuPDF's generic find_tables() can mistake an entire engineering drawing
frame for one huge table. Intersecting stroked edges form closed rectangular
faces, including narrow side tables and irregular merged title blocks.
Dense hatch strokes are filtered before topology recovery; changes in the
column structure separate an adjacent specification from its title block.
Repeated-row CAD seeds remain a fallback, not a condition for every table.

Heuristic discovery never changes text. Closed-face candidates require ruled
borders and rectangular cell merges; editable reconstruction independently
checks actual extracted characters. Synthetic tests exercise this separately.
"""
from __future__ import annotations

from collections import defaultdict
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class GridCell:
    row: int
    col: int
    rowspan: int
    colspan: int
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class VectorGrid:
    bbox: tuple[float, float, float, float]
    xedges: tuple[float, ...]
    yedges: tuple[float, ...]
    cells: tuple[GridCell, ...]
    discovery: str = "cad_vector_grid"


def _union_coverage(segments, start, stop):
    intervals = sorted((max(a,start), min(b,stop))
                       for a,b in segments if a<stop and b>start)
    covered=0.
    previous=None
    for a,b in intervals:
        if previous is None:
            previous=[a,b]
        elif a <= previous[1]+.4:
            previous[1]=max(previous[1],b)
        else:
            covered+=max(0.,previous[1]-previous[0])
            previous=[a,b]
    if previous:
        covered+=max(0.,previous[1]-previous[0])
    return covered


def _cluster(values, tolerance=.8):
    series=sorted(float(v) for v in values)
    clusters=[]
    for value in series:
        if not clusters or value-clusters[-1][-1]>tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [median(group) for group in clusters]


def _strokes(page,drawings=None):
    horizontal=set()
    vertical=set()
    for path in page.get_drawings() if drawings is None else drawings:
        # Only stroked straight edges; fill-only rectangles are not rules.
        if path.get("type") not in ("s","fs"):
            continue
        for item in path.get("items",[]):
            if item[0]=="l":
                segments=[item[1:3]]
            elif item[0]=="re":
                rect=item[1]
                segments=[(rect.tl,rect.tr),(rect.tr,rect.br),
                          (rect.br,rect.bl),(rect.bl,rect.tl)]
            else:
                continue
            for a,b in segments:
                if abs(a.y-b.y)<.2 and abs(a.x-b.x)>5:
                    horizontal.add((round((a.y+b.y)/2,1),
                                    round(min(a.x,b.x),1),round(max(a.x,b.x),1)))
                elif abs(a.x-b.x)<.2 and abs(a.y-b.y)>5:
                    vertical.add((round((a.x+b.x)/2,1),
                                  round(min(a.y,b.y),1),round(max(a.y,b.y),1)))
    return sorted(horizontal),sorted(vertical)


def _seed_groups(horizontal):
    by_endpoints=defaultdict(set)
    for y,a,b in horizontal:
        if b-a>=22:
            by_endpoints[(round(a),round(b))].add(y)
    seeds=[]
    for (x0,x1),distinct in by_endpoints.items():
        ys=sorted(distinct)
        if len(ys)<5 or ys[-1]-ys[0]<85:
            continue
        gaps=[b-a for a,b in zip(ys,ys[1:])]
        pitch=median(gaps)
        # Repeated CAD glyph horizontal strokes occur inside a 20-point
        # character line, not at approximately spaced table row boundaries.
        if pitch<8:
            continue
        seeds.append((x0,x1,ys[0],ys[-1],pitch))
    parent=list(range(len(seeds)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for i,a in enumerate(seeds):
        for j in range(i):
            b=seeds[j]
            xgap=max(0., max(a[0],b[0])-min(a[1],b[1]))
            yoverlap=min(a[3],b[3])-max(a[2],b[2])
            if (xgap<4 and yoverlap>.4*min(a[3]-a[2],b[3]-b[2])):
                parent[root(i)]=root(j)
    groups=defaultdict(list)
    for i,seed in enumerate(seeds):
        groups[root(i)].append(seed)
    # Three or more adjacent repeated strokes: evidence for an actual
    # grid, not a dimension-line chain or the page border.
    return [group for group in groups.values() if
            len(group)>=3 and
            max(g[1] for g in group)-min(g[0] for g in group)>=300 and
            max(g[3] for g in group)-min(g[2] for g in group)>=120]


def _stroke_index(strokes):
    index=defaultdict(list)
    for value,a,b in strokes:
        index[value].append((a,b))
    return index


def _grid_from_seed(group,horizontal,vertical,page):
    sx0=min(g[0] for g in group)
    sx1=max(g[1] for g in group)
    sy0=min(g[2] for g in group)
    sy1=max(g[3] for g in group)
    pitch=median([g[4] for g in group])
    # Include neighboring outer border and thin auxiliary columns, but
    # reject vertical lines belonging to an adjacent table.
    index=_stroke_index(vertical)
    xlines=[x for x,segments in index.items()
            if sx0-55<=x<=sx1+10 and
            _union_coverage(segments,sy0,sy1)>.23*(sy1-sy0)]
    xs=_cluster(xlines, .8)
    if len(xs)<3 or len(xs)>35:
        return None
    x0,x1=xs[0],xs[-1]
    hy=_stroke_index(horizontal)
    useful=[]
    for y,intervals in hy.items():
        # The full-sheet title border is *not* a table row.
        own=[(a,b) for a,b in intervals
             if a>=x0-40 and b<=x1+40]
        amount=_union_coverage(own,x0,x1)
        if amount>=max(20.,(x1-x0)*.18):
            useful.append((y,amount/(x1-x0)))
    useful.sort()
    top,bottom=sy0,sy1
    # A spec-sheet section header may sit one or two row intervals above
    # the repeated data strokes; include it without swallowing title blocks.
    for _ in range(2):
        earlier=[(y,c) for y,c in useful if 1.8<top-y<2.2*pitch]
        if not earlier:break
        top=max(earlier)[0]
    for _ in range(2):
        later=[(y,c) for y,c in useful if 1.8<y-bottom<2.2*pitch]
        if not later:break
        y,c=min(later)
        bottom=y
        if c>.95:break
    if (x1-x0)*(bottom-top)>page.rect.get_area()*.24:
        return None
    ys=_cluster([y for y,c in useful if top-1<=y<=bottom+1], 3.5)
    if len(ys)<5 or len(ys)>150:
        return None
    # Avoid taking neighboring lines just outside the actual grid.
    if ys[0]>top+4 or ys[-1]<bottom-4:
        return None
    nc,nr=len(xs)-1,len(ys)-1
    if not 2<=nc<=34 or nr<4 or nc*nr>1600:
        return None
    vx=_stroke_index(vertical)
    def h_exists(y,a,b):
        segments=[seg for yy,ss in hy.items() if abs(yy-y)<1.4 for seg in ss
                  if seg[0]>=x0-40 and seg[1]<=x1+40]
        return _union_coverage(segments,a,b)>.78*(b-a)
    def v_exists(x,a,b):
        segments=[seg for xx,ss in vx.items() if abs(xx-x)<1.4 for seg in ss]
        return _union_coverage(segments,a,b)>.78*(b-a)
    parent=list(range(nr*nc))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def join(i,j):
        parent[root(i)]=root(j)
    borders=0
    for r in range(nr):
        for c in range(nc):
            here=r*nc+c
            if c+1<nc:
                if v_exists(xs[c+1],ys[r],ys[r+1]):
                    borders+=1
                else:
                    join(here,here+1)
            if r+1<nr:
                if h_exists(ys[r+1],xs[c],xs[c+1]):
                    borders+=1
                else:
                    join(here,here+nc)
    if borders<max(12,nr*nc*.6):
        return None
    components=defaultdict(list)
    for index in range(nr*nc):
        components[root(index)].append((index//nc,index%nc))
    cells=[]
    for component in components.values():
        r0=min(r for r,c in component);r1=max(r for r,c in component)+1
        c0=min(c for r,c in component);c1=max(c for r,c in component)+1
        if len(component)!=(r1-r0)*(c1-c0):
            # A PPTX merged cell must be rectangular. Never fake a
            # nonrectangular merge and change source structure silently.
            return None
        cells.append(GridCell(r0,c0,r1-r0,c1-c0,
                              (xs[c0],ys[r0],xs[c1],ys[r1])))
    cells.sort(key=lambda cell:(cell.row,cell.col))
    return VectorGrid((x0,ys[0],x1,ys[-1]),
                      tuple(xs),tuple(ys),tuple(cells))


def _joined_strokes(strokes):
    result=[]
    for coordinate,segments in _stroke_index(strokes).items():
        merged=[]
        for a,b in sorted(segments):
            if merged and a<=merged[-1][1]+.8:
                merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
            else:
                merged.append((a,b))
        result.extend((coordinate,a,b) for a,b in merged)
    return sorted(result)


def _crossings(coordinate,start,stop,keys,index):
    return [key for key in keys[bisect_left(keys,start-.8):bisect_right(keys,stop+.8)]
            if any(a-.8<=coordinate<=b+.8 for a,b in index[key])]


class LabelRowIndex:
    """Find row attachments without requiring the grid to become a PPTX table."""

    def __init__(self, page, drawings):
        self.horizontal, vertical = _strokes(page, drawings)
        self.ys = [y for y, _, _ in self.horizontal]
        self.vertical = _stroke_index(vertical)
        self.xs = sorted(self.vertical)
        levels = defaultdict(list)
        # Use joined strokes for the closed grid, raw strokes for wire ends.
        for y, start, stop in _joined_strokes(self.horizontal):
            joins = _crossings(y, start, stop, self.xs, self.vertical)
            for left, right in zip(joins, joins[1:]):
                if right - left >= 8:
                    levels[left, right].append(y)
        self.bands = defaultdict(list)
        for (left, right), values in levels.items():
            rows = _cluster(values, tolerance=.8)
            if len(rows) >= 3:
                self.bands[left, 1].append((right, rows))
                self.bands[right, -1].append((left, rows))

    def attachments(self, bounds, font_size):
        """Return (near edge, far edge, row) matches for one horizontal label."""
        left, top, right, bottom = bounds
        center = (top + bottom) / 2
        y_tolerance = max(1., font_size * .23)
        x_tolerance = max(2., font_size * .35)
        first = bisect_left(self.ys, center - y_tolerance)
        last = bisect_right(self.ys, center + y_tolerance)
        matches = set()
        for y, start, stop in self.horizontal[first:last]:
            for side, near_text, near_grid in ((1, start, stop), (-1, stop, start)):
                boundary = right if side == 1 else left
                if abs(near_text - boundary) > x_tolerance:
                    continue
                for edge in _crossings(y, near_grid, near_grid, self.xs, self.vertical):
                    if side * (edge - boundary) <= x_tolerance:
                        continue
                    for other, rows in self.bands.get((edge, side), ()):
                        row = bisect_right(rows, y) - 1
                        if not 0 <= row < len(rows) - 1:
                            continue
                        top_rule, bottom_rule = rows[row:row + 2]
                        # A connector is inside a cell band, not a horizontal rule.
                        if min(y - top_rule, bottom_rule - y) <= .8:
                            continue
                        if all(_union_coverage(self.vertical[x], top_rule, bottom_rule)
                               >= .95 * (bottom_rule - top_rule) for x in (edge, other)):
                            matches.add((edge, other, row))
        return matches


def _without_dense_fill(strokes,perpendicular):
    by_span=defaultdict(list)
    for stroke in strokes:
        by_span[stroke[1:]].append(stroke)
    ends=defaultdict(set)
    for x,a,b in perpendicular:
        ends[a].add(x)
        ends[b].add(x)
    keys=sorted(ends)
    removed=set()
    for series in by_span.values():
        runs=[]
        for stroke in sorted(series):
            if runs and stroke[0]-runs[-1][-1][0]<1.5:
                runs[-1].append(stroke)
            else:
                runs.append([stroke])
        for run in runs:
            if len(run)<4:
                continue
            for y,a,b in run:
                junctions={x for key in keys[bisect_left(keys,y-.4):bisect_right(keys,y+.4)]
                           for x in ends[key] if a-.8<=x<=b+.8}
                if len(junctions)<2:
                    removed.add((y,a,b))
    return [stroke for stroke in strokes if stroke not in removed]


def _connected_rules(horizontal,vertical):
    """Keep actual line junctions rather than floating glyph/hatching strokes."""
    horizontal,vertical=_joined_strokes(horizontal),_joined_strokes(vertical)
    horizontal,vertical=(_without_dense_fill(horizontal,vertical),
                         _without_dense_fill(vertical,horizontal))
    vx=_stroke_index(vertical)
    xs=sorted(vx)
    rules=[]
    for y,a,b in horizontal:
        joins=_crossings(y,a,b,xs,vx)
        if joins and abs(joins[0]-a)<=.8 and abs(joins[-1]-b)<=.8:
            rules.append((y,a,b))
    hy=_stroke_index(rules)
    ys=sorted(hy)
    uprights=[]
    for x,a,b in vertical:
        joins=_crossings(x,a,b,ys,hy)
        if len(joins)>=2:
            uprights.append((x,a,b))
    return rules,uprights


def _closed_grid(bounds,horizontal,vertical):
    """Build rectangular merged faces, including irregular title blocks."""
    x0,y0,x1,y1=bounds
    xs=_cluster([x0,x1]+[x for x,a,b in vertical if x0+.8<x<x1-.8
                         and min(b,y1)-max(a,y0)>3])
    ys=_cluster([y0,y1]+[y for y,a,b in horizontal if y0+.8<y<y1-.8
                         and min(b,x1)-max(a,x0)>3])
    nc,nr=len(xs)-1,len(ys)-1
    if not (2<=nc<=34 and 2<=nr<=150 and nc*nr<=1600):
        return None
    hy,vx=_stroke_index(horizontal),_stroke_index(vertical)
    def covered(index,coordinate,a,b):
        segments=[segment for key,spans in index.items() if abs(key-coordinate)<1.0
                  for segment in spans]
        return _union_coverage(segments,a,b)>.92*(b-a)
    parent=list(range(nr*nc))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for r in range(nr):
        for c in range(nc):
            i=r*nc+c
            if c+1<nc and not covered(vx,xs[c+1],ys[r],ys[r+1]):
                parent[root(i)]=root(i+1)
            if r+1<nr and not covered(hy,ys[r+1],xs[c],xs[c+1]):
                parent[root(i)]=root(i+nc)
    components=defaultdict(list)
    for i in range(nr*nc):
        components[root(i)].append((i//nc,i%nc))
    cells=[]
    for component in components.values():
        r0=min(r for r,c in component);r1=max(r for r,c in component)+1
        c0=min(c for r,c in component);c1=max(c for r,c in component)+1
        if len(component)!=(r1-r0)*(c1-c0):
            return None
        a,b,c,d=xs[c0],ys[r0],xs[c1],ys[r1]
        if not (covered(hy,b,a,c) and covered(hy,d,a,c)
                and covered(vx,a,b,d) and covered(vx,c,b,d)):
            return None
        cells.append(GridCell(r0,c0,r1-r0,c1-c0,(a,b,c,d)))
    if len(cells)<4:
        return None
    return VectorGrid(bounds,tuple(xs),tuple(ys),
                      tuple(sorted(cells,key=lambda cell:(cell.row,cell.col))),"vector_topology")


def _partition_levels(levels,left,right,vertical):
    """Shared outer borders do not make a specification and stamp one table."""
    breaks=[0]
    for i,y in enumerate(levels[1:-1],1):
        before=max(levels[i-1],y-50)
        after=min(levels[i+1],y+50)
        above={round(x,1) for x,a,b in vertical if left-.8<=x<=right+.8
               and min(b,y)-max(a,before)>max(2.,(y-before)*.2)}
        below={round(x,1) for x,a,b in vertical if left-.8<=x<=right+.8
               and min(b,after)-max(a,y)>max(2.,(after-y)*.2)}
        if (len(above)>=3 and len(below)>=3
                and len(above & below)<.4*len(above | below)):
            breaks.append(i)
    breaks.append(len(levels)-1)
    return [levels[a:b+1] for a,b in zip(breaks,breaks[1:])]


def _closed_grids(page,horizontal,vertical):
    horizontal,vertical=_connected_rules(horizontal,vertical)
    vx=_stroke_index(vertical)
    xs=sorted(vx)
    boundaries=defaultdict(set)
    for y,a,b in horizontal:
        joins=_crossings(y,a,b,xs,vx)
        for i,left in enumerate(joins):
            for right in joins[i+1:]:
                if right-left>=8:
                    boundaries[(left,right)].add(y)
    proposals=[]
    for (left,right),levels in boundaries.items():
        groups=[]
        for y in sorted(levels):
            if groups and all(_union_coverage(vx[x],groups[-1][-1],y)>.98*(y-groups[-1][-1])
                              for x in (left,right)):
                groups[-1].append(y)
            else:
                groups.append([y])
        for group in [part for levels in groups
                      for part in _partition_levels(levels,left,right,vertical)]:
            if len(group)<3:
                continue
            bounds=(left,group[0],right,group[-1])
            if _area(bounds)>page.rect.get_area()*.24:
                continue
            own_h=[s for s in horizontal if bounds[1]-.8<=s[0]<=bounds[3]+.8]
            own_v=[s for s in vertical if left-.8<=s[0]<=right+.8]
            grid=_closed_grid(bounds,own_h,own_v)
            if grid is not None:
                proposals.append(grid)
    result=[]
    for grid in sorted(proposals,key=lambda g:-_area(g.bbox)):
        if not any(_overlap(grid.bbox,other.bbox)>.2*min(_area(grid.bbox),_area(other.bbox))
                   for other in result):
            result.append(grid)
    return result


def discover_vector_grids(page,drawings=None):
    """Return only strong ruled-table candidates from vector coordinates."""
    horizontal,vertical=_strokes(page,drawings)
    proposals=_closed_grids(page,horizontal,vertical)
    for group in _seed_groups(horizontal):
        grid=_grid_from_seed(group,horizontal,vertical,page)
        if grid is None:
            continue
        if any(_overlap(grid.bbox,other.bbox)>.2*
               min(_area(grid.bbox),_area(other.bbox)) for other in proposals):
            # Prefer a single unambiguous grid rather than overlapping exports.
            continue
        proposals.append(grid)
    return sorted(proposals,key=lambda grid:(grid.bbox[1],grid.bbox[0]))


def _area(bounds):
    return max(0.,bounds[2]-bounds[0])*max(0.,bounds[3]-bounds[1])


def _overlap(a,b):
    return max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(
        0.,min(a[3],b[3])-max(a[1],b[1]))
