"""Recover CAD table grids from the actual PDF strokes, without visual OCR.

PyMuPDF's generic find_tables() can mistake an entire engineering drawing
frame for one huge table and merge several real columns. Here, repeatedly
drawn horizontal CAD strokes become table *candidates*; continuous vertical
and horizontal strokes determine the precise grid and actual merges.

Heuristic discovery never changes text. Candidates are accepted only when
they have a dense orthogonal grid, rectangular cell merges and sufficient
real extracted characters. Synthetic tests exercise this independently.
"""
from __future__ import annotations

from collections import defaultdict
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


def _strokes(page):
    horizontal=set()
    vertical=set()
    for path in page.get_drawings():
        # Only explicitly drawn straight lines. Ignore filled rectangles,
        # curves and glyph outlines represented by path fill operations.
        if path.get("type") not in ("s","fs"):
            continue
        for item in path.get("items",[]):
            if item[0]!="l":
                continue
            a,b=item[1:3]
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


def discover_vector_grids(page):
    """Return only strong ruled-table candidates from vector coordinates."""
    horizontal,vertical=_strokes(page)
    proposals=[]
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
