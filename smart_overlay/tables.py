from html.parser import HTMLParser

import cv2
import numpy as np
from PIL import Image

from .geometry import contains, intersection, polygon_box, union
from .models import Box, Cell, Fragment, Style, Table, uid


class TableHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in {"td", "th"}:
            if not self.rows:
                self.rows.append([])
            attrs = dict(attrs)
            self.cell = {"text": "", "rowspan": int(attrs.get("rowspan", 1)),
                         "colspan": int(attrs.get("colspan", 1))}
            self.rows[-1].append(self.cell)
        elif tag == "br" and self.cell is not None:
            self.cell["text"] += "\n"

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"] += data

    def handle_endtag(self, tag):
        if tag in {"td", "th"}:
            self.cell = None


def parse_html(html: str) -> tuple[int, int, list[Cell]]:
    parser = TableHTML()
    parser.feed(html)
    used, cells = set(), []
    for r, row in enumerate(parser.rows):
        c = 0
        for raw in row:
            while (r, c) in used:
                c += 1
            cell = Cell(row=r, col=c, **raw)
            cells.append(cell)
            for rr in range(r, r + cell.rowspan):
                for cc in range(c, c + cell.colspan):
                    if (rr, cc) in used:
                        raise ValueError("HTML таблицы содержит пересекающиеся ячейки")
                    used.add((rr, cc))
            c += cell.colspan
    if not used:
        raise ValueError("Пустая HTML структура таблицы")
    return max(r for r, c in used) + 1, max(c for r, c in used) + 1, cells


def grid_lines(image: Image.Image, box: Box) -> tuple[list[float], list[float]]:
    crop = np.array(image.crop((int(box.x), int(box.y), int(box.right), int(box.bottom))))
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 8)

    def peaks(horizontal):
        length = max(10, (crop.shape[1] if horizontal else crop.shape[0]) // 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1) if horizontal else (1, length))
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        counts = np.mean(lines > 0, axis=1 if horizontal else 0)
        indexes = np.flatnonzero(counts > 0.35)
        groups = np.split(indexes, np.flatnonzero(np.diff(indexes) > 2) + 1)
        return [float(np.mean(g)) for g in groups if len(g)]

    return peaks(False), peaks(True)


def geometric_tables(image: Image.Image, fragments: list[Fragment], existing: list[Table]) -> list[Table]:
    """Find ruled tables missed by page layout detection; every proposal is reviewable."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 8)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, image.width//40), 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, image.height//40))))
    mask = cv2.bitwise_or(horizontal, vertical)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    found = []
    for x, y, width, height, area in stats[1:count]:
        if width < 50 or height < 35 or area < 150:
            continue
        bbox = Box(x=int(x), y=int(y), w=int(width), h=int(height))
        if any(intersection(bbox, t.bbox) > 0 for t in existing+found):
            continue
        vx, hy = grid_lines(image, bbox)
        if len(vx) < 3 or len(hy) < 3:
            continue
        # Avoid treating partially connected diagram lines as a closed table.
        if vx[0] > 4 or hy[0] > 4 or width-vx[-1] > 4 or height-hy[-1] > 4:
            continue
        xs, ys = [x+v for v in vx], [y+v for v in hy]
        cols, rows = len(xs)-1, len(ys)-1
        if rows > 500 or cols > 100:
            continue
        parent = list(range(rows*cols))
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        def join(a, b):
            parent[root(b)] = root(a)
        def line_present(lines, coordinate, start, end, along_x):
            c, s, e = round(coordinate), round(start)+3, round(end)-2
            if e <= s:
                return False
            segment = lines[max(0, c-2):c+3, s:e] if along_x else lines[s:e, max(0, c-2):c+3]
            return bool(segment.size and np.mean(np.any(segment > 0, axis=0 if along_x else 1)) > 0.65)
        for r in range(rows):
            for c in range(cols):
                if c+1 < cols and not line_present(vertical, xs[c+1], ys[r], ys[r+1], False):
                    join(r*cols+c, r*cols+c+1)
                if r+1 < rows and not line_present(horizontal, ys[r+1], xs[c], xs[c+1], True):
                    join(r*cols+c, (r+1)*cols+c)
        groups = {}
        for r in range(rows):
            for c in range(cols):
                groups.setdefault(root(r*cols+c), []).append((r, c))
        cells = []
        valid = True
        for members in groups.values():
            r0, r1 = min(r for r,c in members), max(r for r,c in members)
            c0, c1 = min(c for r,c in members), max(c for r,c in members)
            if (r1-r0+1)*(c1-c0+1) != len(members):
                valid = False
                break
            cells.append(Cell(row=r0, col=c0, rowspan=r1-r0+1, colspan=c1-c0+1))
        if not valid:
            continue
        box = Box(x=xs[0], y=ys[0], w=xs[-1]-xs[0], h=ys[-1]-ys[0])
        table = Table(id=uid("table"), bbox=box, rows=rows, columns=cols, cells=cells,
                      row_heights=np.diff(ys).tolist(), col_widths=np.diff(xs).tolist(),
                      needs_review=True, confidence=0.7,
                      review_notes=["Таблица найдена по линиям OpenCV, проверьте сетку и объединения"])
        assign_fragments(table, fragments)
        by_id = {f.id: f for f in fragments}
        for cell in table.cells:
            cell.text = "\n".join(f.text for f in sorted((by_id[i] for i in cell.ocr_ids), key=lambda f: (f.bbox.y, f.bbox.x)))
        if len(table.ocr_ids) >= 2:
            found.append(table)
    return found


def cell_box(table: Table, cell: Cell) -> Box:
    return Box(x=table.bbox.x + sum(table.col_widths[:cell.col]),
               y=table.bbox.y + sum(table.row_heights[:cell.row]),
               w=sum(table.col_widths[cell.col:cell.col + cell.colspan]),
               h=sum(table.row_heights[cell.row:cell.row + cell.rowspan]))


def assign_fragments(table: Table, fragments: list[Fragment]):
    table.ocr_ids = []
    for cell in table.cells:
        cell.ocr_ids = []
    for fragment in fragments:
        overlap = intersection(table.bbox, fragment.bbox) / (fragment.bbox.w * fragment.bbox.h)
        if overlap <= 0:
            continue
        if overlap < 0.5:
            table.needs_review = True
            table.review_notes.append(f"Фрагмент {fragment.id} пересекает границу таблицы")
        ranked = sorted(table.cells, key=lambda c: intersection(cell_box(table, c), fragment.bbox), reverse=True)
        best = ranked[0]
        if not contains(cell_box(table, best), fragment.bbox, tolerance=3):
            table.needs_review = True
            table.review_notes.append(f"Проверьте принадлежность {fragment.id} ячейке")
        best.ocr_ids.append(fragment.id)
        table.ocr_ids.append(fragment.id)


def tables_from_result(pages: list[dict], image: Image.Image, fragments: list[Fragment]) -> list[Table]:
    tables = []
    for page in pages:
        layouts = [b for b in page.get("layout_det_res", {}).get("boxes", []) if b.get("label") == "table"]
        for index, raw in enumerate(page.get("table_res_list", [])):
            boxes = []
            for points in raw.get("cell_box_list", []):
                flat = np.asarray(points).flatten().tolist()
                if len(flat) == 4:
                    x0, y0, x1, y1 = flat
                    boxes.append(Box(x=x0, y=y0, w=max(1, x1-x0), h=max(1, y1-y0)))
                elif len(flat) == 8:
                    boxes.append(polygon_box(list(zip(flat[::2], flat[1::2]))))
            if boxes:
                bbox = union(boxes)
            elif index < len(layouts):
                x0, y0, x1, y1 = layouts[index]["coordinate"]
                bbox = Box(x=x0, y=y0, w=x1-x0, h=y1-y0)
            else:
                raise ValueError("PP-Structure не вернул геометрию найденной таблицы")
            if not contains(Box(x=0, y=0, w=image.width, h=image.height), bbox, 1):
                raise ValueError("Геометрия таблицы вне изображения")
            notes = []
            try:
                rows, cols, cells = parse_html(raw["pred_html"])
                if len(boxes) != len(cells):
                    notes.append("Количество областей ячеек отличается от HTML структуры")
                # Boundaries inferred from spans preserve unequal widths/heights.
                xs, ys = {0: [0.0], cols: [bbox.w]}, {0: [0.0], rows: [bbox.h]}
                if len(boxes) == len(cells):
                    for cell, b in zip(cells, boxes):
                        xs.setdefault(cell.col, []).append(b.x-bbox.x)
                        xs.setdefault(cell.col+cell.colspan, []).append(b.right-bbox.x)
                        ys.setdefault(cell.row, []).append(b.y-bbox.y)
                        ys.setdefault(cell.row+cell.rowspan, []).append(b.bottom-bbox.y)
                def boundaries(values, count, total):
                    result = [float(np.median(values[i])) if i in values else float("nan") for i in range(count+1)]
                    known = [i for i, value in enumerate(result) if np.isfinite(value)]
                    result = np.interp(range(count+1), known, [result[i] for i in known]).tolist()
                    result[0], result[-1] = 0, total
                    if any(b-a <= 0 for a, b in zip(result, result[1:])):
                        raise ValueError("Несогласованные границы ячеек")
                    return result
                x = boundaries(xs, cols, bbox.w)
                y = boundaries(ys, rows, bbox.h)
                vx, hy = grid_lines(image, bbox)
                if len(vx) >= 2 and abs(len(vx)-cols-1) > 0:
                    notes.append("Линии OpenCV расходятся с числом столбцов")
                if len(hy) >= 2 and abs(len(hy)-rows-1) > 0:
                    notes.append("Линии OpenCV расходятся с числом строк")
                if any(c.rowspan > 1 or c.colspan > 1 for c in cells):
                    notes.append("Проверьте объединённые ячейки")
                table = Table(id=uid("table"), bbox=bbox, rows=rows, columns=cols, cells=cells,
                              col_widths=np.diff(x).tolist(), row_heights=np.diff(y).tolist(),
                              needs_review=bool(notes), review_notes=notes, confidence=0.8)
            except (ValueError, KeyError) as exc:
                # Preserve detected extent with an explicit unresolved structure.
                table = new_table(bbox, 1, 1)
                table.needs_review = True
                table.review_notes = [f"Структура не восстановлена: {exc}. Задайте сетку вручную."]
            assign_fragments(table, fragments)
            by_id = {f.id: f for f in fragments}
            for cell in table.cells:
                if not cell.text and cell.ocr_ids:
                    cell.text = "\n".join(f.text for f in sorted((by_id[i] for i in cell.ocr_ids),
                                                               key=lambda f: (f.bbox.y, f.bbox.x)))
            tables.append(table)
    return tables


def new_table(bbox: Box, rows: int, cols: int, style: Style | None = None) -> Table:
    style = style or Style()
    return Table(id=uid("table"), bbox=bbox, rows=rows, columns=cols,
                 row_heights=[bbox.h/rows]*rows, col_widths=[bbox.w/cols]*cols,
                 cells=[Cell(row=r, col=c, style=style.model_copy()) for r in range(rows) for c in range(cols)])


def merge_cells(table: Table, r0: int, c0: int, r1: int, c1: int) -> Table:
    result = table.model_copy(deep=True)
    selected = [c for c in result.cells if r0 <= c.row <= r1 and c0 <= c.col <= c1]
    if not selected or any(c.row+c.rowspan > r1+1 or c.col+c.colspan > c1+1 for c in selected):
        raise ValueError("Выделение пересекает объединённую ячейку")
    if sum(c.rowspan*c.colspan for c in selected) != (r1-r0+1)*(c1-c0+1):
        raise ValueError("Выделите полный прямоугольник ячеек")
    result.cells = [c for c in result.cells if c not in selected]
    selected.sort(key=lambda c: (c.row, c.col))
    result.cells.append(Cell(row=r0, col=c0, rowspan=r1-r0+1, colspan=c1-c0+1,
                             text="\n".join(c.text for c in selected if c.text),
                             ocr_ids=[i for c in selected for i in c.ocr_ids], style=selected[0].style))
    return Table.model_validate(result.model_dump())


def split_cell(table: Table, row: int, col: int) -> Table:
    result = table.model_copy(deep=True)
    target = next(c for c in result.cells if c.row == row and c.col == col)
    result.cells.remove(target)
    for r in range(target.row, target.row+target.rowspan):
        for c in range(target.col, target.col+target.colspan):
            result.cells.append(Cell(row=r, col=c, style=target.style.model_copy(),
                                     text=target.text if (r, c) == (row, col) else "",
                                     ocr_ids=target.ocr_ids if (r, c) == (row, col) else []))
    result.needs_review = True
    result.review_notes.append("После разделения распределите текст по ячейкам")
    return Table.model_validate(result.model_dump())
