from __future__ import annotations

import math
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def uid(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Box(Model):
    x: float
    y: float
    w: float = Field(gt=0)
    h: float = Field(gt=0)

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h


class Source(Model):
    filename: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mode: str
    orientation: int = 1
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    original_b64: str
    raster_b64: str
    raster_mime: Literal["image/png", "image/jpeg"]


class Fragment(Model):
    id: str
    text: str
    bbox: Box
    polygon: list[tuple[float, float]] = Field(min_length=4)
    rotation: float = 0
    confidence: float = Field(ge=0, le=1)
    origin: Literal["paddle", "region", "manual_split"] = "paddle"
    parent_id: str | None = None


class Style(Model):
    fill: str = Field(default="#FFFFFF", pattern=r"^#[0-9A-Fa-f]{6}$")
    color: str = Field(default="#000000", pattern=r"^#[0-9A-Fa-f]{6}$")
    font: str = "Arial"
    font_size: float = Field(default=12, gt=0, le=256)
    align: Literal["left", "center", "right"] = "left"
    border_color: str = Field(default="#000000", pattern=r"^#[0-9A-Fa-f]{6}$")
    border_pt: float = Field(default=0.5, ge=0, le=20)
    bold: bool = False
    italic: bool = False


class NativeRun(Model):
    text: str
    font: str
    font_size: float = Field(gt=0, le=256)
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    bold: bool = False
    italic: bool = False
    advance: float | None = Field(default=None, ge=0)


class NativeLine(Model):
    runs: list[NativeRun]
    line_spacing: float | None = Field(default=None, gt=0)
    indent: float = Field(default=0, ge=0)


class NativeParagraph(Model):
    runs: list[NativeRun]


class NativeLayout(Model):
    """PDF typography: frame in raster pixels, paragraph distances in PDF points.

    Runs are used only while their text still matches the editable plain text.
    Old projects omit this metadata and keep their previous behavior.
    """
    frame: Box
    rotation: float = Field(default=0, ge=-360, le=360)
    valign: Literal["top", "middle", "bottom"] = "top"
    lines: list[NativeLine]
    paragraphs: list[NativeParagraph] | None = None
    margin: tuple[float, float, float, float] = (0, 0, 0, 0)
    reference_style: Style | None = None


class Correction(Model):
    source: Literal["vision", "manual"]
    before: str
    after: str
    confidence: float = Field(ge=0, le=1)
    applied: bool
    reason: str


class Overlay(Model):
    id: str
    bbox: Box
    ocr_ids: list[str] = Field(default_factory=list)
    style: Style = Field(default_factory=Style)
    confidence: float = Field(default=1, ge=0, le=1)
    needs_review: bool = False
    review_notes: list[str] = Field(default_factory=list)
    excluded: bool = False
    corrections: list[Correction] = Field(default_factory=list)


class Text(Overlay):
    type: Literal["text"] = "text"
    text: str
    rotation: float = Field(default=0, ge=-360, le=360)
    native_layout: NativeLayout | None = None


class Cell(Model):
    id: str = Field(default_factory=lambda: uid("cell"))
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    text: str = ""
    ocr_ids: list[str] = Field(default_factory=list)
    style: Style = Field(default_factory=Style)
    native_layout: NativeLayout | None = None


class Table(Overlay):
    type: Literal["table"] = "table"
    rows: int = Field(ge=1, le=500)
    columns: int = Field(ge=1, le=100)
    row_heights: list[float]
    col_widths: list[float]
    cells: list[Cell]

    @model_validator(mode="after")
    def topology(self):
        if len(self.row_heights) != self.rows or len(self.col_widths) != self.columns:
            raise ValueError("Число размеров строк/столбцов не соответствует таблице")
        if any(v <= 0 or not math.isfinite(v) for v in self.row_heights + self.col_widths):
            raise ValueError("Размеры строк и столбцов должны быть положительными")
        occupied = set()
        for cell in self.cells:
            if cell.row + cell.rowspan > self.rows or cell.col + cell.colspan > self.columns:
                raise ValueError("Объединённая ячейка выходит за таблицу")
            for r in range(cell.row, cell.row + cell.rowspan):
                for c in range(cell.col, cell.col + cell.colspan):
                    if (r, c) in occupied:
                        raise ValueError("Ячейки таблицы пересекаются")
                    occupied.add((r, c))
        if len(occupied) != self.rows * self.columns:
            raise ValueError("В структуре таблицы есть незаполненные позиции")
        if abs(sum(self.row_heights) - self.bbox.h) > 1 or abs(sum(self.col_widths) - self.bbox.w) > 1:
            raise ValueError("Размеры сетки не соответствуют области таблицы")
        return self


class Project(Model):
    version: Literal[1] = 1
    source: Source
    fragments: list[Fragment] = Field(default_factory=list)
    objects: list[Annotated[Text | Table, Field(discriminator="type")]] = Field(default_factory=list)
    ignored: dict[str, str] = Field(default_factory=dict)
    raw: dict = Field(default_factory=dict)
    messages: list[str] = Field(default_factory=list)
    usage: list[dict] = Field(default_factory=list)
    settings: dict = Field(default_factory=dict)

