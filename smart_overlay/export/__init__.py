import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from .native_tables import fixed_native_pdf_layout

from ..models import Project, Text
from ..geometry import union
from ..layout import text_layout
from ..settings import Settings
from ..structure import validate_project

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_node() -> str:
    configured = os.environ.get("SMART_OVERLAY_NODE")
    if configured:
        executable = Path(configured).expanduser()
        if not executable.is_file():
            raise RuntimeError("SMART_OVERLAY_NODE указывает на несуществующий файл Node.js")
        return str(executable.resolve())
    found = shutil.which("node")
    if found:
        return found
    candidates = [PROJECT_ROOT / ".runtime" / "node" / ("node.exe" if os.name == "nt" else "bin/node")]
    if os.name == "nt":
        for name, suffix in (("ProgramFiles", "nodejs"), ("LOCALAPPDATA", "Programs/nodejs"),
                             ("LOCALAPPDATA", "nodejs")):
            if os.environ.get(name):
                candidates.append(Path(os.environ[name]) / suffix / "node.exe")
    for executable in candidates:
        if executable.is_file():
            return str(executable.resolve())
    raise RuntimeError("Не найден Node.js. Установите Node.js 22 LTS, поместите node.exe в .runtime/node "
                       "или задайте полный путь в SMART_OVERLAY_NODE. Инструкция: README.md.")


def _page_payload(project: Project, settings: Settings, *, allow_unreviewed, inline_text_fill):
    validate_project(project, for_export=not allow_unreviewed)
    width, height = project.source.width*settings.inches_per_pixel, project.source.height*settings.inches_per_pixel
    if max(width, height) > 56 or min(width, height) < 1:
        raise ValueError("Размер слайда должен быть от 1 до 56 дюймов. Измените коэффициент пикселей.")
    # A PDF may be large and shared by hundreds of slides. It belongs only in
    # portable projects, not in every in-memory generator payload.
    payload = project.model_dump(exclude={"raw":True,"source":{"original_b64"}})
    payload["allow_unreviewed"] = allow_unreviewed
    payload["inline_text_fill"] = inline_text_fill
    # Preserve saved legacy export modes while new PDF projects use filled
    # editable frames/cells on the untouched full-page raster.
    payload["native_pdf_background_mode"] = (
        project.raw.get("native_pdf", {}).get("background_mode", "legacy_cleaned")
        if project.source.mode == "PDF" else None)
    payload["native_pdf_overlay_mode"] = (
        project.raw.get("native_pdf", {}).get("overlay_mode", "ocr_style")
        if project.source.mode == "PDF" else None)
    payload["native_pdf_zoom"] = project.raw.get("native_pdf", {}).get("zoom", 2)
    payload["native_pdf_page"] = project.raw.get("native_pdf", {}).get("page")
    for obj in payload["objects"]:
        if obj["type"] == "text":
            original = next(o for o in project.objects if o.id == obj["id"])
            assert isinstance(original, Text)
            frame, masks = text_layout(project, original)
            if original.native_layout is not None and payload["native_pdf_overlay_mode"] == "editable":
                # Use the true oriented PDF rectangle, not the envelope of
                # unequal-length rotated lines. Preserve user moves/resizes.
                members=[f.bbox for f in project.fragments if f.id in original.ocr_ids]
                if members and all(abs(f.rotation-original.rotation)<.01
                                   for f in project.fragments if f.id in original.ocr_ids):
                    before=union(members)
                    native=original.native_layout.frame
                    sx,sy=original.bbox.w/before.w,original.bbox.h/before.h
                    frame=native.model_copy(update={
                        "x":original.bbox.x+(native.x-before.x)*sx,
                        "y":original.bbox.y+(native.y-before.y)*sy,
                        "w":native.w*sx,"h":native.h*sy})
            obj["frame"] = frame.model_dump()
            obj["masks"] = [b.model_dump() for b in masks]
    payload["settings"] = settings.model_dump()
    return payload


def _write_pptx(payload, output):
    node = resolve_node()
    output = Path(output).resolve()
    with tempfile.TemporaryDirectory(prefix="smart-overlay-") as temp:
        data = Path(temp) / "export.json"
        data.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
        intermediate = Path(temp) / "result.pptx"
        process = subprocess.run([node, str(Path(__file__).with_name("generate.cjs")), str(data), str(intermediate)],
                                 capture_output=True, text=True, encoding="utf-8",
                                 timeout=max(120,30*len(payload.get("slides",[payload]))),
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if process.returncode:
            raise RuntimeError("Ошибка генератора PPTX: " + process.stderr[-2000:])
        if any(p["native_pdf_background_mode"] == "full_original"
               and p["native_pdf_overlay_mode"] == "editable"
               for p in payload.get("slides",[payload])):
            fixed_native_pdf_layout(intermediate, payload)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(intermediate, output)


def export_pptx(project: Project, output: str | Path, settings: Settings, *, allow_unreviewed=False, inline_text_fill=False):
    payload=_page_payload(project,settings,allow_unreviewed=allow_unreviewed,
                          inline_text_fill=inline_text_fill)
    _write_pptx(payload,output)


def export_pptx_pages(projects, output: str | Path, settings: Settings, *, allow_unreviewed=False, inline_text_fill=False):
    """Export one same-format deck; consume portable projects one at a time."""
    slides=[]
    format_size=None
    for project in projects:
        payload=_page_payload(project,settings,allow_unreviewed=allow_unreviewed,
                              inline_text_fill=inline_text_fill)
        meta=project.raw.get("native_pdf",{})
        size=meta.get("page_size_pt") if project.source.mode=="PDF" else None
        if size is None:
            factor=1/meta.get("zoom",2) if project.source.mode=="PDF" else settings.inches_per_pixel*72
            size=(project.source.width*factor,project.source.height*factor)
        if format_size is None:
            format_size=size
        elif any(abs(a-b)>.5 for a,b in zip(format_size,size)):
            raise ValueError("Разные размеры/форматы страниц требуют отдельных PPTX")
        slides.append(payload)
    if not slides:
        raise ValueError("Выберите хотя бы одну страницу для PPTX")
    _write_pptx({"slides":slides},output)

