import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from ..models import Project
from ..geometry import rotated_frame
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


def export_pptx(project: Project, output: str | Path, settings: Settings, *, allow_unreviewed=False):
    validate_project(project, for_export=not allow_unreviewed)
    width, height = project.source.width*settings.inches_per_pixel, project.source.height*settings.inches_per_pixel
    if max(width, height) > 56 or min(width, height) < 1:
        raise ValueError("Размер слайда должен быть от 1 до 56 дюймов. Измените коэффициент пикселей.")
    node = resolve_node()
    payload = project.model_dump()
    payload["allow_unreviewed"] = allow_unreviewed
    fragments = {f.id: f for f in project.fragments}
    for obj in payload["objects"]:
        if obj["type"] == "text":
            from ..models import Box
            obj["frame"] = rotated_frame(Box(**obj["bbox"]), obj["rotation"],
                                          [fragments[i].polygon for i in obj["ocr_ids"]]).model_dump()
    payload["settings"] = settings.model_dump()
    output = Path(output).resolve()
    with tempfile.TemporaryDirectory(prefix="smart-overlay-") as temp:
        data = Path(temp) / "export.json"
        # Only export image, objects and formatting; original bytes and raw OCR stay in the project.
        payload.pop("raw", None)
        payload["source"].pop("original_b64", None)
        data.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
        intermediate = Path(temp) / "result.pptx"
        process = subprocess.run([node, str(Path(__file__).with_name("generate.cjs")), str(data), str(intermediate)],
                                 capture_output=True, text=True, encoding="utf-8", timeout=120,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if process.returncode:
            raise RuntimeError("Ошибка генератора PPTX: " + process.stderr[-2000:])
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(intermediate, output)

