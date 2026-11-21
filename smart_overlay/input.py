import base64
import hashlib
import io
from pathlib import Path

from PIL import Image, ImageChops, ImageOps

from .models import Project, Source

FORMATS = {"PNG", "JPEG", "TIFF", "BMP"}


def load_image(path: str | Path) -> Project:
    path = Path(path)
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        if image.format not in FORMATS:
            raise ValueError("Поддерживаются PNG, JPEG, TIFF и BMP")
        if getattr(image, "n_frames", 1) > 1:
            raise ValueError("Многостраничный TIFF не поддерживается: сохраните отдельное изображение")
        orientation = int(image.getexif().get(274, 1))
        mode = image.mode
        display = ImageOps.exif_transpose(image).convert("RGB")
        if image.format in {"PNG", "JPEG"} and orientation == 1 and image.mode in {"RGB", "L"}:
            raster, mime = raw, "image/jpeg" if image.format == "JPEG" else "image/png"
        else:
            buffer = io.BytesIO()
            display.save(buffer, format="PNG")
            raster, mime = buffer.getvalue(), "image/png"
        source = Source(filename=path.name, width=display.width, height=display.height,
                        mode=mode, orientation=orientation, sha256=hashlib.sha256(raw).hexdigest(),
                        original_b64=base64.b64encode(raw).decode(),
                        raster_b64=base64.b64encode(raster).decode(), raster_mime=mime)
    return Project(source=source)


def source_image(project: Project) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(project.source.raster_b64))).convert("RGB")


def save_project(project: Project, path: str | Path):
    from .structure import validate_project
    validate_project(project)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(project.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def open_project(path: str | Path) -> Project:
    from .structure import validate_project
    project = Project.model_validate_json(Path(path).read_text("utf-8"))
    raw = base64.b64decode(project.source.original_b64, validate=True)
    if hashlib.sha256(raw).hexdigest() != project.source.sha256:
        raise ValueError("Контрольная сумма исходного изображения не совпадает")
    raster = source_image(project)
    if raster.size != (project.source.width, project.source.height):
        raise ValueError("Размер изображения в проекте не совпадает с метаданными")
    if project.source.mode == "PDF":
        from .native_pdf import _clean_background, _pdf_lib, _render_png
        info = project.raw.get("native_pdf")
        if not isinstance(info, dict) or "page" not in info or "zoom" not in info:
            raise ValueError("PDF проект не содержит номера страницы и масштаба")
        try:
            if info.get("text_redacted_background", False):
                # Backward compatibility: projects exported before original
                # backgrounds were retained still contain a cleaned bitmap.
                rendered_png = _clean_background(
                    raw, int(info["page"]), float(info["zoom"]))
            else:
                fitz = _pdf_lib()
                document = fitz.open(stream=raw, filetype="pdf")
                try:
                    rendered_png = _render_png(
                        document[int(info["page"])-1],float(info["zoom"]))
                finally:
                    document.close()
            with Image.open(io.BytesIO(rendered_png)) as rendered:
                expected = rendered.convert("RGB")
                if (expected.size != raster.size or
                        ImageChops.difference(expected, raster).getbbox()):
                    raise ValueError("Фон PDF проекта не соответствует исходнику")
        except (ValueError, RuntimeError, OSError, IndexError) as exc:
            raise ValueError("Не удалось проверить исходный PDF проекта") from exc
    else:
        with Image.open(io.BytesIO(raw)) as original:
            expected = ImageOps.exif_transpose(original).convert("RGB")
            if expected.size != raster.size or ImageChops.difference(expected, raster).getbbox():
                raise ValueError("Отображаемый растр не соответствует исходному изображению")
    validate_project(project)
    return project

