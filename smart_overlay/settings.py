import json
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from .models import Model


class Settings(Model):
    language: str = "en"
    ocr_version: Literal["PP-OCRv5", "PP-OCRv4", "PP-OCRv3"] = "PP-OCRv5"
    endpoint: str = "https://openrouter.ai/api/v1"
    model: str = ""
    retries: int = Field(default=2, ge=0, le=5)
    connect_timeout: float = Field(default=10, ge=1, le=120)
    request_timeout: float = Field(default=120, ge=5, le=600)
    stream_responses: bool = True
    max_output_tokens: int = Field(default=8192, ge=512, le=32768)
    padding_x: float = Field(default=3, ge=0, le=50)
    padding_y: float = Field(default=2, ge=0, le=50)
    fill_mode: Literal["white", "auto"] = "white"
    font: str = "Arial"
    min_font: float = Field(default=5, gt=0, le=256)
    max_font: float = Field(default=36, gt=0, le=256)
    confidence_threshold: float = Field(default=0.85, ge=0, le=1)
    inches_per_pixel: float = Field(default=1 / 150, gt=0, le=1)
    tile_size: int = Field(default=1600, ge=512, le=4096)

    @model_validator(mode="after")
    def valid(self):
        if self.min_font > self.max_font:
            raise ValueError("Минимальный шрифт больше максимального")
        url = urlsplit(self.endpoint)
        if url.scheme != "https" or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError("API endpoint должен быть HTTPS URL без ключей и параметров")
        return self


def data_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share")) / "SmartOverlayOCR"


def load_settings() -> Settings:
    path = data_dir() / "settings.json"
    return Settings.model_validate_json(path.read_text("utf-8")) if path.exists() else Settings()


def save_settings(settings: Settings):
    path = data_dir() / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings.model_dump(), indent=2), "utf-8")

