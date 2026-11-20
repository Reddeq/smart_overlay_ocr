import json
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from .models import Model


class Settings(Model):
    ocr_engine: Literal["smart", "vl_local", "hybrid_local", "cloud_full"] = "smart"
    vl_device: Literal["cpu", "gpu:0"] = "cpu"
    hybrid_max_checks: int = Field(default=3, ge=0, le=30)
    hybrid_confidence: float = Field(default=0.74, ge=0, le=1)
    ensemble_enabled: bool = True
    ensemble_apply_consensus: bool = False
    ensemble_batch_vl: bool = True
    ensemble_batch_size: int = Field(default=8, ge=2, le=16)
    ensemble_max_batches: int = Field(default=1, ge=0, le=4)
    ensemble_batch_timeout: int = Field(default=160, ge=20, le=600)
    vl_strategy: Literal["detect", "tiles"] = "detect"
    vl_max_regions: int = Field(default=80, ge=1, le=1000)
    vl_region_timeout_seconds: int = Field(default=60, ge=10, le=1800)
    vl_preflight_seconds: int = Field(default=45, ge=10, le=1800)
    vl_total_timeout_seconds: int = Field(default=480, ge=30, le=7200)
    vl_tile_size: int = Field(default=896, ge=384, le=2048)
    vl_tile_overlap: int = Field(default=96, ge=0, le=512)
    vl_max_tiles: int = Field(default=64, ge=1, le=512)
    vl_max_new_tokens: int = Field(default=384, ge=64, le=2048)
    vl_tile_timeout_seconds: int = Field(default=180, ge=30, le=1800)
    vl_startup_timeout_seconds: int = Field(default=300, ge=30, le=1800)
    language: str = "en"  # compatibility with the old single-language editor/CLI
    ocr_language_mode: Literal["single", "auto", "manual"] = "single"
    ocr_languages: list[str] = Field(default_factory=list)
    ocr_version: Literal["PP-OCRv5", "PP-OCRv4", "PP-OCRv3"] = "PP-OCRv5"
    endpoint: str = "https://openrouter.ai/api/v1"
    cloud_ocr_model: str = "google/gemini-2.5-pro"
    cloud_ocr_tile_size: int = Field(default=1024, ge=512, le=2048)
    cloud_ocr_overlap: int = Field(default=96, ge=0, le=256)
    cloud_ocr_max_requests: int = Field(default=8, ge=1, le=32)
    cloud_ocr_max_tokens: int = Field(default=8192, ge=1024, le=32768)
    cloud_ocr_timeout: int = Field(default=240, ge=30, le=900)
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
        if self.cloud_ocr_overlap * 2 >= self.cloud_ocr_tile_size:
            raise ValueError("Перекрытие облачных фрагментов слишком велико")
        if not self.cloud_ocr_model.strip() or len(self.cloud_ocr_model)>160:
            raise ValueError("Не указана допустимая модель облачного OCR")
        if self.vl_tile_overlap * 2 >= self.vl_tile_size:
            raise ValueError("Перекрытие VL должно быть меньше половины размера участка")
        if self.ocr_language_mode != "single":
            if self.ocr_version != "PP-OCRv5":
                raise ValueError("Автоопределение и языковые наборы требуют PP-OCRv5")
            from .languages import LANGUAGES
            if self.ocr_language_mode == "manual":
                if not self.ocr_languages or len(self.ocr_languages) != len(set(self.ocr_languages)):
                    raise ValueError("Выберите один или несколько разных языков OCR")
                if set(self.ocr_languages) - set(LANGUAGES):
                    raise ValueError("Неизвестный язык OCR в ручном наборе")
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

