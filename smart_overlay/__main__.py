import argparse
import logging
import os
from pathlib import Path

from .export import export_pptx
from .input import load_image, open_project, save_project
from .pipeline import SmartFailure, local_ocr, smart_ocr
from .settings import Settings, data_dir


def main():
    parser = argparse.ArgumentParser(description="Smart Overlay OCR")
    sub = parser.add_subparsers(dest="command")
    recognize = sub.add_parser("recognize", help="OCR image and save a portable project")
    recognize.add_argument("image", type=Path)
    recognize.add_argument("--project", type=Path, required=True)
    recognize.add_argument("--mode", choices=["local", "smart"], default="local")
    recognize.add_argument("--settings", type=Path)
    recognize.add_argument("--consent-api", action="store_true")
    one = sub.add_parser("convert", help="One-click drawing to editable PPTX")
    one.add_argument("image", type=Path)
    one.add_argument("--output", type=Path)
    one.add_argument("--settings", type=Path)
    one.add_argument("--languages", help="Comma-separated manual OCR languages, e.g. ru,en or ch,ru")
    one.add_argument("--smart", action="store_true", help="Use optional windowed OpenRouter analysis")
    one.add_argument("--consent-api", action="store_true", help="Allow sending this drawing to OpenRouter")
    one.add_argument("--proxy", help="Existing Xray HTTP proxy, e.g. http://127.0.0.1:10809")
    one.add_argument("--ocr-engine", choices=["smart", "vl_local"], default=None,
                     help="Use classic Smart OCR or independent, fully local PaddleOCR-VL 1.6")
    one.add_argument("--vl-device", choices=["cpu", "gpu:0"], default=None,
                     help="PaddleOCR-VL runtime device; CPU is the safe default")
    one.add_argument("--vl-diagnostics", type=Path,
                     help="Save confidential local VL parsing JSON and a box overlay in this directory")
    sub.add_parser("editor", help="Open the legacy correction editor")
    export = sub.add_parser("export", help="Export an already reviewed project without OCR")
    export.add_argument("project", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--allow-unreviewed", action="store_true",
                        help="Export unreviewed objects with a warning; structural validation still applies")
    replay = sub.add_parser("replay-vl", help="Replay a saved VL validation failure locally, without OCR/API")
    replay.add_argument("diagnostic", type=Path)
    from .diagnose_vl import add_parser as add_diagnostic_parser
    add_diagnostic_parser(sub)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not args.command:
        from .quick_app import run_quick_app
        return run_quick_app()
    if args.command == "editor":
        from .app import run_app
        return run_app()
    if args.command == "convert":
        from .one_click import convert_drawing
        if args.smart and args.ocr_engine == "vl_local":
            parser.error("Local PaddleOCR-VL and --smart (cloud) are mutually exclusive")
        if args.smart and not args.consent_api:
            parser.error("Cloud processing requires --consent-api for this drawing")
        settings = (Settings.model_validate_json(args.settings.read_text("utf-8"))
                    if args.settings else Settings(ocr_language_mode="auto"))
        if args.languages is not None:
            codes = [c.strip() for c in args.languages.split(",")]
            settings = Settings.model_validate({**settings.model_dump(),
                "ocr_version": "PP-OCRv5", "ocr_language_mode": "manual",
                "ocr_languages": codes})
        result = convert_drawing(args.image, destination=args.output, settings=settings,
                                 api_key=os.environ.get("OPENROUTER_API_KEY", ""),
                                 consent=args.smart and args.consent_api, proxy=args.proxy,
                                 ocr_engine=args.ocr_engine, vl_device=args.vl_device,
                                 vl_diagnostics=args.vl_diagnostics, progress=print)
        print(result.pptx)
        for warning in result.warnings:
            logging.warning(warning)
        return 0
    if args.command == "diagnose-vl":
        from .diagnose_vl import main as diagnose
        try:
            return diagnose(args)
        except (ValueError, OSError) as exc:
            from .vision import validation_detail
            print("VL diagnostics: " + validation_detail(exc, os.environ.get("OPENROUTER_API_KEY", "")))
            return 2
    if args.command == "replay-vl":
        from .vision import validation_detail
        from .vision_diagnostics import replay as replay_diagnostic
        try:
            replay_diagnostic(args.diagnostic)
        except (ValueError, KeyError, TypeError) as exc:
            print("Локальная проверка VL: " + validation_detail(exc, ""))
            return 2
        print("Локальная проверка VL прошла; обращения к API не выполнялись")
        return 0
    if args.command == "export":
        project = open_project(args.project)
        settings = Settings.model_validate(project.settings) if project.settings else Settings()
        unreviewed = sum(o.needs_review and not o.excluded for o in project.objects)
        if unreviewed and args.allow_unreviewed:
            logging.warning("Экспорт без проверки: %s объектов могут содержать ошибки", unreviewed)
        export_pptx(project, args.output, settings, allow_unreviewed=args.allow_unreviewed)
        print(args.output)
        return 0
    settings = Settings.model_validate_json(args.settings.read_text("utf-8")) if args.settings else Settings()
    if args.mode == "smart" and not args.consent_api:
        parser.error("Smart OCR requires --consent-api to send image and OCR to the external endpoint")
    project = local_ocr(load_image(args.image), settings, progress=print)
    save_project(project, args.project)
    if args.mode == "smart":
        try:
            project = smart_ocr(project, settings, os.environ.get("OPENROUTER_API_KEY", ""),
                                data_dir()/"cache", args.consent_api, progress=print)
        except SmartFailure as exc:
            save_project(exc.project, args.project)
            print("OCR Only сохранён; Smart OCR недоступен:", exc)
            return 2
        save_project(project, args.project)
    print(args.project)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
