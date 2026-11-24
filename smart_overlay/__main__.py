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
    pdf_selection = one.add_mutually_exclusive_group()
    pdf_selection.add_argument("--pdf-page", type=int, default=1,
                     help="Page number of a born-digital PDF (1-based; one slide per conversion)")
    pdf_selection.add_argument("--pdf-pages", type=str, default=None,
                               help="Native PDF batch: 'all' or selection such as '1, 2, 6-9'; "
                                    "--output is the filename base for size-grouped decks")
    one.add_argument("--pdf-overlay-mode", choices=["ocr_style", "source_exact"],
                     help="Deprecated compatibility option; ignored for new conversions")
    one.add_argument("--pdf-tables", action=argparse.BooleanOptionalAction, default=None,
                     help="Detect native PDF tables (default: saved setting, otherwise enabled)")
    one.add_argument("--pdf-ignore-numbers", action=argparse.BooleanOptionalAction, default=None,
                     help="Native PDF only: ignore standalone numeric labels (PNG-only); "
                          "table cells always keep all numbers (default: saved setting, otherwise disabled)")
    one.add_argument("--pdf-split-lines", action=argparse.BooleanOptionalAction, default=None,
                     help="Native PDF only: one frame per text line outside reconstructed tables")
    one.add_argument("--settings", type=Path)
    one.add_argument("--languages", help="Comma-separated manual OCR languages, e.g. ru,en or ch,ru")
    one.add_argument("--smart", action="store_true", help="Use optional windowed OpenRouter analysis")
    one.add_argument("--consent-api", action="store_true", help="Allow sending this drawing to OpenRouter")
    one.add_argument("--proxy", help="Existing Xray HTTP proxy, e.g. http://127.0.0.1:10809")
    one.add_argument("--ocr-engine", choices=["smart", "vl_local", "hybrid_local", "cloud_full", "native_pdf"], default=None,
                     help="Use classic Smart OCR or independent, fully local PaddleOCR-VL 1.6")
    one.add_argument("--cloud-model", help="OpenRouter image-capable model, e.g. google/gemini-2.5-pro")
    one.add_argument("--cloud-max-requests", type=int, help="Maximum paid image requests")
    one.add_argument("--cloud-tile-size", type=int, help="Pixel size of cloud OCR crop (512–2048)")
    one.add_argument("--cloud-max-tokens", type=int, help="Max output tokens per paid image request")
    one.add_argument("--cloud-reasoning-tokens", type=int,
                     help="Gemini 2.5 thinking budget included inside max output tokens")
    one.add_argument("--cloud-timeout", type=int, help="Seconds per cloud OCR image request")
    one.add_argument("--vl-device", choices=["cpu", "gpu:0"], default=None,
                     help="PaddleOCR-VL runtime device; CPU is the safe default")
    one.add_argument("--vl-diagnostics", type=Path,
                     help="Save confidential local VL parsing JSON and a box overlay in this directory")
    one.add_argument("--vl-tile-size", type=int, help="Local VL tile side in pixels (384–2048)")
    one.add_argument("--vl-tile-timeout", type=int, help="Timeout for each VL tile, seconds")
    one.add_argument("--vl-max-new-tokens", type=int, help="Maximum VL output tokens per tile")
    one.add_argument("--vl-max-tiles", type=int, help="Limit number of VL tiles")
    one.add_argument("--vl-strategy", choices=["detect","tiles"],
                     help="detect: text lines only (default); tiles: legacy document layout")
    one.add_argument("--vl-max-regions", type=int, help="Maximum text lines for local VL")
    one.add_argument("--hybrid-max-checks", type=int,
                     help="Maximum suspect lines for local VL re-reading (default: 3)")
    one.add_argument("--ensemble-only", action="store_true",
                     help="Compare PP-OCRv5/v6 on all lines without expensive local VL")
    one.add_argument("--ensemble-batch-size", type=int, help="Numbered VL crop batch size (2-16)")
    one.add_argument("--ensemble-max-batches", type=int, help="Maximum VL batch calls (default 1)")
    one.add_argument("--ensemble-apply-consensus", action="store_true",
                     help="Apply conservative low-confidence OCR consensus (review still required)")
    one.add_argument("--vl-preflight", type=int, help="Maximum seconds for the first text line")
    one.add_argument("--vl-total-timeout", type=int, help="Overall local VL inference budget (seconds)")
    quality = sub.add_parser("evaluate-ocr", help="Compare OCR ensemble with HUMAN-labeled reference CSV")
    quality.add_argument("report", type=Path, help="ocr-ensemble.json from a hybrid run")
    quality.add_argument("--make-template", type=Path,
                         help="Create editable id,original,truth CSV instead of scoring")
    quality.add_argument("--truth", type=Path, help="Completed human-labeled CSV")
    quality.add_argument("--output", type=Path, help="Output JSON with CER and exact-match accuracy")
    bench = sub.add_parser("benchmark-vl", help="Local one-line PaddleOCR-VL benchmark, no PPTX")
    bench.add_argument("image", type=Path)
    bench.add_argument("--device", choices=["cpu","gpu:0"], default="cpu")
    bench.add_argument("--timeout", type=int, default=60,
                       help="Hard limit for one text-line inference in seconds")
    bench.add_argument("--output", type=Path, help="Local diagnostics directory")
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
    if args.command == "evaluate-ocr":
        import json
        from .ocr_eval import truth_template, evaluate_csv
        if args.make_template:
            print(truth_template(args.report,args.make_template))
            return 0
        if not args.truth:
            parser.error("Use --make-template CSV or --truth completed.csv")
        result = evaluate_csv(args.report,args.truth,
                              args.output or args.report.with_name("ocr-evaluation.json"))
        print(json.dumps({k:v for k,v in result.items() if k not in ("note",)},
                         ensure_ascii=False,indent=2))
        return 0
    if args.command == "benchmark-vl":
        import json
        from uuid import uuid4
        from .settings import data_dir
        from .vl_tiled import run_isolated
        settings = Settings(vl_strategy="detect",
                            vl_region_timeout_seconds=args.timeout,
                            vl_preflight_seconds=args.timeout)
        directory = (args.output if args.output else
                     data_dir() / "diagnostics" / "paddle-vl-benchmark"
                     / f"{args.image.stem}-{uuid4().hex[:8]}")
        result, _ = run_isolated(
            args.image, settings, args.device, directory,
            progress=print, check=lambda: None,
            tile_timeout=args.timeout,
            startup_timeout=settings.vl_startup_timeout_seconds,
            benchmark=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("Локальный отчёт: " + str(directory))
        return 0
    if args.command == "convert":
        from .one_click import convert_drawing
        if args.pdf_pages is not None:
            if args.image.suffix.lower() != ".pdf":
                parser.error("--pdf-pages requires a born-digital PDF input")
            if args.smart or args.ocr_engine not in (None, "native_pdf"):
                parser.error("--pdf-pages uses native PDF only, without cloud or OCR")
        if args.smart and args.ocr_engine in ("vl_local", "hybrid_local"):
            parser.error("Local VL/hybrid and --smart (cloud) are mutually exclusive")
        if args.smart and not args.consent_api:
            parser.error("Cloud processing requires --consent-api for this drawing")
        if args.ocr_engine == "cloud_full" and not args.consent_api:
            parser.error("Full cloud OCR requires --consent-api for this drawing")
        settings = (Settings.model_validate_json(args.settings.read_text("utf-8"))
                    if args.settings else Settings(ocr_language_mode="auto"))
        if args.pdf_pages is not None:
            settings.ocr_engine = "native_pdf"
            args.ocr_engine = "native_pdf"
        if args.smart and (args.ocr_engine or settings.ocr_engine) in ("vl_local", "hybrid_local"):
            parser.error("Local PaddleOCR-VL/hybrid cannot be combined with cloud --smart")
        if (args.ocr_engine or settings.ocr_engine) == "cloud_full" and not args.consent_api:
            parser.error("Full cloud OCR requires --consent-api for this drawing")
        if (args.ocr_engine or settings.ocr_engine) == "cloud_full" and args.smart:
            parser.error("--smart and full cloud OCR are mutually exclusive")
        if args.languages is not None:
            codes = [c.strip() for c in args.languages.split(",")]
            settings = Settings.model_validate({**settings.model_dump(),
                "ocr_version": "PP-OCRv5", "ocr_language_mode": "manual",
                "ocr_languages": codes})
        overrides = {"native_pdf_detect_tables": args.pdf_tables,
                     "native_pdf_ignore_numbers": args.pdf_ignore_numbers,
                     "native_pdf_split_lines": args.pdf_split_lines,
                     "vl_tile_size": args.vl_tile_size,
                     "vl_tile_timeout_seconds": args.vl_tile_timeout,
                     "vl_max_new_tokens": args.vl_max_new_tokens,
                     "vl_max_tiles": args.vl_max_tiles,
                     "vl_strategy": args.vl_strategy,
                     "vl_max_regions": args.vl_max_regions,
                     "vl_preflight_seconds": args.vl_preflight,
                     "vl_total_timeout_seconds": args.vl_total_timeout,
                     "hybrid_max_checks": args.hybrid_max_checks,
                     "ensemble_batch_size": args.ensemble_batch_size,
                     "ensemble_max_batches": args.ensemble_max_batches,
                     "cloud_ocr_model": args.cloud_model,
                     "cloud_ocr_max_requests": args.cloud_max_requests,
                     "cloud_ocr_tile_size": args.cloud_tile_size,
                     "cloud_ocr_max_tokens": args.cloud_max_tokens,
                     "cloud_ocr_reasoning_tokens": args.cloud_reasoning_tokens,
                     "cloud_ocr_timeout": args.cloud_timeout}
        if any(v is not None for v in overrides.values()):
            settings = Settings.model_validate({
                **settings.model_dump(),
                **{k: v for k, v in overrides.items() if v is not None}})
        if args.ensemble_only or args.ensemble_apply_consensus:
            settings = Settings.model_validate({
                **settings.model_dump(),
                "ensemble_batch_vl": not args.ensemble_only,
                "ensemble_apply_consensus": args.ensemble_apply_consensus})
        if args.pdf_pages is not None:
            from .pdf_batch import convert_pdf_pages
            try:
                result = convert_pdf_pages(args.image, pages=args.pdf_pages,
                                           destination=args.output, settings=settings, progress=print)
            except (ValueError, OSError) as exc:
                parser.error(f"PDF batch: {exc}")
            for group in result.groups:
                print(group.pptx)
            for warning in result.warnings:
                logging.warning(warning)
            return 0
        result = convert_drawing(args.image, destination=args.output, settings=settings,
                                 api_key=os.environ.get("OPENROUTER_API_KEY", ""),
                                 consent=args.consent_api and
                                         (args.smart or (args.ocr_engine or settings.ocr_engine)=="cloud_full"),
                                 proxy=args.proxy,
                                 ocr_engine=args.ocr_engine, vl_device=args.vl_device,
                                 vl_diagnostics=args.vl_diagnostics,
                                 pdf_page=args.pdf_page, progress=print)
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
