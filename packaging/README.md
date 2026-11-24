# Portable-сборка «Живые чертежи»

## Запуск

На Windows x64 распакуйте ZIP целиком и запустите `SmartOverlayOCR.exe`. Папка `_internal` должна остаться рядом с EXE. Интерпретатор Python, Qt, конвертер PDF и Node.js для генерации PPTX уже в архиве. Лаунчер использует штатный .NET Framework 4.x Windows 10/11. Установка Paddle, доступ в интернет, API-ключ и PowerPoint для конвертации не нужны.

Выберите цифровой PDF с текстовым слоем. Презентация сохраняется рядом с файлом; для исходников на флешке с FAT/exFAT скопируйте PDF в папку на NTFS: публикация результата использует жёсткие ссылки, чтобы не перезаписывать уже имеющиеся файлы. Настройки галочек и промежуточные проекты хранятся в `%LOCALAPPDATA%\SmartOverlayOCR`.

Сведения о лицензиях включены в `PORTABLE-NOTICES.txt`. Перед передачей архива проверьте применимые условия распространения, включая лицензию PyMuPDF.

## Воспроизведение сборки

Нужен обычный установленный CPython 3.12 x64, не embeddable ZIP. Понадобится pnpm для подготовки `node_modules` и компилятор `csc.exe` из штатного .NET Framework. Следующие команды предназначены для сборочной машины, не для коллег:

```powershell
py -3.12 -m venv .venv-portable
.\.venv-portable\Scripts\python.exe -m pip install -c requirements-lock.txt PySide6-Essentials PyMuPDF Pillow numpy opencv-contrib-python pydantic httpx sniffio
npx --yes pnpm@11.25.0 install --frozen-lockfile
```

Из официального Windows x64 ZIP Node.js (проверено с `v24.22.3`) положите `node.exe` и `LICENSE` в `.runtime/node/`. Сверьте SHA-256 архива с `SHASUMS256.txt` из того же релиза.

```powershell
.\.venv-portable\Scripts\python.exe packaging\build_portable.py
```

Сборщик запускает EXE с пустым `PATH`, проводит тестовую конвертацию через GUI и публикует папку `dist/SmartOverlayOCR-win-x64` и одноимённый ZIP. Отдельный тест `tests/test_portable_packaging.py` распаковывает ZIP и запускает EXE из нового места; задайте `SMART_OVERLAY_PORTABLE_TEST=dist/SmartOverlayOCR-win-x64` для его включения.
