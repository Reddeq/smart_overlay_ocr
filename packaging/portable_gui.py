"""Windowed portable entry point; regular source CLI remains unchanged."""
import ctypes
import os
from pathlib import Path
import sys
import tempfile
import traceback
from unittest.mock import patch
from xml.etree import ElementTree as ET
import zipfile


def smoke_test(marker):
    import cv2  # Verify that the optional raster-grid detector can load its DLLs.
    import pymupdf
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

    from smart_overlay.one_click import OneClickResult
    from smart_overlay.quick_app import QuickWindow

    app = QApplication([])
    app.setStyle('Fusion')
    with tempfile.TemporaryDirectory(dir=marker.parent) as work:
        os.environ['LOCALAPPDATA'] = work
        pdf = Path(work) / 'Цифровой чертёж.pdf'
        with pymupdf.open() as document:
            # This sheet crosses the pixel limit at the default render zoom.
            page = document.new_page(width=4768, height=1684)
            page.insert_text((40, 50), 'CONTROL PANEL')
            page.insert_text((300, 50), '1200')
            page.insert_text((400, 70), 'KEEP POWER', fontsize=12)
            page.insert_text((400, 87), 'OFF HERE', fontsize=12)
            for y in (110, 143, 176, 209):
                page.draw_line((60, y), (350, y))
            for x in (60, 185, 350):
                page.draw_line((x, 110 if x != 185 else 143), (x, 209))
            for x, y, text in ((72, 130, 'TERMINALS'), (72, 163, '8'), (198, 163, 'POWER'),
                               (72, 195, '7'), (198, 195, 'GROUND')):
                page.insert_text((x, y), text, fontsize=10)
            document.save(pdf)

        window = QuickWindow()
        window.show()
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(str(pdf), '')), \
             patch.object(QMessageBox, 'warning') as warning:
            window.pick.click()
            assert window.path == pdf and window.run_button.isEnabled(), warning.call_args
            assert [box.text() for box in (window.pdf_tables_checkbox, window.pdf_ignore_numbers_checkbox,
                                           window.pdf_split_lines_checkbox)] == [
                'Реконструировать таблицы', 'Игнорировать числа', 'Не объединять строки']
            window.pdf_ignore_numbers_checkbox.click()
            window.pdf_split_lines_checkbox.click()
            window.run_button.click()
            assert window.worker is not None
            loop = QEventLoop()
            window.worker.finished.connect(loop.quit)
            QTimer.singleShot(60_000, loop.quit)
            loop.exec()
            assert window.worker is None and isinstance(window.result, OneClickResult), warning.call_args
            assert not warning.called, warning.call_args
            window.review.click()
            assert window.editor is not None and window.editor.native_only
            window.editor.close()
            pptx = window.result.pptx
        window.close()
        with zipfile.ZipFile(pptx) as archive:
            slide = ET.fromstring(archive.read('ppt/slides/slide1.xml'))
            table = slide.find('.//{*}tbl')
            assert table is not None
            assert {t.text for t in table.findall('.//{*}t')} >= {'TERMINALS', '8', 'POWER', '7', 'GROUND'}
            text = {t.text for t in slide.findall('.//{*}t')}
            assert 'CONTROL PANEL' in text and '1200' not in text
            shapes = {''.join(t.text or '' for t in shape.findall('.//{*}t'))
                      for shape in slide.findall('.//{*}sp')}
            assert {'KEEP POWER', 'OFF HERE'} <= shapes
    marker.write_text('OK\n', encoding='utf-8')
    return 0


def main():
    marker = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 and sys.argv[1] == '--smoke-test' else None
    try:
        if marker is not None:
            return smoke_test(marker)
        from smart_overlay.quick_app import run_quick_app

        return run_quick_app()
    except Exception as exc:
        if marker is not None:
            marker.write_text(traceback.format_exc(), encoding='utf-8')
        else:
            ctypes.windll.user32.MessageBoxW(None, str(exc), 'Smart Overlay OCR', 0x10)
        return 1


if __name__ == '__main__':
    sys.exit(main())
