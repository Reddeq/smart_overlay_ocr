import os
from pathlib import Path
import struct
import subprocess
import zipfile

import pytest


def test_portable_executable_from_build(tmp_path):
    if not (directory := os.environ.get('SMART_OVERLAY_PORTABLE_TEST')):
        pytest.skip('Set SMART_OVERLAY_PORTABLE_TEST to the built portable directory')
    directory = Path(directory).resolve()
    unpacked = tmp_path / 'Проверка архива'
    with zipfile.ZipFile(directory.with_suffix('.zip')) as archive:
        assert not any('/paddle' in name.lower() for name in archive.namelist())
        assert not any(name.endswith(('.pdmodel', '.pdiparams', '.safetensors')) for name in archive.namelist())
        archive.extractall(unpacked)
    exe = unpacked / directory.name / 'SmartOverlayOCR.exe'
    data = exe.read_bytes()
    pe = struct.unpack_from('<I', data, 0x3C)[0]
    assert data[pe:pe + 6] == b'PE\0\0\x64\x86'
    env = {k: v for k, v in os.environ.items() if not k.startswith('QT_')}
    env.update(PATH='', PYTHONPATH=str(tmp_path / 'not-installed'), QT_QPA_PLATFORM='windows')
    marker = tmp_path / 'smoke result.txt'
    process = subprocess.run([exe, '--smoke-test', marker], cwd=tmp_path, env=env, timeout=90)
    report = marker.read_text('utf-8') if marker.is_file() else 'No marker from launcher'
    assert process.returncode == 0 and report.strip() == 'OK', report
