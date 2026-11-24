"""Run with Windows x64 .venv/Scripts/python.exe packaging/build_portable.py.

Requires the portable requirements, pnpm install --frozen-lockfile, and
.runtime/node/{node.exe,LICENSE} from the official Windows x64 distribution.
"""
from importlib.metadata import distribution
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = Path(__file__).resolve().parent
NAME = 'SmartOverlayOCR-win-x64'
WHEELS = ('PySide6-Essentials', 'shiboken6', 'pymupdf', 'Pillow', 'numpy',
          'opencv-contrib-python', 'pydantic', 'pydantic-core', 'annotated-types',
          'typing-extensions', 'typing-inspection', 'httpx', 'httpcore', 'h11',
          'anyio', 'idna', 'certifi', 'sniffio')
IGNORE = shutil.ignore_patterns('__pycache__', '*.pyc')


def require_x64(path):
    with path.open('rb') as executable:
        executable.seek(0x3C)
        executable.seek(struct.unpack('<I', executable.read(4))[0])
        if executable.read(6) != b'PE\0\0\x64\x86':
            raise RuntimeError(f'Expected a Windows x64 executable: {path}')


def copy_python(target):
    base = Path(sys.base_prefix)
    target.mkdir()
    tag = f'{sys.version_info.major}{sys.version_info.minor}'
    require_x64(base / 'pythonw.exe')
    for path in [base / 'pythonw.exe', base / 'LICENSE.txt', *base.glob('*.dll')]:
        shutil.copy2(path, target)
    shutil.copytree(base / 'DLLs', target / 'DLLs', ignore=IGNORE)
    shutil.copytree(base / 'Lib', target / 'Lib',
                    ignore=shutil.ignore_patterns('site-packages', '__pycache__', 'test', 'ensurepip'))
    (target / f'python{tag}._pth').write_text('.\nDLLs\nLib\npackages\n..\n', encoding='ascii')
    packages = target / 'packages'
    for name in WHEELS:
        wheel = distribution(name)
        for item in wheel.files:
            if '..' in item.parts or '__pycache__' in item.parts:
                continue
            source = wheel.locate_file(item)
            if not source.is_file():
                continue
            destination = packages / item
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    # Make the wheel's app-local C++ runtime available to non-Qt extensions too.
    for pattern in ('msvcp140*.dll', 'concrt140.dll', 'vcomp140.dll'):
        for runtime in (packages / 'PySide6').glob(pattern):
            shutil.copy2(runtime, target)


def copy_node(internal):
    node = ROOT / '.runtime' / 'node' / 'node.exe'
    require_x64(node)
    (internal / 'node').mkdir()
    shutil.copy2(node, internal / 'node' / 'node.exe')
    shutil.copy2(node.with_name('LICENSE'), internal / 'node' / 'LICENSE.txt')
    store = ROOT / 'node_modules' / '.pnpm'
    modules = internal / 'node_modules'
    for manifest in sorted(store.glob('*/node_modules/*/package.json')) + sorted(store.glob('*/node_modules/@*/*/package.json')):
        relative = manifest.relative_to(store)
        if relative.parts[0] == 'node_modules':
            continue
        package = manifest.parent
        # pnpm's real package belongs to its own store slot; skip dependency links.
        key = '/'.join(relative.parts[2:-1]).replace('/', '+')
        if not relative.parts[0].startswith(key + '@'):
            continue
        destination = modules.joinpath(*relative.parts[2:-1])
        if destination.exists():
            raise RuntimeError(f'Multiple pnpm versions cannot be flattened: {key}')
        shutil.copytree(package, destination)
    subprocess.run([node, '-e', "require('pptxgenjs')"], cwd=internal, check=True,
                   env={k: v for k, v in os.environ.items() if k not in ('NODE_PATH', 'NODE_OPTIONS')})


def remove_readonly(function, path, exc):
    # pnpm's copied Windows store directories can retain the read-only bit.
    if not isinstance(exc, PermissionError):
        raise exc
    os.chmod(path, stat.S_IWRITE)
    function(path)


def build():
    if sys.platform != 'win32' or struct.calcsize('P') != 8:
        raise RuntimeError('Run the build with Windows x64 CPython')

    dist = ROOT / 'dist'
    dist.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='portable-', dir=dist) as temp:
        staging = Path(temp)
        package = staging / NAME
        internal = package / '_internal'
        internal.mkdir(parents=True)
        copy_python(internal / 'python')
        copy_node(internal)
        shutil.copytree(ROOT / 'smart_overlay', internal / 'smart_overlay', ignore=IGNORE)
        shutil.copy2(PACKAGING / 'portable_gui.py', internal)
        shutil.copy2(PACKAGING / 'PORTABLE-NOTICES.txt', package)
        csc = Path(os.environ['WINDIR']) / 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
        launcher = package / 'SmartOverlayOCR.exe'
        subprocess.run([csc, '/nologo', '/target:winexe', '/platform:x64',
                        '/reference:System.Windows.Forms.dll', f'/out:{launcher}',
                        PACKAGING / 'portable_launcher.cs'], cwd=staging, check=True)
        require_x64(launcher)
        marker = staging / 'smoke proof.txt'
        env = {k: v for k, v in os.environ.items() if not k.startswith('QT_')}
        result = subprocess.run([launcher, '--smoke-test', marker], cwd=staging,
                                env={**env, 'PATH': '', 'QT_QPA_PLATFORM': 'offscreen'}, timeout=90)
        if result.returncode or not marker.exists() or marker.read_text().strip() != 'OK':
            raise RuntimeError(marker.read_text(encoding='utf-8') if marker.exists() else f'Smoke test exited {result.returncode}')
        archive_path = staging / (NAME + '.zip')
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            for path in sorted(package.rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(staging).as_posix())
        final_dir = dist / NAME
        if final_dir.exists():
            shutil.rmtree(final_dir, onexc=remove_readonly)
        package.rename(final_dir)
        final_zip = dist / archive_path.name
        archive_path.replace(final_zip)
        print(final_dir, final_zip, sep='\n')


if __name__ == '__main__':
    build()
