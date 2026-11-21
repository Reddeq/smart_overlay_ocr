"""Dependency diagnostics must identify the Python used by the launcher."""
import builtins
from pathlib import Path
import sys

from packaging.requirements import Requirement
import pytest

from smart_overlay.native_pdf import NativePDFError, _pdf_lib


@pytest.mark.parametrize("detail",["No module named 'pymupdf'","DLL load failed while importing _extra"])
def test_pdf_import_error_identifies_running_interpreter(monkeypatch,detail):
    real_import=builtins.__import__
    def failed_import(name,*args,**kwargs):
        if name=="pymupdf":
            raise ImportError(detail)
        return real_import(name,*args,**kwargs)
    executable="C:/Program Files/Smart Overlay/.venv/Scripts/python.exe"
    monkeypatch.setattr(sys,"executable",executable)
    monkeypatch.setattr(builtins,"__import__",failed_import)
    with pytest.raises(NativePDFError) as caught:
        _pdf_lib()
    message=str(caught.value)
    assert executable in message
    assert detail in message
    assert "-m pip install" in message
    assert "PyMuPDF>=1.26.4,<2" in message


def test_default_install_lock_includes_supported_pymupdf():
    root=Path(__file__).resolve().parents[1]
    def requirements(path):
        return {r.name.casefold():r for line in path.read_text(encoding="utf8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
                for r in [Requirement(line)]}
    core=requirements(root/"requirements.txt")["pymupdf"]
    locked=requirements(root/"requirements-lock.txt")
    assert "pymupdf" in locked,"The documented default installation must include native PDF support"
    versions=[spec.version for spec in locked["pymupdf"].specifier if spec.operator=="=="]
    assert len(versions)==1 and versions[0] in core.specifier
