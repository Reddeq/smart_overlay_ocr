from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.create_example import create


@pytest.fixture
def project(tmp_path):
    return create(tmp_path)

