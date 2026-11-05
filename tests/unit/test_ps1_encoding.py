"""Windows PowerShell 5.1은 BOM 없는 UTF-8을 시스템 코드 페이지(CP949)로 읽어 한글이 깨지고 따옴표가 어긋난다."""

from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[2] / "tools").glob("*.ps1"))


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_ps1_has_utf8_bom(path: Path):
    assert path.read_bytes().startswith(b"\xef\xbb\xbf"), f"{path.name}는 UTF-8 BOM으로 저장해야 한다"
