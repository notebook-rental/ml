"""테스트 공통 준비물.

- tiny_backbone_path: DINOv2와 같은 입출력 모양의 아주 작은 ViT ONNX(무작위 가중치). 테스트를 빠르게 하려고 쓴다.
- storage: 임시 폴더를 저장소처럼 쓰는 LocalStorage
- put: 저장소에 사진을 넣는 도우미(테스트에서는 originals/·derived/ 어디든 넣을 수 있어야 하므로 직접 파일로 쓴다)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from build_vit_onnx import build  # noqa: E402

from common.storage import LocalStorage  # noqa: E402
from inference.compare import Models  # noqa: E402
from inference.config import Settings  # noqa: E402
from inference.pipeline.engine import create_session  # noqa: E402
from inference.pipeline.patch_features import Backbone  # noqa: E402
from inference.pipeline.segment import GrabCutSegmenter  # noqa: E402
from tests.fixtures import synth  # noqa: E402


@pytest.fixture(scope="session")
def tiny_backbone_path(tmp_path_factory) -> Path:
    import onnx

    p = tmp_path_factory.mktemp("models") / "tiny-vit.onnx"
    onnx.save(build(448, 336, dim=32, depth=1, heads=2, mlp=64), p)
    return p


@pytest.fixture(scope="session")
def patch_mean_path(tmp_path_factory) -> Path:
    import onnx

    from tests.fixtures.backbones import build_patch_mean

    p = tmp_path_factory.mktemp("models") / "patch-mean.onnx"
    onnx.save(build_patch_mean(), p)
    return p


@pytest.fixture(scope="session")
def models(patch_mean_path) -> Models:
    return Models(Backbone(create_session(patch_mean_path), "patch-mean@test"), GrabCutSegmenter())


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        app_profile="test",
        storage_backend="local",
        local_storage_root=str(tmp_path / "storage"),
        model_cache_dir=str(tmp_path / "cache"),
        internal_token="",
    )


@pytest.fixture
def storage(settings) -> LocalStorage:
    return LocalStorage(settings.local_storage_root)


@pytest.fixture
def put(storage):
    def _put(key: str, data: bytes) -> str:
        p = storage.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return key

    return _put


@pytest.fixture(scope="session")
def scene() -> synth.Scene:
    return synth.base_scene(seed=0)


@pytest.fixture(scope="session")
def baseline_img(scene) -> np.ndarray:
    return synth.view(scene.image, seed=1, shift=10)[0]
