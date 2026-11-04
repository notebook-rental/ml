"""모델 내려받기·sha256 검증·교체 테스트 (contracts ml-service.md "모델 교체")."""

from __future__ import annotations

import hashlib

import pytest

from common.messages import ActiveModel, ActiveModels
from inference.model_registry import ActiveModelPoller, ModelIntegrityError, ModelRegistry


@pytest.fixture
def registry(settings, storage, put, patch_mean_path):
    data = patch_mean_path.read_bytes()
    put("models/dinov2-s14/model.onnx", data)
    settings.backbone_sha256 = hashlib.sha256(data).hexdigest()
    return ModelRegistry(storage, settings)


def test_load_and_cache(registry, settings):
    assert not registry.ready
    with pytest.raises(RuntimeError):
        _ = registry.models
    registry.load_initial()
    assert registry.ready and registry.models.backbone.size == (448, 336)
    assert registry.models.segmenter.name == "grabcut@1"
    # 두 번째는 로컬 캐시를 그대로 쓴다
    p = registry.fetch(settings.backbone_key, settings.backbone_sha256)
    assert p.is_file()


def test_integrity_error(registry, settings):
    settings.backbone_sha256 = "0" * 64
    with pytest.raises(ModelIntegrityError):
        registry.load_initial()
    assert not registry.ready


def test_backbone_input_override(registry, settings):
    settings.backbone_input = "448x336"
    assert registry.build().backbone.size == (448, 336)


def test_apply_active_models(registry, put):
    registry.load_initial()
    put("models/dmgseg-v3/model.onnx", b"onnx-bytes")
    sha = hashlib.sha256(b"onnx-bytes").hexdigest()
    a = ActiveModels(production=ActiveModel(id=3, name="changedet-v1"))
    assert registry.apply(a) is True
    assert registry.models.model_version_id == 3 and registry.active == a
    assert registry.apply(a) is False  # 같은 내용이면 바꾸지 않는다
    b = ActiveModels(
        production=ActiveModel(
            id=4,
            name="dmgseg-v3",
            kind="trained_segmentation",
            artifactKey="models/dmgseg-v3/model.onnx",
            artifactSha256=sha,
        )
    )
    assert registry.apply(b) is True
    assert registry.apply(ActiveModels(production=None)) is False


class _Client:
    def __init__(self, value=None, error=None):
        self.value, self.error = value, error

    def active_models(self):
        if self.error:
            raise self.error
        return self.value


def test_poller(registry):
    registry.load_initial()
    ok = ActiveModelPoller(
        registry, _Client(ActiveModels(production=ActiveModel(id=1, name="changedet-v1"))), 30
    )
    assert ok.poll_once() is True
    bad = ActiveModelPoller(registry, _Client(error=ConnectionError("down")), 30)
    assert bad.poll_once() is False
    bad.stop()


def test_seed_from_url_when_bucket_empty(settings, storage, patch_mean_path):
    """버킷에 백본이 없으면 공개 주소에서 받아 해시를 확인하고, 버킷 models/ 에도 한 번 채운다."""
    data = patch_mean_path.read_bytes()
    settings.backbone_sha256 = hashlib.sha256(data).hexdigest()
    settings.backbone_url = "https://example.invalid/model.onnx"
    reg = ModelRegistry(storage, settings)
    calls = []
    reg.download_url = lambda url: calls.append(url) or data
    reg.load_initial()
    assert reg.ready and calls == [settings.backbone_url]
    assert storage.get_bytes(settings.backbone_key) == data


def test_seed_rejects_bad_hash_and_never_seeds(settings, storage, patch_mean_path):
    settings.backbone_sha256 = "0" * 64
    settings.backbone_url = "https://example.invalid/model.onnx"
    reg = ModelRegistry(storage, settings)
    reg.download_url = lambda url: b"not a model"
    with pytest.raises(ModelIntegrityError):
        reg.load_initial()
    assert not storage.exists(settings.backbone_key)


def test_no_url_or_hash_keeps_not_found(settings, storage):
    from common.storage import ObjectNotFound

    settings.backbone_url = "https://example.invalid/model.onnx"
    settings.backbone_sha256 = ""  # 해시 없이 인터넷 파일을 믿지 않는다
    with pytest.raises(ObjectNotFound):
        ModelRegistry(storage, settings).load_initial()


def test_seed_failure_is_not_fatal(settings, storage, patch_mean_path):
    data = patch_mean_path.read_bytes()
    settings.backbone_sha256 = hashlib.sha256(data).hexdigest()
    settings.backbone_url = "https://example.invalid/model.onnx"

    class NoWrite:
        def get_bytes(self, key):
            return storage.get_bytes(key)

        def put_new_model(self, key, data):
            raise PermissionError("AccessDenied")

    reg = ModelRegistry(NoWrite(), settings)
    reg.download_url = lambda url: data
    reg.load_initial()
    assert reg.ready
    reg2 = ModelRegistry(type("Plain", (), {"get_bytes": NoWrite.get_bytes})(), settings)
    reg2._seed("models/x.onnx", data)  # put_new_model이 없는 저장소는 건너뛴다
