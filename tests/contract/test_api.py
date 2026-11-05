"""ML 서비스 HTTP 계약 테스트 (tasks T051, contracts ml-service.md·api-response.md).

- 모든 응답이 공통 형식(header/result)인지
- /compare: done / not_comparable / failed, 동시 처리 상한 초과 503
- /quality, /baseline-features, /prefetch, /models/reload, 헬스체크, 내부 토큰
"""

from __future__ import annotations

import hashlib
import threading
import time

import pytest
from fastapi.testclient import TestClient

import inference.compare as compare_mod
from inference.app import create_app
from inference.model_registry import ModelRegistry
from inference.pipeline.baseline_cache import cache_key
from inference.pipeline.qr import detect_qr
from tests.fixtures import synth


@pytest.fixture
def app_factory(settings, storage, put, patch_mean_path):
    data = patch_mean_path.read_bytes()
    put("models/dinov2-s14/model.onnx", data)
    settings.backbone_sha256 = hashlib.sha256(data).hexdigest()

    def make(**overrides):
        for k, v in overrides.items():
            setattr(settings, k, v)
        registry = ModelRegistry(storage, settings)
        registry.load_initial()
        return create_app(settings, storage, registry=registry, load_models=False)

    return make


@pytest.fixture
def client(app_factory):
    with TestClient(app_factory()) as c:
        yield c


@pytest.fixture
def photos(put, scene, baseline_img):
    target = synth.add_damage(scene.image, synth.Damage("sticker", (500, 400, 620, 480)))
    target, _ = synth.view(target, seed=2, shift=25, brightness=0.9)
    put("derived/work/base.jpg", synth.jpeg(baseline_img))
    put("derived/work/target.jpg", synth.jpeg(target))
    put("derived/work/broken.jpg", b"not a jpeg")
    return {"qr": [q.model_dump() for q in detect_qr(baseline_img)]}


def _ok(r, status=200):
    assert r.status_code == status, r.text
    body = r.json()
    assert body["header"] == {"isSuccessful": True, "resultCode": 0, "resultMessage": "SUCCESS"}
    return body["result"]


def _fail(r, status, code):
    assert r.status_code == status, r.text
    h = r.json()["header"]
    assert h["isSuccessful"] is False and h["resultCode"] == code and h["resultMessage"]
    return r.json()["result"]


def compare_body(**kw):
    body = {
        "schemaVersion": 1,
        "comparisonId": 101,
        "mode": "production",
        "modelVersion": {"id": 1, "name": "changedet-v1", "artifactKey": None, "artifactSha256": None},
        "shotType": "top",
        "deviceId": 9,
        "baselinePhotoId": 55,
        "baselineImageKey": "derived/work/base.jpg",
        "targetImageKey": "derived/work/target.jpg",
        "existingDamages": [],
        "baselineQr": [],
        "params": {"threshold": None},
    }
    body.update(kw)
    return body


def test_health(client):
    assert _ok(client.get("/health/live"))["status"] == "UP"
    r = _ok(client.get("/health/ready"))
    assert r["backbone"] == "dinov2-s14@1" and r["segmenter"] == "grabcut@1"


def test_not_ready(settings, storage):
    app = create_app(settings, storage, load_models=False)
    with TestClient(app) as c:
        _fail(c.get("/health/ready"), 503, -1098)
        _fail(c.post("/compare", json=compare_body()), 503, -1098)
        _fail(c.post("/prefetch", json={"baselinePhotoIds": [1]}), 503, -1098)
        _fail(
            c.post("/baseline-features", json={"photoId": 1, "imageKey": "x", "shotType": "top"}), 503, -1098
        )


def test_quality(client, photos):
    res = _ok(
        client.post(
            "/quality",
            json={
                "schemaVersion": 1,
                "imageKey": "derived/work/base.jpg",
                "shotType": "top",
                "deviceModelId": 3,
            },
        )
    )
    assert res["pass"] is True and set(res["scores"]) == {"blur", "exposure", "coverage"}
    assert res["qr"][0]["value"] == synth.QR_VALUE and len(res["qr"][0]["corners"]) == 4
    _fail(client.post("/quality", json={"imageKey": "derived/none.jpg", "shotType": "top"}), 404, -1002)
    _fail(
        client.post("/quality", json={"imageKey": "derived/work/broken.jpg", "shotType": "top"}), 400, -1001
    )
    err = _fail(client.post("/quality", json={"imageKey": "x", "shotType": "lid"}), 400, -1001)
    assert err["errors"][0]["field"] == "shotType"


def test_compare_done(client, photos):
    res = _ok(client.post("/compare", json=compare_body(baselineQr=photos["qr"])))
    assert res["status"] == "done"
    assert len(res["alignment"]["homography"]) == 3 and res["alignment"]["reprojErrorPx"] >= 0
    assert res["derivedKeys"]["alignedTarget"].startswith("derived/comparisons/101/")
    assert res["modelVersionId"] == 1 and res["runtimeMs"] > 0
    f = [x for x in res["findings"] if x["predictedType"] != "qr_tampered"]
    assert len(f) == 1 and f[0]["predictedType"] == "sticker" and 0 < f[0]["confidence"] <= 1
    assert f[0]["geometry"]["type"] == "polygon" and f[0]["overlapsDamageId"] is None


def test_compare_bank_picks_matching_baseline(client, put, photos):
    """기준 사진 목록(bankPhotos) 중 반납 사진과 닮은 사진을 고른다. 컷 이름이 틀려도 된다(004 US2)."""
    import numpy as np

    other = np.full((1200, 1600, 3), (30, 120, 30), np.uint8)
    other[300:900, 400:1200] = (20, 200, 220)
    put("derived/work/other.jpg", synth.jpeg(other))
    bank = [
        {"photoId": 70, "imageKey": "derived/work/other.jpg", "shotType": "top"},
        {
            "photoId": 71,
            "imageKey": "derived/work/base.jpg",
            "shotType": "bottom",
        },  # 이름은 bottom, 실제는 상판
    ]
    res = _ok(
        client.post(
            "/compare",
            json=compare_body(baselinePhotoId=70, baselineImageKey="derived/work/other.jpg", bankPhotos=bank),
        )
    )
    assert res["status"] == "done"
    assert res["matchedBaselinePhotoId"] == 71
    assert res["method"] in ("aligned", "patch")
    assert any(f["predictedType"] != "qr_tampered" for f in res["findings"])


def test_compare_done_reports_method(client, photos):
    res = _ok(client.post("/compare", json=compare_body()))
    assert res["method"] == "aligned" and res["matchedBaselinePhotoId"] == 55


def test_compare_string_ids_and_unknown_fields(client, photos):
    res = _ok(client.post("/compare", json=compare_body(comparisonId="c-1", futureField=True)))
    assert res["status"] == "done"


def test_compare_not_comparable(client, put, photos, caplog):
    import logging

    import numpy as np

    caplog.set_level(logging.INFO, logger="inference.compare")

    noise = np.full((900, 1200, 3), 128, "uint8")  # 노트북이 안 보이는 사진 → laptop_not_found
    put("derived/work/noise.jpg", synth.jpeg(noise))
    res = _ok(
        client.post("/compare", json=compare_body(targetImageKey="derived/work/noise.jpg", shotType="hinge"))
    )
    assert res["status"] == "not_comparable" and res["reason"]
    # 왜 자동 비교 불가인지 운영 로그에 남는다
    assert any(f"reason={res['reason']}" in m for m in caplog.messages)


def test_compare_failed(client, photos):
    res = _ok(client.post("/compare", json=compare_body(targetImageKey="derived/work/missing.jpg")))
    assert res["status"] == "failed" and "ObjectNotFound" in res["error"]
    res = _ok(client.post("/compare", json=compare_body(targetImageKey="derived/work/broken.jpg")))
    assert res["status"] == "failed"


def test_compare_unexpected_error_is_failed(client, photos, monkeypatch):
    def boom(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(compare_mod.bank_mod, "match", boom)
    res = _ok(client.post("/compare", json=compare_body()))
    assert res["status"] == "failed" and "boom" in res["error"]


def test_compare_busy_returns_503(app_factory, photos, monkeypatch):
    app = app_factory(max_concurrent_compares=1)
    started, release = threading.Event(), threading.Event()
    real = compare_mod.compare

    def slow(*a, **k):
        started.set()
        release.wait(5)
        return real(*a, **k)

    monkeypatch.setattr(compare_mod, "compare", slow)
    with TestClient(app) as c:
        t = threading.Thread(target=lambda: c.post("/compare", json=compare_body()))
        t.start()
        assert started.wait(5)
        _fail(c.post("/compare", json=compare_body()), 503, -4003)
        release.set()
        t.join(10)


def test_baseline_features_and_prefetch(client, storage, photos):
    r = _ok(
        client.post(
            "/baseline-features",
            json={
                "schemaVersion": 1,
                "photoId": 55,
                "imageKey": "derived/work/base.jpg",
                "shotType": "top",
                "qr": photos["qr"],
            },
        ),
        202,
    )
    assert r["featuresKey"] == cache_key(55)
    svc = client.app.state.svc
    for _ in range(100):
        if storage.exists(cache_key(55)):
            break
        time.sleep(0.05)
    assert storage.exists(cache_key(55))
    svc.baselines._lru.clear()
    assert (
        _ok(
            client.post(
                "/prefetch",
                json={
                    "schemaVersion": 1,
                    "deviceId": 9,
                    "baselinePhotoIds": [55, 56],
                    "backbone": "dinov2-s14@1",
                },
            ),
            202,
        )
        is None
    )
    for _ in range(100):
        if svc.baselines.cached(55) is not None:
            break
        time.sleep(0.05)
    assert svc.baselines.cached(55) is not None
    # 계산 실패(이미지 없음)는 로그만 남기고 서비스는 계속 산다
    _ok(
        client.post(
            "/baseline-features", json={"photoId": 77, "imageKey": "derived/none.jpg", "shotType": "top"}
        ),
        202,
    )
    svc.background.submit(lambda: None).result(5)
    assert not storage.exists(cache_key(77))


def test_prefetch_errors_are_swallowed(client, monkeypatch):
    svc = client.app.state.svc

    def boom(*a, **k):
        raise OSError("storage down")

    monkeypatch.setattr(svc.baselines, "load", boom)
    from common.messages import PrefetchRequest

    svc.prefetch(PrefetchRequest(baselinePhotoIds=[1]))  # 예외가 밖으로 나오지 않는다


def test_models_reload(client):
    body = {
        "schemaVersion": 1,
        "production": {"id": 7, "name": "changedet-v1", "kind": "zero_shot_change"},
        "shadow": None,
    }
    assert _ok(client.post("/models/reload", json=body))["changed"] is True
    assert _ok(client.post("/models/reload", json=body))["changed"] is False


def test_internal_token(app_factory, photos):
    app = app_factory(internal_token="s3cret")
    with TestClient(app) as c:
        _fail(c.post("/quality", json={"imageKey": "derived/work/base.jpg", "shotType": "top"}), 401, -1004)
        _ok(
            c.post(
                "/quality",
                json={"imageKey": "derived/work/base.jpg", "shotType": "top"},
                headers={"X-Internal-Token": "s3cret"},
            )
        )
        _ok(c.get("/health/live"))  # 헬스체크는 토큰 없이(kubelet이 부른다)


def test_spring_poll_on_start(settings, storage, put, patch_mean_path):
    from common.messages import ActiveModel, ActiveModels

    class FakeSpring:
        def active_models(self):
            return ActiveModels(production=ActiveModel(id=11, name="changedet-v1"))

    put("models/dinov2-s14/model.onnx", patch_mean_path.read_bytes())
    from inference.app import Service

    svc = Service(settings, storage, spring=FakeSpring())
    svc.start(load_models=True, poll=False)
    assert svc.registry.models.model_version_id == 11
    svc.stop()


def test_probe_paths(client):
    """deploy의 startupProbe·livenessProbe(/healthz), readinessProbe(/readyz)와 경로가 맞아야 한다."""
    assert _ok(client.get("/healthz"))["status"] == "UP"
    assert _ok(client.get("/readyz"))["backbone"] == "dinov2-s14@1"


def test_model_load_retries_until_file_appears(settings, storage, put, patch_mean_path):
    """MinIO에 모델 파일이 없으면 503 + 실패 이유를 보이고, 파일을 올리면 다시 띄우지 않아도 Ready가 된다."""
    import threading

    from inference.app import Service

    svc = Service(settings, storage)
    t = threading.Thread(target=svc.start, kwargs={"poll": False, "retry_seconds": 0.05}, daemon=True)
    t.start()
    for _ in range(100):
        if svc.load_error:
            break
        threading.Event().wait(0.02)
    assert svc.load_error and "ObjectNotFound" in svc.load_error
    app = create_app(settings, storage, registry=svc.registry, load_models=False)
    app.state.svc.load_error = svc.load_error
    with TestClient(app) as c:
        r = c.get("/readyz")
        assert r.status_code == 503 and "ObjectNotFound" in r.json()["result"]["loadError"]
    put("models/dinov2-s14/model.onnx", patch_mean_path.read_bytes())
    t.join(timeout=30)
    assert svc.registry.ready and svc.load_error is None
    svc.stop()


def test_model_load_stops_on_shutdown(settings, storage):
    from inference.app import Service

    svc = Service(settings, storage)
    svc.stop()  # 이미 종료 중이면 한 번 실패한 뒤 바로 멈춘다
    svc.start(poll=False, retry_seconds=10)
    assert not svc.registry.ready and svc.load_error
