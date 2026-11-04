"""기준 특징 사전 계산·저장·메모리 캐시 테스트 (tasks T065, T069)."""

from __future__ import annotations

import numpy as np
import pytest

from common.messages import QrCode
from inference.pipeline import baseline_cache as bc
from tests.fixtures import synth


@pytest.fixture
def features(models, baseline_img):
    return bc.compute(
        1,
        baseline_img,
        models.backbone,
        models.segmenter,
        1024,
        [QrCode(value="NR1:X", corners=[[1, 2], [3, 4], [5, 6], [7, 8]])],
    )


def test_compute_contents(features):
    f = features
    assert f.orig_size == (1600, 1200) and f.scale == pytest.approx(0.64)
    assert f.image.shape == (768, 1024, 3) and f.mask.shape == (768, 1024)
    assert len(f.keypoints.pts) > 100
    assert f.dino.shape[:2] == (24, 32)
    x1, y1, x2, y2 = f.crop
    assert 0 <= x1 < x2 <= 1024 and 0 <= y1 < y2 <= 768
    assert f.nbytes() > 0 and f.gray.ndim == 2


def test_serialize_roundtrip(features):
    data = bc.serialize(features)
    g = bc.deserialize(data)
    assert g.photo_id == "1" and g.backbone == features.backbone and g.crop == features.crop
    assert g.qr[0].value == "NR1:X"
    assert np.allclose(g.dino, features.dino, atol=1e-3)
    assert g.keypoints.desc.dtype == np.float32
    assert len(data) < 4_000_000


def test_store_lru_and_backbone_mismatch(storage, put, models, baseline_img):
    put("derived/work/b.jpg", synth.jpeg(baseline_img))
    store = bc.BaselineStore(storage, capacity=1)
    f = store.get_or_compute(1, "derived/work/b.jpg", models.backbone, models.segmenter, 1024)
    assert storage.exists(bc.cache_key(1))
    assert store.cached(1) is f
    assert store.load(1, models.backbone.name) is f and store.hits == 1
    assert store.load(1, "other-backbone") is None  # 백본이 바뀌면 다시 계산해야 한다
    assert store.load(2, models.backbone.name) is None
    # 용량 1: 다른 사진을 넣으면 앞의 것이 메모리에서 빠진다
    f2 = bc.deserialize(bc.serialize(f))
    f2.photo_id = "2"
    store._remember(f2)
    assert store.cached(1) is None
    # 메모리에 없어도 저장소에서 다시 읽는다
    again = store.load(1, models.backbone.name)
    assert again is not None and again.photo_id == "1"


def test_corrupted_cache_is_ignored(storage, put, models):
    put(bc.cache_key(9), b"garbage")
    assert bc.BaselineStore(storage, 2).load(9, models.backbone.name) is None


def test_old_format_rejected(features):
    import io
    import json

    data = bc.serialize(features)
    z = dict(np.load(io.BytesIO(data)))
    meta = json.loads(bytes(z["meta"]).decode())
    meta["format"] = 99
    z["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    buf = io.BytesIO()
    np.savez(buf, **z)
    with pytest.raises(ValueError):
        bc.deserialize(buf.getvalue())


def test_mask_crop_and_warp_mask():
    assert bc.mask_crop(np.zeros((10, 20), np.uint8)) == (0, 0, 20, 10)
    m = np.zeros((100, 100), np.uint8)
    m[20:40, 30:60] = 255
    assert bc.mask_crop(m, pad=0) == (30, 20, 60, 40)
    w = bc.warp_mask_to_target(m, np.eye(3), (100, 100))
    assert np.array_equal(w, m)
