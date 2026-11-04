"""1단계 변화 탐지(changedet-v1) 회귀 테스트 (tasks T049).

고정된 합성 전/후 사진으로 "어디에, 몇 개" 후보가 나오는지 확인한다. 파이프라인을 고치다가 성능이 나빠지면
여기서 걸린다. 실제 사진 평가(재현율·오탐)는 초기 평가 세트(research R11)로 따로 한다.

경우:
- 스티커 붙음 → 그 자리에 후보 1개(유형 sticker)
- 테이프 선(가늘고 긴 변화) → 후보 1개(유형 scratch)
- 흰 스크래치 → 후보 1개
- 조명만 다름(어둡게·밝게) → 후보 0개
- 손상 없음, 다른 각도 → 후보 0개
- QR 스티커 제거 → qr_tampered 후보
- QR 영역만 바뀜(다른 QR) → QR 영역은 손상 후보가 아니고 qr_tampered만
"""

from __future__ import annotations

import numpy as np
import pytest

from common.messages import CompareRequest, QrCode
from inference.compare import compare
from inference.pipeline.baseline_cache import BaselineStore
from tests.fixtures import synth

pytestmark = pytest.mark.regression

BASE_QR = None


def _run(storage, put, models, settings, baseline_img, target_img, qr=True, shot="top"):
    put("derived/work/base.jpg", synth.jpeg(baseline_img))
    put("derived/work/target.jpg", synth.jpeg(target_img))
    baseline_qr = []
    if qr:
        from inference.pipeline.qr import detect_qr

        baseline_qr = detect_qr(baseline_img)
        assert baseline_qr, "기준 사진에서 QR이 읽혀야 한다"
    req = CompareRequest(
        comparisonId=1,
        shotType=shot,
        baselinePhotoId=1,
        baselineImageKey="derived/work/base.jpg",
        targetImageKey="derived/work/target.jpg",
        baselineQr=baseline_qr,
    )
    return compare(req, storage, BaselineStore(storage, 4), models, settings)


def _center(f) -> np.ndarray:
    return np.array(f.geometry.coords).mean(axis=0)


def _near(f, point, tol=60.0) -> bool:
    return float(np.linalg.norm(_center(f) - np.array(point))) < tol


def damaged_view(scene, damages, seed=2, brightness=0.9):
    img = scene.image
    for d in damages:
        img = synth.add_damage(img, d)
    return synth.view(img, seed=seed, shift=25, brightness=brightness)


def test_sticker_found(storage, put, models, settings, scene, baseline_img):
    d = synth.Damage("sticker", (500, 400, 620, 480))
    target, M = damaged_view(scene, [d])
    r = _run(storage, put, models, settings, baseline_img, target)
    assert r.status == "done"
    damage = [f for f in r.findings if f.predictedType != "qr_tampered"]
    assert len(damage) == 1
    assert _near(damage[0], synth.map_box(d.box, M))
    assert damage[0].predictedType == "sticker"
    assert damage[0].baselineGeometry is not None


def test_tape_line_found_as_scratch(storage, put, models, settings, scene, baseline_img):
    d = synth.Damage("tape", (400, 750, 760, 790))
    target, M = damaged_view(scene, [d])
    r = _run(storage, put, models, settings, baseline_img, target)
    damage = [f for f in r.findings if f.predictedType != "qr_tampered"]
    assert len(damage) == 1
    assert _near(damage[0], synth.map_box(d.box, M), tol=90)
    assert damage[0].predictedType == "scratch"


def test_thin_scratch_found(storage, put, models, settings, scene, baseline_img):
    d = synth.Damage("scratch", (700, 700, 950, 760))
    target, M = damaged_view(scene, [d])
    r = _run(storage, put, models, settings, baseline_img, target)
    damage = [f for f in r.findings if f.predictedType != "qr_tampered"]
    assert len(damage) == 1
    assert _near(damage[0], synth.map_box(d.box, M), tol=90)


def test_two_damages_found(storage, put, models, settings, scene, baseline_img):
    ds = [synth.Damage("sticker", (500, 400, 620, 480)), synth.Damage("stain", (900, 800, 980, 860))]
    target, M = damaged_view(scene, ds)
    r = _run(storage, put, models, settings, baseline_img, target)
    damage = [f for f in r.findings if f.predictedType != "qr_tampered"]
    assert len(damage) == 2
    for d in ds:
        assert any(_near(f, synth.map_box(d.box, M)) for f in damage)


@pytest.mark.parametrize("brightness", [0.7, 0.85, 1.15])
def test_lighting_only_no_findings(storage, put, models, settings, scene, baseline_img, brightness):
    target, _ = synth.view(scene.image, seed=3, shift=25, brightness=brightness)
    r = _run(storage, put, models, settings, baseline_img, target)
    assert r.status == "done"
    assert r.findings == []


def test_no_damage_other_angle_no_findings(storage, put, models, settings, scene, baseline_img):
    target, _ = synth.view(scene.image, seed=4, shift=35)
    r = _run(storage, put, models, settings, baseline_img, target)
    assert r.status == "done"
    assert r.findings == []
    assert r.alignment.reprojErrorPx < 3


def test_qr_removed_is_tampered(storage, put, models, settings, baseline_img):
    no_qr = synth.base_scene(seed=0, qr_value=None)
    target, M = synth.view(no_qr.image, seed=5, shift=20)
    r = _run(storage, put, models, settings, baseline_img, target)
    assert r.status == "done"
    tampered = [f for f in r.findings if f.predictedType == "qr_tampered"]
    assert len(tampered) == 1
    qr_box = (
        synth.QR_POS[0],
        synth.QR_POS[1],
        synth.QR_POS[0] + synth.QR_SIZE,
        synth.QR_POS[1] + synth.QR_SIZE,
    )
    assert _near(tampered[0], synth.map_box(qr_box, M))
    assert tampered[0].confidence == pytest.approx(0.6)


def test_qr_replaced_only_tampered(storage, put, models, settings, baseline_img):
    other = synth.base_scene(seed=0, qr_value="NR1:AAAAAAAAAA")
    target, _ = synth.view(other.image, seed=6, shift=20)
    r = _run(storage, put, models, settings, baseline_img, target)
    assert [f.predictedType for f in r.findings] == ["qr_tampered"]
    assert r.findings[0].confidence == pytest.approx(0.9)


def test_existing_damage_overlap(storage, put, models, settings, scene, baseline_img):
    d = synth.Damage("sticker", (500, 400, 620, 480))
    target, _ = damaged_view(scene, [d])
    # 기존 손상은 기준 사진 좌표로 준다: 장면 좌표 상자를 기준 사진으로 옮긴다
    _, Mb = synth.view(scene.image, seed=1, shift=10)
    import cv2

    pts = cv2.perspectiveTransform(np.float32([[[500, 400]], [[620, 400]], [[620, 480]], [[500, 480]]]), Mb)
    existing = [{"id": 77, "geometry": {"type": "polygon", "coords": pts.reshape(-1, 2).tolist()}}]
    put("derived/work/base.jpg", synth.jpeg(baseline_img))
    put("derived/work/target.jpg", synth.jpeg(target))
    req = CompareRequest.model_validate(
        {
            "comparisonId": 2,
            "shotType": "keyboard",
            "baselinePhotoId": 2,
            "baselineImageKey": "derived/work/base.jpg",
            "targetImageKey": "derived/work/target.jpg",
            "existingDamages": existing,
        }
    )
    r = compare(req, storage, BaselineStore(storage, 4), models, settings)
    damage = [f for f in r.findings if f.predictedType != "qr_tampered"]
    assert damage and damage[0].overlapsDamageId == 77


def test_unrelated_photo_not_comparable(storage, put, models, settings, baseline_img):
    rng = np.random.default_rng(0)
    noise = (rng.random((1200, 1600, 3)) * 255).astype(np.uint8)
    r = _run(storage, put, models, settings, baseline_img, noise, qr=False, shot="keyboard")
    assert r.status == "not_comparable"
    assert r.reason in {
        "too_few_matches",
        "too_few_inliers",
        "alignment_error_too_large",
        "homography_failed",
    }
    assert r.findings == []


def test_idempotent(storage, put, models, settings, scene, baseline_img):
    target, _ = damaged_view(scene, [synth.Damage("sticker", (500, 400, 620, 480))])
    a = _run(storage, put, models, settings, baseline_img, target)
    b = _run(storage, put, models, settings, baseline_img, target)
    assert [f.geometry for f in a.findings] == [f.geometry for f in b.findings]


def test_qr_model_roundtrip():
    q = QrCode(value="NR1:X", corners=[[0, 0], [1, 0], [1, 1], [0, 1]])
    assert QrCode.model_validate(q.model_dump()) == q
