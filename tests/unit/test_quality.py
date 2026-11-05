"""품질 검사 단위 테스트 (tasks T036): 흐림, 어두움, 밝음, 구도 이탈, QR 없음·여러 개."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from inference.quality import check_quality, laptop_box
from tests.fixtures import synth


def test_good_photo_passes(baseline_img, settings):
    r = check_quality(baseline_img, "top", settings)
    assert r.passed and r.reasons == []
    assert r.qr and r.qr[0].value == synth.QR_VALUE
    x1, y1, x2, y2 = r.laptopBox
    assert x1 < 300 and y1 < 260 and x2 > 1300 and y2 > 950
    assert r.model_dump(by_alias=True)["pass"] is True


def test_blurry(baseline_img, settings):
    r = check_quality(cv2.GaussianBlur(baseline_img, (0, 0), 6), "keyboard", settings)
    assert "blurry" in r.reasons and not r.passed


@pytest.mark.parametrize(("factor", "reason"), [(0.15, "too_dark"), (3.0, "too_bright")])
def test_exposure(baseline_img, settings, factor, reason):
    img = np.clip(baseline_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    assert reason in check_quality(img, "keyboard", settings).reasons


def test_out_of_frame(settings):
    img = np.full((1200, 1600, 3), 90, np.uint8)
    cv2.rectangle(img, (40, 40), (200, 150), (200, 200, 200), -1)
    r = check_quality(img, "keyboard", settings)
    assert "out_of_frame" in r.reasons


def test_missing_label_does_not_reject(settings):
    img = synth.view(synth.base_scene(0, qr_value=None).image, seed=1, shift=10)[0]
    assert "qr_not_found" not in check_quality(img, "top", settings).reasons  # 라벨이 바닥면에 있을 수 있다
    assert "qr_not_found" not in check_quality(img, "hinge", settings).reasons


def test_multiple_qr(settings):
    img = synth.base_scene(0).image.copy()
    img[700:880, 400:580] = synth._qr_image("NR1:BBBBBBBBBB", 180)
    r = check_quality(img, "top", settings)
    assert "multiple_qr" in r.reasons


def test_laptop_box_none_on_blank():
    assert laptop_box(np.zeros((300, 400), np.uint8)) is None


def test_barcode_label_counts_as_label_on_top(settings):
    """자산 번호 라벨이 바코드여도 상판 컷을 통과시키고 값을 돌려준다(spec 002)."""
    img = synth.base_scene(0, qr_value=None).image.copy()
    bc = synth.barcode_image("NB26110001")
    h, w = bc.shape[:2]
    img[300 : 300 + h, 500 : 500 + w] = bc
    r = check_quality(img, "top", settings)
    assert "qr_not_found" not in r.reasons
    assert [q.value for q in r.qr] == ["NB26110001"]
    assert len(r.qr[0].corners) == 4


def test_barcode_not_searched_on_other_shots(settings):
    img = synth.base_scene(0, qr_value=None).image.copy()
    bc = synth.barcode_image("NB26110001")
    h, w = bc.shape[:2]
    img[300 : 300 + h, 500 : 500 + w] = bc
    assert check_quality(img, "hinge", settings).qr == []
