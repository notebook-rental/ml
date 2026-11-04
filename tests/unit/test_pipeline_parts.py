"""파이프라인 부품 단위 테스트: 조명 정규화, SSIM, 패치 거리, 후처리, 유형 추정, 기존 손상 겹침, QR 마스크."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from common.messages import ExistingDamage, Geometry, QrCode
from inference.pipeline import existing_damage, normalize, postprocess, typing
from inference.pipeline.imageutil import ImageDecodeError, decode, encode_png, resize_long_side, to_gray
from inference.pipeline.patch_features import l2_normalize, neighbor_distance
from inference.pipeline.qr import qr_polygon_mask
from inference.pipeline.ssim import ssim_diff, ssim_map


def test_match_histogram_restores_brightness():
    rng = np.random.default_rng(0)
    ref = (rng.random((100, 100)) * 200).astype(np.uint8)
    dark = (ref * 0.6).astype(np.uint8)
    mask = np.full((100, 100), 255, np.uint8)
    out = normalize.match_histogram(dark, ref, mask)
    assert abs(float(out.mean()) - float(ref.mean())) < 3
    assert np.array_equal(normalize.match_histogram(dark, ref, np.zeros_like(mask)), dark)
    col = normalize.match_color(cv2.merge([dark] * 3), cv2.merge([ref] * 3), mask)
    assert col.shape == (100, 100, 3)
    t, b = normalize.normalize_pair(dark, ref, mask)
    assert b is ref


def test_ssim_identical_and_different():
    rng = np.random.default_rng(0)
    a = (rng.random((64, 64)) * 255).astype(np.uint8)
    assert np.allclose(ssim_map(a, a), 1.0, atol=1e-4)
    assert ssim_diff(a, a).max() < 0.01
    b = a.copy()
    b[20:40, 20:40] = 255 - b[20:40, 20:40]
    d = ssim_diff(a, b)
    assert d[30, 30] > 0.3 and d[5, 5] < 0.05


def test_ssim_tolerates_one_pixel_shift():
    rng = np.random.default_rng(1)
    a = cv2.GaussianBlur((rng.random((80, 80)) * 255).astype(np.uint8), (0, 0), 1)
    b = np.roll(a, 1, axis=1)
    assert ssim_diff(a, b)[10:-10, 10:-10].max() < 0.05
    assert ssim_diff(a, b, tolerance=0)[10:-10, 10:-10].max() > 0.05


def test_neighbor_distance_tolerates_shift():
    rng = np.random.default_rng(0)
    f = l2_normalize(rng.normal(size=(8, 8, 16)).astype(np.float32))
    assert neighbor_distance(f, f, 1).max() < 1e-5
    shifted = np.roll(f, 1, axis=1)
    assert neighbor_distance(shifted, f, 1)[:, 1:].max() < 1e-5
    assert neighbor_distance(shifted, f, 0)[:, 1:].max() > 0.1


def _blob(shape, pts):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.int32)], 255)
    return m


def test_extract_regions_filters():
    score = np.zeros((200, 200), np.float32)
    score[20:60, 20:60] = 0.9  # 후보
    score[100:102, 100:102] = 0.9  # 너무 작음
    score[150:190, 150:190] = 0.9  # 반사
    valid = np.full((200, 200), 255, np.uint8)
    refl = np.zeros_like(valid)
    refl[150:190, 150:190] = 255
    regions = postprocess.extract_regions(score, valid, 0.5, 60, refl)
    assert len(regions) == 1 and regions[0].bbox[:2] == (20, 20)
    assert regions[0].score_max == pytest.approx(0.9)
    assert postprocess.extract_regions(score, np.zeros_like(valid), 0.5, 10) == []


def test_combine_and_specular_and_coords():
    d = np.array([[0.0, 1.0]], np.float32)
    s = np.array([[0.3, 0.0]], np.float32)
    assert postprocess.combine(d, s, 0.5, 2.0).tolist() == [[pytest.approx(0.6), 1.0]]
    t = np.array([[250, 100]], np.uint8)
    b = np.array([[100, 100]], np.uint8)
    assert postprocess.specular(t, b).tolist() == [[255, 0]]
    poly = np.array([[10, 10], [20, 10], [20, 20]], np.float32)
    assert postprocess.to_coords(poly, None, 0.5)[0] == [20.0, 20.0]
    H = np.array([[1, 0, 5], [0, 1, 0], [0, 0, 1]], np.float64)
    assert postprocess.to_coords(poly, H, 1.0)[0] == [15.0, 10.0]


def _region(mask):
    score = (mask > 0).astype(np.float32)
    return postprocess.extract_regions(score, np.full_like(mask, 255), 0.5, 1)[0]


def test_guess_type_rules():
    shape = (300, 300)
    base = np.full((300, 300, 3), 150, np.uint8)
    line = _blob(shape, [[20, 100], [260, 100], [260, 108], [20, 108]])
    assert typing.guess_type(_region(line), base, base)[0] == "scratch"
    square = _blob(shape, [[50, 50], [120, 50], [120, 110], [50, 110]])
    colored = base.copy()
    colored[square > 0] = (40, 170, 230)
    assert typing.guess_type(_region(square), colored, base)[0] == "sticker"
    disk = np.zeros(shape, np.uint8)
    cv2.circle(disk, (150, 150), 40, 255, -1)
    stained = base.copy()
    stained[disk > 0] = (120, 135, 150)
    assert typing.guess_type(_region(disk), stained, base)[0] == "stain"
    assert typing.guess_type(_region(disk), base, base)[0] == "other"
    assert typing.color_difference(base, base, np.zeros(shape, np.uint8)) == 0.0


def test_existing_damage_overlap():
    mask = _blob((100, 100), [[10, 10], [50, 10], [50, 50], [10, 50]])
    near = ExistingDamage(id=5, geometry=Geometry(type="box", coords=[[0, 0], [40, 40]]))
    far = ExistingDamage(id=6, geometry=Geometry(type="polygon", coords=[[80, 80], [90, 80], [90, 90]]))
    bad = ExistingDamage(id=7, geometry=Geometry(type="polygon", coords=[[1, 1]]))
    assert existing_damage.overlapping_damage(mask, [far, near, bad], 1.0) == 5
    assert existing_damage.overlapping_damage(mask, [far], 1.0) is None
    assert existing_damage.overlapping_damage(np.zeros_like(mask), [near], 1.0) is None


def test_qr_polygon_mask():
    q = [QrCode(value="A", corners=[[10, 10], [30, 10], [30, 30], [10, 30]]), QrCode(value="B", corners=[])]
    m = qr_polygon_mask((50, 50), q, 1.0)
    assert m[20, 20] == 255 and m[45, 45] == 0


def test_imageutil():
    with pytest.raises(ImageDecodeError):
        decode(b"not an image")
    img = np.zeros((10, 20, 3), np.uint8)
    assert decode(encode_png(img)).shape == (10, 20, 3)
    small, s = resize_long_side(img, 100)
    assert s == 1.0 and small is img
    small, s = resize_long_side(img, 10)
    assert small.shape[:2] == (5, 10) and s == 0.5
    gray = to_gray(img)
    assert to_gray(gray) is gray
