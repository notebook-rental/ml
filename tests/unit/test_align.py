"""정렬 단위 테스트 (tasks T050): 호모그래피 정합 오차, 실패하면 not_comparable."""

from __future__ import annotations

import cv2
import numpy as np

from common.messages import QrCode
from inference.pipeline import align
from inference.pipeline.imageutil import resize_long_side, to_gray
from tests.fixtures import synth


def _kps(img):
    small, s = resize_long_side(img, 1024)
    return align.extract(to_gray(small)), s


def test_alignment_matches_ground_truth(scene):
    b, Mb = synth.view(scene.image, seed=1, shift=10)
    t, Mt = synth.view(scene.image, seed=7, shift=30)
    kb, s = _kps(b)
    kt, _ = _kps(t)
    r = align.align(kt, kb)
    assert r.ok and r.reproj_error_px < 1.0 and r.inliers >= 25
    S = np.diag([s, s, 1.0])
    truth = S @ Mb @ np.linalg.inv(Mt) @ np.linalg.inv(S)
    lid = np.float32([[[260, 200]], [[1340, 200]], [[1340, 1000]], [[260, 1000]]])
    pts = cv2.perspectiveTransform(lid, Mt) * s
    err = np.linalg.norm(
        cv2.perspectiveTransform(pts, r.homography) - cv2.perspectiveTransform(pts, truth), axis=2
    )
    assert err.max() < 2.0


def test_too_few_matches():
    empty = align.Keypoints(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32))
    r = align.align(empty, empty)
    assert not r.ok and r.reason == "too_few_matches"


def test_unrelated_images_fail(baseline_img):
    rng = np.random.default_rng(1)
    noise = (rng.random((900, 1200)) * 255).astype(np.uint8)
    kb, _ = _kps(baseline_img)
    kt = align.extract(noise)
    assert not align.align(kt, kb).ok


def test_implausible_homography_rejected():
    assert not align._plausible(np.diag([10.0, 10.0, 1.0]))
    assert align._plausible(np.eye(3))


def test_strict_error_limit_rejects(scene):
    b, _ = synth.view(scene.image, seed=1, shift=10)
    t, _ = synth.view(scene.image, seed=2, shift=30)
    kb, _ = _kps(b)
    kt, _ = _kps(t)
    r = align.align(kt, kb, max_reproj_error_px=0.0)
    assert not r.ok and r.reason == "alignment_error_too_large"
    r = align.align(kt, kb, min_inliers=10_000)
    assert not r.ok and r.reason in {"too_few_matches", "too_few_inliers"}


def test_qr_pairs():
    corners = [[0, 0], [10, 0], [10, 10], [0, 10]]
    t = [QrCode(value="A", corners=corners), QrCode(value="B", corners=corners)]
    b = [QrCode(value="A", corners=[[1, 1], [11, 1], [11, 11], [1, 11]])]
    src, dst = align.qr_pairs(t, b, 0.5, 0.5)
    assert src.shape == (4, 2) and dst[0].tolist() == [0.5, 0.5]
    assert align.qr_pairs([], b, 1, 1) is None


def test_extract_with_mask_and_blank():
    blank = np.zeros((100, 100), np.uint8)
    k = align.extract(blank, np.full((100, 100), 255, np.uint8))
    assert k.pts.shape == (0, 2)
    s, d = align.match(k, k)
    assert len(s) == 0


def test_warp_identity():
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    assert np.array_equal(align.warp(img, np.eye(3), (10, 10), nearest=True), img)
