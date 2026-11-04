"""노트북 영역 분리 단위 테스트(GrabCut 대체 방법, 경량 SAM 래퍼)."""

from __future__ import annotations

import numpy as np

from inference.pipeline.segment import GrabCutSegmenter, SamSegmenter, fill_holes, largest_component


def test_grabcut_finds_lid(baseline_img):
    from inference.pipeline.imageutil import resize_long_side

    small, s = resize_long_side(baseline_img, 1024)
    m = GrabCutSegmenter().segment(small, (int(250 * s), int(190 * s), int(1350 * s), int(1010 * s)))
    assert m.shape == small.shape[:2]
    cy, cx = small.shape[0] // 2, small.shape[1] // 2
    assert m[cy, cx] == 255 and m[5, 5] == 0


def test_grabcut_tiny_box_returns_all():
    img = np.zeros((100, 100, 3), np.uint8)
    assert GrabCutSegmenter().segment(img, (10, 10, 12, 12)).min() == 255


def test_grabcut_default_box_flat_image():
    img = np.full((120, 160, 3), 128, np.uint8)
    m = GrabCutSegmenter().segment(img, None)
    assert m.shape == (120, 160) and m.max() == 255


def test_mask_helpers():
    m = np.zeros((50, 50), np.uint8)
    m[5:10, 5:10] = 255
    m[20:45, 20:45] = 255
    m[30:35, 30:35] = 0
    big = largest_component(m)
    assert big[7, 7] == 0 and big[25, 25] == 255
    assert fill_holes(big)[32, 32] == 255
    assert largest_component(np.zeros((5, 5), np.uint8)).sum() == 0


class _FakeEncoder:
    def run(self, _o, feeds):
        x = feeds["image"]
        assert x.shape == (1, 3, 1024, 1024)
        return [np.zeros((1, 256, 64, 64), np.float32)]


class _FakeDecoder:
    def run(self, _o, feeds):
        h, w = (int(v) for v in feeds["orig_im_size"])
        coords = feeds["point_coords"][0]
        scale = 1024 / max(h, w)
        (x1, y1), (x2, y2) = coords[0] / scale, coords[1] / scale
        masks = np.full((1, 3, h, w), -1.0, np.float32)
        masks[0, 1, int(y1) : int(y2), int(x1) : int(x2)] = 1.0
        return [masks, np.array([[0.1, 0.9, 0.2]], np.float32)]


def test_sam_segmenter_uses_box_and_best_mask():
    img = np.zeros((300, 400, 3), np.uint8)
    m = SamSegmenter(_FakeEncoder(), _FakeDecoder()).segment(img, (100, 50, 300, 250))
    assert m.shape == (300, 400) and m[150, 200] == 255 and m[10, 10] == 0
    m2 = SamSegmenter(_FakeEncoder(), _FakeDecoder()).segment(np.zeros((2000, 1500, 3), np.uint8), None)
    assert m2.shape == (2000, 1500) and m2[1000, 750] == 255
