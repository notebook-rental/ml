"""조각 맞추기 기억 장소(inference/pipeline/bank.py, docs specs/004) 단위 테스트."""

from __future__ import annotations

import numpy as np

from inference.pipeline import bank as bank_mod
from inference.pipeline.bank import Bank, bank_features, letterbox, match, tiles


class FakeBackbone:
    """조각마다 (평균 B, G, R, 세로 위치 무늬) 특징을 내는 대역. 방향(회전)을 구분할 수 있게 위치 무늬를 섞는다."""

    size = (448, 336)
    name = "fake"

    def features(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        gh, gw = h // 14, w // 14
        f = img[: gh * 14, : gw * 14].reshape(gh, 14, gw, 14, 3).mean(axis=(1, 3)).astype(np.float32)
        f = np.concatenate([f, np.full((gh, gw, 1), 40, np.float32)], axis=2)
        return f / np.linalg.norm(f, axis=2, keepdims=True)


def _scene() -> np.ndarray:
    img = np.zeros((300, 400, 3), np.uint8)
    img[:, :200] = (200, 40, 40)  # 왼쪽 파랑
    img[:150, 200:] = (40, 200, 40)  # 오른쪽 위 초록
    img[150:, 200:] = (40, 40, 200)  # 오른쪽 아래 빨강
    return img


def test_letterbox_keeps_aspect():
    img = np.full((400, 100, 3), 255, np.uint8)  # 세로로 긴 사진
    out, box = letterbox(img, (448, 336))
    assert out.shape == (336, 448, 3)
    assert box.h == 336 and abs(box.w - 84) <= 1  # 1:4 비율 유지
    assert out[:, : box.x].max() == 0  # 여백은 검정


def test_tiles_cover_image():
    img = np.zeros((300, 400), np.uint8)
    cover = np.zeros_like(img)
    ts = tiles(img)
    assert len(ts) == 4
    for x, y, t in ts:
        cover[y : y + t.shape[0], x : x + t.shape[1]] += 1
    assert cover.min() >= 1


def test_same_scene_rotated_matches_with_small_distance():
    fb = FakeBackbone()
    base = _scene()
    bank = Bank.build([("p1", bank_features(fb, base))])
    m = match(fb, bank, np.ascontiguousarray(np.rot90(base)))
    assert m.mean_distance < 0.02
    assert m.votes == {"p1": m.votes["p1"]} and m.votes["p1"] > 0


def test_new_patch_is_far_and_votes_pick_right_photo():
    fb = FakeBackbone()
    base = _scene()
    other = np.full((300, 400, 3), (128, 128, 128), np.uint8)
    bank = Bank.build([("gray", bank_features(fb, other)), ("scene", bank_features(fb, base))])
    target = base.copy()
    target[100:160, 60:140] = (0, 230, 230)  # 노란 스티커(기억에 없는 색)
    m = match(fb, bank, target)
    assert max(m.votes, key=m.votes.get) == "scene"
    assert m.distance[130, 100] > 5 * m.distance[250, 300]  # 스티커 자리가 훨씬 멀다
    assert m.distance.shape == target.shape[:2]


def test_rotations_constant():
    assert bank_mod.ROTATIONS == (0, 1, 2, 3)
