"""조각 맞추기 비교: 기준 사진 전부를 기억 장소로 (docs specs/004-free-angle-compare, plan D1~D4).

지금까지는 반납 사진을 대여 사진에 정확히 겹친 뒤 같은 자리끼리 비교했다. 관리자가 대충 찍으면(세로로 들고,
비스듬히, 가까이) 겹치기가 안 되어 "자동 비교 불가"가 됐다.

여기서는 겹치지 않는다. 반납 사진의 노트북을 DINOv2 조각(14×14 픽셀)으로 나누고, 조각마다 그 노트북 **기준 사진
전부**의 조각(기억 장소) 중 가장 닮은 것과의 거리를 이상 점수로 쓴다(PatchCore와 같은 생각). 어느 기준 사진
어디에도 닮은 조각이 없으면 새로 생긴 것이다. 그래서 각도·거리·방향·컷 이름과 상관없다.

- D2 방향: 반납 크롭을 0·90·180·270°로 돌려 보고 가장 잘 맞는 방향 하나를 쓴다(조각 특징은 회전에 약하다).
- D3 크기: 크롭 통째 1장 + 2×2 겹친 조각 4장. 가까이 찍은 사진도 맞는 크기가 있다.
- D4 비율: 늘이지 않고 검은 여백을 붙여 모델 입력 비율(4:3)에 맞춘다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from inference.pipeline.patch_features import Backbone

TILE_GRID = 2
TILE_OVERLAP = 0.25
ROTATIONS = (0, 1, 2, 3)  # np.rot90 횟수(반시계 90°씩)


@dataclass
class Box:
    """여백을 붙인 그림 안에서 원래 그림이 있는 자리(모델 입력 좌표)."""

    x: int
    y: int
    w: int
    h: int


def letterbox(img: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, Box]:
    """img를 늘이지 않고 size=(W,H)에 맞춘다. 남는 곳은 검은 여백. (결과, 원래 그림 자리)."""
    W, H = size
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((H, W) + img.shape[2:], img.dtype)
    x, y = (W - nw) // 2, (H - nh) // 2
    out[y : y + nh, x : x + nw] = resized
    return out, Box(x, y, nw, nh)


def tiles(
    img: np.ndarray, n: int = TILE_GRID, overlap: float = TILE_OVERLAP
) -> list[tuple[int, int, np.ndarray]]:
    """n×n 겹친 조각. [(x, y, 조각)], 조각이 그림 전체를 덮는다."""
    h, w = img.shape[:2]
    tw = int(np.ceil(w / (n - (n - 1) * overlap)))
    th = int(np.ceil(h / (n - (n - 1) * overlap)))
    out = []
    for i in range(n):
        for j in range(n):
            x = min(w - tw, int(round(j * tw * (1 - overlap)))) if n > 1 else 0
            y = min(h - th, int(round(i * th * (1 - overlap)))) if n > 1 else 0
            out.append((max(0, x), max(0, y), img[max(0, y) : max(0, y) + th, max(0, x) : max(0, x) + tw]))
    return out


def _grid_features(backbone: Backbone, img: np.ndarray) -> tuple[np.ndarray, Box]:
    """여백을 붙여 넣은 특징 [gh,gw,C]와 원래 그림 자리."""
    lb, box = letterbox(img, backbone.size)
    return backbone.features(lb), box


def _inside(feats: np.ndarray, box: Box, size: tuple[int, int]) -> np.ndarray:
    """여백이 아닌 조각만(원래 그림과 반 이상 겹치는 조각). [K,C]."""
    gh, gw, c = feats.shape
    W, H = size
    px, py = W / gw, H / gh
    xs = (np.arange(gw) + 0.5) * px
    ys = (np.arange(gh) + 0.5) * py
    mx = (xs >= box.x) & (xs < box.x + box.w)
    my = (ys >= box.y) & (ys < box.y + box.h)
    return feats[np.ix_(my, mx)].reshape(-1, c)


def bank_features(backbone: Backbone, crop: np.ndarray) -> np.ndarray:
    """기준 사진 노트북 크롭 하나의 기억 장소 조각(통째 + 2×2). [K,C] float16, 길이 1."""
    parts = []
    for view in [crop] + [t for _, _, t in tiles(crop)]:
        f, box = _grid_features(backbone, view)
        parts.append(_inside(f, box, backbone.size))
    return np.vstack(parts).astype(np.float16)


@dataclass
class Bank:
    feats: np.ndarray  # [N,C] float32
    owner: np.ndarray  # [N] int, 몇 번째 기준 사진의 조각인지
    photo_ids: list[str]

    @classmethod
    def build(cls, entries: list[tuple[str, np.ndarray]]) -> Bank:
        feats = np.vstack([e for _, e in entries]).astype(np.float32)
        owner = np.concatenate([np.full(len(e), i, np.int32) for i, (_, e) in enumerate(entries)])
        return cls(feats, owner, [pid for pid, _ in entries])


@dataclass
class BankMatch:
    distance: np.ndarray  # [ch,cw] float32, 크롭 크기, 0이면 똑같음
    rotation: int  # 반납 크롭을 rot90 몇 번 돌렸을 때 가장 잘 맞았는지
    mean_distance: float  # 통째 조각 기준 평균 거리
    votes: dict[str, int]  # 기준 사진별로 가장 닮은 조각을 준 횟수


def _nearest(feats: np.ndarray, bank: Bank) -> tuple[np.ndarray, np.ndarray]:
    """조각마다 (가장 가까운 거리, 그 조각의 주인). feats [gh,gw,C] → [gh,gw] 두 개."""
    gh, gw, c = feats.shape
    sim = feats.reshape(-1, c) @ bank.feats.T
    idx = sim.argmax(1)
    best = sim[np.arange(len(idx)), idx]
    return (1.0 - best).reshape(gh, gw).astype(np.float32), bank.owner[idx].reshape(gh, gw)


def _to_image(grid: np.ndarray, box: Box, size: tuple[int, int], out_hw: tuple[int, int]) -> np.ndarray:
    """조각 격자 값 → 모델 입력 크기로 키움 → 여백 잘라 냄 → 원래 그림 크기."""
    W, H = size
    full = cv2.resize(grid, (W, H), interpolation=cv2.INTER_CUBIC)
    inner = full[box.y : box.y + box.h, box.x : box.x + box.w]
    return cv2.resize(inner, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)


def match(backbone: Backbone, bank: Bank, crop: np.ndarray, valid: np.ndarray | None = None) -> BankMatch:
    """반납 노트북 크롭을 기억 장소와 맞춘다. valid(크롭 크기 마스크)가 있으면 그 안만 평균·투표에 쓴다."""
    best = None
    for k in ROTATIONS:
        r = np.ascontiguousarray(np.rot90(crop, k))
        f, box = _grid_features(backbone, r)
        d, own = _nearest(f, bank)
        dimg = _to_image(d, box, backbone.size, r.shape[:2])
        dimg = np.rot90(dimg, -k)
        m = valid > 0 if valid is not None else np.ones(dimg.shape, bool)
        mean = float(dimg[m].mean()) if m.any() else float(dimg.mean())
        if best is None or mean < best[0]:
            best = (mean, k, r, dimg, _inside(own[..., None], box, backbone.size).ravel())
    mean, k, r, whole, own = best

    # 2×2 조각(가까이 찍은 것 대비). 어느 크기에서든 닮았으면 정상 → 두 맵 중 작은 값
    fine = np.full(r.shape[:2], np.inf, np.float32)
    for x, y, t in tiles(r):
        f, box = _grid_features(backbone, t)
        d, _ = _nearest(f, bank)
        dimg = _to_image(d, box, backbone.size, t.shape[:2])
        th, tw = dimg.shape
        fine[y : y + th, x : x + tw] = np.minimum(fine[y : y + th, x : x + tw], dimg)
    fine = np.rot90(fine, -k)
    dist = np.where(np.isfinite(fine), np.minimum(whole, fine), whole).astype(np.float32)

    counts = np.bincount(own, minlength=len(bank.photo_ids))
    votes = {pid: int(counts[i]) for i, pid in enumerate(bank.photo_ids)}
    return BankMatch(np.ascontiguousarray(dist), k, mean, votes)
