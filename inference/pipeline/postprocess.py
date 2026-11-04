"""점수 결합 → 임계값 → 덩어리(연결 요소) → 다각형 (research R2 6단계, tasks T060).

두 점수 맵(DINOv2 패치 거리, SSIM 차분)을 합쳐 "바뀐 정도" 맵을 만들고, 임계값을 넘는 픽셀끼리 이어진 덩어리를
손상 후보 하나로 본다. 작은 잡음 덩어리는 버리고, 화면 반사(빛 번짐)로 보이는 덩어리도 버린다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Region:
    polygon: np.ndarray  # [K,2] float32, compare 해상도(기준 사진 좌표계)
    mask: np.ndarray  # 덩어리 마스크(uint8, compare 해상도)
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area: int
    score_max: float
    score_mean: float


def combine(dino: np.ndarray, ssim: np.ndarray, dino_scale: float, ssim_gain: float) -> np.ndarray:
    """dino(코사인 거리, 0~2)를 dino_scale로 나눠 0~1로, SSIM 차분은 ssim_gain을 곱해 둘 중 큰 값."""
    d = (dino / dino_scale).clip(0, 1)
    s = (ssim * ssim_gain).clip(0, 1)
    return np.maximum(d, s).astype(np.float32)


def specular(target_gray: np.ndarray, baseline_gray: np.ndarray, level: int = 235) -> np.ndarray:
    """반납 사진에서만 거의 하얗게(235 이상) 날아간 곳 = 조명 반사로 본다. 덩어리의 절반 이상이 여기면 버린다."""
    return ((target_gray >= level) & (baseline_gray < level)).astype(np.uint8) * 255


def extract_regions(
    score: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    min_area: int,
    reflection: np.ndarray | None = None,
) -> list[Region]:
    binary = ((score >= threshold) & (valid > 0)).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    regions: list[Region] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        m = (labels == i).astype(np.uint8) * 255
        if reflection is not None and (reflection[m > 0] > 0).mean() > 0.5:
            continue
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)
        eps = 0.01 * cv2.arcLength(c, True)
        poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
        if len(poly) < 3:
            x, y, w, h = cv2.boundingRect(c)
            poly = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
        vals = score[m > 0]
        x, y, w, h = (int(v) for v in stats[i, :4])
        regions.append(Region(poly, m, (x, y, w, h), area, float(vals.max()), float(vals.mean())))
    regions.sort(key=lambda r: r.score_max, reverse=True)
    return regions


def to_coords(poly: np.ndarray, H: np.ndarray | None, scale: float) -> list[list[float]]:
    """compare 해상도 좌표 → (H로 옮긴 뒤) → 원본 픽셀 좌표(÷ scale)."""
    pts = poly.reshape(-1, 1, 2).astype(np.float32)
    if H is not None:
        pts = cv2.perspectiveTransform(pts, H)
    pts = pts.reshape(-1, 2) / scale
    return [[round(float(x), 1), round(float(y), 1)] for x, y in pts]
