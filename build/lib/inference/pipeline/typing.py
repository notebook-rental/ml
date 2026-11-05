"""손상 유형 추정 (research R2 7단계, tasks T063).

1단계는 학습 데이터가 없으므로 모양 규칙(휴리스틱)으로 추정한다. 직원이 판정하면서 바로잡고, 그 판정이 라벨로 쌓여
2단계 학습 모델이 대신하게 된다. 확신이 없으면 "other(기타)"로 둔다.

- scratch(스크래치): 가늘고 길다(긴 변 / 짧은 변 ≥ 4, 짧은 변 ≤ 48px @1024. DINOv2 패치 점수가 선을 넓게 퍼뜨리므로 여유를 둔다).
- sticker(스티커): 꽉 찬 사각형에 가깝고(최소 외접 사각형 채움 비율 ≥ 0.85, 원·타원은 약 0.79) 색이 확실히 다르다.
- stain(얼룩): 둥근 덩어리이고 색은 조금 다르다.
- 그 밖(찍힘 dent, 깨짐 crack, 부품 없음 missing_part)은 모양만으로 구분이 어려워 other로 두고 직원이 고른다.
"""

from __future__ import annotations

import cv2
import numpy as np

from inference.pipeline.postprocess import Region


def shape_features(region: Region) -> dict[str, float]:
    contours, _ = cv2.findContours(region.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    long_side, short_side = max(rw, rh), max(1.0, min(rw, rh))
    fill = region.area / max(1.0, rw * rh)
    return {"elongation": long_side / short_side, "fill": float(fill), "short": float(short_side)}


def color_difference(target_bgr: np.ndarray, baseline_bgr: np.ndarray, mask: np.ndarray) -> float:
    """Lab 색공간에서 덩어리 안 평균 색 차이(ΔE 근사)."""
    m = mask > 0
    if not m.any():
        return 0.0
    t = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[m].mean(axis=0)
    b = cv2.cvtColor(baseline_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[m].mean(axis=0)
    return float(np.linalg.norm(t - b))


def guess_type(region: Region, target_bgr: np.ndarray, baseline_bgr: np.ndarray) -> tuple[str, float]:
    """(유형, 유형 신뢰도 0~1)."""
    f = shape_features(region)
    de = color_difference(target_bgr, baseline_bgr, region.mask)
    if f["elongation"] >= 4.0 and f["short"] <= 48:
        return "scratch", min(1.0, 0.5 + f["elongation"] / 20)
    if f["fill"] >= 0.85 and de >= 20:
        return "sticker", min(1.0, 0.5 + de / 100)
    if f["elongation"] < 2.5 and de >= 8:
        return "stain", 0.5
    return "other", 0.3
