"""조명 정규화 (research R2 3단계, tasks T057).

대여 때와 반납 때 조명이 다르면 사진 전체 밝기·대비가 달라져 "바뀐 곳"처럼 보인다.
노트북 마스크 안의 밝기 분포를 기준 사진에 맞춘다(히스토그램 매칭).

CLAHE(국소 대비 평활화)도 시험했지만, 평평한 상판의 미세한 잡음까지 키워 SSIM 오탐이 늘어서 쓰지 않는다
(합성 회귀 테스트에서 밝기 0.7배일 때 오탐 2건 → 0건).
"""

from __future__ import annotations

import cv2
import numpy as np


def match_histogram(src: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """회색조 src의 마스크 안 분포를 ref의 마스크 안 분포에 맞춘다."""
    m = mask > 0
    if m.sum() < 16:
        return src
    s_hist = np.bincount(src[m], minlength=256).astype(np.float64)
    r_hist = np.bincount(ref[m], minlength=256).astype(np.float64)
    s_cdf = np.cumsum(s_hist) / s_hist.sum()
    r_cdf = np.cumsum(r_hist) / r_hist.sum()
    lut = np.searchsorted(r_cdf, s_cdf, side="left").clip(0, 255).astype(np.uint8)
    return cv2.LUT(src, lut)


def normalize_pair(
    target_gray: np.ndarray, baseline_gray: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return match_histogram(target_gray, baseline_gray, mask), baseline_gray


def match_color(target_bgr: np.ndarray, baseline_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """DINOv2 입력용: 채널마다 히스토그램을 맞춘 컬러 이미지."""
    return cv2.merge([match_histogram(target_bgr[..., c], baseline_bgr[..., c], mask) for c in range(3)])
