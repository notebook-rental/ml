"""SSIM 차분 맵 (research R2 5단계, tasks T059).

SSIM(구조적 유사도)은 두 사진의 작은 창(가우시안 σ=1.5)마다 밝기·대비·구조가 얼마나 같은지 0~1로 낸다.
DINOv2 패치(14픽셀)보다 작은 변화, 예를 들어 가는 스크래치를 잡는 보조 신호로 쓴다.
차분 = (1 - SSIM) / 2 → 0(같음) ~ 1(완전히 다름).
OpenCV 가우시안 블러로 직접 계산해 scikit-image보다 빠르다.
"""

from __future__ import annotations

import cv2
import numpy as np

C1 = (0.01 * 255) ** 2
C2 = (0.03 * 255) ** 2


def ssim_map(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    def blur(x):
        return cv2.GaussianBlur(x, (0, 0), sigma)

    mu_a, mu_b = blur(a), blur(b)
    aa, bb, ab = blur(a * a) - mu_a**2, blur(b * b) - mu_b**2, blur(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (aa + bb + C2)
    return num / den


def _shift(x: np.ndarray, dy: int, dx: int) -> np.ndarray:
    if dy == 0 and dx == 0:
        return x
    h, w = x.shape
    p = np.pad(x, 1, mode="edge")
    return p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]


def ssim_diff(
    a: np.ndarray, b: np.ndarray, tolerance: int = 1, sigma: float = 1.5, pre_blur: float = 1.0
) -> np.ndarray:
    """정렬이 1픽셀쯤 어긋나도 괜찮게, b를 상하좌우 ±tolerance 픽셀 옮겨 가며 잰 차분 중 가장 작은 값을 쓴다.

    가는 선·경계는 1픽셀만 어긋나도 SSIM이 크게 떨어진다(실측: 합성 사진의 매트 격자선이 모두 후보로 잡힘).
    진짜 손상은 옮겨도 사라지지 않으므로 최솟값을 쓰면 정렬 오차에 의한 오탐만 줄어든다.

    속도: 블러는 옮기기와 순서를 바꿔도 같으므로(경계 제외) a·b 각각의 평균·분산은 한 번만 계산하고,
    옮길 때마다 a×b 블러 하나만 새로 계산한다(9번 전부 다시 계산하는 것보다 약 4배 빠름).
    """
    # 1픽셀 미만(서브픽셀) 정렬 오차로 생기는 경계선 차이를 줄이려 먼저 살짝 흐린다(σ=1, 가는 스크래치는 남는다)
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), pre_blur) if pre_blur else a.astype(np.float32)
    b = cv2.GaussianBlur(b.astype(np.float32), (0, 0), pre_blur) if pre_blur else b.astype(np.float32)

    def blur(x):
        return cv2.GaussianBlur(x, (0, 0), sigma)

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a * mu_a
    var_b = blur(b * b) - mu_b * mu_b
    d = None
    for dy in range(-tolerance, tolerance + 1):
        for dx in range(-tolerance, tolerance + 1):
            mb, vb = _shift(mu_b, dy, dx), _shift(var_b, dy, dx)
            cov = blur(a * _shift(b, dy, dx)) - mu_a * mb
            num = (2 * mu_a * mb + C1) * (2 * cov + C2)
            den = (mu_a * mu_a + mb * mb + C1) * (var_a + vb + C2)
            m = (1.0 - num / den) * 0.5
            d = m if d is None else np.minimum(d, m)
    # 한 점짜리 잡음을 줄이려 살짝 펴 준다
    return cv2.GaussianBlur(d, (0, 0), 1.0).clip(0, 1)
