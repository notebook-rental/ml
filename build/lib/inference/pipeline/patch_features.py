"""DINOv2 패치 특징과 기준 사진 기반 이상 점수 (research R2 4단계, tasks T058).

DINOv2 ViT-S/14는 사진을 14×14 픽셀 조각(패치)으로 나누고 조각마다 384개 숫자(특징 벡터)를 낸다.
비슷하게 생긴 조각은 벡터도 비슷하다. 조명이 조금 달라도 벡터는 거의 그대로라 픽셀 차이보다 오탐이 적다.

PatchCore 변형: 대여(기준) 사진의 패치 특징을 "정상" 기억 장소(메모리 뱅크)로 두고, 반납 사진의 각 패치가
**같은 위치 주변(반경 r 패치)** 의 기준 패치 중 가장 가까운 것과 얼마나 다른지(코사인 거리)를 이상 점수로 쓴다.
정렬이 몇 픽셀 어긋나도 주변을 같이 보므로 괜찮다.

모델 파일 형식(tools/export_dinov2.py로 만든다):
- 입력 `pixel_values`: float32 [1,3,H,W], RGB, ImageNet 평균·표준편차 정규화, H·W는 14의 배수
- 출력 `patch_tokens`: float32 [1, (H/14)·(W/14), C]  (CLS 토큰 제외)
"""

from __future__ import annotations

import cv2
import numpy as np

PATCH = 14
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


DEFAULT_SIZE = (
    448,
    336,
)  # (W, H). 32×24 패치. 대여·반납 크롭을 같은 크기로 넣으므로 비율이 조금 달라도 괜찮다


class Backbone:
    """입력 크기가 고정된 모델이면 그 크기를, 아니면 DEFAULT_SIZE를 쓴다."""

    def __init__(self, session, name: str, size: tuple[int, int] | None = None):
        self.session = session
        self.name = name
        inp = session.get_inputs()[0]
        self.input_name = inp.name
        shape = list(inp.shape)
        if size is None and len(shape) == 4 and all(isinstance(v, int) for v in shape[2:]):
            size = (shape[3], shape[2])
        self.size = size or DEFAULT_SIZE

    def features(self, bgr: np.ndarray) -> np.ndarray:
        """bgr 크롭 → [h, w, C] float32, 패치마다 길이 1로 정규화한 특징."""
        w, h = self.size
        img = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        x = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) - MEAN) / STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        tokens = self.session.run(None, {self.input_name: x})[0]
        gh, gw = h // PATCH, w // PATCH
        feats = tokens.reshape(gh, gw, -1).astype(np.float32)
        return l2_normalize(feats)

    def warmup(self) -> None:
        self.features(np.zeros((self.size[1], self.size[0], 3), np.uint8))


def l2_normalize(f: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(f, axis=-1, keepdims=True)
    return f / np.maximum(n, 1e-6)


def neighbor_distance(target: np.ndarray, baseline: np.ndarray, radius: int) -> np.ndarray:
    """target[i,j]와 baseline[i+di, j+dj] (|di|,|dj| ≤ radius) 사이 코사인 거리의 최솟값. [h,w] float32.

    특징은 이미 길이 1이라 코사인 유사도 = 내적. 거리 = 1 - 유사도.
    """
    h, w, _ = target.shape
    best = np.full((h, w), -2.0, np.float32)
    pad = np.pad(baseline, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = pad[radius + dy : radius + dy + h, radius + dx : radius + dx + w]
            sim = np.einsum("ijc,ijc->ij", target, shifted)
            np.maximum(best, sim, out=best)
    return (1.0 - best).astype(np.float32)
