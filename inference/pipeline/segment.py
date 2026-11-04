"""노트북 영역 분리 (research R2 1단계, tasks T055).

배경(책상, 매트, 손)을 비교에서 빼려고 노트북 마스크를 만든다. 두 가지 방법이 있다.

1. SamSegmenter: 경량 SAM(MobileSAM/EfficientSAM) ONNX. 노트북 상자를 프롬프트로 주면 윤곽을 정확히 딴다.
   인코더 1회 ≈ CPU 0.3~1초라 **대여 사진 사전 계산(rental time)** 에서만 돌린다. 반납 사진은 대여 마스크를
   정렬해서 재사용하므로(R12) 반납 때 비교 시간에 들어가지 않는다.
2. GrabCutSegmenter: 모델 없이 OpenCV GrabCut. SAM 모델 파일이 없을 때의 대체 방법.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from inference.pipeline.imageutil import resize_long_side, to_gray


class Segmenter(Protocol):
    name: str

    def segment(self, img: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
        """img와 같은 크기의 uint8 마스크(255=노트북)."""
        ...


def default_box(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    return int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == idx, 255, 0).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """키보드 무늬·로고 때문에 생긴 구멍을 메운다(노트북 윤곽 안은 모두 노트북)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)
    return out


class GrabCutSegmenter:
    name = "grabcut@1"

    def __init__(self, work_side: int = 384, iterations: int = 3):
        self.work_side = work_side
        self.iterations = iterations

    def segment(self, img: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
        h, w = img.shape[:2]
        small, s = resize_long_side(img, self.work_side)
        if box is None:
            box = default_box(img.shape)
        x1, y1, x2, y2 = (round(v * s) for v in box)
        sh, sw = small.shape[:2]
        x1, y1 = max(1, x1), max(1, y1)
        x2, y2 = min(sw - 2, x2), min(sh - 2, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return np.full((h, w), 255, np.uint8)
        mask = np.zeros((sh, sw), np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(
                small, mask, (x1, y1, x2 - x1, y2 - y1), bgd, fgd, self.iterations, cv2.GC_INIT_WITH_RECT
            )
            fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except cv2.error:  # pragma: no cover - 단색 이미지 등에서 GrabCut이 실패하는 경우
            fg = np.zeros((sh, sw), np.uint8)
            fg[y1:y2, x1:x2] = 255
        fg = fill_holes(largest_component(fg))
        if fg.sum() < 0.05 * 255 * sh * sw:  # 너무 작으면 상자 전체를 쓴다(비교를 막지 않는 쪽으로)
            fg = np.zeros((sh, sw), np.uint8)
            fg[y1:y2, x1:x2] = 255
        return cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)


class SamSegmenter:
    """경량 SAM ONNX(인코더 + 프롬프트 디코더).

    인코더 입력 `image`: float32 [1,3,1024,1024], ImageNet 평균·표준편차로 정규화, 긴 변 1024로 줄이고
    오른쪽·아래를 0으로 채움. 출력 `image_embeddings` [1,256,64,64].
    디코더는 segment-anything 공식 ONNX 내보내기와 같은 입력 이름을 쓴다.
    """

    name = "mobilesam@1"
    SIDE = 1024
    MEAN = np.array([123.675, 116.28, 103.53], np.float32)
    STD = np.array([58.395, 57.12, 57.375], np.float32)

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def segment(self, img: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
        h, w = img.shape[:2]
        if box is None:
            box = default_box(img.shape)
        small, s = resize_long_side(img, self.SIDE)
        if s == 1.0 and max(h, w) < self.SIDE:
            s = self.SIDE / max(h, w)
            small = cv2.resize(img, (round(w * s), round(h * s)))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = np.zeros((self.SIDE, self.SIDE, 3), np.float32)
        x[: rgb.shape[0], : rgb.shape[1]] = (rgb - self.MEAN) / self.STD
        x = x.transpose(2, 0, 1)[None]
        emb = self.encoder.run(None, {"image": x})[0]

        x1, y1, x2, y2 = box
        coords = np.array([[[x1 * s, y1 * s], [x2 * s, y2 * s], [0.0, 0.0]]], np.float32)
        labels = np.array([[2, 3, -1]], np.float32)  # 2·3 = 상자의 왼쪽 위·오른쪽 아래, -1 = 빈 점
        masks, scores = self.decoder.run(
            None,
            {
                "image_embeddings": emb,
                "point_coords": coords,
                "point_labels": labels,
                "mask_input": np.zeros((1, 1, 256, 256), np.float32),
                "has_mask_input": np.zeros(1, np.float32),
                "orig_im_size": np.array([h, w], np.float32),
            },
        )[:2]
        best = int(np.argmax(scores[0]))
        m = (masks[0, best] > 0).astype(np.uint8) * 255
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return fill_holes(largest_component(m))


def laptop_box_from_gray(img: np.ndarray) -> tuple[int, int, int, int] | None:
    from inference.quality import laptop_box

    small, s = resize_long_side(img, 1024)
    b = laptop_box(to_gray(small))
    return None if b is None else tuple(round(v / s) for v in b)  # type: ignore[return-value]
