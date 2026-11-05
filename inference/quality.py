"""촬영 품질 검사 (`POST /quality`, FR-003, research R5).

폰이 1차로 검사한 뒤 서버가 같은 지표로 최종 확인한다. 계산은 모두 긴 변 1024px로 줄인 사진에서 한다
(작업본 2560px를 그대로 쓰면 느리고, 흐림·노출 판정에는 이 정도 해상도면 충분하다).

- 흐림(blur): 라플라시안 분산. 경계가 또렷할수록 값이 크다. 작으면 흐린 사진.
- 노출(exposure): 평균 밝기(0~1). 너무 어둡거나 밝으면 거절.
- 구도(coverage): 노트북 영역 상자가 사진에서 차지하는 비율. 작으면 너무 멀리서 찍었거나 벗어난 사진.
- QR: 상판(top) 컷은 QR이 하나 보여야 한다. 둘 이상이면 거절(FR-009).
"""

from __future__ import annotations

import cv2
import numpy as np

from common.messages import QrCode, QualityResult, QualityScores
from inference.config import Settings
from inference.pipeline.imageutil import resize_long_side, to_gray
from inference.pipeline.qr import detect_barcodes, detect_qr

ANALYSIS_SIDE = 1024


def laptop_box(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """노트북 영역의 대략적인 상자(x1,y1,x2,y2). 경계선이 모인 가장 큰 덩어리를 찾는다.

    배경은 보통 단색 매트·책상이라 경계가 적고, 노트북은 테두리·키·로고 때문에 경계가 많다.
    """
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    k = max(3, (max(gray.shape) // 64) | 1)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w * h < 0.01 * gray.shape[0] * gray.shape[1]:
        return None
    return x, y, x + w, y + h


LABEL_SHOTS = ("top", "bottom")


def check_quality(img: np.ndarray, shot_type: str, settings: Settings) -> QualityResult:
    small, scale = resize_long_side(img, ANALYSIS_SIDE)
    gray = to_gray(small)

    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    exposure = float(gray.mean() / 255.0)
    box = laptop_box(gray)
    coverage = 0.0
    if box is not None:
        coverage = (box[2] - box[0]) * (box[3] - box[1]) / float(gray.shape[0] * gray.shape[1])

    qrs: list[QrCode] = detect_qr(img)
    # QR이 없으면 바코드 라벨을 찾는다(라벨은 QR 또는 바코드, spec 002).
    # 라벨은 상판이나 바닥면에 붙어 있다. 바코드는 이 두 컷에서만 찾아 다른 컷의 품질 검사 시간을 늘리지 않는다.
    labels: list[QrCode] = qrs if qrs or shot_type not in LABEL_SHOTS else detect_barcodes(img)

    reasons: list[str] = []
    if blur < settings.quality_blur_min:
        reasons.append("blurry")
    if exposure < settings.quality_dark_max:
        reasons.append("too_dark")
    elif exposure > settings.quality_bright_min:
        reasons.append("too_bright")
    if coverage < settings.quality_coverage_min:
        reasons.append("out_of_frame")
    if shot_type == "top" and len({q.value for q in qrs}) > 1:
        reasons.append("multiple_qr")
    # 라벨이 안 보여도 거절하지 않는다. 라벨이 상판이 아니라 바닥면에 있는 노트북도 있다(2026-11-05 marco).
    # 노트북 확인은 촬영 시작 때 라벨 스캔으로 한다. 찾은 라벨은 qr로 돌려줘 api가 다른 노트북인지 비교한다.

    box_orig = [round(v / scale) for v in box] if box is not None else None
    return QualityResult(
        passed=not reasons,
        reasons=reasons,
        scores=QualityScores(blur=round(blur, 1), exposure=round(exposure, 3), coverage=round(coverage, 3)),
        laptopBox=box_orig,
        qr=labels,
    )
