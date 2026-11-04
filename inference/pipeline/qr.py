"""QR 검출·해독 (OpenCV QRCodeDetector).

식별용 QR 해독은 폰과 Spring(ZXing)이 맡는다. 여기서는
- 품질 검사에서 상판 컷의 QR 개수 확인,
- 반납 비교에서 QR 영역을 비교 대상에서 빼고(스티커가 바뀌는 건 손상이 아님),
  QR이 사라졌거나 값이 바뀌었는지 확인(qr_tampered)
에 쓴다.
"""

from __future__ import annotations

import cv2
import numpy as np

from common.messages import QrCode
from inference.pipeline.imageutil import resize_long_side, to_gray

DETECT_SIDE = 1600


def _decode_multi(det, gray: np.ndarray, scale: float) -> list[QrCode]:
    try:
        ok, values, points, _ = det.detectAndDecodeMulti(gray)
    except cv2.error:  # pragma: no cover - OpenCV가 특이한 입력에서 예외를 내는 경우
        return []
    if not ok or points is None:
        return []
    out = []
    for value, pts in zip(values, points, strict=False):
        if value:
            corners = [[round(float(x) / scale, 1), round(float(y) / scale, 1)] for x, y in pts]
            out.append(QrCode(value=value, corners=corners))
    return out


def detect_qr(img: np.ndarray) -> list[QrCode]:
    """사진 속 QR들(값이 읽힌 것만). 꼭짓점은 입력 사진 픽셀 좌표.

    OpenCV 검출기는 각도·크기에 따라 가끔 놓친다. 그래서 (1) 기본 검출기 → (2) ArUco 기반 검출기 →
    (3) 더 줄인 사진 순서로, 값이 하나라도 읽힐 때까지 시도한다. 대부분 (1)에서 끝난다.
    """
    small, scale = resize_long_side(img, DETECT_SIDE)
    gray = to_gray(small)
    attempts = [
        (cv2.QRCodeDetector(), gray, scale),
        (cv2.QRCodeDetectorAruco(), gray, scale),
    ]
    half, s2 = resize_long_side(img, DETECT_SIDE // 2)
    attempts.append((cv2.QRCodeDetector(), to_gray(half), s2))
    for det, g, sc in attempts:
        found = _decode_multi(det, g, sc)
        if found:
            uniq: dict[str, QrCode] = {}
            for q in found:
                uniq.setdefault(q.value, q)
            return list(uniq.values())
    return []


def qr_polygon_mask(
    shape: tuple[int, int], qrs: list[QrCode], scale: float, margin: float = 0.25
) -> np.ndarray:
    """QR 꼭짓점(원본 좌표 × scale) 주변을 margin 비율만큼 넓힌 마스크(255=QR 영역)."""
    mask = np.zeros(shape, np.uint8)
    for q in qrs:
        if len(q.corners) != 4:
            continue
        pts = np.array(q.corners, np.float32) * scale
        c = pts.mean(axis=0)
        pts = c + (pts - c) * (1.0 + margin)
        cv2.fillConvexPoly(mask, pts.astype(np.int32), 255)
    return mask
