"""자산 번호 라벨 검출·해독. 라벨은 QR 또는 1차원 바코드(Code128 등)다(docs spec 002).

식별용 해독은 폰과 Spring(ZXing)이 맡는다. 여기서는
- 품질 검사에서 상판 컷에 라벨이 보이는지, QR이 여러 개인지 확인,
- 반납 비교에서 라벨 영역을 비교 대상에서 빼고(스티커가 바뀌는 건 손상이 아님),
  라벨이 사라졌거나 값이 바뀌었는지 확인(qr_tampered)
에 쓴다.

- QR: OpenCV QRCodeDetector (지금까지 쓰던 것).
- 바코드: zxing-cpp. OpenCV 바코드 검출기는 상품 바코드(EAN/UPC)만 읽고 Code128을 못 읽기 때문이다.
  zxing-cpp는 C++로 된 ZXing이라 빠르고(1600px 사진 한 장에 수십 ms), 설치 파일 하나로 끝난다.
"""

from __future__ import annotations

import cv2
import numpy as np
import zxingcpp

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


# 자산 번호 라벨로 쓰는 1차원 바코드 형식. 상품 바코드(EAN 등)는 넣지 않아 노트북 상자 바코드 등을 덜 읽는다.
BARCODE_FORMATS = (zxingcpp.BarcodeFormat.Code128, zxingcpp.BarcodeFormat.Code39)


def detect_barcodes(img: np.ndarray) -> list[QrCode]:
    """사진 속 1차원 바코드 라벨들(값이 읽힌 것만). 꼭짓점은 입력 사진 픽셀 좌표."""
    small, scale = resize_long_side(img, DETECT_SIDE)
    try:
        results = zxingcpp.read_barcodes(to_gray(small), formats=BARCODE_FORMATS)
    except Exception:  # pragma: no cover - 라이브러리가 특이한 입력에서 예외를 내는 경우
        return []
    uniq: dict[str, QrCode] = {}
    for r in results:
        if not r.text:
            continue
        p = r.position
        pts = [p.top_left, p.top_right, p.bottom_right, p.bottom_left]
        corners = [[round(pt.x / scale, 1), round(pt.y / scale, 1)] for pt in pts]
        uniq.setdefault(r.text, QrCode(value=r.text, corners=corners))
    return list(uniq.values())


def detect_labels(img: np.ndarray) -> list[QrCode]:
    """QR과 바코드 라벨을 모두 찾는다. QR을 먼저 보고, 같은 값은 한 번만 넣는다."""
    out = {q.value: q for q in detect_qr(img)}
    for b in detect_barcodes(img):
        out.setdefault(b.value, b)
    return list(out.values())
