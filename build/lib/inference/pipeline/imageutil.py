"""이미지 읽기·크기 조정 같은 공통 도구."""

from __future__ import annotations

import cv2
import numpy as np


class ImageDecodeError(ValueError):
    pass


def decode(data: bytes) -> np.ndarray:
    """JPEG/PNG 바이트 → BGR uint8 배열. EXIF 회전은 폰이 작업본을 만들 때 이미 반영한다."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError("이미지를 읽을 수 없습니다")
    return img


def resize_long_side(img: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    """긴 변이 long_side가 되게 줄인다(늘리지는 않는다). (줄인 이미지, 배율) 반환. 배율 = 새 크기 / 원래 크기."""
    h, w = img.shape[:2]
    scale = min(1.0, long_side / max(h, w))
    if scale >= 1.0:
        return img, 1.0
    out = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return out, scale


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:  # pragma: no cover - OpenCV 내부 오류
        raise RuntimeError("JPEG 인코딩 실패")
    return buf.tobytes()


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:  # pragma: no cover
        raise RuntimeError("PNG 인코딩 실패")
    return buf.tobytes()
