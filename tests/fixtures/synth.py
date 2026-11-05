"""테스트용 합성 사진 만들기.

실제 노트북 사진은 저장소에 넣지 않는다(공개 저장소, 용량, 개인정보). 대신 코드로 "노트북처럼 생긴" 장면을 만든다.
- 바닥: 무늬가 있는 매트
- 노트북 상판: 회색 판 + 질감 + 로고 + 나사 구멍 같은 점들(특징점이 잡히도록)
- 상판 QR 스티커(OpenCV QR 인코더)
반납 사진은 같은 장면을 살짝 다른 각도·밝기로 다시 "찍고"(호모그래피 + 밝기 변화), 손상(스티커, 스크래치 등)을 넣는다.
같은 seed면 항상 같은 그림이 나온다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

W, H = 1600, 1200  # 작업본 크기(테스트를 빠르게 하려고 2560보다 작게)
LID = (260, 200, 1340, 1000)  # 노트북 상판 위치(x1,y1,x2,y2)
QR_VALUE = "NR1:7K3M9QX2TD"
QR_POS = (1080, 260)  # QR 왼쪽 위
QR_SIZE = 180


@dataclass
class Damage:
    kind: str  # sticker, scratch, stain, tape
    box: tuple[int, int, int, int]  # 기준 장면 좌표(x1,y1,x2,y2)


@dataclass
class Scene:
    image: np.ndarray
    qr_corners: list[list[float]] = field(default_factory=list)


def _qr_image(value: str, size: int) -> np.ndarray:
    enc = cv2.QRCodeEncoder.create()
    q = enc.encode(value)
    q = cv2.resize(q, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(q, cv2.COLOR_GRAY2BGR)


def base_scene(seed: int = 0, qr_value: str | None = QR_VALUE) -> Scene:
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.uint8)
    # 매트: 큰 격자 + 잡음
    img[:] = (70, 90, 60)
    for x in range(0, W, 80):
        cv2.line(img, (x, 0), (x, H), (60, 80, 50), 2)
    for y in range(0, H, 80):
        cv2.line(img, (0, y), (W, y), (60, 80, 50), 2)
    # 상판
    x1, y1, x2, y2 = LID
    lid = np.full((y2 - y1, x2 - x1, 3), 150, np.uint8)
    noise = rng.normal(0, 6, lid.shape[:2]).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 3)
    lid = np.clip(lid.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)
    img[y1:y2, x1:x2] = lid
    cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 40), 6)
    # 로고와 특징점이 될 무늬
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(img, (cx, cy), 70, (210, 210, 210), -1)
    cv2.circle(img, (cx, cy), 40, (120, 120, 120), -1)
    cv2.putText(img, "NR", (cx - 45, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (60, 60, 60), 6)
    for _ in range(60):
        px, py = int(rng.integers(x1 + 30, x2 - 30)), int(rng.integers(y1 + 30, y2 - 30))
        if abs(px - QR_POS[0] - QR_SIZE / 2) < QR_SIZE and abs(py - QR_POS[1] - QR_SIZE / 2) < QR_SIZE:
            continue
        r = int(rng.integers(3, 9))
        c = int(rng.integers(60, 200))
        cv2.circle(img, (px, py), r, (c, c, c), -1)
    for i in range(6):
        cv2.rectangle(img, (x1 + 40 + i * 50, y2 - 70), (x1 + 75 + i * 50, y2 - 40), (90, 90, 90), -1)
    corners: list[list[float]] = []
    if qr_value:
        qx, qy = QR_POS
        img[qy : qy + QR_SIZE, qx : qx + QR_SIZE] = _qr_image(qr_value, QR_SIZE)
        corners = [[qx, qy], [qx + QR_SIZE, qy], [qx + QR_SIZE, qy + QR_SIZE], [qx, qy + QR_SIZE]]
    return Scene(img, [[float(a), float(b)] for a, b in corners])


def add_damage(img: np.ndarray, d: Damage, seed: int = 0) -> np.ndarray:
    out = img.copy()
    x1, y1, x2, y2 = d.box
    if d.kind == "sticker":
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 170, 230), -1)
        cv2.putText(out, "SALE", (x1 + 5, (y1 + y2) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    elif d.kind == "scratch":
        cv2.line(out, (x1, y1), (x2, y2), (225, 225, 225), 3, lineType=cv2.LINE_AA)
    elif d.kind == "tape":
        cv2.line(out, (x1, y1), (x2, y2), (30, 30, 30), 12)
    elif d.kind == "stain":
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.ellipse(out, (cx, cy), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, (95, 110, 125), -1)
    return out


def view(
    img: np.ndarray,
    seed: int = 1,
    shift: float = 25.0,
    brightness: float = 1.0,
    blur: float = 0.0,
    noise: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """같은 장면을 다시 "찍은" 사진. (사진, 장면→사진 호모그래피)."""
    rng = np.random.default_rng(seed)
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = src + rng.uniform(-shift, shift, src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    out = np.clip(out.astype(np.float32) * brightness + rng.normal(0, noise, out.shape), 0, 255).astype(
        np.uint8
    )
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    return out, M


def jpeg(img: np.ndarray, q: int = 92) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def map_box(box: tuple[int, int, int, int], M: np.ndarray) -> tuple[float, float]:
    """장면 좌표 상자의 중심을 사진 좌표로."""
    x1, y1, x2, y2 = box
    c = np.float32([[[(x1 + x2) / 2, (y1 + y2) / 2]]])
    p = cv2.perspectiveTransform(c, M)[0, 0]
    return float(p[0]), float(p[1])


def barcode_image(value: str, scale: int = 3) -> np.ndarray:
    """Code128 바코드 라벨 이미지(BGR). 자산 번호 바코드 라벨 테스트용."""
    import zxingcpp

    img = np.array(zxingcpp.create_barcode(value, zxingcpp.BarcodeFormat.Code128).to_image(scale=scale))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
