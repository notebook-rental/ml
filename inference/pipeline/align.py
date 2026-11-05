"""정렬: 반납 사진을 대여 사진 좌표로 맞춘다 (research R2 2단계, tasks T056).

같은 노트북을 같은 구도로 찍어도 손 위치·각도가 조금씩 다르다. 두 사진에서 특징점(SIFT)을 뽑아 짝을 찾고,
RANSAC으로 엉뚱한 짝을 걸러 낸 뒤 호모그래피(평면 대 평면 투영 변환 3×3 행렬)를 구한다.
노트북 상판·바닥은 평면에 가깝기 때문에 호모그래피 하나로 잘 맞는다.

- 상판(top) 컷은 QR 네 꼭짓점을 추가 짝으로 넣어 더 안정적으로 맞춘다.
- 맞춘 결과의 오차(재투영 오차, 픽셀)가 크거나 짝이 너무 적으면 "자동 비교 불가(not_comparable)".
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SIFT_MAX_FEATURES = 4000
RATIO = 0.75
QR_REPEAT = 6  # QR 꼭짓점 짝에 주는 가중치(같은 짝을 여러 번 넣는다)


@dataclass
class Keypoints:
    pts: np.ndarray  # [N,2] float32, compare 해상도 좌표
    desc: np.ndarray  # [N,128] float32


@dataclass
class AlignResult:
    ok: bool
    homography: np.ndarray | None  # target(compare 해상도) → baseline(compare 해상도)
    reproj_error_px: float
    inliers: int
    reason: str | None = None


_sift = None


def _get_sift():
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
    return _sift


def extract(gray: np.ndarray, mask: np.ndarray | None = None) -> Keypoints:
    if mask is not None:
        mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
    kps, desc = _get_sift().detectAndCompute(gray, mask)
    if desc is None or not kps:
        return Keypoints(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32))
    pts = np.array([k.pt for k in kps], np.float32)
    return Keypoints(pts, desc.astype(np.float32))


def extract_asift(gray: np.ndarray, mask: np.ndarray | None = None) -> Keypoints:
    """ASIFT: 사진을 여러 방향으로 미리 눌러(기울인 것처럼) 본 뒤 SIFT를 뽑아 원래 좌표로 되돌린다 (docs specs/004).

    비스듬히 찍으면 같은 무늬가 한쪽으로 찌그러져 SIFT 짝이 안 맞는다. 반대로 찌그러뜨린 사진을 여러 장 만들어 두면
    그중 하나는 기준 사진과 비슷해진다. 모델 없이 OpenCV만 쓴다.
    """
    pts, desc = [], []
    h, w = gray.shape
    for t in (1.0, 1.41, 2.0):
        for phi in (0,) if t == 1.0 else (0, 45, 90, 135):
            A = cv2.getRotationMatrix2D((w / 2, h / 2), phi, 1.0)
            A[0] /= t  # 돌린 뒤 가로로 1/t 눌러 기울기 흉내
            A[0, 2] += (w - w / t) / 2
            g = cv2.warpAffine(gray, A, (w, h), flags=cv2.INTER_LINEAR)
            m = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST) if mask is not None else None
            k = extract(g, m)
            if not len(k.pts):
                continue
            back = np.c_[k.pts, np.ones(len(k.pts))] @ cv2.invertAffineTransform(A).T
            pts.append(back.astype(np.float32))
            desc.append(k.desc)
    if not pts:
        return Keypoints(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32))
    return Keypoints(np.vstack(pts), np.vstack(desc))


def match(target: Keypoints, baseline: Keypoints) -> tuple[np.ndarray, np.ndarray]:
    """Lowe 비율 검사로 확실한 짝만 남긴다. (target 점들, baseline 점들)."""
    if len(target.pts) < 2 or len(baseline.pts) < 2:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = bf.knnMatch(target.desc, baseline.desc, k=2)
    src, dst = [], []
    for p in pairs:
        if len(p) == 2 and p[0].distance < RATIO * p[1].distance:
            src.append(target.pts[p[0].queryIdx])
            dst.append(baseline.pts[p[0].trainIdx])
    return np.array(src, np.float32).reshape(-1, 2), np.array(dst, np.float32).reshape(-1, 2)


def align(
    target: Keypoints,
    baseline: Keypoints,
    extra_pairs: tuple[np.ndarray, np.ndarray] | None = None,
    min_inliers: int = 25,
    max_reproj_error_px: float = 8.0,
    wide: bool = False,
) -> AlignResult:
    src, dst = match(target, baseline)
    if extra_pairs is not None and len(extra_pairs[0]):
        src = np.vstack([src, np.repeat(extra_pairs[0], QR_REPEAT, axis=0)])
        dst = np.vstack([dst, np.repeat(extra_pairs[1], QR_REPEAT, axis=0)])
    if len(src) < max(4, min_inliers):
        return AlignResult(False, None, float("inf"), len(src), "too_few_matches")
    cv2.setRNGSeed(0)  # RANSAC 계열은 무작위를 쓴다. 같은 입력이면 같은 결과(멱등)
    H, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 4.0, maxIters=5000, confidence=0.999)
    if H is None or inl is None:
        return AlignResult(False, None, float("inf"), 0, "homography_failed")
    inl = inl.ravel().astype(bool)
    n_in = int(inl.sum())
    proj = cv2.perspectiveTransform(src[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.median(np.linalg.norm(proj - dst[inl], axis=1))) if n_in else float("inf")
    if n_in < min_inliers:
        return AlignResult(False, H, err, n_in, "too_few_inliers")
    if err > max_reproj_error_px or not (_plausible_wide(H) if wide else _plausible(H)):
        return AlignResult(False, H, err, n_in, "alignment_error_too_large")
    return AlignResult(True, H, err, n_in)


def _plausible(H: np.ndarray) -> bool:
    """찍는 구도가 정해져 있으므로 크게 뒤집히거나 찌그러진 변환은 잘못 맞춘 것이다."""
    det = float(np.linalg.det(H[:2, :2]))
    return 0.25 < det < 4.0 and abs(H[2, 0]) < 0.005 and abs(H[2, 1]) < 0.005


def _plausible_wide(H: np.ndarray) -> bool:
    """대충 찍은 사진용 넓은 범위(크기 약 0.3~3배, 원근 항 0.01까지). 뒤집힘(행렬식 음수)만 확실히 막는다."""
    det = float(np.linalg.det(H[:2, :2]))
    return 0.1 < det < 10.0 and abs(H[2, 0]) < 0.01 and abs(H[2, 1]) < 0.01


def qr_pairs(target_qr, baseline_qr, t_scale: float, b_scale: float) -> tuple[np.ndarray, np.ndarray] | None:
    """같은 값의 QR이 두 사진에 다 있으면 네 꼭짓점을 짝으로 쓴다(원본 좌표 → compare 해상도)."""
    by_value = {q.value: q for q in baseline_qr if len(q.corners) == 4}
    src, dst = [], []
    for q in target_qr:
        b = by_value.get(q.value)
        if b is None or len(q.corners) != 4:
            continue
        src.extend(np.array(q.corners, np.float32) * t_scale)
        dst.extend(np.array(b.corners, np.float32) * b_scale)
    if not src:
        return None
    return np.array(src, np.float32), np.array(dst, np.float32)


def warp(img: np.ndarray, H: np.ndarray, size: tuple[int, int], nearest: bool = False) -> np.ndarray:
    """img를 H로 옮겨 size=(w,h) 캔버스에 그린다. 바깥은 0."""
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpPerspective(img, H, size, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
