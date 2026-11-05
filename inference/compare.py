"""컷 하나 비교 (`POST /compare`, research R2·R12, tasks T060·T061·T064).

순서(괄호는 대략의 CPU 시간 목표, 4코어 기준은 벤치마크 결과 문서 참고):
1. 기준 특징 꺼내기 — 메모리(prefetch) 또는 저장소 npz, 없으면 계산
2. 반납 사진 읽기·줄이기(긴 변 1024)
3. SIFT 특징점 → 기준과 짝 찾기 → RANSAC 호모그래피 (정렬)
4. 반납 사진을 기준 좌표로 옮기기, 조명 맞추기
5. DINOv2 패치 특징 → 기준과 주변 최근접 거리(이상 점수)
6. SSIM 차분(가는 변화)
7. 점수 합치기 → 임계값 → 덩어리 → 다각형, QR·노트북 밖·반사 제외
8. 라벨(QR·바코드) 사라짐/바뀜 → qr_tampered 후보, 기존 손상 겹침, 유형 추정
9. 좌표를 원본 픽셀로 바꿔 응답, 정렬 결과·점수 맵을 derived/comparisons/에 저장

같은 입력이면 결과가 같다(멱등). api가 재시도해도 안전하다.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np

from common.messages import Alignment, CompareRequest, CompareResult, Finding, Geometry, QrCode
from common.storage import ObjectNotFound, Storage
from inference.config import Settings
from inference.pipeline import align
from inference.pipeline.baseline_cache import BaselineFeatures, BaselineStore
from inference.pipeline.existing_damage import overlapping_damage
from inference.pipeline.imageutil import (
    ImageDecodeError,
    decode,
    encode_jpeg,
    encode_png,
    resize_long_side,
    to_gray,
)
from inference.pipeline.normalize import match_color, normalize_pair
from inference.pipeline.patch_features import Backbone, neighbor_distance
from inference.pipeline.postprocess import combine, extract_regions, specular, to_coords
from inference.pipeline.qr import detect_labels, qr_polygon_mask
from inference.pipeline.segment import Segmenter
from inference.pipeline.ssim import ssim_diff
from inference.pipeline.typing import guess_type

log = logging.getLogger(__name__)

MAX_FINDINGS = 20


@dataclass
class Models:
    backbone: Backbone
    segmenter: Segmenter
    model_version_id: object = None


class Timer:
    def __init__(self):
        self.ms: dict[str, int] = {}

    @contextmanager
    def step(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = self.ms.get(name, 0) + round((time.perf_counter() - t) * 1000)


def compare(
    req: CompareRequest, storage: Storage, baselines: BaselineStore, models: Models, settings: Settings
) -> CompareResult:
    t0 = time.perf_counter()
    timer = Timer()
    try:
        result = _compare(req, storage, baselines, models, settings, timer)
    except (ObjectNotFound, ImageDecodeError) as e:
        result = CompareResult(status="failed", error=f"{type(e).__name__}: {e}")
    except Exception as e:  # 예상 못 한 오류도 응답 본문으로 돌려준다(api가 재시도·사람 검토로 넘김)
        log.exception("compare failed comparisonId=%s", req.comparisonId)
        result = CompareResult(status="failed", error=f"{type(e).__name__}: {e}")
    result.runtimeMs = round((time.perf_counter() - t0) * 1000)
    result.timingsMs = timer.ms
    result.modelVersionId = (
        req.modelVersion.id if req.modelVersion.id is not None else models.model_version_id
    )
    a = result.alignment
    log.info(
        "compare comparisonId=%s status=%s reason=%s inliers=%s reprojErrorPx=%s findings=%d runtimeMs=%d timings=%s",
        req.comparisonId,
        result.status,
        result.reason
        or result.error
        or "-",  # 자동 비교 불가·실패의 이유(정렬 실패면 inliers가 기준보다 적다)
        a.inliers if a else "-",
        a.reprojErrorPx if a else "-",
        len(result.findings),
        result.runtimeMs,
        timer.ms,
    )
    return result


def _compare(
    req: CompareRequest, storage: Storage, baselines: BaselineStore, models: Models, s: Settings, timer: Timer
) -> CompareResult:
    with timer.step("baseline"):
        base: BaselineFeatures = baselines.get_or_compute(
            req.baselinePhotoId,
            req.baselineImageKey,
            models.backbone,
            models.segmenter,
            s.compare_long_side,
            req.baselineQr,
        )
    baseline_qr = req.baselineQr or base.qr

    with timer.step("decode"):
        target_full = decode(storage.get_bytes(req.targetImageKey))
        target, t_scale = resize_long_side(target_full, s.compare_long_side)
        t_gray = to_gray(target)

    target_qr: list[QrCode] = []
    if req.shotType == "top" or baseline_qr:
        with timer.step("qr"):
            target_qr = detect_labels(target_full)

    with timer.step("align"):
        kps = align.extract(t_gray)
        extra = align.qr_pairs(target_qr, baseline_qr, t_scale, base.scale)
        ar = align.align(kps, base.keypoints, extra, s.min_inliers, s.max_reproj_error_px)
    if not ar.ok:
        return CompareResult(
            status="not_comparable",
            reason=ar.reason,
            alignment=_alignment(ar.homography, ar.reproj_error_px, ar.inliers, t_scale, base.scale),
        )
    H = ar.homography  # target(compare) → baseline(compare)
    bh, bw = base.mask.shape

    with timer.step("warp"):
        warped = align.warp(target, H, (bw, bh))
        valid = align.warp(np.full(t_gray.shape, 255, np.uint8), H, (bw, bh), nearest=True)
        valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
        region_mask = cv2.bitwise_and(cv2.erode(base.mask, np.ones((7, 7), np.uint8)), valid)
        w_gray, b_gray = normalize_pair(to_gray(warped), base.gray, region_mask)
        warped_color = match_color(warped, base.image, region_mask)

    with timer.step("dino"):
        x1, y1, x2, y2 = base.crop
        t_feats = models.backbone.features(warped_color[y1:y2, x1:x2])
        dist = neighbor_distance(t_feats, base.dino, s.neighbor_radius)
        dino_map = np.zeros((bh, bw), np.float32)
        dino_map[y1:y2, x1:x2] = cv2.resize(dist, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)

    with timer.step("ssim"):
        ssim_map = ssim_diff(w_gray, b_gray)

    with timer.step("postprocess"):
        threshold = req.params.threshold if req.params.threshold is not None else s.threshold
        score = combine(dino_map, ssim_map, s.dino_scale, s.ssim_gain)
        qr_mask = qr_polygon_mask((bh, bw), baseline_qr, base.scale)
        region_mask[qr_mask > 0] = 0
        reflection = specular(to_gray(warped), base.gray)
        regions = extract_regions(score, region_mask, threshold, s.min_region_px, reflection)[:MAX_FINDINGS]

        H_inv = np.linalg.inv(H)  # baseline(compare) → target(compare)
        findings: list[Finding] = []
        for r in regions:
            dtype, _type_conf = guess_type(r, warped, base.image)
            findings.append(
                Finding(
                    geometry=Geometry(type="polygon", coords=to_coords(r.polygon, H_inv, t_scale)),
                    baselineGeometry=Geometry(type="polygon", coords=to_coords(r.polygon, None, base.scale)),
                    predictedType=dtype,
                    confidence=round(min(1.0, r.score_max), 3),
                    overlapsDamageId=overlapping_damage(r.mask, req.existingDamages, base.scale),
                )
            )
        findings.extend(_qr_findings(baseline_qr, target_qr, H_inv, base.scale, t_scale))

    with timer.step("store"):
        derived = _store_derived(storage, req.comparisonId, warped, score)

    return CompareResult(
        status="done",
        alignment=_alignment(H, ar.reproj_error_px, ar.inliers, t_scale, base.scale),
        derivedKeys=derived,
        findings=findings,
    )


def _qr_findings(
    baseline_qr: list[QrCode], target_qr: list[QrCode], H_inv: np.ndarray, b_scale: float, t_scale: float
) -> list[Finding]:
    """기준 사진에 있던 QR이 반납 사진에서 안 보이거나 값이 다르면 qr_tampered 후보(FR-015, tasks T061)."""
    out = []
    target_values = {q.value for q in target_qr}
    for q in baseline_qr:
        if len(q.corners) != 4 or q.value in target_values:
            continue
        poly = np.array(q.corners, np.float32) * b_scale
        # 다른 값의 QR이 보이면 바뀐 것(확실), 아무것도 안 읽히면 가려졌거나 흐린 것일 수도 있다
        conf = 0.9 if target_qr else 0.6
        out.append(
            Finding(
                geometry=Geometry(type="polygon", coords=to_coords(poly, H_inv, t_scale)),
                baselineGeometry=Geometry(
                    type="polygon", coords=[[float(x), float(y)] for x, y in q.corners]
                ),
                predictedType="qr_tampered",
                confidence=conf,
            )
        )
    return out


def _alignment(H: np.ndarray | None, err: float, inliers: int, t_scale: float, b_scale: float) -> Alignment:
    """compare 해상도 호모그래피를 원본 픽셀 좌표 기준(반납 원본 → 기준 원본)으로 바꿔 돌려준다."""
    if H is None:
        return Alignment(homography=[], reprojErrorPx=_finite(err), inliers=inliers)
    S_t = np.diag([t_scale, t_scale, 1.0])
    S_b_inv = np.diag([1 / b_scale, 1 / b_scale, 1.0])
    Ho = S_b_inv @ H @ S_t
    Ho = Ho / Ho[2, 2]
    return Alignment(
        homography=[[round(float(v), 8) for v in row] for row in Ho],
        reprojErrorPx=round(_finite(err) / b_scale, 2),
        inliers=inliers,
    )


def _finite(v: float) -> float:
    return float(v) if np.isfinite(v) else -1.0


def _store_derived(storage: Storage, comparison_id, warped: np.ndarray, score: np.ndarray) -> dict[str, str]:
    base = f"derived/comparisons/{comparison_id}"
    keys = {"alignedTarget": f"{base}/aligned-target.jpg", "scoreMap": f"{base}/score-map.png"}
    storage.put_bytes(keys["alignedTarget"], encode_jpeg(warped, 85), "image/jpeg")
    storage.put_bytes(keys["scoreMap"], encode_png((score * 255).astype(np.uint8)), "image/png")
    return keys
