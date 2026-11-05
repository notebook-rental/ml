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

from common.messages import Alignment, BankPhoto, CompareRequest, CompareResult, Finding, Geometry, QrCode
from common.storage import ObjectNotFound, Storage
from inference.config import Settings
from inference.pipeline import align
from inference.pipeline import bank as bank_mod
from inference.pipeline.baseline_cache import BaselineFeatures, BaselineStore, mask_crop
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
from inference.pipeline.postprocess import Region, combine, extract_regions, specular, to_coords
from inference.pipeline.qr import detect_labels, qr_polygon_mask
from inference.pipeline.segment import Segmenter, laptop_box_from_gray
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
        "compare comparisonId=%s status=%s method=%s matched=%s reason=%s inliers=%s reprojErrorPx=%s findings=%d "
        "runtimeMs=%d timings=%s",
        req.comparisonId,
        result.status,
        result.method or "-",
        result.matchedBaselinePhotoId if result.matchedBaselinePhotoId is not None else "-",
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
    if s.compare_mode == "bank":
        return _compare_bank(req, storage, baselines, models, s, timer)
    return _compare_aligned(req, storage, baselines, models, s, timer)


def _compare_aligned(
    req: CompareRequest, storage: Storage, baselines: BaselineStore, models: Models, s: Settings, timer: Timer
) -> CompareResult:
    """예전 방식: 같은 컷 기준 사진 하나에 정확히 겹친 뒤 같은 자리끼리 비교(compare_mode=aligned)."""
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


def _merge_close(regions: list[Region], gap: int = 15) -> list[Region]:
    """가까운(gap 픽셀 안) 덩어리를 하나로 합친다. 조각 점수와 SSIM이 한 손상을 두 조각으로 나누는 것을 막는다."""
    if len(regions) < 2:
        return regions
    union = np.zeros_like(regions[0].mask)
    for r in regions:
        union |= r.mask
    n, labels = cv2.connectedComponents(cv2.dilate(union, np.ones((gap, gap), np.uint8)), connectivity=8)
    groups: dict[int, list[Region]] = {}
    for r in regions:
        ys, xs = np.nonzero(r.mask)
        groups.setdefault(int(labels[ys[0], xs[0]]), []).append(r)
    out = []
    for g in groups.values():
        if len(g) == 1:
            out.append(g[0])
            continue
        m = np.zeros_like(union)
        for r in g:
            m |= r.mask
        contours, _ = cv2.findContours(
            cv2.dilate(m, np.ones((gap, gap), np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        c = max(contours, key=cv2.contourArea)
        poly = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True).reshape(-1, 2).astype(np.float32)
        x, y, w, h = cv2.boundingRect(m)
        area = int((m > 0).sum())
        out.append(
            Region(
                poly,
                m,
                (x, y, w, h),
                area,
                max(r.score_max for r in g),
                sum(r.score_mean * r.area for r in g) / area,
            )
        )
    out.sort(key=lambda r: r.score_max, reverse=True)
    return out


def _sharpen(r: Region, target: np.ndarray, reference: np.ndarray) -> Region:
    """조각 점수 덩어리는 경계가 뭉툭하다(조각 14px). 유형(모양) 추정에는 덩어리 주변에서 색이 실제로 다른 픽셀만 쓴다."""
    near = cv2.dilate(r.mask, np.ones((15, 15), np.uint8)) > 0
    t = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = (np.linalg.norm(t - b, axis=2) > 18) & near
    m = cv2.morphologyEx(diff.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return r
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < 0.2 * r.area:
        return r
    sm = (labels == i).astype(np.uint8) * 255
    return Region(r.polygon, sm, tuple(int(v) for v in stats[i, :4]), area, r.score_max, r.score_mean)


def _bank_refs(req: CompareRequest) -> list[BankPhoto]:
    refs = list(req.bankPhotos)
    if not any(str(b.photoId) == str(req.baselinePhotoId) for b in refs):
        refs.insert(
            0, BankPhoto(photoId=req.baselinePhotoId, imageKey=req.baselineImageKey, shotType=req.shotType)
        )
    return refs


def _compare_bank(
    req: CompareRequest, storage: Storage, baselines: BaselineStore, models: Models, s: Settings, timer: Timer
) -> CompareResult:
    """조각 맞추기(docs specs/004): 기준 사진 전부를 기억 장소로, 겹치지 않고 비교한다. 겹치기는 되면 덤으로."""
    refs = _bank_refs(req)
    with timer.step("baseline"):
        bases: dict[str, BaselineFeatures] = {}
        ids = {}
        for b in refs:
            qr = req.baselineQr if str(b.photoId) == str(req.baselinePhotoId) else None
            bases[str(b.photoId)] = baselines.get_or_compute(
                b.photoId, b.imageKey, models.backbone, models.segmenter, s.compare_long_side, qr
            )
            ids[str(b.photoId)] = b.photoId
        bank = bank_mod.Bank.build(
            [(pid, f.bank) for pid, f in bases.items() if f.bank is not None and len(f.bank)]
        )
    req_base = bases[str(req.baselinePhotoId)]
    baseline_qr = req.baselineQr or req_base.qr

    with timer.step("decode"):
        target_full = decode(storage.get_bytes(req.targetImageKey))
        target, t_scale = resize_long_side(target_full, s.compare_long_side)
        t_gray = to_gray(target)

    with timer.step("segment"):
        cv2.setRNGSeed(0)  # GrabCut 초기화도 무작위를 쓴다(멱등)
        box = laptop_box_from_gray(target_full)
        mask = models.segmenter.segment(
            target, None if box is None else tuple(round(v * t_scale) for v in box)
        )
    if box is None or float((mask > 0).mean()) < s.bank_min_laptop_ratio:
        return CompareResult(status="not_comparable", reason="laptop_not_found")
    x1, y1, x2, y2 = mask_crop(mask)

    with timer.step("bank"):
        m = bank_mod.match(models.backbone, bank, target[y1:y2, x1:x2], mask[y1:y2, x1:x2])
    if m.mean_distance > s.bank_max_mean_distance:
        return CompareResult(status="not_comparable", reason="no_matching_baseline")
    matched_id = max(m.votes, key=lambda k: m.votes[k])
    matched = bases[matched_id]
    matched_is_req = matched_id == str(req.baselinePhotoId)

    target_qr: list[QrCode] = []
    if req.shotType == "top" or baseline_qr:
        with timer.step("qr"):
            target_qr = detect_labels(target_full)

    with timer.step("align"):
        # 덤: 고른 기준 사진과 정확히 겹쳐지면 대여 좌표·기존 손상·라벨 바뀜까지 낸다(되지 않아도 비교는 done)
        kps = align.extract_asift(t_gray, mask)
        extra = align.qr_pairs(target_qr, baseline_qr, t_scale, matched.scale) if matched_is_req else None
        ar = align.align(kps, matched.keypoints, extra, s.min_inliers, s.max_reproj_error_px, wide=True)
    H = ar.homography if ar.ok else None  # target(compare) → matched baseline(compare)

    with timer.step("postprocess"):
        th, tw = mask.shape
        score = np.zeros((th, tw), np.float32)
        score[y1:y2, x1:x2] = (m.distance / s.dino_scale).clip(0, 1)
        valid = cv2.erode(mask, np.ones((7, 7), np.uint8))
        valid[qr_polygon_mask((th, tw), target_qr, t_scale) > 0] = 0  # 라벨 자리는 후보에서 뺀다
        reflection = specular(t_gray, np.zeros_like(t_gray))
        threshold = req.params.threshold if req.params.threshold is not None else s.bank_threshold
        bh, bw = matched.mask.shape
        reference = None
        if H is not None:
            # 겹쳐졌으면 가는 변화(SSIM)도 본다: 조각(14px)보다 가는 스크래치·테이프를 놓치지 않게.
            # 대여 사진을 반납 좌표로 옮겨 같은 자리끼리 비교하고, 임계값에 맞춰 조각 점수와 같은 눈금으로 합친다
            Hinv = np.linalg.inv(H)
            reference = align.warp(matched.image, Hinv, (tw, th))
            seen = align.warp(np.full(matched.mask.shape, 255, np.uint8), Hinv, (tw, th), nearest=True)
            seen = cv2.erode(seen, np.ones((9, 9), np.uint8))
            t_n, r_n = normalize_pair(t_gray, to_gray(reference), cv2.bitwise_and(valid, seen))
            fine = (ssim_diff(t_n, r_n) * s.ssim_gain).clip(0, 1)
            fine[seen == 0] = 0
            score = np.maximum(score, fine * (threshold / s.bank_ssim_threshold)).clip(0, 1)
        regions = _merge_close(extract_regions(score, valid, threshold, s.min_region_px, reflection))[
            :MAX_FINDINGS
        ]

        # 유형 추정용 "손상이 없었다면" 그림: 겹쳐졌으면 대여 사진을 반납 좌표로 옮긴 것,
        # 아니면 후보 자리를 주변 색으로 메운 것(대여 사진에 같은 자리가 없으므로)
        if reference is not None:
            pass
        elif regions:
            union = np.zeros((th, tw), np.uint8)
            for r in regions:
                union |= r.mask
            reference = cv2.inpaint(
                target, cv2.dilate(union, np.ones((5, 5), np.uint8)), 5, cv2.INPAINT_TELEA
            )
        else:
            reference = target

        findings: list[Finding] = []
        for r in regions:
            dtype, _ = guess_type(_sharpen(r, target, reference), target, reference)
            base_geom, overlaps = None, None
            if H is not None:
                base_geom = Geometry(type="polygon", coords=to_coords(r.polygon, H, matched.scale))
                if matched_is_req and req.existingDamages:
                    rm = align.warp(r.mask, H, (bw, bh), nearest=True)
                    overlaps = overlapping_damage(rm, req.existingDamages, matched.scale)
            findings.append(
                Finding(
                    geometry=Geometry(type="polygon", coords=to_coords(r.polygon, None, t_scale)),
                    baselineGeometry=base_geom,
                    predictedType=dtype,
                    confidence=round(min(1.0, r.score_max), 3),
                    overlapsDamageId=overlaps,
                )
            )
        if H is not None and matched_is_req:
            findings.extend(_qr_findings(baseline_qr, target_qr, np.linalg.inv(H), matched.scale, t_scale))

    with timer.step("store"):
        derived = _store_derived(
            storage, req.comparisonId, align.warp(target, H, (bw, bh)) if H is not None else None, score
        )

    return CompareResult(
        status="done",
        method="aligned" if H is not None else "patch",
        matchedBaselinePhotoId=ids[matched_id],
        alignment=_alignment(H, ar.reproj_error_px, ar.inliers, t_scale, matched.scale),
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


def _store_derived(
    storage: Storage, comparison_id, warped: np.ndarray | None, score: np.ndarray
) -> dict[str, str]:
    """점수 맵과, 겹쳐졌으면 대여 좌표로 옮긴 반납 사진을 저장한다(조각 맞추기만 했으면 점수 맵만, 반납 사진 좌표)."""
    base = f"derived/comparisons/{comparison_id}"
    keys = {"scoreMap": f"{base}/score-map.png"}
    if warped is not None:
        keys["alignedTarget"] = f"{base}/aligned-target.jpg"
        storage.put_bytes(keys["alignedTarget"], encode_jpeg(warped, 85), "image/jpeg")
    storage.put_bytes(keys["scoreMap"], encode_png((score * 255).astype(np.uint8)), "image/png")
    return keys
