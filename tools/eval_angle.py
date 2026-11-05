"""각도에 강한 비교 방식 실험 (docs specs/004, 실제 사진으로 후보 방식 비교).

사진 폴더 하나를 받아, 같은 컷끼리(기준 ↔ 반납) 여러 비교 방식을 돌리고 숫자만 출력한다.
운영 저장소·DB에는 아무것도 쓰지 않는다. 임시 폴더(LocalStorage)에서만 계산한다.
사진 내용(픽셀)은 출력하지 않는다. 결과는 표(markdown)와 JSON 한 줄이다.

폴더 모양:
    <dir>/baseline/<shot>.jpg   대여(기준) 사진. shot = top, bottom, left, right, hinge, front, keyboard, screen_off
    <dir>/return/<shot>.jpg     반납 사진(있는 컷만)

비교하는 방식:
- A current      지금 운영 방식(SIFT 전체 → 호모그래피, 기준 25점/8px)
- B crop         반납 사진도 노트북 영역만 잘라 SIFT를 뽑는다(배경 짝이 섞이지 않게)
- C crop_loose   B + 기준 완화(12점/16px)
- E asift       B + 반납 사진을 여러 방향으로 미리 기울여 SIFT(ASIFT, 모델 없음)
- F hybrid      E + 넓은 허용 범위(크기·원근) + 주변 반경 3, 그래도 겹치기가 안 되면 D로 대신
- G best_match  실제 사진만: 컷 이름을 무시하고 기준 8장 중 가장 잘 맞는 사진과 F로 비교
- D dino_global  겹치기(호모그래피) 없이 DINOv2 패치끼리 "가장 닮은 조각"을 노트북 전체에서 찾는다
                 각도·위치가 달라도 조각 단위로 맞춘다. 정렬 실패가 없다.

합성 각도 시험(--synthetic): 기준 사진을 0°·10°·20°·30°·40° 기울여 "다시 찍은" 사진을 만들고
가짜 스크래치 1개를 그려, 방식마다 비교 성공 여부·가짜 후보 수·스크래치를 찾았는지 센다.

사용법:
    python tools/eval_angle.py <dir> --backbone <dinov2 onnx> [--synthetic] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.messages import CompareRequest  # noqa: E402
from common.storage import LocalStorage  # noqa: E402
from inference import compare as compare_mod  # noqa: E402
from inference.compare import Models, compare  # noqa: E402
from inference.config import Settings  # noqa: E402
from inference.pipeline import align  # noqa: E402
from inference.pipeline.baseline_cache import BaselineStore, mask_crop  # noqa: E402
from inference.pipeline.engine import create_session  # noqa: E402
from inference.pipeline.imageutil import decode, resize_long_side, to_gray  # noqa: E402
from inference.pipeline.patch_features import Backbone  # noqa: E402
from inference.pipeline.postprocess import extract_regions, specular  # noqa: E402
from inference.pipeline.segment import GrabCutSegmenter, laptop_box_from_gray  # noqa: E402

SHOTS = ("top", "bottom", "left", "right", "hinge", "front", "keyboard", "screen_off")
TILTS = (0, 10, 20, 30, 40)


@dataclass
class Outcome:
    method: str
    shot: str
    case: str  # real 또는 tilt20 등
    comparable: bool
    reason: str | None = None
    inliers: int | None = None
    reproj: float | None = None
    candidates: int = 0
    hit: bool | None = None  # 합성 스크래치를 찾았는지(합성 시험만)
    ms: int = 0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- 공통


def view_angle(H: list[list[float]]) -> dict:
    """호모그래피에서 사람이 읽을 수 있는 차이: 크기 비율, 돌아간 각도, 기울기(원근) 정도."""
    if not H:
        return {}
    M = np.array(H, float)
    a = M[:2, :2]
    scale = math.sqrt(abs(np.linalg.det(a)))
    rot = math.degrees(math.atan2(a[1, 0], a[0, 0]))
    persp = float(np.hypot(M[2, 0], M[2, 1]) * 1000)
    return {"scale": round(scale, 2), "rotDeg": round(rot, 1), "perspective": round(persp, 3)}


def tilt(img: np.ndarray, deg: float, roll: float = 6.0, gain: float = 0.88) -> tuple[np.ndarray, np.ndarray]:
    """사진을 deg만큼 옆으로(요) 기울이고 roll만큼 돌려 "다른 각도에서 다시 찍은" 것처럼 만든다. (사진, H 기준→결과)."""
    h, w = img.shape[:2]
    f = 1.1 * max(w, h)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    y, p, r = math.radians(deg), math.radians(deg * 0.4), math.radians(roll if deg else 0)
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    H = K @ Rz @ Ry @ Rx @ np.linalg.inv(K)
    # 결과가 화면 안에 들어오도록 옮기고 줄인다
    corners = cv2.perspectiveTransform(np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], np.float32), H)
    x0, y0 = corners.reshape(-1, 2).min(0)
    x1, y1 = corners.reshape(-1, 2).max(0)
    s = min(w / (x1 - x0), h / (y1 - y0)) * 0.95
    T = np.array(
        [[s, 0, -x0 * s + (w - (x1 - x0) * s) / 2], [0, s, -y0 * s + (h - (y1 - y0) * s) / 2], [0, 0, 1]]
    )
    H = T @ H
    out = cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
    out = np.clip(out.astype(np.float32) * gain + 6, 0, 255).astype(np.uint8)
    return out, H


def draw_scratch(img: np.ndarray, seed: int) -> np.ndarray:
    """노트북 안쪽 아무 곳에 가는 스크래치 하나. 그린 자리 마스크(원본 좌표)를 돌려준다."""
    rng = np.random.default_rng(seed)
    box = laptop_box_from_gray(img)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box if box else (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
    bw, bh = x2 - x1, y2 - y1
    cx = int(x1 + bw * rng.uniform(0.3, 0.7))
    cy = int(y1 + bh * rng.uniform(0.3, 0.7))
    L = 0.15 * bw
    ang = rng.uniform(0, math.pi)
    p1 = (int(cx - L / 2 * math.cos(ang)), int(cy - L / 2 * math.sin(ang)))
    p2 = (int(cx + L / 2 * math.cos(ang)), int(cy + L / 2 * math.sin(ang)))
    thick = max(2, w // 600)
    mean = float(cv2.mean(to_gray(img)[max(0, cy - 20) : cy + 20, max(0, cx - 20) : cx + 20])[0])
    color = (235, 235, 235) if mean < 128 else (30, 30, 30)
    cv2.line(img, p1, p2, color, thick, cv2.LINE_AA)
    m = np.zeros((h, w), np.uint8)
    cv2.line(m, p1, p2, 255, thick + 6)
    return m


# ---------------------------------------------------------------- A·B·C: 지금 방식(+변형)


class Harness:
    def __init__(self, backbone_path: str, work: Path):
        self.work = work
        self.storage = LocalStorage(work)
        session = create_session(backbone_path, "onnxruntime", int(os.environ.get("EVAL_THREADS", "0")))
        self.models = Models(backbone=Backbone(session, "dinov2-s14@1"), segmenter=GrabCutSegmenter())
        self.baselines = BaselineStore(self.storage, 64)
        self.n = 0

    def put(self, key: str, img: np.ndarray) -> None:
        p = self.work / key
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

    def run_pipeline(
        self,
        shot: str,
        base_key: str,
        base_id: str,
        target_key: str,
        settings: Settings,
        crop_target: bool,
        asift: bool = False,
        wide: bool = False,
    ) -> dict:
        self.n += 1
        req = CompareRequest(
            comparisonId=f"eval-{self.n}",
            shotType=shot,
            baselinePhotoId=base_id,
            baselineImageKey=base_key,
            targetImageKey=target_key,
        )
        orig_extract = align.extract
        orig_plausible = align._plausible
        if wide:
            compare_mod.align._plausible = wide_plausible
        if crop_target:
            seg = self.models.segmenter

            def extract(gray, mask=None):
                if mask is None:  # 반납 사진(기준 사진은 이미 마스크를 넘긴다)
                    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    box = laptop_box_from_gray(img)
                    mask = seg.segment(img, box)
                    if asift:
                        return asift_extract(gray, mask, orig_extract)
                return orig_extract(gray, mask)

            compare_mod.align.extract = extract
        try:
            t = time.perf_counter()
            r = compare(req, self.storage, self.baselines, self.models, settings)
            ms = round((time.perf_counter() - t) * 1000)
        finally:
            compare_mod.align.extract = orig_extract
            compare_mod.align._plausible = orig_plausible
        return {"result": r, "ms": ms}


def wide_plausible(H: np.ndarray) -> bool:
    """가까이·멀리, 비스듬히 찍어도 받아 주는 넓은 범위(크기 0.3~3배, 원근 항 0.01까지). 뒤집힘만 막는다."""
    det = float(np.linalg.det(H[:2, :2]))
    return 0.1 < det < 10.0 and abs(H[2, 0]) < 0.01 and abs(H[2, 1]) < 0.01


def asift_extract(gray: np.ndarray, mask: np.ndarray, extract) -> align.Keypoints:
    """ASIFT: 반납 사진을 여러 방향으로 미리 눌러(기울인 것처럼) 본 뒤 SIFT를 뽑아 원래 좌표로 되돌린다.

    비스듬히 찍으면 같은 무늬가 한쪽으로 찌그러져 SIFT 짝이 안 맞는다. 반대로 찌그러뜨린 사진을 여러 장 만들어 두면
    그중 하나는 기준 사진과 비슷해진다. 모델 없이 OpenCV만 쓴다.
    """
    pts, desc = [], []
    h, w = gray.shape
    for t in (1.0, 1.41, 2.0):
        for phi in (0,) if t == 1.0 else (0, 45, 90, 135):
            A = cv2.getRotationMatrix2D((w / 2, h / 2), phi, 1.0)
            A[0] /= t  # 회전 뒤 가로로 1/t 눌러서 기울기 흉내
            A[0, 2] += (w - w / t) / 2
            g = cv2.warpAffine(gray, A, (w, h), flags=cv2.INTER_LINEAR)
            m = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST) if mask is not None else None
            k = extract(g, m)
            if not len(k.pts):
                continue
            Ai = cv2.invertAffineTransform(A)
            back = (np.c_[k.pts, np.ones(len(k.pts))] @ Ai.T).astype(np.float32)
            pts.append(back)
            desc.append(k.desc)
    if not pts:
        return align.Keypoints(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32))
    return align.Keypoints(np.vstack(pts), np.vstack(desc))


def pipeline_outcome(
    method: str, shot: str, case: str, out: dict, scratch_mask: np.ndarray | None
) -> Outcome:
    r = out["result"]
    a = r.alignment
    o = Outcome(
        method,
        shot,
        case,
        r.status == "done",
        r.reason or r.error,
        a.inliers if a else None,
        a.reprojErrorPx if a else None,
        ms=out["ms"],
    )
    o.extra = view_angle(a.homography if a else [])
    if r.status == "done":
        hits = 0
        for f in r.findings:
            if scratch_mask is not None and _poly_hits(f.geometry.coords, scratch_mask):
                hits += 1
        o.candidates = len(r.findings) - hits
        o.hit = (hits > 0) if scratch_mask is not None else None
    elif scratch_mask is not None:
        o.hit = False
    return o


def _poly_hits(coords, mask: np.ndarray) -> bool:
    m = np.zeros_like(mask)
    cv2.fillPoly(m, [np.array(coords, np.int32)], 255)
    return bool(((m > 0) & (mask > 0)).any())


# ---------------------------------------------------------------- D: 겹치기 없는 DINOv2 조각 맞추기


def dino_global(
    h: Harness,
    base_key: str,
    base_id: str,
    target: np.ndarray,
    s: Settings,
    scratch_mask: np.ndarray | None,
    shot: str,
    case: str,
) -> Outcome:
    t0 = time.perf_counter()
    base = h.baselines.get_or_compute(
        base_id, base_key, h.models.backbone, h.models.segmenter, s.compare_long_side
    )
    small, scale = resize_long_side(target, s.compare_long_side)
    box = laptop_box_from_gray(target)
    mask = h.models.segmenter.segment(small, None if box is None else tuple(round(v * scale) for v in box))
    x1, y1, x2, y2 = mask_crop(mask)
    feats = h.models.backbone.features(small[y1:y2, x1:x2])  # [gh,gw,C]
    gh, gw, c = feats.shape
    bank = base.dino.reshape(-1, c)
    # 조각마다 기준 노트북 전체에서 가장 닮은 조각(코사인 유사도 최대)
    sim = feats.reshape(-1, c) @ bank.T
    best = sim.max(1).reshape(gh, gw)
    dist = (1.0 - best).astype(np.float32)
    dmap = np.zeros(mask.shape, np.float32)
    dmap[y1:y2, x1:x2] = cv2.resize(dist, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
    score = (dmap / s.dino_scale).clip(0, 1)
    valid = cv2.erode(mask, np.ones((7, 7), np.uint8))
    refl = specular(to_gray(small), np.full(mask.shape, 0, np.uint8))
    regions = extract_regions(score, valid, s.threshold, s.min_region_px, refl)
    ms = round((time.perf_counter() - t0) * 1000)
    hits = 0
    if scratch_mask is not None:
        sm = cv2.resize(scratch_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        hits = sum(1 for r in regions if ((r.mask > 0) & (sm > 0)).any())
    o = Outcome("D dino_global", shot, case, True, None, ms=ms)
    o.candidates = len(regions) - hits
    o.hit = (hits > 0) if scratch_mask is not None else None
    o.extra = {"meanDist": round(float(dist.mean()), 3)}

    return o


# ---------------------------------------------------------------- 실행


def methods(base: Settings):
    loose = base.model_copy(update={"min_inliers": 12, "max_reproj_error_px": 16.0})
    return [
        ("A current", base, False, False, False),
        ("B crop", base, True, False, False),
        ("C crop_loose", loose, True, False, False),
        ("E asift", base, True, True, False),
        ("F hybrid", base.model_copy(update={"neighbor_radius": 3}), True, True, True),
    ]


def best_match(h: Harness, dir_: Path, s: Settings, prev: list[Outcome]) -> list[Outcome]:
    """G: 컷 이름을 믿지 않는다. 반납 사진마다 기준 사진 전부와 맞춰 보고 가장 잘 맞는(짝이 많은) 기준을 고른다.

    기준 컷 이름이 틀렸거나(바닥 자리에 상판), 관리자가 다른 면을 찍어도 맞는 기준을 찾는다.
    """
    _, st, crop, asift, wide = methods(s)[-1]
    bases = [sh for sh in SHOTS if (dir_ / "baseline" / f"{sh}.jpg").exists()]
    out = []
    for shot in SHOTS:
        if not (dir_ / "return" / f"{shot}.jpg").exists():
            continue
        tkey = f"photos/target/{shot}-real.jpg"
        best = None
        for b in bases:
            r = h.run_pipeline(shot, f"photos/baseline/{b}.jpg", f"b-{b}", tkey, st, crop, asift, wide)
            o = pipeline_outcome("G best_match", shot, "real", r, None)
            o.extra = {**o.extra, "matched": b}
            if o.comparable and (best is None or (o.inliers or 0) > (best.inliers or 0)):
                best = o
        if best is None:  # 어느 기준과도 겹치기가 안 되면 같은 이름 컷과 조각 맞추기(D)
            d = next(
                (o for o in prev if o.method == "D dino_global" and o.shot == shot and o.case == "real"), None
            )
            if d is None:
                continue
            best = Outcome(
                "G best_match",
                shot,
                "real",
                True,
                "fallback:dino",
                candidates=d.candidates,
                ms=d.ms,
                extra={**d.extra, "matched": shot},
            )
        out.append(best)
        print("g", end="", file=sys.stderr, flush=True)
    return out


def run(dir_: Path, backbone: str, synthetic: bool) -> list[Outcome]:
    work = Path(tempfile.mkdtemp(prefix="eval-angle-"))
    try:
        h = Harness(backbone, work)
        s = Settings(storage_backend="local", local_storage_root=str(work), app_profile="test")
        outs: list[Outcome] = []
        for shot in SHOTS:
            bp = dir_ / "baseline" / f"{shot}.jpg"
            if not bp.exists():
                continue
            bimg = decode(bp.read_bytes())
            bkey, bid = f"photos/baseline/{shot}.jpg", f"b-{shot}"
            h.put(bkey, bimg)
            rp = dir_ / "return" / f"{shot}.jpg"
            cases: list[tuple[str, np.ndarray, np.ndarray | None]] = []
            if rp.exists():
                cases.append(("real", decode(rp.read_bytes()), None))
            if synthetic:
                for d in TILTS:
                    img, _ = tilt(bimg, d)
                    m = draw_scratch(img, seed=SHOTS.index(shot) * 100 + d)
                    cases.append((f"tilt{d}", img, m))
            for case, timg, smask in cases:
                tkey = f"photos/target/{shot}-{case}.jpg"
                h.put(tkey, timg)
                for name, st, crop, asift, wide in methods(s):
                    out = h.run_pipeline(shot, bkey, bid, tkey, st, crop, asift, wide)
                    outs.append(pipeline_outcome(name, shot, case, out, smask))
                d = dino_global(h, bkey, bid, timg, s, smask, shot, case)
                outs.append(d)
                f = outs[-2]
                if not f.comparable:  # F: 겹치기가 안 되면 조각 맞추기(D)로 대신한다. "비교 불가"가 없다
                    outs[-2] = Outcome(
                        "F hybrid",
                        shot,
                        case,
                        True,
                        f"fallback:{f.reason}",
                        f.inliers,
                        f.reproj,
                        d.candidates,
                        d.hit,
                        f.ms + d.ms,
                        d.extra,
                    )
                print(".", end="", file=sys.stderr, flush=True)
        outs += best_match(h, dir_, s, outs)
        print(file=sys.stderr)
        return outs
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------- H: 기준 8장 전체를 기억 장소로 쓰는 조각 맞추기

SWEEP = (0.35, 0.45, 0.55, 0.65, 0.75)


def letterbox(img: np.ndarray, aspect: float) -> np.ndarray:
    """가로세로 비율을 aspect(W/H)로 맞추도록 검은 여백을 붙인다(늘이거나 찌그러뜨리지 않는다)."""
    h, w = img.shape[:2]
    if w / h > aspect:
        nh = round(w / aspect)
        top = (nh - h) // 2
        return cv2.copyMakeBorder(img, top, nh - h - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
    nw = round(h * aspect)
    left = (nw - w) // 2
    return cv2.copyMakeBorder(img, 0, 0, left, nw - w - left, cv2.BORDER_CONSTANT, value=0)


def tiles(img: np.ndarray, n: int = 2, overlap: float = 0.25):
    """n×n 조각(겹침 overlap). (x, y, 조각) 목록."""
    h, w = img.shape[:2]
    tw, th = int(w / (n - (n - 1) * overlap)), int(h / (n - (n - 1) * overlap))
    out = []
    for i in range(n):
        for j in range(n):
            x = int(j * tw * (1 - overlap))
            y = int(i * th * (1 - overlap))
            out.append((x, y, img[y : y + th, x : x + tw]))
    return out


def laptop_crop(h: Harness, img: np.ndarray, s: Settings):
    small, scale = resize_long_side(img, s.compare_long_side)
    box = laptop_box_from_gray(img)
    mask = h.models.segmenter.segment(small, None if box is None else tuple(round(v * scale) for v in box))
    x1, y1, x2, y2 = mask_crop(mask)
    return small, mask, (x1, y1, x2, y2)


def feats_lb(h: Harness, img: np.ndarray) -> np.ndarray:
    bw, bh = h.models.backbone.size
    return h.models.backbone.features(letterbox(img, bw / bh))


def build_bank(h: Harness, images: list[np.ndarray], s: Settings) -> np.ndarray:
    """기준 사진마다 노트북 크롭을 0·90·180·270도로 돌리고, 통째로 + 2×2 조각(가까이 찍은 것 대비)으로 특징을 모은다."""
    parts = []
    for img in images:
        small, _mask, (x1, y1, x2, y2) = laptop_crop(h, img, s)
        crop = small[y1:y2, x1:x2]
        for k in range(4):
            r = np.ascontiguousarray(np.rot90(crop, k))
            views = [r] + [t for _, _, t in tiles(r)]
            for v in views:
                f = feats_lb(h, v)
                parts.append(f.reshape(-1, f.shape[-1]))
    return np.vstack(parts).astype(np.float32)


def bank_score(h: Harness, bank: np.ndarray, target: np.ndarray, s: Settings):
    """반납 사진 노트북 영역의 이상 점수 맵(0~1). 통째로·2×2 조각 두 크기 중 더 잘 맞는(작은) 값을 쓴다."""
    small, mask, (x1, y1, x2, y2) = laptop_crop(h, target, s)
    crop = small[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    bw, bh = h.models.backbone.size

    def dist_map(view: np.ndarray) -> np.ndarray:
        f = feats_lb(h, view)
        gh, gw, c = f.shape
        best = (f.reshape(-1, c) @ bank.T).max(1).reshape(gh, gw)
        d = cv2.resize((1.0 - best).astype(np.float32), (bw, bh), interpolation=cv2.INTER_CUBIC)
        # 여백을 붙였던 만큼 잘라 원래 크롭 비율로 되돌린다
        vh, vw = view.shape[:2]
        sc = min(bw / vw, bh / vh)
        ox, oy = int((bw - vw * sc) / 2), int((bh - vh * sc) / 2)
        d = d[oy : oy + int(vh * sc), ox : ox + int(vw * sc)]
        return cv2.resize(d, (vw, vh), interpolation=cv2.INTER_LINEAR)

    whole = dist_map(crop)
    fine = np.full_like(whole, np.inf)
    for x, y, t in tiles(crop):
        d = dist_map(t)
        th, tw = d.shape
        fine[y : y + th, x : x + tw] = np.minimum(fine[y : y + th, x : x + tw], d)
    fine[~np.isfinite(fine)] = whole[~np.isfinite(fine)]
    dist = np.minimum(whole, fine)
    score = np.zeros(mask.shape, np.float32)
    score[y1:y2, x1:x2] = (dist / s.dino_scale).clip(0, 1)[:ch, :cw]
    valid = cv2.erode(mask, np.ones((7, 7), np.uint8))
    refl = specular(to_gray(small), np.zeros(mask.shape, np.uint8))
    return score, valid, refl, float(dist.mean())


def run_bank(dir_: Path, backbone: str, synthetic: bool) -> str:
    work = Path(tempfile.mkdtemp(prefix="eval-bank-"))
    try:
        h = Harness(backbone, work)
        s = Settings(storage_backend="local", local_storage_root=str(work), app_profile="test")
        bases = {
            sh: decode((dir_ / "baseline" / f"{sh}.jpg").read_bytes())
            for sh in SHOTS
            if (dir_ / "baseline" / f"{sh}.jpg").exists()
        }
        t = time.perf_counter()
        bank = build_bank(h, list(bases.values()), s)
        bank_ms = round((time.perf_counter() - t) * 1000)
        cases = []
        for sh in SHOTS:
            rp = dir_ / "return" / f"{sh}.jpg"
            if rp.exists():
                cases.append((sh, "real", decode(rp.read_bytes()), None))
            if synthetic and sh in bases:
                for d in TILTS:
                    img, _ = tilt(bases[sh], d)
                    cases.append((sh, f"tilt{d}", img, draw_scratch(img, seed=SHOTS.index(sh) * 100 + d)))
        rows = []
        for sh, case, img, smask in cases:
            t = time.perf_counter()
            score, valid, refl, mean = bank_score(h, bank, img, s)
            ms = round((time.perf_counter() - t) * 1000)
            sm = None
            if smask is not None:
                sm = cv2.resize(smask, (score.shape[1], score.shape[0]), interpolation=cv2.INTER_NEAREST)
            res = []
            for th in SWEEP:
                regs = extract_regions(score, valid, th, s.min_region_px, refl)
                hits = sum(1 for r in regs if sm is not None and ((r.mask > 0) & (sm > 0)).any())
                res.append((len(regs) - hits, hits > 0))
            rows.append((sh, case, mean, ms, res))
            print("h", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        head = " | ".join(f"가짜@{t}" for t in SWEEP)
        lines = [
            f"## H bank: 기준 {len(bases)}장 × 4방향 × 2크기 기억 장소({len(bank)}조각, {bank_ms}ms)",
            "",
            f"| 컷 | 경우 | 평균 거리 | ms | {head} |",
            "|---|---|---|---|" + "---|" * len(SWEEP),
        ]
        for sh, case, mean, ms, res in rows:
            if case == "real":
                lines.append(
                    f"| {sh} | real | {mean:.3f} | {ms} | " + " | ".join(str(c) for c, _ in res) + " |"
                )
        if synthetic:
            lines += [
                "",
                "| 기울기 | " + " | ".join(f"찾음·가짜@{t}" for t in SWEEP) + " |",
                "|---|" + "---|" * len(SWEEP),
            ]
            for d in TILTS:
                g = [r for r in rows if r[1] == f"tilt{d}"]
                cells = []
                for i in range(len(SWEEP)):
                    hit = sum(r[4][i][1] for r in g)
                    fp = sum(r[4][i][0] for r in g) / max(1, len(g))
                    cells.append(f"{hit}/{len(g)} · {fp:.1f}")
                lines.append(f"| {d}° | " + " | ".join(cells) + " |")
        tot = [
            "",
            "실제 컷 가짜 후보 합계: "
            + ", ".join(
                f"@{t}={sum(r[4][i][0] for r in rows if r[1] == 'real')}" for i, t in enumerate(SWEEP)
            ),
        ]
        return "\n".join(lines + tot)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def report(outs: list[Outcome]) -> str:
    lines = [
        "## 실제 반납 사진",
        "",
        "| 컷 | 방식 | 비교 | 이유 | 짝(inliers) | 오차px | 가짜 후보 | 크기·회전·원근 | ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for o in outs:
        if o.case != "real":
            continue
        lines.append(
            f"| {o.shot} | {o.method} | {'성공' if o.comparable else '불가'} | {o.reason or '-'} | "
            f"{o.inliers if o.inliers is not None else '-'} | {o.reproj if o.reproj is not None else '-'} | "
            f"{o.candidates if o.comparable else '-'} | {o.extra} | {o.ms} |"
        )
    syn = [o for o in outs if o.case.startswith("tilt")]
    if syn:
        lines += [
            "",
            "## 합성 각도 시험(기준 사진을 기울이고 스크래치 1개)",
            "",
            "| 방식 | 기울기 | 비교 성공 | 스크래치 찾음 | 가짜 후보(평균) |",
            "|---|---|---|---|---|",
        ]
        for m in sorted({o.method for o in syn}):
            for d in TILTS:
                g = [o for o in syn if o.method == m and o.case == f"tilt{d}"]
                if not g:
                    continue
                ok = sum(o.comparable for o in g)
                hit = sum(bool(o.hit) for o in g)
                fp = [o.candidates for o in g if o.comparable]
                lines.append(
                    f"| {m} | {d}° | {ok}/{len(g)} | {hit}/{len(g)} | "
                    f"{(sum(fp) / len(fp)) if fp else float('nan'):.1f} |"
                )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--bank", action="store_true", help="H(기준 전체 기억 장소)만 임계값별로 돌린다")
    a = ap.parse_args()
    if a.bank:
        print(run_bank(Path(a.dir), a.backbone, a.synthetic))
        return
    outs = run(Path(a.dir), a.backbone, a.synthetic)
    print(report(outs))
    if a.json:
        Path(a.json).write_text(json.dumps([o.__dict__ for o in outs], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
