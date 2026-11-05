"""컷 1장 비교 속도 측정 (tasks T001, research R12 "개발 첫 주에 운영 CPU에서 실측").

무엇을 재나:
- 대여 사진 사전 계산(노트북 마스크 + SIFT + DINOv2) 시간 — 대여 때 뒤에서 하므로 직원은 기다리지 않는다
- 반납 컷 1장 비교 시간(`/compare` 본체), 단계별(정렬, DINOv2, SSIM, 후처리 …) 시간
- 동시에 여러 건이 들어올 때(동시 반납 3건 가정) 한 건당 시간

입력은 tests/fixtures/synth.py의 합성 장면을 작업본 크기(긴 변 2560px)로 키운 사진이다.
속도는 사진 내용이 아니라 크기·모델 구조로 거의 정해지므로 실제 사진과 큰 차이가 없다.

모델: --backbone에 ONNX 경로를 준다. 공식 DINOv2 가중치 파일이 없으면 tools/build_vit_onnx.py로 같은 구조의
무작위 가중치 모델을 만들어 쓴다(속도는 같고, 정확도는 의미 없음).

예:
    python tools/build_vit_onnx.py --out .data/models/vit-s14-448x336.onnx
    python -m bench.bench_compare --backbone .data/models/vit-s14-448x336.onnx --runs 20 \
        --threads 0 4 --long-side 1024 1280 --concurrency 1 3 --out bench-result.json

결과 표(Markdown)를 표준 출력으로, 원자료를 --out JSON으로 남긴다. 운영 서버에서 돌린 결과는
docs/benchmark.md에 붙여 넣는다.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.messages import CompareRequest  # noqa: E402
from common.storage import LocalStorage  # noqa: E402
from inference.compare import Models, compare  # noqa: E402
from inference.config import Settings  # noqa: E402
from inference.pipeline import baseline_cache as bc  # noqa: E402
from inference.pipeline.engine import create_session  # noqa: E402
from inference.pipeline.patch_features import Backbone  # noqa: E402
from inference.pipeline.qr import detect_qr  # noqa: E402
from inference.pipeline.segment import GrabCutSegmenter  # noqa: E402
from tests.fixtures import synth  # noqa: E402

WORK_LONG_SIDE = 2560  # 폰이 올리는 작업본(R15)


def make_photos(root: Path) -> list[str]:
    """기준 1장 + 반납 3장(스티커, 스크래치, 손상 없음)을 2560px로 만들어 저장소에 넣는다."""
    scene = synth.base_scene(0)
    scale = WORK_LONG_SIDE / synth.W

    def up(img):
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    base, _ = synth.view(scene.image, seed=1, shift=10)
    targets = [
        synth.add_damage(scene.image, synth.Damage("sticker", (500, 400, 620, 480))),
        synth.add_damage(scene.image, synth.Damage("scratch", (700, 700, 950, 760))),
        scene.image,
    ]
    keys = []
    views = [synth.view(t, seed=2 + j, shift=25, brightness=0.9)[0] for j, t in enumerate(targets)]
    for i, t in enumerate([base, *views]):
        key = f"derived/work/bench-{i}.jpg"
        p = root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(synth.jpeg(up(t), 90))
        keys.append(key)
    return keys


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(np.floor(k)), int(np.ceil(k))
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_config(
    backbone_path: str, threads: int, long_side: int, concurrency: int, runs: int, engine: str
) -> dict:
    root = Path(tempfile.mkdtemp(prefix="bench-"))
    storage = LocalStorage(root)
    keys = make_photos(root)
    settings = Settings(
        app_profile="test",
        storage_backend="local",
        compare_long_side=long_side,
        intra_op_threads=threads,
        engine=engine,
    )
    backbone = Backbone(create_session(backbone_path, engine, settings.threads()), "bench-backbone")
    backbone.warmup()
    models = Models(backbone, GrabCutSegmenter())

    base_img = cv2.imdecode(np.frombuffer(storage.get_bytes(keys[0]), np.uint8), cv2.IMREAD_COLOR)
    qr = detect_qr(base_img)
    # 사전 계산 시간(대여 시점)
    pre = []
    for _ in range(3):
        t = time.perf_counter()
        f = bc.compute(1, base_img, backbone, models.segmenter, long_side, qr)
        pre.append((time.perf_counter() - t) * 1000)
    store = bc.BaselineStore(storage, 8)
    store.save(f)

    def one(i: int) -> tuple[float, dict, str]:
        req = CompareRequest(
            comparisonId=i,
            shotType="top",
            baselinePhotoId=1,
            baselineImageKey=keys[0],
            targetImageKey=keys[1 + i % 3],
            baselineQr=qr,
        )
        t = time.perf_counter()
        r = compare(req, storage, store, models, settings)
        return (time.perf_counter() - t) * 1000, r.timingsMs, r.status

    one(0)  # 워밍업
    totals, stages, statuses = [], {}, []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for ms, timings, status in ex.map(one, range(runs)):
            totals.append(ms)
            statuses.append(status)
            for k, v in timings.items():
                stages.setdefault(k, []).append(v)
    return {
        "engine": engine,
        "threads": settings.threads(),
        "longSide": long_side,
        "concurrency": concurrency,
        "runs": runs,
        "backboneInput": list(backbone.size),
        "precomputeMs": {"p50": round(statistics.median(pre))},
        "compareMs": {
            "p50": round(pct(totals, 0.5)),
            "p95": round(pct(totals, 0.95)),
            "max": round(max(totals)),
        },
        "stagesP50Ms": {k: round(statistics.median(v)) for k, v in stages.items()},
        "statuses": {s: statuses.count(s) for s in set(statuses)},
    }


def table(results: list[dict]) -> str:
    stage_names = ["baseline", "decode", "qr", "align", "warp", "dino", "ssim", "postprocess", "store"]
    head = (
        "| 엔진 | 스레드 | 비교 해상도 | 동시 | 사전계산 p50 | 비교 p50 | 비교 p95 | "
        + " | ".join(stage_names)
        + " |"
    )
    sep = "|" + "---|" * (8 + len(stage_names))
    rows = [head, sep]
    for r in results:
        st = r["stagesP50Ms"]
        rows.append(
            f"| {r['engine']} | {r['threads']} | {r['longSide']} | {r['concurrency']} | "
            f"{r['precomputeMs']['p50']} | {r['compareMs']['p50']} | {r['compareMs']['p95']} | "
            + " | ".join(str(st.get(s, "-")) for s in stage_names)
            + " |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True, help="DINOv2-S ONNX 경로")
    ap.add_argument("--engine", nargs="+", default=["onnxruntime"], choices=["onnxruntime", "openvino"])
    ap.add_argument("--threads", nargs="+", type=int, default=[0], help="0 = CPU 수")
    ap.add_argument("--long-side", nargs="+", type=int, default=[1024])
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1])
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--out", help="결과 JSON 경로")
    a = ap.parse_args(argv)

    results = []
    for engine in a.engine:
        for th in a.threads:
            for ls in a.long_side:
                for cc in a.concurrency:
                    r = run_config(a.backbone, th, ls, cc, a.runs, engine)
                    results.append(r)
                    print(json.dumps(r, ensure_ascii=False), file=sys.stderr)
    env = {
        "cpu": platform.processor() or platform.machine(),
        "cpuCount": os.cpu_count(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
    }
    print(f"환경: {env}\n")
    print(table(results))
    if a.out:
        Path(a.out).write_text(json.dumps({"env": env, "results": results}, ensure_ascii=False, indent=2))
    return results


if __name__ == "__main__":
    main()
