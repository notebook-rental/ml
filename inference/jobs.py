"""ML 작업 처리기와 실행 루프 (docs specs/003-gpu-worker, tasks T311·T319).

같은 코드를 두 곳에서 쓴다.

- 집 AI PC GPU 워커(`python -m worker run`): 엔진 `cuda`, 파일은 서명 URL(`PresignedStorage`)
- 클러스터 ml 파드: 엔진 CPU, 파일은 MinIO 직접. GPU 워커가 꺼져 있을 때 2분 뒤 대신 처리한다

결과 파일(기준 특징 npz) 형식이 같아야 ml 파드가 GPU 워커가 만든 파일을 그대로 읽는다. 그래서 처리 코드를 하나로 둔다.

작업 종류:
- `selftest`: 백본을 몇 번 돌려 실행 공급자(CUDA인지 CPU인지)와 시간을 잰다. GPU가 정말 쓰이는지 확인하는 용도
- `baseline_features`: 기준 사진 하나의 특징(노트북 영역, 특징점, DINOv2 패치 특징)을 계산해 npz로 저장
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass

import httpx
import numpy as np

from common.jobs_client import JobsClient, PresignedStorage, WorkerAuthError
from common.messages import QrCode
from common.storage import ObjectNotFound, Storage
from inference.config import Settings
from inference.model_registry import Models
from inference.pipeline import baseline_cache
from inference.pipeline.imageutil import decode

log = logging.getLogger(__name__)

SELFTEST = "selftest"
BASELINE_FEATURES = "baseline_features"
TYPES = [SELFTEST, BASELINE_FEATURES]


class JobFailed(Exception):
    """작업 하나가 실패했다(사진 없음 등). 실패로 보고하고 다음 작업을 계속한다."""


@dataclass
class JobContext:
    models: Models
    settings: Settings
    storage: Storage  # 파드: MinIO, 워커: 작업마다 PresignedStorage로 바꿔 끼운다
    worker_id: str


def providers(models: Models) -> list[str]:
    get = getattr(models.backbone.session, "get_providers", None)
    return list(get()) if get else [type(models.backbone.session).__name__]


def gpu_name() -> str | None:
    """NVIDIA GPU 이름. nvidia-smi(드라이버와 같이 설치됨)가 없으면 None."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - 드라이버 문제
        return None


def run_selftest(job: dict, ctx: JobContext) -> dict:
    import onnxruntime as ort

    backbone = ctx.models.backbone
    runs = max(1, min(int(job.get("runs", 10)), 100))
    w, h = backbone.size
    img = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    backbone.features(img)  # 첫 실행: GPU는 여기서 커널을 준비하느라 느리다
    warmup_ms = (time.perf_counter() - t0) * 1000
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        backbone.features(img)
        times.append((time.perf_counter() - t) * 1000)
    return {
        "workerId": ctx.worker_id,
        "engine": ctx.settings.engine,
        "providers": providers(ctx.models),
        "device": gpu_name(),
        "backbone": backbone.name,
        "input": [w, h],
        "runs": runs,
        "warmupMs": round(warmup_ms, 1),
        "p50Ms": round(float(np.median(times)), 2),
        "minMs": round(float(np.min(times)), 2),
        "onnxruntime": ort.__version__,
    }


def run_baseline_features(job: dict, ctx: JobContext) -> dict:
    photo_id = job["photoId"]
    image_key = job["imageKey"]
    key = job.get("featuresKey") or baseline_cache.cache_key(photo_id)
    m = ctx.models
    t0 = time.perf_counter()
    base = {
        "workerId": ctx.worker_id,
        "featuresKey": key,
        "backbone": m.backbone.name,
        "providers": providers(m),
    }
    if not job.get("force", False):
        try:
            existing = baseline_cache.deserialize(ctx.storage.get_bytes(key))
            if existing.backbone == m.backbone.name:
                return {**base, "status": "skipped", "ms": round((time.perf_counter() - t0) * 1000, 1)}
        except (ObjectNotFound, ValueError, KeyError):
            pass  # 없거나 형식이 옛것이면 새로 계산
    try:
        img = decode(ctx.storage.get_bytes(image_key))
    except ObjectNotFound as e:
        raise JobFailed(f"사진이 없습니다: {image_key}") from e
    qr = [QrCode.model_validate(q) for q in job.get("qr") or []]
    f = baseline_cache.compute(photo_id, img, m.backbone, m.segmenter, ctx.settings.compare_long_side, qr)
    ctx.storage.put_bytes(key, baseline_cache.serialize(f))
    return {**base, "status": "computed", "ms": round((time.perf_counter() - t0) * 1000, 1)}


HANDLERS: dict[str, Callable[[dict, JobContext], dict]] = {
    SELFTEST: run_selftest,
    BASELINE_FEATURES: run_baseline_features,
}


class JobRunner:
    """가져가기 → 처리 → 완료/실패 보고를 반복한다.

    use_presigned=True(GPU 워커)이면 작업마다 붙어 온 서명 URL로 저장소를 바꿔 끼운다.
    """

    def __init__(
        self,
        client: JobsClient,
        context: Callable[[], JobContext],
        worker_id: str,
        types: list[str] | None = None,
        poll_seconds: float = 5.0,
        use_presigned: bool = False,
        files: httpx.Client | None = None,
    ):
        self.client = client
        self.context = context
        self.worker_id = worker_id
        self.types = types or TYPES
        self.poll_seconds = poll_seconds
        self.use_presigned = use_presigned
        # 서명 URL로 파일을 주고받을 HTTP 클라이언트(연결을 다시 쓰려고 하나를 계속 쓴다)
        self.files = files or (
            httpx.Client(timeout=httpx.Timeout(30.0, read=120.0)) if use_presigned else None
        )
        self.done = 0
        self.failed = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run_once(self) -> bool:
        """작업 하나를 처리했으면 True, 가져갈 작업이 없으면 False."""
        job = self.client.claim(self.types, self.worker_id)
        if job is None:
            return False
        job_id, jtype = job["jobId"], job["type"]
        log.info("작업 시작 jobId=%s type=%s attempts=%s", job_id, jtype, job.get("attempts"))
        try:
            handler = HANDLERS.get(jtype)
            if handler is None:
                raise JobFailed(f"이 작업자가 모르는 작업 종류: {jtype}")
            ctx = self.context()
            if self.use_presigned:
                ctx = JobContext(
                    ctx.models,
                    ctx.settings,
                    PresignedStorage(job.get("urls") or {}, self.files),
                    ctx.worker_id,
                )
            result = handler(job, ctx)
        except WorkerAuthError:
            raise
        except Exception as e:
            self.failed += 1
            msg = f"{type(e).__name__}: {e}"
            log.exception("작업 실패 jobId=%s type=%s", job_id, jtype)
            status = self.client.fail(job_id, self.worker_id, msg)
            log.info("실패 보고 jobId=%s → %s", job_id, status)
            return True
        self.client.complete(job_id, self.worker_id, result)
        self.done += 1
        log.info("작업 완료 jobId=%s type=%s %s", job_id, jtype, result.get("status", ""))
        return True

    def run_forever(self) -> None:
        """작업이 있으면 쉬지 않고, 없으면 poll_seconds마다 묻는다. 인증 실패면 멈춘다."""
        while not self._stop.is_set():
            try:
                if self.run_once():
                    continue
            except WorkerAuthError:
                log.error("api 인증 실패로 작업 루프를 멈춥니다. 토큰을 확인하세요")
                raise
            except Exception:  # 네트워크 끊김 등. 잠깐 쉬고 다시 묻는다
                log.exception("작업 가져가기 실패(%.0f초 뒤 다시)", self.poll_seconds)
            self._stop.wait(self.poll_seconds)

    def start(self) -> None:  # pragma: no cover - 스레드 루프는 run_once로 테스트
        def loop():
            with suppress(WorkerAuthError):  # 로그는 run_forever가 남긴다
                self.run_forever()

        self._thread = threading.Thread(target=loop, name="ml-job-runner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
