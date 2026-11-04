"""ML 추론 서비스 HTTP API (FastAPI). 계약: docs contracts/ml-service.md

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET  | /health/live | 프로세스가 살아 있는지(liveness) |
| GET  | /health/ready | 모델 로드·워밍업이 끝났는지(readiness). 끝나기 전에는 503 |
| POST | /quality | 촬영 품질 검사(동기, 3초 이내) |
| POST | /baseline-features | 대여 사진 기준 특징 사전 계산(202, 뒤에서 계산) |
| POST | /prefetch | 반납 스캔 순간 기준 특징 8개를 메모리에 올림(202) |
| POST | /compare | 컷 하나 비교(동기). 동시 처리 상한을 넘으면 503 |
| POST | /models/reload | 모델 교체 알림(본문은 GET /internal/models/active와 같음) |

모든 응답은 공통 형식(header/result)이다. 클러스터 안에서만 열리고(Ingress 미등록),
INTERNAL_TOKEN이 설정되어 있으면 POST 요청에 헤더 X-Internal-Token이 맞아야 한다.

실행: `uvicorn --factory inference.app:default_app --host 0.0.0.0 --port 8000`
CPU를 많이 쓰는 처리는 동기 함수(def)로 두어 FastAPI가 스레드 풀에서 돌리게 한다. ONNX Runtime·OpenCV는
계산 중 GIL을 풀기 때문에 스레드로도 여러 요청이 동시에 돈다.
"""

from __future__ import annotations

import hmac
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from common.callback import SpringClient
from common.messages import (
    ActiveModels,
    BaselineFeaturesRequest,
    CompareRequest,
    PrefetchRequest,
    QualityRequest,
)
from common.storage import LocalStorage, ObjectNotFound, S3Storage, Storage
from inference import compare as compare_mod
from inference.config import Settings, get_settings
from inference.model_registry import ActiveModelPoller, ModelRegistry
from inference.pipeline.baseline_cache import BaselineStore, cache_key
from inference.pipeline.imageutil import ImageDecodeError, decode
from inference.quality import check_quality
from inference.response import ResultCode, fail, ok

log = logging.getLogger("inference")


def make_storage(s: Settings) -> Storage:
    if s.storage_backend == "local":
        return LocalStorage(s.local_storage_root)
    return S3Storage(
        s.minio_endpoint, s.minio_access_key, s.minio_secret_key, s.minio_bucket
    )  # pragma: no cover


class Service:
    """앱 하나가 들고 있는 상태(저장소, 모델, 캐시, 동시 처리 상한)."""

    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        registry: ModelRegistry | None = None,
        spring: SpringClient | None = None,
    ):
        self.settings = settings
        self.storage = storage
        self.registry = registry or ModelRegistry(storage, settings)
        self.baselines = BaselineStore(storage, settings.baseline_lru_size)
        self.compare_slots = threading.BoundedSemaphore(settings.max_concurrent_compares)
        # 사전 계산·prefetch는 비교보다 덜 급하다. 워커 1개로 CPU를 비교에 양보한다.
        self.background = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bg")
        self.spring = spring
        self.poller: ActiveModelPoller | None = None

    def start(self, load_models: bool = True, poll: bool = True) -> None:
        if load_models and not self.registry.ready:
            self.registry.load_initial()
        if self.spring is not None:
            self.poller = ActiveModelPoller(
                self.registry, self.spring, self.settings.active_model_poll_seconds
            )
            self.poller.poll_once()
            if poll:  # pragma: no cover - 실제 주기 확인은 운영에서
                self.poller.start()

    def stop(self) -> None:
        if self.poller:
            self.poller.stop()
        self.background.shutdown(wait=False, cancel_futures=True)

    # ---- 뒤에서 하는 일 ----

    def precompute(self, req: BaselineFeaturesRequest) -> None:
        m = self.registry.models
        try:
            self.baselines.get_or_compute(
                req.photoId, req.imageKey, m.backbone, m.segmenter, self.settings.compare_long_side, req.qr
            )
        except Exception:
            log.exception("기준 특징 계산 실패 photoId=%s", req.photoId)

    def prefetch(self, req: PrefetchRequest) -> None:
        name = self.registry.models.backbone.name
        for pid in req.baselinePhotoIds:
            try:
                self.baselines.load(pid, name)
            except Exception:  # 못 올려도 비교 때 저장소에서 읽으므로 결과는 같다
                log.exception("prefetch 실패 photoId=%s", pid)


def create_app(
    settings: Settings | None = None,
    storage: Storage | None = None,
    registry: ModelRegistry | None = None,
    spring: SpringClient | None = None,
    load_models: bool = True,
) -> FastAPI:
    settings = settings or get_settings()
    storage = storage or make_storage(settings)
    if spring is None and settings.internal_token and settings.app_profile != "test":
        spring = SpringClient(settings.api_base_url, settings.internal_token)  # pragma: no cover
    svc = Service(settings, storage, registry, spring)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # 모델 로드는 뒤에서. 그동안 liveness는 통과, readiness는 503
        t = threading.Thread(target=svc.start, kwargs={"load_models": load_models}, daemon=True)
        t.start()
        yield
        svc.stop()

    app = FastAPI(title="notebook-rental ml-inference", version="0.1.0", lifespan=lifespan)
    app.state.svc = svc

    @app.middleware("http")
    async def internal_token(request: Request, call_next):
        token = settings.internal_token
        if token and request.method == "POST":
            given = request.headers.get("X-Internal-Token", "")
            if not hmac.compare_digest(given.encode(), token.encode()):
                return fail(ResultCode.UNAUTHORIZED)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def on_validation(_req, exc: RequestValidationError):
        errors = [
            {"field": ".".join(str(p) for p in e["loc"][1:]), "reason": e["type"]} for e in exc.errors()
        ]
        return fail(ResultCode.INVALID_INPUT, {"errors": errors})

    @app.exception_handler(ValidationError)
    async def on_model_validation(_req, exc: ValidationError):
        return fail(ResultCode.INVALID_INPUT, {"errors": [{"field": "body", "reason": "invalid"}]})

    @app.exception_handler(Exception)
    async def on_error(_req, exc: Exception):  # pragma: no cover - 마지막 안전망
        log.exception("처리하지 못한 오류")
        return fail(ResultCode.SERVER_ERROR)

    @app.get("/health/live")
    def live():
        return ok({"status": "UP"})

    @app.get("/health/ready")
    def ready():
        if not svc.registry.ready:
            return fail(ResultCode.NOT_READY)
        m = svc.registry.models
        return ok(
            {
                "status": "UP",
                "backbone": m.backbone.name,
                "segmenter": m.segmenter.name,
                "baselineCache": {
                    "size": len(svc.baselines._lru),
                    "hits": svc.baselines.hits,
                    "misses": svc.baselines.misses,
                },
            }
        )

    @app.post("/quality")
    def quality(req: QualityRequest):
        try:
            img = decode(storage.get_bytes(req.imageKey))
        except ObjectNotFound:
            return fail(ResultCode.NOT_FOUND, {"imageKey": req.imageKey})
        except ImageDecodeError:
            return fail(
                ResultCode.INVALID_INPUT, {"errors": [{"field": "imageKey", "reason": "not_an_image"}]}
            )
        return ok(check_quality(img, req.shotType, settings))

    @app.post("/baseline-features")
    def baseline_features(req: BaselineFeaturesRequest):
        if not svc.registry.ready:
            return fail(ResultCode.NOT_READY)
        svc.background.submit(svc.precompute, req)
        return ok({"featuresKey": cache_key(req.photoId)}, ResultCode.ACCEPTED)

    @app.post("/prefetch")
    def prefetch(req: PrefetchRequest):
        if not svc.registry.ready:
            return fail(ResultCode.NOT_READY)
        svc.background.submit(svc.prefetch, req)
        return ok(None, ResultCode.ACCEPTED)

    @app.post("/compare")
    def compare(req: CompareRequest):
        if not svc.registry.ready:
            return fail(ResultCode.NOT_READY)
        if not svc.compare_slots.acquire(blocking=False):
            return fail(ResultCode.BUSY)
        try:
            result = compare_mod.compare(req, storage, svc.baselines, svc.registry.models, settings)
        finally:
            svc.compare_slots.release()
        return ok(result)

    @app.post("/models/reload")
    def reload(body: ActiveModels):
        changed = svc.registry.apply(body)
        return ok({"changed": changed})

    return app


def default_app() -> FastAPI:  # pragma: no cover - uvicorn --factory 진입점
    """`uvicorn --factory inference.app:default_app`로 실행한다. 설정은 환경변수에서 읽는다."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return create_app()
