"""모델 파일 내려받기·검증·로드, 무중단 교체 (contracts ml-service.md "모델 교체", tasks T113의 1단계 부분).

- 모델 파일은 컨테이너 이미지에 넣지 않는다. 시작할 때 저장소 models/에서 받아 로컬 캐시 폴더에 둔다.
- sha256을 알면 검증하고, 다르면 로드하지 않는다(엉뚱한 파일·손상된 파일 방지).
- 새 세션을 만들고 워밍업(빈 이미지로 한 번 추론)까지 끝난 뒤에 참조를 한 번에 바꾼다.
  처리 중이던 비교는 예전 세션으로 끝난다(Python 참조가 남아 있으므로).
- 1단계 운영 모델 `changedet-v1`(kind zero_shot_change)은 별도 파일 없이 DINOv2 백본을 쓴다.
  2단계 학습 모델(kind trained_segmentation)은 아직 이 서비스가 실행하지 않는다(US5 범위).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

from common.messages import ActiveModels
from common.storage import ObjectNotFound, Storage
from inference.compare import Models
from inference.config import Settings
from inference.pipeline.engine import create_session
from inference.pipeline.patch_features import Backbone
from inference.pipeline.segment import GrabCutSegmenter, SamSegmenter

log = logging.getLogger(__name__)


class ModelIntegrityError(Exception):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelRegistry:
    def __init__(self, storage: Storage, settings: Settings):
        self.storage = storage
        self.settings = settings
        self.cache_dir = Path(settings.model_cache_dir)
        self._models: Models | None = None
        self._active: ActiveModels | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._models is not None

    @property
    def models(self) -> Models:
        if self._models is None:
            raise RuntimeError("모델이 아직 로드되지 않았습니다")
        return self._models

    @property
    def active(self) -> ActiveModels | None:
        return self._active

    def fetch(self, key: str, sha256: str | None = None) -> Path:
        """저장소 키 → 로컬 캐시 파일. 이미 있고 해시가 맞으면 다시 받지 않는다."""
        local = self.cache_dir / key
        if local.is_file() and (not sha256 or sha256_file(local) == sha256):
            return local
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(local.suffix + ".part")
        log.info("모델 파일 내려받는 중 key=%s → %s", key, local)
        seeded = False
        try:
            data = self.storage.get_bytes(key)
        except ObjectNotFound:
            url = self.settings.backbone_url if key == self.settings.backbone_key else ""
            if not url or not sha256:
                raise
            log.warning("버킷에 %s 가 없습니다. 공개 주소에서 받습니다: %s", key, url)
            data = self.download_url(url)
            seeded = True
        log.info("모델 파일 받음 key=%s (%.1f MB)", key, len(data) / 1e6)
        tmp.write_bytes(data)
        if sha256 and sha256_file(tmp) != sha256:
            tmp.unlink(missing_ok=True)
            raise ModelIntegrityError(f"sha256 불일치: {key}")
        tmp.replace(local)
        if seeded:
            self._seed(key, data)
        return local

    def download_url(self, url: str) -> bytes:  # pragma: no cover - 실제 인터넷 다운로드는 운영에서
        import httpx

        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0)) as c:
            r = c.get(url)
            r.raise_for_status()
            return r.content

    def _seed(self, key: str, data: bytes) -> None:
        """받은 모델(해시 확인 끝)을 버킷에도 넣어 둔다. 실패해도 로컬 캐시로 계속 동작한다."""
        put = getattr(self.storage, "put_new_model", None)
        if put is None:
            return
        try:
            if put(key, data):
                log.info("버킷에 모델 파일을 채웠습니다 key=%s", key)
        except Exception as e:
            log.warning("버킷에 모델 파일을 채우지 못했습니다(로컬 캐시로 계속) key=%s: %s", key, e)

    def _session(self, key: str, sha256: str | None = None):
        s = self.settings
        return create_session(self.fetch(key, sha256), s.engine, s.threads())

    def build(self, model_version_id=None) -> Models:
        s = self.settings
        size = None
        if s.backbone_input:
            w, h = (int(v) for v in s.backbone_input.lower().split("x"))
            size = (w, h)
        backbone = Backbone(self._session(s.backbone_key, s.backbone_sha256 or None), s.backbone_name, size)
        backbone.warmup()
        if s.sam_encoder_key and s.sam_decoder_key:
            segmenter = SamSegmenter(self._session(s.sam_encoder_key), self._session(s.sam_decoder_key))
        else:
            segmenter = GrabCutSegmenter()
        return Models(backbone=backbone, segmenter=segmenter, model_version_id=model_version_id)

    def load_initial(self) -> None:
        log.info(
            "모델 로드 시작 backbone=%s key=%s cache=%s",
            self.settings.backbone_name,
            self.settings.backbone_key,
            self.cache_dir,
        )
        with self._lock:
            self._models = self.build()
        log.info(
            "모델 로드 완료 backbone=%s segmenter=%s", self._models.backbone.name, self._models.segmenter.name
        )

    def apply(self, active: ActiveModels) -> bool:
        """활성 모델 정보를 반영한다. 바뀐 것이 있으면 True."""
        prod = active.production
        if prod is None:
            log.warning("운영 모델이 없습니다(resultCode -6005). 현재 모델을 유지합니다")
            return False
        if (
            self._active is not None
            and self._active.production == prod
            and self._active.shadow == active.shadow
        ):
            return False
        if prod.kind != "zero_shot_change":
            log.warning(
                "아직 지원하지 않는 모델 종류 %s(%s). 1단계 변화 탐지를 계속 씁니다", prod.kind, prod.name
            )
        if prod.artifactKey and prod.artifactSha256:
            # 2단계 모델 파일을 미리 받아 해시를 확인해 둔다(실행은 US5에서)
            self.fetch(prod.artifactKey, prod.artifactSha256)
        with self._lock:
            new = self.build(model_version_id=prod.id)
            self._models = new  # 원자적 교체
            self._active = active
        log.info(
            "모델 교체 완료 production=%s shadow=%s", prod.name, active.shadow.name if active.shadow else None
        )
        return True


class ActiveModelPoller:
    """30초마다 GET /internal/models/active로 놓친 교체 알림을 보완한다."""

    def __init__(self, registry: ModelRegistry, client, interval: float):
        self.registry = registry
        self.client = client
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def poll_once(self) -> bool:
        try:
            return self.registry.apply(self.client.active_models())
        except Exception as e:  # 네트워크 오류 등은 다음 주기에 다시
            log.warning("활성 모델 확인 실패: %s", e)
            return False

    def _run(self) -> None:  # pragma: no cover - 스레드 루프는 poll_once로 테스트
        while not self._stop.wait(self.interval):
            self.poll_once()

    def start(self) -> None:  # pragma: no cover
        self._thread = threading.Thread(target=self._run, name="active-model-poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
