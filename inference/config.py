"""설정. 프로파일은 APP_PROFILE=local|prod (Spring의 local/prod와 같은 구분).

값을 정하는 순서(뒤가 앞을 덮어쓴다):
1. 이 파일의 기본값
2. inference/config/<APP_PROFILE>.env  (비밀값이 아닌 값만. 저장소가 공개라서 비밀값은 넣지 않는다)
3. 환경변수  (운영에서는 Kubernetes Secret `notebook-rental-secrets` → envFrom)

비밀값(MINIO_ACCESS_KEY, MINIO_SECRET_KEY, INTERNAL_TOKEN)은 환경변수로만 받는다.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path(__file__).parent / "config"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file_encoding="utf-8", extra="ignore")

    app_profile: Literal["local", "prod", "test"] = "local"

    # 저장소: s3(MinIO) 또는 local(디스크 폴더, 테스트·벤치마크용)
    storage_backend: Literal["s3", "local"] = "s3"
    local_storage_root: str = "./.data/storage"
    minio_endpoint: str = "http://localhost:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "notebook-rental"

    # 업무 API(Spring) 내부 API
    api_base_url: str = "http://localhost:8080"
    internal_token: str = ""
    active_model_poll_seconds: int = 30
    # 작업 대기열(ml_job)을 몇 초마다 볼지. 0이면 보지 않는다. 운영(prod.env)은 10초.
    # api는 파드에 만든 지 2분 지난 작업만 준다(집 AI PC GPU 워커가 먼저, docs specs/003-gpu-worker)
    job_poll_seconds: float = 0

    # 모델 파일: 저장소 키 → 로컬 캐시 폴더에 내려받아 쓴다. 이미지에 모델을 넣지 않는다.
    model_cache_dir: str = "./.data/models"
    backbone_name: str = "dinov2-s14@1"
    backbone_key: str = "models/dinov2-s14/model.onnx"
    backbone_sha256: str = ""
    # 버킷에 백본 파일이 없을 때 받아 올 공개 주소(ml repo Release, export-model 워크플로가 만든다).
    # 받은 파일은 sha256을 확인하고 버킷 models/ 에 한 번 채운다. 비우면 이 기능을 쓰지 않는다.
    backbone_url: str = ""
    sam_encoder_key: str = ""  # 비우면 고전 방식(GrabCut) 노트북 영역 분리
    sam_decoder_key: str = ""

    # 추론 튜닝 값(inference-profile.json으로 덮어쓸 수 있다)
    engine: Literal["onnxruntime", "cuda", "openvino"] = "onnxruntime"  # cuda: NVIDIA GPU(onnxruntime-gpu)
    intra_op_threads: int = 0  # 0이면 파드 CPU 수
    max_concurrent_compares: int = 2  # 넘으면 503
    compare_long_side: int = 1024  # 정렬·DINOv2 비교 해상도(긴 변)
    neighbor_radius: int = 2  # 패치 최근접 탐색 반경(패치 단위)
    threshold: float = 0.35  # 결합 점수 임계값(0~1)
    min_region_px: int = 60  # compare 해상도에서 최소 후보 면적
    dino_scale: float = 0.5  # 패치 코사인 거리를 0~1로 바꿀 때 나누는 값
    ssim_gain: float = 1.6  # SSIM 차분에 곱하는 값(가는 스크래치 민감도)
    backbone_input: str = ""  # "448x336"처럼 주면 그 크기로(입력 크기가 고정이 아닌 모델일 때)
    max_reproj_error_px: float = 8.0
    min_inliers: int = 25
    baseline_lru_size: int = 240  # 기준 특징 메모리 캐시(사진 수). 노트북 30대 × 8컷

    # 품질 검사 임계값(초기 평가 세트로 보정한다, research R5)
    quality_blur_min: float = 60.0
    quality_dark_max: float = 0.18
    quality_bright_min: float = 0.88
    quality_coverage_min: float = 0.25

    def threads(self) -> int:
        return self.intra_op_threads or (os.cpu_count() or 1)


def _profile_env_file(profile: str) -> Path | None:
    p = CONFIG_DIR / f"{profile}.env"
    return p if p.is_file() else None


@lru_cache
def get_settings() -> Settings:
    profile = os.environ.get("APP_PROFILE", "local")
    env_file = _profile_env_file(profile)
    return Settings(_env_file=env_file) if env_file else Settings()
