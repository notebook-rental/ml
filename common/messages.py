"""업무 API(Spring) ↔ ML 서비스 사이에 오가는 요청·응답 본문.

계약 문서: notebook-rental/docs specs/001-return-damage-inspection/contracts/ml-service.md

- 필드 이름은 JSON 그대로 camelCase를 쓴다(Spring Jackson 기본값과 맞춤).
- ID는 api가 JSON number로 보낸다(BIGINT). 문자열로 와도 받아 준다.
- 모르는 필드는 무시한다(필드 추가는 하위 호환, 계약 5절).
- 좌표는 항상 원본 사진 픽셀 좌표다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

ShotType = Literal["top", "bottom", "left", "right", "hinge", "front", "keyboard", "screen_off"]
DamageType = Literal["scratch", "dent", "crack", "sticker", "stain", "missing_part", "qr_tampered", "other"]
QualityReason = Literal["blurry", "too_dark", "too_bright", "out_of_frame", "qr_not_found", "multiple_qr"]
Id = int | str


class Message(BaseModel):
    """모든 본문의 부모. 모르는 필드는 버리고, 이름 그대로(camelCase) 읽고 쓴다."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class QrCode(Message):
    value: str
    corners: list[list[float]] = Field(default_factory=list, description="네 꼭짓점 [[x,y]×4], 원본 픽셀")


class Geometry(Message):
    type: Literal["polygon", "box"] = "polygon"
    coords: list[list[float]] = Field(default_factory=list)


# ---------- /quality ----------


class QualityRequest(Message):
    schemaVersion: int = SCHEMA_VERSION
    imageKey: str
    shotType: ShotType
    deviceModelId: Id | None = None


class QualityScores(Message):
    blur: float
    exposure: float
    coverage: float


class QualityResult(Message):
    passed: bool = Field(serialization_alias="pass", validation_alias="pass")
    reasons: list[QualityReason]
    scores: QualityScores
    laptopBox: list[int] | None
    qr: list[QrCode]


# ---------- /baseline-features, /prefetch ----------


class BaselineFeaturesRequest(Message):
    schemaVersion: int = SCHEMA_VERSION
    photoId: Id
    imageKey: str
    shotType: ShotType
    qr: list[QrCode] = Field(default_factory=list)


class PrefetchRequest(Message):
    schemaVersion: int = SCHEMA_VERSION
    deviceId: Id | None = None
    baselinePhotoIds: list[Id]
    backbone: str | None = None


# ---------- /compare ----------


class ModelVersionRef(Message):
    id: Id | None = None
    name: str = "changedet-v1"
    artifactKey: str | None = None
    artifactSha256: str | None = None


class ExistingDamage(Message):
    id: Id
    geometry: Geometry


class CompareParams(Message):
    threshold: float | None = None


class BankPhoto(Message):
    """조각 맞추기 기억 장소에 넣을 기준 사진 하나(docs specs/004). 보통 그 노트북의 현재 기준 사진 전부."""

    photoId: Id
    imageKey: str
    shotType: str | None = None


class CompareRequest(Message):
    schemaVersion: int = SCHEMA_VERSION
    comparisonId: Id
    mode: Literal["production", "shadow"] = "production"
    modelVersion: ModelVersionRef = Field(default_factory=ModelVersionRef)
    shotType: ShotType
    deviceId: Id | None = None
    baselinePhotoId: Id
    baselineImageKey: str
    targetImageKey: str
    existingDamages: list[ExistingDamage] = Field(default_factory=list)
    baselineQr: list[QrCode] = Field(default_factory=list)
    params: CompareParams = Field(default_factory=CompareParams)
    bankPhotos: list[BankPhoto] = Field(
        default_factory=list, description="비우면 baselinePhotoId 한 장만 기억 장소로 쓴다(옛 api와 호환)"
    )


class Alignment(Message):
    homography: list[list[float]]
    reprojErrorPx: float
    inliers: int = 0


class Finding(Message):
    geometry: Geometry = Field(description="반납(target) 원본 사진 픽셀 좌표")
    baselineGeometry: Geometry | None = Field(default=None, description="대여(baseline) 원본 사진 픽셀 좌표")
    predictedType: DamageType
    confidence: float
    overlapsDamageId: Id | None = None


class CompareResult(Message):
    status: Literal["done", "not_comparable", "failed"]
    reason: str | None = None
    error: str | None = None
    alignment: Alignment | None = None
    derivedKeys: dict[str, str] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    runtimeMs: int = 0
    timingsMs: dict[str, int] = Field(default_factory=dict)
    modelVersionId: Id | None = None
    method: Literal["aligned", "patch"] | None = Field(
        default=None,
        description="aligned: 정확히 겹쳐 대여 좌표도 있음, patch: 조각 맞추기만(대여 좌표 없음)",
    )
    matchedBaselinePhotoId: Id | None = Field(default=None, description="반납 사진과 가장 닮은 기준 사진")


# ---------- /models/reload, GET /internal/models/active ----------


class ActiveModel(Message):
    id: Id | None = None
    name: str
    kind: str = "zero_shot_change"
    artifactKey: str | None = None
    artifactSha256: str | None = None
    inferenceProfileKey: str | None = None


class ActiveModels(Message):
    schemaVersion: int = SCHEMA_VERSION
    production: ActiveModel | None = None
    shadow: ActiveModel | None = None
