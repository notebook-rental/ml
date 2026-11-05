"""ML 서비스 → 업무 API(Spring) 내부 API 클라이언트.

내부 API(`/internal/**`)는 클러스터 안에서만 열리고 헤더 `X-Internal-Token`으로 인증한다.
토큰 값은 환경변수 INTERNAL_TOKEN으로만 받는다(저장소는 공개이므로 코드·설정 파일에 넣지 않는다).
"""

from __future__ import annotations

import httpx

from common.messages import ActiveModels


class SpringClient:
    def __init__(
        self, base_url: str, token: str, timeout: float = 3.0, transport: httpx.BaseTransport | None = None
    ):
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-Internal-Token": token},
            timeout=timeout,
            transport=transport,
        )

    def active_models(self) -> ActiveModels:
        """현재 운영·시험 모델. 응답은 공통 형식(header/result)이다."""
        r = self._client.get("/internal/models/active")
        r.raise_for_status()
        body = r.json()
        header = body.get("header", {})
        if not header.get("isSuccessful", False):
            raise RuntimeError(
                f"활성 모델 조회 실패: {header.get('resultCode')} {header.get('resultMessage')}"
            )
        return ActiveModels.model_validate(body.get("result") or {})

    def close(self) -> None:
        self._client.close()
