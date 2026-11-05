"""ML 작업 대기열(api `ml_job`) 클라이언트와 서명 URL 저장소 (docs specs/003-gpu-worker).

작업자는 두 종류이고 같은 코드로 api에 붙는다. 다른 것은 주소와 인증 헤더뿐이다.

| 작업자 | 경로 | 인증 | 파일 읽기·쓰기 |
|--------|------|------|----------------|
| 클러스터 ml 파드(CPU) | `/internal/jobs/**` | `X-Internal-Token` | MinIO 직접(키 있음) |
| 집 AI PC GPU 워커 | `/api/v1/worker/jobs/**` | `Authorization: Bearer <워커 토큰>` | 작업에 붙어 온 서명 URL만(`PresignedStorage`) |

api는 작업이 없으면 본문 없는 204를 준다. 그러면 `claim()`은 None을 돌려준다.
"""

from __future__ import annotations

import httpx

from common.storage import ObjectNotFound, StorageError


class WorkerAuthError(RuntimeError):
    """토큰이 없거나 만료(401), 또는 권한 없음(403). 재시도해도 소용없으니 워커를 멈추고 다시 login 해야 한다."""


class ApiError(RuntimeError):
    pass


class JobsClient:
    def __init__(
        self,
        base_url: str,
        prefix: str,
        headers: dict[str, str],
        timeout: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.prefix = prefix
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    @classmethod
    def internal(cls, base_url: str, token: str, **kw) -> JobsClient:
        """클러스터 ml 파드용. api는 만든 지 2분 지난 작업만 준다(GPU 워커에게 먼저 양보)."""
        return cls(base_url, "/internal/jobs", {"X-Internal-Token": token}, **kw)

    @classmethod
    def worker(cls, base_url: str, token: str, **kw) -> JobsClient:
        """집 AI PC GPU 워커용. base_url은 Ingress 주소(예: https://notebook-rental.java21.net)."""
        return cls(base_url, "/api/v1/worker/jobs", {"Authorization": f"Bearer {token}"}, **kw)

    def _post(self, path: str, body: dict) -> httpx.Response:
        r = self._client.post(self.prefix + path, json=body)
        if r.status_code in (401, 403):
            raise WorkerAuthError(f"api 인증 실패({r.status_code}). 토큰이 만료됐으면 login을 다시 하세요")
        return r

    @staticmethod
    def _result(r: httpx.Response) -> dict:
        try:
            body = r.json()
        except ValueError as e:
            raise ApiError(f"api 응답이 JSON이 아닙니다(HTTP {r.status_code})") from e
        header = body.get("header") or {}
        if r.status_code >= 400 or not header.get("isSuccessful", False):
            raise ApiError(
                f"api 오류 HTTP {r.status_code} {header.get('resultCode')} {header.get('resultMessage')}"
            )
        return body.get("result") or {}

    def claim(self, types: list[str], worker_id: str) -> dict | None:
        r = self._post("/claim", {"types": types, "workerId": worker_id})
        if r.status_code == 204:
            return None
        return self._result(r)

    def complete(self, job_id: int, worker_id: str, result: dict) -> None:
        self._result(self._post(f"/{job_id}/complete", {"workerId": worker_id, "result": result}))

    def fail(self, job_id: int, worker_id: str, error: str) -> str:
        return self._result(
            self._post(f"/{job_id}/fail", {"workerId": worker_id, "error": error[:2000]})
        ).get("status", "")

    def close(self) -> None:
        self._client.close()


class PresignedStorage:
    """작업에 붙어 온 서명 URL로만 읽고 쓰는 저장소. `common.storage.Storage`와 같은 모양이다.

    AI PC에는 MinIO 키가 없다. api가 그 작업이 읽을 키(`urls.get`)와 쓸 키(`urls.put`)의 서명 URL만 주므로,
    목록에 없는 키는 읽거나 쓸 수 없다(`StorageError`).
    쓰기 서명은 `Content-Type: application/octet-stream`으로 만들어졌다. 다른 값을 보내면 MinIO가 서명 불일치(403)로 거절한다.
    """

    CONTENT_TYPE = "application/octet-stream"

    def __init__(self, urls: dict, client: httpx.Client | None = None):
        self.get_urls: dict[str, str] = dict((urls or {}).get("get") or {})
        self.put_urls: dict[str, str] = dict((urls or {}).get("put") or {})
        self._client = client or httpx.Client(timeout=httpx.Timeout(30.0, read=120.0))

    def get_bytes(self, key: str) -> bytes:
        url = self.get_urls.get(key)
        if url is None:
            raise StorageError(f"이 작업에서 읽을 수 없는 키: {key}")
        r = self._client.get(url)
        if r.status_code == 404:
            raise ObjectNotFound(key)
        r.raise_for_status()
        return r.content

    def put_bytes(self, key: str, data: bytes, content_type: str = CONTENT_TYPE) -> None:
        url = self.put_urls.get(key)
        if url is None:
            raise StorageError(f"이 작업에서 쓸 수 없는 키: {key}")
        r = self._client.put(url, content=data, headers={"Content-Type": self.CONTENT_TYPE})
        r.raise_for_status()

    def exists(self, key: str) -> bool:
        try:
            self.get_bytes(key)
            return True
        except ObjectNotFound:
            return False
