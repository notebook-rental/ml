"""테스트용 가짜 api + 가짜 MinIO(서명 URL). httpx MockTransport로 실제 네트워크 없이 작업 대기열 한 바퀴를 흉내 낸다."""

from __future__ import annotations

import json

import httpx


class FakeApi:
    def __init__(self, token: str = "worker-token", prefix: str = "/api/v1/worker/jobs"):
        self.token = token
        self.prefix = prefix
        self.queue: list[dict] = []
        self.completed: dict[int, dict] = {}
        self.failed: dict[int, str] = {}
        self.objects: dict[str, bytes] = {}
        self.put_content_types: list[str] = []
        self.claims: list[dict] = []
        self.expired = False

    def add(self, job: dict) -> dict:
        self.queue.append(job)
        return job

    def ok(self, result) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "header": {"isSuccessful": True, "resultCode": 0, "resultMessage": "SUCCESS"},
                "result": result,
            },
        )

    def handler(self, req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if url.startswith("https://storage.test/"):
            key = url.split("https://storage.test/notebook-rental/")[1].split("?")[0]
            if req.method == "GET":
                return (
                    httpx.Response(200, content=self.objects[key])
                    if key in self.objects
                    else httpx.Response(404)
                )
            self.put_content_types.append(req.headers.get("content-type", ""))
            self.objects[key] = req.content
            return httpx.Response(200)
        path = req.url.path
        if path == "/api/v1/auth/login":
            body = json.loads(req.content)
            if body["password"] != "admin":
                return httpx.Response(
                    401, json={"header": {"isSuccessful": False, "resultCode": -1006, "resultMessage": "x"}}
                )
            return self.ok({"accessToken": "admin-token"})
        if path.startswith("/api/v1/admin/"):
            assert req.headers["authorization"] == "Bearer admin-token"
            if path == "/api/v1/admin/worker-tokens":
                name = json.loads(req.content)["name"]
                return self.ok({"token": self.token, "workerId": name, "expiresAt": "2027-02-03T00:00:00Z"})
            if path == "/api/v1/admin/ml-jobs" and req.method == "POST":
                return self.ok({"jobIds": [1, 2], "skippedPhotos": 1})
            if path == "/api/v1/admin/ml-jobs":
                return self.ok(
                    [
                        {
                            "jobId": 1,
                            "type": "selftest",
                            "status": "done",
                            "lockedBy": "ai-pc",
                            "result": {"providers": ["CUDAExecutionProvider"], "p50Ms": 3.1},
                            "error": None,
                        }
                    ]
                )
            return self.ok({"jobId": int(path.rsplit("/", 1)[1]), "status": "done"})
        if (
            self.expired
            or req.headers.get("authorization") not in (f"Bearer {self.token}", None)
            and req.headers.get("x-internal-token") is None
        ):
            return httpx.Response(401, json={"header": {"isSuccessful": False, "resultCode": -1005}})
        if path == self.prefix + "/claim":
            self.claims.append(json.loads(req.content))
            if not self.queue:
                return httpx.Response(204)
            return self.ok(self.queue.pop(0))
        job_id = int(path.split("/")[-2])
        body = json.loads(req.content)
        if path.endswith("/complete"):
            self.completed[job_id] = body["result"]
            return self.ok({"jobId": job_id, "status": "done"})
        self.failed[job_id] = body["error"]
        return self.ok({"jobId": job_id, "status": "queued"})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def urls_for(inputs: list[str], outputs: list[str]) -> dict:
    base = "https://storage.test/notebook-rental/"
    return {
        "get": {k: base + k + "?X-Amz-Signature=get" for k in inputs},
        "put": {k: base + k + "?X-Amz-Signature=put" for k in outputs},
    }
