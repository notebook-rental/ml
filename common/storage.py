"""객체 저장소(MinIO, S3 호환) 접근.

버킷은 `notebook-rental` 하나이고 폴더로 나눈다(docs infra/README.md).

- originals/  원본 사진. ML 서비스는 **읽기만** 한다(헌법 II, 원본 불변).
- derived/    가공물(작업본, 정렬 결과, 점수 맵, 기준 특징 캐시). ML 서비스가 쓴다.
- datasets/   학습 스냅샷(1단계에서는 쓰지 않음).
- models/     모델 파일(.onnx)과 추론 프로필.

두 가지 구현이 있다.
- S3Storage: 운영(prod)과 local 프로파일. boto3로 MinIO에 붙는다.
- LocalStorage: 테스트·벤치마크용. 같은 키 구조를 디스크 폴더에 둔다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

WRITABLE_PREFIXES = ("derived/",)


class StorageError(Exception):
    pass


class ObjectNotFound(StorageError):
    pass


class Storage(Protocol):
    def get_bytes(self, key: str) -> bytes: ...

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None: ...

    def exists(self, key: str) -> bool: ...


def check_writable(key: str) -> None:
    """ML 서비스가 쓸 수 있는 폴더인지 확인한다. 원본(originals/)·모델 덮어쓰기를 코드에서도 막는다."""
    if ".." in key.split("/") or key.startswith("/"):
        raise StorageError(f"잘못된 키: {key}")
    if not key.startswith(WRITABLE_PREFIXES):
        raise StorageError(f"ML 서비스는 derived/ 폴더에만 쓸 수 있습니다: {key}")


class LocalStorage:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        if ".." in key.split("/") or key.startswith("/"):
            raise StorageError(f"잘못된 키: {key}")
        return self.root / key

    def get_bytes(self, key: str) -> bytes:
        p = self._path(key)
        if not p.is_file():
            raise ObjectNotFound(key)
        return p.read_bytes()

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        check_writable(key)
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()


class S3Storage:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str, client=None):
        if client is None:  # pragma: no cover - 실제 MinIO 연결은 통합 환경에서 확인
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name="us-east-1",
                config=Config(
                    s3={"addressing_style": "path"},
                    retries={"max_attempts": 3, "mode": "standard"},
                    max_pool_connections=32,
                ),
            )
        self.client = client
        self.bucket = bucket

    def get_bytes(self, key: str) -> bytes:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
        except self.client.exceptions.NoSuchKey as e:
            raise ObjectNotFound(key) from e
        return obj["Body"].read()

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        check_writable(key)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
