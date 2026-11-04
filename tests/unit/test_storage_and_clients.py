"""저장소(로컬·S3), Spring 내부 API 클라이언트, 설정 테스트."""

from __future__ import annotations

import boto3
import httpx
import pytest
from moto import mock_aws

from common.callback import SpringClient
from common.storage import LocalStorage, ObjectNotFound, S3Storage, StorageError, check_writable
from inference import config


def test_check_writable():
    check_writable("derived/x.npz")
    for bad in ["originals/a.jpg", "models/m.onnx", "derived/../originals/a", "/etc/passwd"]:
        with pytest.raises(StorageError):
            check_writable(bad)


def test_local_storage(tmp_path):
    s = LocalStorage(tmp_path)
    s.put_bytes("derived/a/b.bin", b"hi")
    assert s.get_bytes("derived/a/b.bin") == b"hi" and s.exists("derived/a/b.bin")
    assert not s.exists("derived/none")
    with pytest.raises(ObjectNotFound):
        s.get_bytes("derived/none")
    with pytest.raises(StorageError):
        s.put_bytes("originals/x.jpg", b"x")  # 원본은 쓸 수 없다(헌법 II)
    with pytest.raises(StorageError):
        s.get_bytes("../secret")


@mock_aws
def test_s3_storage():
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="notebook-rental")
    client.put_object(Bucket="notebook-rental", Key="originals/p.jpg", Body=b"orig")
    s = S3Storage("http://unused", "k", "s", "notebook-rental", client=client)
    assert s.get_bytes("originals/p.jpg") == b"orig"
    s.put_bytes("derived/x.bin", b"d")
    assert s.exists("derived/x.bin") and not s.exists("derived/none")
    with pytest.raises(ObjectNotFound):
        s.get_bytes("derived/none")
    with pytest.raises(StorageError):
        s.put_bytes("originals/p.jpg", b"overwrite")
    s.client = boto3.client("s3", region_name="us-east-1")
    s.bucket = "no-such-bucket"
    with pytest.raises(Exception):  # noqa: B017 - 404가 아닌 오류는 그대로 올린다
        s.exists("derived/x.bin")


def _spring(handler):
    return SpringClient("http://api:8080/", "tok", transport=httpx.MockTransport(handler))


def test_spring_client_active_models():
    seen = {}

    def handler(req: httpx.Request):
        seen["token"] = req.headers.get("X-Internal-Token")
        seen["path"] = req.url.path
        return httpx.Response(
            200,
            json={
                "header": {"isSuccessful": True, "resultCode": 0, "resultMessage": "SUCCESS"},
                "result": {
                    "schemaVersion": 1,
                    "production": {"id": 1, "name": "changedet-v1"},
                    "shadow": None,
                },
            },
        )

    c = _spring(handler)
    m = c.active_models()
    assert m.production.name == "changedet-v1" and m.production.kind == "zero_shot_change"
    assert seen == {"token": "tok", "path": "/internal/models/active"}
    c.close()


def test_spring_client_failure():
    c = _spring(
        lambda r: httpx.Response(
            200,
            json={"header": {"isSuccessful": False, "resultCode": -6005, "resultMessage": "운영 모델 없음"}},
        )
    )
    with pytest.raises(RuntimeError, match="-6005"):
        c.active_models()
    c = _spring(lambda r: httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        c.active_models()


def test_settings_profiles(monkeypatch):
    config.get_settings.cache_clear()
    monkeypatch.setenv("APP_PROFILE", "prod")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "from-env")
    s = config.get_settings()
    assert (
        s.app_profile == "prod" and s.api_base_url == "http://api:8080" and s.minio_access_key == "from-env"
    )
    config.get_settings.cache_clear()
    monkeypatch.setenv("APP_PROFILE", "test")  # test.env는 없다 → 기본값
    s = config.get_settings()
    assert s.app_profile == "test" and s.threads() >= 1
    assert config.Settings(intra_op_threads=3).threads() == 3
    config.get_settings.cache_clear()
