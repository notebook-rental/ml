"""작업 처리기와 실행 루프 (docs specs/003-gpu-worker, T307·T315·T319)."""

import httpx
import pytest

from common.jobs_client import JobsClient, PresignedStorage, WorkerAuthError
from common.storage import StorageError
from inference import jobs
from inference.pipeline import baseline_cache
from tests.fakeapi import FakeApi, urls_for
from tests.fixtures import synth


def ctx(models, settings, storage, worker="ai-pc"):
    return jobs.JobContext(models, settings, storage, worker)


def test_selftest_reports_provider_and_timing(models, settings, storage):
    r = jobs.run_selftest({"runs": 3}, ctx(models, settings, storage))
    assert r["providers"] == ["CPUExecutionProvider"]
    assert r["backbone"] == "patch-mean@test"
    assert r["runs"] == 3 and r["p50Ms"] >= 0 and r["warmupMs"] >= 0
    assert r["workerId"] == "ai-pc"
    assert r["onnxruntime"]


def test_baseline_features_computes_then_skips_then_force(models, settings, storage, put, baseline_img):
    put("derived/work/1/7.jpg", synth.jpeg(baseline_img))
    job = {"photoId": 7, "imageKey": "derived/work/1/7.jpg", "featuresKey": "derived/baseline-features/7.npz"}
    r = jobs.run_baseline_features(job, ctx(models, settings, storage))
    assert r["status"] == "computed" and r["featuresKey"] == "derived/baseline-features/7.npz"
    f = baseline_cache.deserialize(storage.get_bytes("derived/baseline-features/7.npz"))
    assert f.backbone == "patch-mean@test" and f.photo_id == "7"

    assert jobs.run_baseline_features(job, ctx(models, settings, storage))["status"] == "skipped"
    assert (
        jobs.run_baseline_features({**job, "force": True}, ctx(models, settings, storage))["status"]
        == "computed"
    )


def test_baseline_features_missing_photo_fails(models, settings, storage):
    with pytest.raises(jobs.JobFailed, match="사진이 없습니다"):
        jobs.run_baseline_features(
            {"photoId": 9, "imageKey": "derived/work/1/9.jpg"}, ctx(models, settings, storage)
        )


def test_presigned_storage_only_touches_listed_keys():
    api = FakeApi()
    api.objects["derived/work/1/7.jpg"] = b"img"
    s = PresignedStorage(
        urls_for(["derived/work/1/7.jpg"], ["derived/x.npz"]), httpx.Client(transport=api.transport())
    )
    assert s.get_bytes("derived/work/1/7.jpg") == b"img"
    s.put_bytes("derived/x.npz", b"npz", content_type="text/plain")
    assert api.objects["derived/x.npz"] == b"npz"
    assert api.put_content_types == ["application/octet-stream"]  # 서명할 때와 같은 값
    with pytest.raises(StorageError):
        s.get_bytes("originals/secret.jpg")
    with pytest.raises(StorageError):
        s.put_bytes("originals/x.jpg", b"")


def runner(api, models, settings, storage, presigned=True):
    client = JobsClient.worker("https://api.test", api.token, transport=api.transport())
    c = ctx(models, settings, storage)
    files = httpx.Client(transport=api.transport())
    return jobs.JobRunner(client, lambda: c, "ai-pc", poll_seconds=0, use_presigned=presigned, files=files)


def test_worker_runner_full_cycle_with_presigned_urls(models, settings, storage, baseline_img):
    api = FakeApi()
    api.objects["derived/work/1/7.jpg"] = synth.jpeg(baseline_img)
    keys = ["derived/work/1/7.jpg", "derived/baseline-features/7.npz"]
    api.add(
        {
            "jobId": 11,
            "type": "baseline_features",
            "photoId": 7,
            "imageKey": keys[0],
            "featuresKey": keys[1],
            "urls": urls_for(keys, [keys[1]]),
        }
    )
    api.add({"jobId": 12, "type": "selftest", "runs": 1, "urls": urls_for([], [])})
    r = runner(api, models, settings, storage)
    assert r.run_once() and r.run_once()
    assert not r.run_once()  # 대기열이 비면 False
    assert api.completed[11]["status"] == "computed"
    assert baseline_cache.deserialize(api.objects[keys[1]]).photo_id == "7"
    assert api.completed[12]["providers"] == ["CPUExecutionProvider"]
    assert api.claims[0] == {"types": ["selftest", "baseline_features"], "workerId": "ai-pc"}
    assert r.done == 2 and r.failed == 0
    assert not storage.exists(keys[1])  # 워커는 MinIO 직접 쓰기를 하지 않는다


def test_runner_reports_failure_and_continues(models, settings, storage):
    api = FakeApi()
    api.add(
        {
            "jobId": 21,
            "type": "baseline_features",
            "photoId": 1,
            "imageKey": "derived/work/1/1.jpg",
            "urls": urls_for(["derived/work/1/1.jpg", "derived/baseline-features/1.npz"], []),
        }
    )
    api.add({"jobId": 22, "type": "propose", "urls": urls_for([], [])})
    r = runner(api, models, settings, storage)
    r.run_once()
    r.run_once()
    assert "사진이 없습니다" in api.failed[21]
    assert "모르는 작업 종류" in api.failed[22]
    assert r.failed == 2


def test_runner_stops_on_auth_error(models, settings, storage):
    api = FakeApi()
    api.expired = True
    r = runner(api, models, settings, storage)
    with pytest.raises(WorkerAuthError):
        r.run_forever()


def test_pod_runner_uses_internal_token_and_direct_storage(models, settings, storage, put, baseline_img):
    from inference.app import Service

    put("derived/work/1/7.jpg", synth.jpeg(baseline_img))
    api = FakeApi(prefix="/internal/jobs")
    api.add({"jobId": 31, "type": "baseline_features", "photoId": 7, "imageKey": "derived/work/1/7.jpg"})
    svc = Service(settings, storage)
    svc.registry._models = models
    client = JobsClient.internal("http://api:8080", "internal", transport=api.transport())
    r = svc.start_jobs(client)
    assert r.run_once()
    assert api.completed[31]["status"] == "computed"
    assert storage.exists("derived/baseline-features/7.npz")
    svc.stop()
