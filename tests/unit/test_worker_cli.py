"""AI PC 워커 CLI (docs specs/003-gpu-worker, T308)."""

import json

import pytest

from tests.fakeapi import FakeApi
from worker import cli


@pytest.fixture(autouse=True)
def worker_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NR_WORKER_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def args(*argv):
    return cli.build_parser().parse_args(list(argv))


def test_login_saves_token_only_in_user_folder(worker_home, capsys):
    api = FakeApi()
    rc = cli.cmd_login(
        args(
            "login", "--api", "https://api.test/", "--user", "admin", "--password", "admin", "--name", "ai-pc"
        ),
        transport=api.transport(),
    )
    assert rc == 0
    cfg = json.loads((worker_home / "worker.json").read_text(encoding="utf-8"))
    assert cfg == {
        "api": "https://api.test",
        "workerId": "ai-pc",
        "token": "worker-token",
        "expiresAt": "2027-02-03T00:00:00Z",
    }
    out = capsys.readouterr().out
    assert "worker-token" not in out  # 토큰을 화면에 찍지 않는다


def test_login_wrong_password_fails():
    with pytest.raises(SystemExit, match="HTTP 401"):
        cli.cmd_login(
            args("login", "--api", "https://api.test", "--user", "admin", "--password", "nope"),
            transport=FakeApi().transport(),
        )


def test_run_without_login_tells_what_to_do():
    with pytest.raises(SystemExit, match="worker login"):
        cli.load_config()


def test_enqueue_and_jobs(worker_home, capsys):
    api = FakeApi()
    cli.save_config({"api": "https://api.test", "workerId": "ai-pc", "token": "t", "expiresAt": "x"})
    cli.cmd_enqueue(
        args("enqueue", "baseline_features", "--user", "admin", "--password", "admin"),
        transport=api.transport(),
    )
    cli.cmd_jobs(args("jobs", "--user", "admin", "--password", "admin"), transport=api.transport())
    out = capsys.readouterr().out
    assert "작업 2개" in out and "건너뛴 사진: 1장" in out
    assert "CUDAExecutionProvider p50=3.1ms" in out


def test_worker_settings_follow_prod_backbone(worker_home):
    s = cli.worker_settings("cuda")
    assert s.engine == "cuda"
    assert s.backbone_name == "dinov2-s14@1"
    assert s.backbone_url.startswith("https://github.com/notebook-rental/ml/releases/")
    assert len(s.backbone_sha256) == 64
    assert s.model_cache_dir == str(worker_home / "models")


def test_runner_from_config_claims_with_worker_token(models, settings):
    api = FakeApi()
    api.add({"jobId": 5, "type": "selftest", "runs": 1, "urls": {"get": {}, "put": {}}})
    r = cli.make_runner(
        {"api": "https://api.test", "token": "worker-token", "workerId": "ai-pc"},
        settings,
        models,
        0,
        transport=api.transport(),
    )
    assert r.run_once()
    assert api.completed[5]["workerId"] == "ai-pc"
