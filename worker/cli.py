"""집 AI PC GPU 워커 CLI (docs specs/003-gpu-worker, tasks T312).

AI PC는 집 공유기 뒤에 있어서 서버가 먼저 연결할 수 없다. 그래서 이 프로그램이 몇 초마다 api에
"할 일 있어요?"라고 묻고(풀링), 받은 작업을 GPU로 처리해 결과를 돌려준다. 집으로 들어오는 포트를 열 필요가 없다.

명령:
    python -m worker login --api https://notebook-rental.java21.net   # 관리자 로그인 → 워커 토큰 저장(처음 한 번)
    python -m worker check                                           # GPU가 잡히는지, 백본이 GPU로 도는지 확인
    python -m worker run                                             # 작업 처리 시작(Ctrl+C로 멈춤)
    python -m worker enqueue selftest                                # (관리자) GPU 자가 진단 작업 넣기
    python -m worker enqueue baseline_features [--force]             # (관리자) 기준 특징 전체 다시 계산
    python -m worker jobs [--id 12]                                  # (관리자) 최근 작업 / 작업 하나 보기

워커 토큰은 `~/.notebook-rental/worker.json`(Windows: `C:\\Users\\<사용자>\\.notebook-rental\\worker.json`)에만 저장한다.
이 파일은 git에 올리거나 채팅에 붙이지 않는다. 토큰은 작업 API만 쓸 수 있고 90일 뒤 만료된다.
AI PC에는 MinIO 키가 없다. 파일은 api가 작업마다 붙여 주는 서명 URL로만 읽고 쓴다.
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import socket
import sys
from contextlib import contextmanager, suppress
from pathlib import Path

import httpx

from common.jobs_client import JobsClient, WorkerAuthError
from common.storage import ObjectNotFound
from inference.config import Settings, _profile_env_file

log = logging.getLogger("worker")

DEFAULT_API = "https://notebook-rental.java21.net"


def home() -> Path:
    """워커 설정·모델 캐시 폴더. NR_WORKER_HOME 환경변수로 바꿀 수 있다(테스트용)."""
    return Path(os.environ.get("NR_WORKER_HOME") or Path.home() / ".notebook-rental")


def config_path() -> Path:
    return home() / "worker.json"


def save_config(cfg: dict) -> Path:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    # 나만 읽게(Windows에서는 효과가 작다. 사용자 폴더라 다른 계정은 기본으로 못 읽는다)
    with suppress(OSError):
        p.chmod(0o600)
    return p


def load_config() -> dict:
    p = config_path()
    if not p.is_file():
        raise SystemExit(f"워커 설정이 없습니다({p}). 먼저 `python -m worker login`을 실행하세요")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------- 관리자 로그인(토큰 발급, 작업 넣기·보기) ----------


def _result(r: httpx.Response) -> dict | list:
    body = r.json()
    header = body.get("header") or {}
    if r.status_code >= 400 or not header.get("isSuccessful", False):
        raise SystemExit(
            f"실패: HTTP {r.status_code} {header.get('resultCode')} {header.get('resultMessage')}"
        )
    return body.get("result")


@contextmanager
def admin_client(api: str, login_id: str | None = None, password: str | None = None, transport=None):
    """관리자 계정으로 로그인한 클라이언트. 비밀번호는 화면에 보이지 않게 묻고 어디에도 저장하지 않는다."""
    login_id = login_id or input("관리자 아이디: ").strip()
    password = password if password is not None else getpass.getpass("비밀번호: ")
    with httpx.Client(base_url=api.rstrip("/"), timeout=15.0, transport=transport) as c:
        token = _result(c.post("/api/v1/auth/login", json={"loginId": login_id, "password": password}))[
            "accessToken"
        ]
        c.headers["Authorization"] = f"Bearer {token}"
        yield c


def cmd_login(a, transport=None) -> int:
    name = a.name or socket.gethostname().split(".")[0].lower().replace("_", "-")[:30]
    with admin_client(a.api, a.user, a.password, transport) as c:
        t = _result(c.post("/api/v1/admin/worker-tokens", json={"name": name}))
    p = save_config(
        {
            "api": a.api.rstrip("/"),
            "workerId": t["workerId"],
            "token": t["token"],
            "expiresAt": t["expiresAt"],
        }
    )
    print(f"워커 토큰을 저장했습니다: {p}")
    print(f"워커 이름: {t['workerId']}, 만료: {t['expiresAt']}")
    print(
        "이 파일은 git에 올리거나 다른 사람에게 보내지 마세요. 다음: python -m worker check → python -m worker run"
    )
    return 0


def cmd_enqueue(a, transport=None) -> int:
    body: dict = {"type": a.type}
    if a.type == "selftest":
        body["runs"] = a.runs
    else:
        body["force"] = a.force
    api = a.api or load_config().get("api", DEFAULT_API)
    with admin_client(api, a.user, a.password, transport) as c:
        r = _result(c.post("/api/v1/admin/ml-jobs", json=body))
    print(
        f"작업 {len(r['jobIds'])}개를 넣었습니다: {r['jobIds'][:20]}{' …' if len(r['jobIds']) > 20 else ''}"
    )
    if r.get("skippedPhotos"):
        print(f"이미 대기·진행 중이라 건너뛴 사진: {r['skippedPhotos']}장")
    return 0


def cmd_jobs(a, transport=None) -> int:
    api = a.api or load_config().get("api", DEFAULT_API)
    with admin_client(api, a.user, a.password, transport) as c:
        if a.id:
            print(json.dumps(_result(c.get(f"/api/v1/admin/ml-jobs/{a.id}")), ensure_ascii=False, indent=2))
            return 0
        rows = _result(c.get("/api/v1/admin/ml-jobs", params={"limit": a.limit}))
    for j in rows:
        res = j.get("result") or {}
        extra = res.get("providers", [""])[0] if res.get("providers") else ""
        if "p50Ms" in res:
            extra += f" p50={res['p50Ms']}ms"
        print(
            f"#{j['jobId']:<6} {j['type']:<18} {j['status']:<8} by={j.get('lockedBy') or '-':<14} {extra} {j.get('error') or ''}"
        )
    return 0


# ---------- 모델과 작업 처리 ----------


class _NoBucket:
    """워커에는 MinIO 키가 없다. 모델은 공개 Release 주소(BACKBONE_URL)에서 받고 sha256을 확인한다."""

    def get_bytes(self, key: str) -> bytes:
        raise ObjectNotFound(key)


def worker_settings(engine: str) -> Settings:
    """운영(prod.env)과 같은 백본 이름·주소·해시를 쓴다. 그래야 ml 파드가 워커가 만든 결과를 그대로 읽는다."""
    env = _profile_env_file("prod")
    overrides = {
        "app_profile": "prod",
        "engine": engine,
        "storage_backend": "local",
        "model_cache_dir": str(home() / "models"),
        "intra_op_threads": 0,
    }
    return Settings(_env_file=env, **overrides) if env else Settings(**overrides)


def load_models(settings: Settings):
    from inference.model_registry import ModelRegistry

    registry = ModelRegistry(_NoBucket(), settings)
    registry.load_initial()
    return registry.models


def cmd_check(a) -> int:
    import onnxruntime as ort

    from inference.jobs import JobContext, gpu_name, run_selftest

    print(f"ONNX Runtime {ort.__version__}, 설치된 실행 공급자: {ort.get_available_providers()}")
    print(f"GPU: {gpu_name() or '찾지 못함(nvidia-smi 없음)'}")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print(
            "경고: CUDA 실행 공급자가 없습니다. onnxruntime(CPU판)이 같이 깔려 있으면 지우고 onnxruntime-gpu를 다시 까세요(docs/gpu-worker.md)"
        )
    s = worker_settings(a.engine)
    models = load_models(s)
    r = run_selftest({"runs": a.runs}, JobContext(models, s, _NoBucket(), "local-check"))
    print(json.dumps(r, ensure_ascii=False, indent=2))
    if a.engine == "cuda" and r["providers"][:1] != ["CUDAExecutionProvider"]:
        print("결과: GPU를 쓰지 못하고 CPU로 돌았습니다")
        return 1
    print("결과: 정상")
    return 0


def make_runner(cfg: dict, settings: Settings, models, poll: float, transport=None):
    from inference.jobs import JobContext, JobRunner

    client = JobsClient.worker(cfg["api"], cfg["token"], transport=transport)
    ctx = JobContext(models, settings, _NoBucket(), cfg["workerId"])
    files = httpx.Client(timeout=httpx.Timeout(30.0, read=120.0), transport=transport)
    return JobRunner(client, lambda: ctx, cfg["workerId"], poll_seconds=poll, use_presigned=True, files=files)


def cmd_run(a) -> int:
    cfg = load_config()
    s = worker_settings(a.engine)
    models = load_models(s)
    runner = make_runner(cfg, s, models, a.poll)
    print(
        f"워커 시작: {cfg['workerId']} → {cfg['api']} (엔진 {a.engine}, {a.poll:.0f}초마다 확인, 멈추려면 Ctrl+C)"
    )
    try:
        if a.once:
            runner.run_once()
        else:
            runner.run_forever()
    except WorkerAuthError as e:
        print(f"{e}. `python -m worker login`을 다시 실행하세요", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(f"\n멈춤. 처리 {runner.done}개, 실패 {runner.failed}개")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m worker", description="집 AI PC GPU 워커")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def admin_opts(p, api_default=None):
        p.add_argument("--api", default=api_default, help=f"서비스 주소(기본: 저장된 값 또는 {DEFAULT_API})")
        p.add_argument("--user", help="관리자 아이디(안 주면 물어봄)")
        p.add_argument("--password", help=argparse.SUPPRESS)  # 테스트용. 평소에는 화면에서 묻는다

    p = sub.add_parser("login", help="관리자 로그인 → 워커 토큰 저장")
    admin_opts(p, DEFAULT_API)
    p.add_argument("--name", help="워커 이름(기본: 컴퓨터 이름). 영문·숫자·-")

    for name, h in (("check", "GPU 확인(api 없이)"), ("run", "작업 처리 시작")):
        p = sub.add_parser(name, help=h)
        p.add_argument(
            "--engine",
            default="cuda",
            choices=["cuda", "onnxruntime"],
            help="cuda(GPU) 또는 onnxruntime(CPU)",
        )
        if name == "check":
            p.add_argument("--runs", type=int, default=20)
        else:
            p.add_argument("--poll", type=float, default=5.0, help="작업이 없을 때 다시 묻는 간격(초)")
            p.add_argument("--once", action="store_true", help="작업 하나만 처리하고 끝")

    p = sub.add_parser("enqueue", help="(관리자) 작업 넣기")
    p.add_argument("type", choices=["selftest", "baseline_features"])
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--force", action="store_true", help="이미 같은 백본으로 계산된 것도 다시")
    admin_opts(p)

    p = sub.add_parser("jobs", help="(관리자) 최근 작업 보기")
    p.add_argument("--id", type=int)
    p.add_argument("--limit", type=int, default=20)
    admin_opts(p)
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = build_parser().parse_args(argv)
    return {"login": cmd_login, "check": cmd_check, "run": cmd_run, "enqueue": cmd_enqueue, "jobs": cmd_jobs}[
        a.cmd
    ](a)
