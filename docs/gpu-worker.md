# 집 AI PC를 GPU 워커로 쓰기

운영 클러스터에는 GPU가 없다. 집 AI PC(Windows, NVIDIA RTX 5060 Ti 16GB)의 GPU로 **무거운 ML 작업**을 처리하게 하는 방법이다.
설계와 이유는 docs 저장소 `specs/003-gpu-worker/`(spec, plan, 계약)에 있다.

## 1. 어떻게 동작하나

```
관리자 ── 작업 넣기 ──▶ api ── ml_job 표(작업 대기열)
                        ▲                      ▲
   5초마다 "할 일 있어요?" │                      │ 2분 넘게 아무도 안 가져가면
   (AI PC → 밖으로만)      │                      │ 클러스터 ml 파드가 CPU로 대신
                  ┌───────┴────────┐         ┌───┴───────────┐
                  │ AI PC GPU 워커    │         │ ml-inference 파드 │
                  └───────┬────────┘         └───────────────┘
                          │ 서명 URL로 사진 읽기, 결과 쓰기
                          ▼
                        MinIO
```

- **풀링(polling)**: AI PC가 먼저 api에 묻는다. 집 공유기의 포트를 열거나 VPN을 깔 필요가 없다. 휴대폰이 웹 페이지를 여는 것과 같은 방향의 연결이다.
- **워커 토큰**: AI PC가 api에 자기를 증명하는 열쇠(JWT). 작업 API만 쓸 수 있고 사람용 화면 API는 못 쓴다. 90일 뒤 만료된다.
- **서명 URL**: AI PC에는 MinIO 비밀번호(키)가 없다. api가 작업마다 "이 파일만, 30분 동안" 읽고 쓸 수 있는 주소를 붙여 준다.
- **CPU 대체 처리**: AI PC가 꺼져 있으면 작업이 2분 기다린 뒤 클러스터가 CPU로 처리한다. 느리지만 멈추지는 않는다.
- **반납 비교는 GPU로 보내지 않는다.** CPU로 컷 1장 0.5초라 집까지 사진을 보내는 시간이 더 길다([benchmark.md](benchmark.md)).

지금 GPU로 처리하는 작업:

| 종류 | 언제 쓰나 | 하는 일 |
|------|-----------|---------|
| `selftest` | 설치 확인, 속도 비교 | 백본(DINOv2)을 여러 번 돌려 **어떤 장치로(CUDA/CPU)** 몇 ms 걸렸는지 남긴다 |
| `baseline_features` | 백본을 바꾼 뒤, 계산 방식을 고친 뒤 | 모든 기준 사진(대여·등록 사진)의 특징을 다시 계산해 `derived/baseline-features/<photoId>.npz`에 저장 |

앞으로 자동 초벌 제안(`propose`, 큰 모델)도 같은 길로 붙인다.

## 2. 준비물

| 항목 | 확인 방법 |
|------|-----------|
| NVIDIA 드라이버 **최신판**(580 이상) | PowerShell에서 `nvidia-smi` → GPU 이름과 드라이버 버전이 보이면 된다. 지금 받는 `onnxruntime-gpu`가 CUDA 13을 쓰므로 드라이버가 오래되면 GPU를 못 찾는다 |
| Python **3.12** | `py -3.12 --version`. 없으면 `winget install Python.Python.3.12` |
| git | `git --version`. 없으면 `winget install Git.Git` |
| 인터넷 | 브라우저에서 `https://notebook-rental.java21.net`이 열리면 된다 |

CUDA Toolkit은 **따로 깔지 않는다.** `onnxruntime-gpu[cuda,cudnn]`이 필요한 CUDA·cuDNN 실행 파일을 pip로 같이 받는다.

## 3. 설치 (처음 한 번)

PowerShell을 열고:

```powershell
cd $HOME
git clone https://github.com/notebook-rental/ml
cd ml
powershell -ExecutionPolicy Bypass -File tools\setup-gpu-worker.ps1
```

스크립트가 하는 일:

1. `.venv` 가상환경(이 폴더 전용 Python 꾸러미 칸)을 만든다.
2. ml 패키지를 설치한다.
3. CPU판 `onnxruntime`을 지우고 GPU판 `onnxruntime-gpu[cuda,cudnn]`을 깐다. 둘이 같은 이름으로 불러와져서 둘 다 있으면 꼬인다.
4. `python -m worker check`로 GPU 확인까지 한다. 처음에는 모델 파일(약 90MB)을 GitHub Release에서 받아 `~/.notebook-rental/models`에 둔다(sha256 확인).

끝에 `"providers": ["CUDAExecutionProvider", ...]`와 `결과: 정상`이 보이면 성공이다.

## 4. 로그인 (처음 한 번, 90일마다)

```powershell
.\.venv\Scripts\python.exe -m worker login
```

관리자 아이디와 비밀번호를 묻는다(비밀번호는 화면에 안 보인다). 비밀번호는 저장하지 않고, 받은 **워커 토큰**만
`C:\Users\<사용자>\.notebook-rental\worker.json`에 저장한다. 이 파일은 git에 올리거나 채팅에 붙이지 않는다.
워커 이름은 컴퓨터 이름으로 정해진다(`--name ai-pc`로 바꿀 수 있다).

## 5. 실행

```powershell
.\.venv\Scripts\python.exe -m worker run
```

`워커 시작: ...`이 보이고 5초마다 조용히 api에 묻는다. 작업이 오면 `작업 시작 jobId=...`, `작업 완료 ...`가 찍힌다. 멈추려면 Ctrl+C.
같은 기록이 `C:\Users\<사용자>\.notebook-rental\worker.log`에도 남는다(1MB씩 3개까지). 작업 스케줄러로 띄워 창이 안 보일 때는 이 파일로 확인한다.

## 6. 원격에서 GPU가 쓰이는지 확인

다른 컴퓨터(또는 AI PC 자신)에서 관리자 계정으로 자가 진단 작업을 넣고 결과를 본다.

```powershell
.\.venv\Scripts\python.exe -m worker enqueue selftest
.\.venv\Scripts\python.exe -m worker jobs
```

`selftest  done  by=<워커 이름>  CUDAExecutionProvider p50=…ms`처럼 나오면 원격 GPU 처리가 되는 것이다.
워커를 끈 채 넣으면 2분 뒤 `by=ml-inference-…  CPUExecutionProvider`로 처리된다(클러스터 CPU 대체). 두 시간을 비교하면 GPU 이득이 보인다.

curl로도 된다(관리자 토큰 `$T`): `POST /api/v1/admin/ml-jobs {"type":"selftest"}`, `GET /api/v1/admin/ml-jobs`.

기준 특징 전체 다시 계산(백본을 바꾼 뒤):

```powershell
.\.venv\Scripts\python.exe -m worker enqueue baseline_features          # 이미 같은 백본이면 건너뜀
.\.venv\Scripts\python.exe -m worker enqueue baseline_features --force  # 무조건 다시
```

## 7. 켜 두기 (선택)

로그인할 때마다 자동으로 켜려면 **작업 스케줄러**에 등록한다.

1. 시작 메뉴 → "작업 스케줄러" → 오른쪽 "작업 만들기".
2. 일반: 이름 `notebook-rental GPU 워커`, "사용자가 로그온할 때만 실행".
3. 트리거: 새로 만들기 → "로그온할 때".
4. 동작: 새로 만들기 → 프로그램 `powershell.exe`, 인수 `-ExecutionPolicy Bypass -File "C:\Users\<사용자>\ml\tools\run-gpu-worker.ps1"`.
5. 조건: "컴퓨터의 AC 전원이 켜져 있는 경우에만" 해제(데스크톱이면 상관없음).

`run-gpu-worker.ps1`은 워커가 네트워크 오류로 끝나면 30초 뒤 다시 켜고, 토큰이 만료되면 멈춘다.

## 8. 업데이트

운영 ml 파드와 **같은 버전**을 써야 한다(같은 백본 이름·해시). 버전이 다르면 ml 파드가 결과를 "다른 백본"으로 보고 다시 계산한다(틀린 결과가 쓰이지는 않는다).

```powershell
git pull
.\.venv\Scripts\python.exe -m pip install --no-deps .
```

## 9. 문제 해결

| 증상 | 원인 | 해결 |
|------|------|------|
| `providers`에 `CUDAExecutionProvider`가 없음 | CPU판 `onnxruntime`이 같이 깔림 | `pip uninstall -y onnxruntime onnxruntime-gpu` 후 `pip install "onnxruntime-gpu[cuda,cudnn]>=1.22"` |
| `CUDA 실행 파일을 미리 불러오지 못했습니다`, CPU로 실행 | 드라이버가 오래됨 | NVIDIA 드라이버 최신판 설치 후 재부팅, `nvidia-smi` 확인 |
| `api 인증 실패(401)` 후 멈춤 | 워커 토큰 만료(90일) | `python -m worker login` 다시 |
| `api 인증 실패(403)` | 관리자 토큰을 워커에 넣음 등 | `login`으로 받은 워커 토큰을 쓴다 |
| 작업이 모두 `by=ml-inference-…` | 워커가 꺼져 있었음 | `python -m worker run`이 떠 있는지 확인 |
| 서명 URL로 올리기 403 | 서명 만료(30분) 또는 시계가 틀림 | Windows 시간 자동 동기화를 켠다. 작업은 실패 보고 후 다시 대기열로 간다 |

## 10. 보안 정리

- AI PC에 있는 비밀은 **워커 토큰 하나**다. MinIO 키, DB 비밀번호, 내부 토큰은 없다.
- 워커 토큰은 작업 가져가기·완료·실패 보고만 할 수 있다. 쓰기 서명 URL은 `derived/` 아래만 나와서 원본 사진은 바꿀 수 없다.
- 토큰이 새었으면: 90일 만료를 기다리기 어렵다면 Rancher Secret의 `JWT_SECRET`을 바꾸고 api를 재시작한다(모든 사용자가 다시 로그인해야 한다).
