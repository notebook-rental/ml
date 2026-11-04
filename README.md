# notebook-rental ML 추론 서비스 (ml-inference)

노트북 대여 서비스에서 **반납된 노트북 사진을 대여 때 사진과 비교해 새로 생긴 손상 후보를 찾아 주는** Python 서비스다.
업무 API(Spring Boot, `notebook-rental/api`)가 HTTP로 부르고, 결과는 직원이 화면에서 보고 최종 판정한다(사람이 최종 판정).

설계 문서는 `notebook-rental/docs`의 `specs/001-return-damage-inspection/`에 있다.
이 저장소는 그중 **1단계(사전학습 모델 + 변화 탐지, 학습 없음)** 를 구현한다. 사용자 스토리 US1~US3에 해당하는 ML 작업이다.

| 문서 | 내용 |
|------|------|
| [docs/pipeline.md](docs/pipeline.md) | 비교 파이프라인이 단계별로 무엇을 왜 하는지(초보자용 설명) |
| [docs/benchmark.md](docs/benchmark.md) | 속도 측정 방법과 결과(tasks T001) |
| 계약 | docs 저장소 `contracts/ml-service.md`, `contracts/api-response.md` |

## 한눈에 보기

```
직원 폰 ──사진──▶ api(Spring) ──HTTP──▶ ml-inference(이 저장소) ──읽기/쓰기──▶ MinIO(notebook-rental 버킷)
                                   ◀── 후보 목록 ───┘
```

1. **대여할 때**: 사진이 저장되면 api가 `POST /baseline-features`를 부른다. ml이 그 사진의 특징(노트북 영역, 특징점,
   DINOv2 패치 특징)을 **미리 계산**해 `derived/baseline-features/<photoId>.npz`에 둔다. 반납 때 기다리는 시간을 줄이려는 것이다.
2. **반납 QR을 스캔할 때**: api가 모든 ml 파드에 `POST /prefetch`를 보내 그 노트북의 기준 특징 8개를 **메모리에 미리 올린다.**
3. **반납 컷 1장이 올라올 때마다**: api가 `POST /compare`를 부르면, ml이 정렬 → 비교 → 후보 추출을 하고 후보 목록을 돌려준다.
   4 vCPU에서 컷 1장 **약 0.5초**(목표 5초, 상한 10초). 측정 방법과 표는 [docs/benchmark.md](docs/benchmark.md).

## 처음 쓰는 용어

| 용어 | 뜻 |
|------|----|
| 기준 사진(baseline) | 대여할 때 찍은 사진. "원래 상태"로 본다 |
| 반납 사진(target) | 반납할 때 같은 컷으로 찍은 사진 |
| 컷(shot) | 정해진 촬영 구도. top, bottom, left, right, hinge, front, keyboard, screen_off 8가지 |
| 작업본 | 폰이 원본을 긴 변 2560px로 줄인 사진. 비교는 작업본으로 하고, 원본은 증거로 보관한다 |
| 정렬(alignment) | 반납 사진을 기준 사진과 같은 위치·각도로 맞추는 것. 특징점(SIFT)과 호모그래피(3×3 투영 변환)를 쓴다 |
| DINOv2 | Meta가 공개한 사전학습 이미지 모델(Apache-2.0). 사진을 14×14px 조각(패치)으로 나눠 조각마다 "생김새"를 숫자 벡터로 낸다 |
| ONNX / ONNX Runtime | 신경망을 저장하는 표준 파일 형식 / 그 파일을 CPU에서 빠르게 실행하는 라이브러리. 운영 서버에 PyTorch가 필요 없다 |
| OpenVINO | 인텔 CPU용 추론 엔진. 인텔 CPU에서는 ONNX Runtime보다 빠를 수 있어 벤치마크로 비교한다 |
| SSIM | 두 사진의 작은 창마다 밝기·대비·구조가 같은 정도(0~1). 가는 스크래치처럼 작은 변화를 잡는다 |
| 후보(finding) | 모델이 "바뀐 것 같다"고 표시한 영역. 손상인지는 직원이 판정한다 |
| 멱등(idempotent) | 같은 요청을 여러 번 보내도 결과가 같다. api가 재시도해도 안전하다 |

## 폴더 구조

```
common/                 api와 주고받는 본문(스키마), 저장소(MinIO) 접근, Spring 내부 API 클라이언트
inference/
  app.py                FastAPI HTTP 서버(엔드포인트 목록은 아래)
  config.py             설정(APP_PROFILE=local|prod, 환경변수)
  config/{local,prod}.env  프로파일별 비밀이 아닌 기본값
  quality.py            촬영 품질 검사(흐림·노출·구도·QR)
  compare.py            컷 하나 비교(파이프라인 조립)
  model_registry.py     모델 내려받기·sha256 검증·워밍업·무중단 교체
  response.py           공통 응답 형식(header/result)
  pipeline/
    segment.py          노트북 영역 분리(경량 SAM 또는 GrabCut)
    align.py            SIFT + RANSAC 호모그래피 정렬, QR 꼭짓점 보조
    normalize.py        조명 정규화(히스토그램 매칭)
    patch_features.py   DINOv2 패치 특징, 기준 대비 최근접 거리
    ssim.py             1px 어긋남을 허용하는 SSIM 차분
    postprocess.py      점수 결합 → 임계값 → 덩어리 → 다각형, 반사 제외
    qr.py               QR 검출, QR 영역 마스크
    existing_damage.py  기존 손상과 겹침 판정
    typing.py           유형 추정(모양 규칙)
    baseline_cache.py   기준 특징 사전 계산·저장(npz)·메모리 LRU
    engine.py           ONNX Runtime / OpenVINO 실행
bench/bench_compare.py  속도 측정(tasks T001)
tools/
  build_vit_onnx.py     DINOv2-S와 같은 구조의 무작위 가중치 ONNX(속도 측정·CI용)
  export_dinov2.py      공식 DINOv2 가중치 → ONNX(운영용, PyTorch 있는 PC에서 실행)
tests/
  unit/                 부품별 테스트
  contract/             HTTP 계약 테스트(응답 형식, 상태 코드, 503)
  regression/           합성 전/후 사진으로 "어디에 몇 개" 후보가 나오는지 고정
  fixtures/             합성 사진 생성기, 테스트용 가짜 백본
```

## HTTP API

모든 JSON 응답은 공통 형식이다: `{"header": {"isSuccessful", "resultCode", "resultMessage"}, "result": ...}`.
클러스터 안에서만 열리고(Ingress에 등록하지 않음), `INTERNAL_TOKEN`이 설정되어 있으면 POST 요청에 헤더 `X-Internal-Token`이 맞아야 한다.

| 메서드 | 경로 | 설명 | 정상 | 실패 |
|--------|------|------|------|------|
| GET | `/health/live` | 프로세스 생존(liveness) | 200 | |
| GET | `/health/ready` | 모델 로드·워밍업 완료(readiness) | 200 | 503 `-1098` |
| POST | `/quality` | 촬영 품질 검사 `{imageKey, shotType}` | 200 `{pass, reasons, scores, laptopBox, qr}` | 404 `-1002`, 400 `-1001` |
| POST | `/baseline-features` | 대여 사진 특징 사전 계산 `{photoId, imageKey, shotType, qr?}` | 202 `{featuresKey}` | 503 |
| POST | `/prefetch` | 기준 특징을 메모리에 `{baselinePhotoIds}` | 202 | 503 |
| POST | `/compare` | 컷 하나 비교(계약 문서 본문) | 200 `{status: done/not_comparable/failed, findings, …}` | 503 `-4003`(바쁨) |
| POST | `/models/reload` | 모델 교체 알림 | 200 `{changed}` | |

- `/compare`는 계산이 실패해도 HTTP 200에 `status=failed`(`error` 포함)로 답한다. 정렬이 안 되면 `status=not_comparable`(`reason`).
  api는 이 경우 "자동 비교 실패, 사람 검토"로 넘긴다.
- 동시에 처리하는 비교 수가 `MAX_CONCURRENT_COMPARES`를 넘으면 **503 + resultCode -4003**. api는 잠깐 뒤 재시도한다.
- 후보 좌표: `geometry`는 **반납 원본 사진 픽셀 좌표**, `baselineGeometry`는 **기준 원본 사진 픽셀 좌표**(device_damage 등록용).
  `existingDamages[].geometry`는 기준 사진 픽셀 좌표로 보낸다.
- ID는 숫자(JSON number)로 보내면 된다. 문자열도 받는다.
- 단계별 시간 `timingsMs`(baseline, decode, qr, align, warp, dino, ssim, postprocess, store)를 함께 돌려준다. 느려지면 어디가 문제인지 바로 보인다.

## 설정(환경변수)

프로파일은 `APP_PROFILE=local|prod`. 값은 `inference/config/<프로파일>.env` → 환경변수 순서로 덮어쓴다.
**비밀값은 저장소에 넣지 않는다(공개 저장소).** 운영에서는 Kubernetes Secret `notebook-rental-secrets`가 환경변수로 넣는다.

| 이름 | 기본값 | 설명 |
|------|--------|------|
| `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` | (Secret) | 이미 제공되는 MinIO |
| `MINIO_BUCKET` | `notebook-rental` | 버킷 하나, 폴더로 구분(originals/ derived/ datasets/ models/) |
| `API_BASE_URL`, `INTERNAL_TOKEN` | `http://api:8080`, (Secret) | Spring 내부 API(활성 모델 30초마다 확인) |
| `BACKBONE_KEY`, `BACKBONE_SHA256` | `models/dinov2-s14/model.onnx`, 빈 값 | DINOv2 ONNX 위치와 해시(해시를 주면 검증) |
| `SAM_ENCODER_KEY`, `SAM_DECODER_KEY` | 빈 값 | 경량 SAM ONNX. 비우면 GrabCut으로 노트북 영역을 딴다 |
| `ENGINE` | `onnxruntime` | `openvino`도 가능(`pip install ".[openvino]"`) |
| `INTRA_OP_THREADS` | 0(=CPU 수) | 요청 하나가 쓰는 스레드 |
| `MAX_CONCURRENT_COMPARES` | 2 | 넘으면 503 |
| `BASELINE_LRU_SIZE` | 240 | 메모리에 올려 둘 기준 사진 수(노트북 30대 × 8컷, 약 1.5GB). 파드 메모리에 맞춰 줄인다 |
| `COMPARE_LONG_SIDE` | 1024 | 정렬·비교 해상도(긴 변) |
| `THRESHOLD`, `DINO_SCALE`, `SSIM_GAIN`, `MIN_REGION_PX` | 0.35, 0.5, 1.6, 60 | 후보 판정 값. **초기 평가 세트(실제 사진)로 보정해야 한다** |

## 개발자 PC에서 실행

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# 모델 파일 준비: 운영과 같은 결과를 보려면 공식 가중치(tools/export_dinov2.py),
# 속도·동작만 보려면 같은 구조의 무작위 가중치 모델
python tools/build_vit_onnx.py --out vit.onnx

# MinIO(deploy/local/docker-compose.yml)의 models/dinov2-s14/model.onnx 에 올린 뒤
export APP_PROFILE=local MINIO_ACCESS_KEY=... MINIO_SECRET_KEY=...
uvicorn --factory inference.app:default_app --port 8000
```

MinIO 없이 디스크 폴더로 돌려 보려면 `STORAGE_BACKEND=local LOCAL_STORAGE_ROOT=./.data/storage`.

## 테스트

```bash
pytest --cov                 # 단위 + 계약 + 회귀, 커버리지 80% 미만이면 실패(pyproject.toml fail_under)
pytest tests/regression -v   # 합성 전/후 사진 회귀 테스트만
ruff check . && ruff format --check .
```

- 실제 노트북 사진은 저장소에 넣지 않는다(공개 저장소, 개인정보). 대신 `tests/fixtures/synth.py`가 노트북처럼 생긴 장면을 코드로 만들고,
  각도·밝기를 바꿔 "다시 찍은" 사진에 스티커·스크래치·얼룩·테이프를 넣는다.
- 회귀 테스트가 지키는 것: 손상은 그 자리에 정확히 1개, **조명만 다르거나 각도만 다르면 후보 0개**, QR 제거·교체는 `qr_tampered`,
  기존 손상과 겹치면 `overlapsDamageId`, 같은 요청은 같은 결과.
- 테스트에서는 DINOv2 대신 "패치 평균 색"을 내는 작은 가짜 백본을 쓴다(빠르고 결과가 안정적). 실제 DINOv2의 정확도는
  초기 평가 세트(docs research R11)로 따로 잰다.
- CI(`.github/workflows/ci.yml`): 린트 → 테스트(커버리지 80% 미만 실패) → 속도 측정 표를 실행 요약에 남김.

## 컨테이너

`Dockerfile`: Python 3.12 slim, 비루트 사용자, 모델·비밀값은 이미지에 넣지 않음, 포트 8000.
워커 프로세스는 1개로 두고(모델을 프로세스마다 올리면 메모리가 배로 든다) 동시 처리는 스레드로 한다.
ONNX Runtime과 OpenCV는 계산 중 Python GIL을 풀기 때문에 스레드로도 여러 요청이 실제로 동시에 돈다.

## 1단계에서 하지 않은 것(다음 작업)

- 경량 SAM ONNX 내보내기: 래퍼(`SamSegmenter`)와 테스트는 있지만 모델 파일은 아직 없다. 지금은 GrabCut을 쓴다.
  반납 사진에는 어차피 SAM을 돌리지 않으므로(대여 마스크를 정렬해 재사용) 반납 속도에는 영향이 없다.
- `/assist/segment`(라벨링 클릭 보조), 초벌 제안 작업 루프(`/internal/jobs/claim`), 학습 CLI: US4·US5 범위.
- 유형 추정 k-NN: 판정 라벨이 쌓이면 추가한다. 지금은 모양 규칙만 쓴다.
- 임계값 보정: 실제 사진 초기 평가 세트(300쌍)로 `THRESHOLD`, `DINO_SCALE`, `SSIM_GAIN`을 맞춘다.
