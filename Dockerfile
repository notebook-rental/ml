# ml-inference 컨테이너 이미지 (tasks T007)
# - 모델 파일은 이미지에 넣지 않는다. 시작할 때 MinIO models/에서 받는다(이미지가 작고, 모델 교체에 재빌드가 필요 없음).
# - 비밀값은 이미지에 넣지 않는다. Kubernetes Secret → 환경변수.
# - 비루트 사용자로 실행한다.
FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md ./
COPY common ./common
COPY inference ./inference
RUN pip install --no-cache-dir --prefix=/install . \
    && PYTHONPATH=$(echo /install/lib/python3.12/site-packages) python -c \
    "from inference.config import _profile_env_file as f; assert f('prod'), 'prod.env가 패키지에 없습니다'"

FROM python:3.12-slim
# OpenCV(headless)가 쓰는 공유 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /var/cache/models && chown app /var/cache/models
COPY --from=build /install /usr/local
WORKDIR /app
USER app
ENV APP_PROFILE=prod \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1
EXPOSE 8000
# 워커 1개: 모델을 프로세스마다 따로 올리면 메모리가 늘어난다. 동시 처리는 스레드(MAX_CONCURRENT_COMPARES)로.
CMD ["uvicorn", "--factory", "inference.app:default_app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
