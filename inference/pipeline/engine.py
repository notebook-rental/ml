"""ONNX 모델 실행 엔진: ONNX Runtime(기본), ONNX Runtime + NVIDIA GPU(cuda), 또는 OpenVINO.

- onnxruntime: CPU에서 신경망을 빠르게 돌리는 라이브러리. 운영 클러스터(CPU 전용)의 기본값.
- cuda: 같은 ONNX Runtime이 NVIDIA GPU(CUDA)로 계산한다. GPU PC(RTX 5060 Ti)에서 쓴다.
  `onnxruntime-gpu` 패키지가 필요하다(`pip install ".[gpu]"`). GPU를 못 찾으면 **CPU로 자동으로 내려가고** 로그를 남긴다.
  "실행 공급자(Execution Provider)"는 ONNX Runtime이 계산을 맡기는 장치별 부품 이름이다(CUDAExecutionProvider, CPUExecutionProvider).
- openvino: 인텔 CPU용 엔진. 인텔 CPU에서는 보통 더 빠르다.
어느 쪽이 빠른지는 벤치마크(bench/bench_compare.py --engine)로 정한다(research R15 추론 튜닝).
세 엔진을 같은 모양(run(None, feeds) → 출력 목록)으로 감싸서 나머지 코드는 엔진을 몰라도 된다.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


def providers_for(engine: str, available: list[str]) -> list[str]:
    """엔진 이름 → ONNX Runtime 실행 공급자 목록(앞이 우선). 설치된 공급자(available)에 없는 것은 뺀다."""
    if engine == "cuda" and CUDA in available:
        return [CUDA, CPU]
    return [CPU]


def create_session(model_path: str | Path, engine: str = "onnxruntime", threads: int = 0):
    if engine == "openvino":  # pragma: no cover - openvino는 선택 설치(extra)라 CI에서는 돌리지 않는다
        return _OpenVinoSession(str(model_path), threads)
    import onnxruntime as ort

    if engine == "cuda" and hasattr(ort, "preload_dlls"):
        # pip로 받은 CUDA·cuDNN 실행 파일(onnxruntime-gpu[cuda,cudnn])을 먼저 불러온다. CUDA Toolkit을 따로 깔지 않아도 된다
        try:
            ort.preload_dlls()
        except Exception as e:  # pragma: no cover - GPU PC 환경 문제. 아래에서 CPU로 내려간다
            log.warning("CUDA 실행 파일을 미리 불러오지 못했습니다: %s", e)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    providers = providers_for(engine, ort.get_available_providers())
    if engine == "cuda" and CUDA not in providers:
        log.warning(
            "ENGINE=cuda지만 CUDA 실행 공급자가 없어 CPU로 실행한다(onnxruntime-gpu 설치·드라이버 확인)"
        )
    sess = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
    if engine == "cuda":
        log.info("ONNX Runtime 실행 공급자: %s", sess.get_providers())
    return sess


class _OpenVinoSession:  # pragma: no cover - 위와 같은 이유
    def __init__(self, path: str, threads: int):
        import openvino as ov

        core = ov.Core()
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if threads:
            cfg["INFERENCE_NUM_THREADS"] = str(threads)
        self._model = core.compile_model(path, "CPU", cfg)
        self._inputs = [i.get_any_name() for i in self._model.inputs]

    def run(self, _outputs, feeds: dict):
        req = self._model.create_infer_request()
        res = req.infer(feeds)
        return [res[o] for o in self._model.outputs]
