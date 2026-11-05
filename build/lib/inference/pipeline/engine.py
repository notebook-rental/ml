"""ONNX 모델 실행 엔진: ONNX Runtime(기본) 또는 OpenVINO.

둘 다 CPU에서 신경망을 빠르게 돌리는 라이브러리다. 인텔 CPU에서는 OpenVINO가 보통 더 빠르다.
어느 쪽이 빠른지는 운영 서버에서 벤치마크(bench/bench_compare.py --engine)로 정한다(research R15 추론 튜닝).
두 엔진을 같은 모양(run(None, feeds) → 출력 목록)으로 감싸서 나머지 코드는 엔진을 몰라도 된다.
"""

from __future__ import annotations

from pathlib import Path


def create_session(model_path: str | Path, engine: str = "onnxruntime", threads: int = 0):
    if engine == "openvino":  # pragma: no cover - openvino는 선택 설치(extra)라 CI에서는 돌리지 않는다
        return _OpenVinoSession(str(model_path), threads)
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])


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
