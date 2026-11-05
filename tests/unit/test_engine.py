"""실행 엔진 선택: ENGINE=cuda일 때 GPU가 있으면 GPU, 없으면 CPU로 내려가는지."""

import logging

from inference.pipeline import engine


def test_cuda_engine_prefers_gpu_then_cpu():
    assert engine.providers_for("cuda", [engine.CUDA, engine.CPU]) == [engine.CUDA, engine.CPU]


def test_cuda_engine_falls_back_to_cpu_without_gpu():
    assert engine.providers_for("cuda", [engine.CPU]) == [engine.CPU]


def test_default_engine_is_cpu_even_if_gpu_installed():
    assert engine.providers_for("onnxruntime", [engine.CUDA, engine.CPU]) == [engine.CPU]


def test_cuda_session_without_gpu_runs_on_cpu_and_warns(patch_mean_path, caplog):
    with caplog.at_level(logging.WARNING):
        sess = engine.create_session(patch_mean_path, "cuda")
    assert sess.get_providers() == [engine.CPU]
    assert "CPU로 실행" in caplog.text
