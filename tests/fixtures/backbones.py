"""테스트용 대역(가짜) 백본.

무작위 가중치 ViT는 평평하고 어두운 곳에서 특징 길이가 작아져, 길이 1로 정규화하면 잡음이 커진다(진짜 DINOv2에는
없는 현상). 회귀 테스트가 그 잡음에 흔들리지 않도록, 패치마다 "평균 색 + 상수"를 특징으로 내는 아주 단순한 ONNX를
쓴다. 입출력 모양은 DINOv2 내보내기와 같다(pixel_values → patch_tokens).
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_patch_mean(width: int = 448, height: int = 336, patch: int = 14) -> onnx.ModelProto:
    n = (width // patch) * (height // patch)
    w = np.zeros((4, 3, patch, patch), np.float32)
    for c in range(3):
        w[c, c] = 1.0 / (patch * patch)
    b = np.array([0, 0, 0, 1.0], np.float32)  # 4번째 = 상수(어두운 곳에서도 길이가 0이 되지 않게)
    nodes = [
        helper.make_node("Conv", ["pixel_values", "w", "b"], ["c"], strides=[patch, patch]),
        helper.make_node("Reshape", ["c", "shape"], ["r"]),
        helper.make_node("Transpose", ["r"], ["patch_tokens"], perm=[0, 2, 1]),
    ]
    graph = helper.make_graph(
        nodes,
        "patch_mean",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, height, width])],
        [helper.make_tensor_value_info("patch_tokens", TensorProto.FLOAT, [1, n, 4])],
        [
            numpy_helper.from_array(w, "w"),
            numpy_helper.from_array(b, "b"),
            numpy_helper.from_array(np.array([1, 4, n], np.int64), "shape"),
        ],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 9
    return m
