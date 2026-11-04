"""DINOv2 ViT-S/14와 **구조가 같은** ONNX 모델을 무작위 가중치로 만든다 (속도 측정·테스트 전용).

왜 필요한가:
- 추론 속도는 가중치 값이 아니라 구조(층 수, 크기, 입력 해상도)로 정해진다.
  공식 가중치를 받을 수 없는 환경(CI, 네트워크 제한)에서도 속도를 정확히 잴 수 있다.
- 정확도(손상을 잘 찾는지)는 이 모델로 판단할 수 없다. 운영에는 tools/export_dinov2.py로 공식 가중치를 내보낸
  모델을 써야 한다.

PyTorch 없이 onnx 패키지만으로 그래프를 직접 조립한다.

사용법:
    python tools/build_vit_onnx.py --out .data/models/vit-s14-random.onnx --width 448 --height 336
    python tools/build_vit_onnx.py --out tiny.onnx --dim 32 --depth 1 --heads 2   # 테스트용 아주 작은 모델
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(
    width: int,
    height: int,
    dim: int = 384,
    depth: int = 12,
    heads: int = 6,
    mlp: int = 1536,
    patch: int = 14,
    seed: int = 0,
) -> onnx.ModelProto:
    assert width % patch == 0 and height % patch == 0, "입력 크기는 14의 배수여야 합니다"
    rng = np.random.default_rng(seed)
    gh, gw = height // patch, width // patch
    n = gh * gw
    t = n + 1
    hd = dim // heads
    inits: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []

    def w(name: str, *shape: int, std: float = 0.02, value: float | None = None) -> str:
        arr = (
            np.full(shape, value, np.float32)
            if value is not None
            else rng.normal(0, std, shape).astype(np.float32)
        )
        inits.append(numpy_helper.from_array(arr, name))
        return name

    def const_i64(name: str, vals: list[int]) -> str:
        inits.append(numpy_helper.from_array(np.array(vals, np.int64), name))
        return name

    def node(op: str, inputs: list[str], out: str, **attrs) -> str:
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # 패치 임베딩
    x = node(
        "Conv",
        ["pixel_values", w("patch.w", dim, 3, patch, patch), w("patch.b", dim, value=0.0)],
        "pe",
        strides=[patch, patch],
    )
    x = node("Reshape", [x, const_i64("shape.pe", [1, dim, n])], "pe.r")
    x = node("Transpose", [x], "pe.t", perm=[0, 2, 1])
    x = node("Concat", [w("cls", 1, 1, dim), x], "tok", axis=1)
    x = node("Add", [x, w("pos", 1, t, dim)], "tok.pos")

    shape_qkv = const_i64("shape.qkv", [1, t, 3, heads, hd])
    shape_out = const_i64("shape.out", [1, t, dim])
    scale = w("attn.scale", 1, value=1.0 / math.sqrt(hd))
    half = w("half", 1, value=0.5)
    one = w("one", 1, value=1.0)
    rsqrt2 = w("rsqrt2", 1, value=1.0 / math.sqrt(2.0))

    for i in range(depth):
        p = f"b{i}"
        y = node(
            "LayerNormalization",
            [x, w(f"{p}.ln1.g", dim, value=1.0), w(f"{p}.ln1.b", dim, value=0.0)],
            f"{p}.ln1",
            axis=-1,
            epsilon=1e-6,
        )
        y = node("MatMul", [y, w(f"{p}.qkv.w", dim, 3 * dim)], f"{p}.qkv.mm")
        y = node("Add", [y, w(f"{p}.qkv.b", 3 * dim, value=0.0)], f"{p}.qkv")
        y = node("Reshape", [y, shape_qkv], f"{p}.qkv.r")
        y = node("Transpose", [y], f"{p}.qkv.t", perm=[2, 0, 3, 1, 4])  # [3,1,h,T,hd]
        nodes.append(helper.make_node("Split", [y], [f"{p}.q0", f"{p}.k0", f"{p}.v0"], axis=0, num_outputs=3))
        sq = const_i64(f"{p}.sq", [0])
        q = node("Squeeze", [f"{p}.q0", sq], f"{p}.q")
        k = node("Squeeze", [f"{p}.k0", sq], f"{p}.k")
        v = node("Squeeze", [f"{p}.v0", sq], f"{p}.v")
        kt = node("Transpose", [k], f"{p}.kt", perm=[0, 1, 3, 2])
        a = node("MatMul", [q, kt], f"{p}.qk")
        a = node("Mul", [a, scale], f"{p}.qk.s")
        a = node("Softmax", [a], f"{p}.attn", axis=-1)
        o = node("MatMul", [a, v], f"{p}.av")
        o = node("Transpose", [o], f"{p}.av.t", perm=[0, 2, 1, 3])
        o = node("Reshape", [o, shape_out], f"{p}.av.r")
        o = node("MatMul", [o, w(f"{p}.proj.w", dim, dim)], f"{p}.proj.mm")
        o = node("Add", [o, w(f"{p}.proj.b", dim, value=0.0)], f"{p}.proj")
        o = node("Mul", [o, w(f"{p}.ls1", dim, value=1.0)], f"{p}.ls1o")
        x = node("Add", [x, o], f"{p}.res1")

        y = node(
            "LayerNormalization",
            [x, w(f"{p}.ln2.g", dim, value=1.0), w(f"{p}.ln2.b", dim, value=0.0)],
            f"{p}.ln2",
            axis=-1,
            epsilon=1e-6,
        )
        y = node("MatMul", [y, w(f"{p}.fc1.w", dim, mlp)], f"{p}.fc1.mm")
        y = node("Add", [y, w(f"{p}.fc1.b", mlp, value=0.0)], f"{p}.fc1")
        # GELU = 0.5·x·(1 + erf(x/√2))
        e = node("Mul", [y, rsqrt2], f"{p}.g1")
        e = node("Erf", [e], f"{p}.g2")
        e = node("Add", [e, one], f"{p}.g3")
        e = node("Mul", [y, e], f"{p}.g4")
        y = node("Mul", [e, half], f"{p}.gelu")
        y = node("MatMul", [y, w(f"{p}.fc2.w", mlp, dim)], f"{p}.fc2.mm")
        y = node("Add", [y, w(f"{p}.fc2.b", dim, value=0.0)], f"{p}.fc2")
        y = node("Mul", [y, w(f"{p}.ls2", dim, value=1.0)], f"{p}.ls2o")
        x = node("Add", [x, y], f"{p}.res2")

    x = node(
        "LayerNormalization",
        [x, w("norm.g", dim, value=1.0), w("norm.b", dim, value=0.0)],
        "norm",
        axis=-1,
        epsilon=1e-6,
    )
    nodes.append(
        helper.make_node(
            "Slice",
            [x, const_i64("s.start", [1]), const_i64("s.end", [t]), const_i64("s.axes", [1])],
            ["patch_tokens"],
        )
    )

    graph = helper.make_graph(
        nodes,
        "vit_s14_random",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, height, width])],
        [helper.make_tensor_value_info("patch_tokens", TensorProto.FLOAT, [1, n, dim])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--mlp", type=int, default=None)
    a = ap.parse_args()
    m = build(a.width, a.height, a.dim, a.depth, a.heads, a.mlp or a.dim * 4)
    onnx.save(m, a.out)
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
