"""공식 DINOv2 ViT-S/14 가중치를 ONNX로 내보낸다 (운영용 모델 파일 만들기).

이 스크립트는 **PyTorch가 있는 PC(GPU PC 또는 개발자 PC)에서 한 번** 실행한다. 운영 서버·컨테이너에는
PyTorch를 넣지 않는다(이미지 크기·보안·시작 시간). 결과 파일을 MinIO `models/dinov2-s14/model.onnx`에 올리고
sha256을 설정값 BACKBONE_SHA256에 넣는다.

가중치 출처: facebookresearch/dinov2 (Apache-2.0). torch.hub가 GitHub에서 코드를, dl.fbaipublicfiles.com에서
가중치를 내려받는다.

    pip install ".[export]"
    python tools/export_dinov2.py --out model.onnx --width 448 --height 336
    sha256sum model.onnx

출력 형식은 inference/pipeline/patch_features.py 설명과 같다
(입력 pixel_values [1,3,H,W] → 출력 patch_tokens [1, H/14·W/14, 384]).
"""

from __future__ import annotations

import argparse


def main() -> None:  # pragma: no cover - PyTorch와 인터넷이 필요한 수동 도구
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()

    class PatchTokens(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m.forward_features(pixel_values)["x_norm_patchtokens"]

    x = torch.zeros(1, 3, a.height, a.width)
    torch.onnx.export(
        PatchTokens(model),
        x,
        a.out,
        input_names=["pixel_values"],
        output_names=["patch_tokens"],
        opset_version=a.opset,
        do_constant_folding=True,
    )
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
