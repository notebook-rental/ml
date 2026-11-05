# 모델 파일만 담은 작은 이미지(코드 없음). 운영 파드의 initContainer가 이 파일을 모델 캐시(emptyDir)로 복사한다.
# 파드가 인터넷(github.com)에 나가지 못해도, 이미지는 ghcr에서 받을 수 있으므로 모델을 준비할 수 있다.
FROM busybox:1.36
COPY model.onnx /models/dinov2-s14/model.onnx
USER 10001
