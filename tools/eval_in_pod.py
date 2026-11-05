"""실험(eval_angle.py)을 ml-inference 파드 안에서 돌리는 실행 파일을 만든다 (Mac에 파이썬 도구를 깔 수 없을 때).

Mac에서 표준 라이브러리만으로 실행하고, 그 출력(파이썬 프로그램)을 파드의 python에 넘긴다:

    python3 tools/eval_in_pod.py | kubectl -n notebook-rental exec -i deploy/ml-inference -- python -

파드 안에서 하는 일:
- 사진을 저장소에서 **읽기만** 해서 파드의 /tmp(임시 폴더, 파드가 다시 시작하면 사라진다)에 둔다
- 파드에 이미 있는 DINOv2 모델과 라이브러리로 실험을 돌린다. 낮은 우선순위(nice 19), 스레드 2개
- 결과 표(숫자)만 출력하고 /tmp의 실험 파일을 지운다. 저장소·DB에는 쓰지 않는다
"""

import base64
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
args = sys.argv[1:] or ["--synthetic"]
src = base64.b64encode((HERE / "eval_angle.py").read_bytes()).decode()
fetch = base64.b64encode((HERE / "fetch_eval_photos.py").read_bytes()).decode()

print(f'''
import base64, glob, io, os, runpy, shutil, subprocess, sys, tarfile
os.nice(19)
root = "/tmp/nr-eval"
shutil.rmtree(root, ignore_errors=True)
os.makedirs(root + "/tools")
open(root + "/tools/eval_angle.py", "wb").write(base64.b64decode("{src}"))
open(root + "/fetch.py", "wb").write(base64.b64decode("{fetch}"))
tar = subprocess.run([sys.executable, root + "/fetch.py"], capture_output=True, check=True)
sys.stderr.write(tar.stderr.decode())
tarfile.open(fileobj=io.BytesIO(tar.stdout)).extractall(root + "/photos")
models = glob.glob("/var/cache/models/**/*.onnx", recursive=True)
model = next((m for m in models if "dinov2" in m), models[0])
os.environ.setdefault("EVAL_THREADS", "2")
sys.argv = ["eval_angle.py", root + "/photos", "--backbone", model] + {args!r}
try:
    runpy.run_path(root + "/tools/eval_angle.py", run_name="__main__")
finally:
    shutil.rmtree(root, ignore_errors=True)
''')
