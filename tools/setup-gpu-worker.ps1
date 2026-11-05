# 집 AI PC(Windows + NVIDIA GPU)에 GPU 워커를 설치한다 (docs/gpu-worker.md).
# 실행: ml 저장소 폴더에서 PowerShell을 열고
#   powershell -ExecutionPolicy Bypass -File tools\setup-gpu-worker.ps1
# 하는 일: 가상환경(.venv) 만들기 → ml 패키지 설치 → CPU판 onnxruntime을 GPU판으로 바꾸기 → GPU 확인
$ErrorActionPreference = "Stop"

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host "nvidia-smi가 없습니다. NVIDIA 드라이버(최신판)를 먼저 설치하세요: https://www.nvidia.com/drivers" -ForegroundColor Red
    exit 1
}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Host "Python 실행기(py)가 없습니다. 'winget install Python.Python.3.12'로 Python 3.12를 설치하세요" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path .venv)) {
    py -3.12 -m venv .venv
}
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install .
# onnxruntime(CPU판)과 onnxruntime-gpu는 같은 이름(onnxruntime)으로 불러와져서 둘 다 있으면 꼬인다. CPU판을 지운다
& $py -m pip uninstall -y onnxruntime onnxruntime-gpu
& $py -m pip install "onnxruntime-gpu[cuda,cudnn]>=1.22"
& $py -m worker check
Write-Host ""
Write-Host "설치 끝. 다음 순서: .\.venv\Scripts\python.exe -m worker login  →  .\.venv\Scripts\python.exe -m worker run" -ForegroundColor Green
