# GPU 워커를 계속 켜 둔다(작업 스케줄러가 로그온할 때 이 파일을 실행하게 등록한다, docs/gpu-worker.md 7절).
# 워커가 네트워크 오류 등으로 끝나면 30초 뒤 다시 켠다. 토큰 만료(종료 코드 2)면 멈춘다(login을 다시 해야 함).
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = ".\.venv\Scripts\python.exe"
while ($true) {
    & $py -m worker run
    if ($LASTEXITCODE -eq 2) {
        Write-Host "워커 토큰이 만료되었거나 거절되었습니다. 'python -m worker login'을 다시 실행하세요" -ForegroundColor Red
        break
    }
    Start-Sleep -Seconds 30
}
