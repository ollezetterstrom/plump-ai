# Launch overnight training detached from any terminal.
# Usage: .\start_overnight.ps1     (log: train_log.txt)
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $dir
$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$log = "train_log_$stamp.txt"
Start-Process -FilePath "python" -ArgumentList "-u", "train_transformer.py" `
    -WorkingDirectory $dir -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
    -WindowStyle Hidden
Write-Host "Started. Log: $log"
Write-Host "Watch progress:   Get-Content $log -Wait"
Write-Host "Check checkpoints: Get-ChildItem plump_transformer*.pt"
Write-Host "Stop:              Stop-Process -Name python"
