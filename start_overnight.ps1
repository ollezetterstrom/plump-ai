param(
    [int]$Rounds = 800000,
    [string]$Device = "auto",
    [string]$Resume = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $logDir "train_$stamp.log"

$arguments = @("train.py", "--rounds", "$Rounds", "--device", $Device)
if ($Resume -ne "") {
    $arguments += @("--resume", $Resume)
}

Write-Host "Logging training output to $log"
python @arguments 2>&1 | Tee-Object -FilePath $log
