# ============================================================
# Deploy tu may DEV len SERVER qua network share + PSRemoting
# ------------------------------------------------------------
# Chay tren MAY DEV (PowerShell as Administrator):
#   .\deploy-remote.ps1
#   .\deploy-remote.ps1 -InstallDeps
#   .\deploy-remote.ps1 -IncludeModel
#
# Yeu cau:
#   1. Da map o Z: -> \\10.10.212.1\d$
#      (chay 1 lan: net use Z: \\10.10.212.1\d$ /user:Administrator <pass>)
#   2. PSRemoting da bat tren server
#      (chay 1 lan tren server: Enable-PSRemoting -Force)
# ============================================================

param(
    [switch]$InstallDeps,
    [switch]$IncludeModel,
    [string]$ServerHost   = "10.10.212.1",
    [string]$ServiceName  = "CaptchaService",
    [string]$SourceDir    = "C:\Users\PC\Desktop\Source\Tax\CaptchaService",
    [string]$RemoteDir    = "Z:\IIS WEB\decapcha.win-tech.vn\CaptchaService",
    [string]$RemoteDirLocal = "D:\IIS WEB\decapcha.win-tech.vn\CaptchaService",
    [string]$PythonExe    = "C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "===== Deploy DEV -> SERVER =====" -ForegroundColor Cyan
Write-Host "Source: $SourceDir"
Write-Host "Server: $ServerHost ($RemoteDirLocal)"
Write-Host ""

if (-not (Test-Path $SourceDir)) { throw "Source khong ton tai: $SourceDir" }
if (-not (Test-Path $RemoteDir))  { throw "Khong vao duoc remote folder (kiem tra Z: da map chua): $RemoteDir" }

# Hoi mat khau Administrator server 1 lan
$cred = Get-Credential -Message "Nhap user/pass Administrator cua server $ServerHost" -UserName "Administrator"

# --- 1. Stop service tren server ---
Write-Host "[1/4] Stopping service tren server..." -ForegroundColor Yellow
Invoke-Command -ComputerName $ServerHost -Credential $cred -ScriptBlock {
    param($name)
    Stop-Service -Name $name -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
} -ArgumentList $ServiceName
Write-Host "      Stopped." -ForegroundColor Green

# --- 2. Copy file qua network share ---
Write-Host "[2/4] Copying files..." -ForegroundColor Yellow

# Robocopy: copy .py + cac file phu, bo qua venv/cache/.env
$exclude = @(".env", ".env.example", ".gitignore", "web.config", "deploy.ps1", "deploy-remote.ps1")
$excludeDirs = @("venv", "__pycache__", ".git", "training_data", "training_data_5", "collected_captchas", "debug_images")

$args = @(
    $SourceDir, $RemoteDir,
    "*.py", "labels.csv", "requirements.txt", "README.md", "SETUP.md",
    "/XF") + $exclude + @("/XD") + $excludeDirs + @("/NJH", "/NJS", "/NP", "/NDL")

robocopy @args | Out-Host

# Model weights (.pth) - luon copy (bat buoc)
$modelFiles = @("captcha_model.pth", "captcha_model_v6.pth")
foreach ($name in $modelFiles) {
    $p = Join-Path $SourceDir $name
    if (Test-Path $p) {
        $size = [math]::Round((Get-Item $p).Length / 1MB, 1)
        Write-Host "      Copying $name (${size}MB)..." -ForegroundColor DarkGray
        Copy-Item $p $RemoteDir -Force
    }
}
Write-Host "      Done." -ForegroundColor Green

# --- 3. Cai thu vien tren server (neu can) ---
if ($InstallDeps) {
    Write-Host "[3/4] Installing pip packages tren server..." -ForegroundColor Yellow
    Invoke-Command -ComputerName $ServerHost -Credential $cred -ScriptBlock {
        param($py, $dir)
        & $py -m pip install -r (Join-Path $dir "requirements.txt")
        # Cai Playwright Chromium (can cho SVG renderer)
        & $py -m playwright install chromium
    } -ArgumentList $PythonExe, $RemoteDirLocal
    Write-Host "      Done." -ForegroundColor Green
} else {
    Write-Host "[3/4] Skip pip install" -ForegroundColor DarkGray
}

# --- 4. Start lai service ---
Write-Host "[4/4] Starting service..." -ForegroundColor Yellow
Invoke-Command -ComputerName $ServerHost -Credential $cred -ScriptBlock {
    param($name)
    Start-Service -Name $name
    Start-Sleep -Seconds 5
    $s = Get-Service -Name $name
    Write-Host "      Status: $($s.Status)"
} -ArgumentList $ServiceName

# --- Health check ---
Write-Host ""
Write-Host "Health check..." -ForegroundColor Cyan
$ok = $false
for ($i = 1; $i -le 6; $i++) {
    try {
        $resp = Invoke-RestMethod -Uri "https://decapcha.win-tech.vn/health" -TimeoutSec 5
        Write-Host "  OK -> $($resp | ConvertTo-Json -Compress)" -ForegroundColor Green
        $ok = $true
        break
    } catch {
        Write-Host "  Thu $i/6, doi 5s..." -ForegroundColor DarkGray
        Start-Sleep -Seconds 5
    }
}

if (-not $ok) { Write-Host "  FAIL - RDP vao server xem log service.log" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "===== Done =====" -ForegroundColor Green
