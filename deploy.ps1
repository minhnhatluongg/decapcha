# ============================================================
# CaptchaService - Deploy Script (chay tren SERVER)
# ------------------------------------------------------------
# Setup thuc te:
#   - NSSM service ten: CaptchaService
#   - Python: C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe
#   - IIS reverse proxy: decapcha.win-tech.vn -> localhost:8000
#
# Cach dung (PowerShell as Administrator tren server):
#   .\deploy.ps1                  # update code thuong
#   .\deploy.ps1 -InstallDeps     # khi requirements.txt doi
# ============================================================

param(
    [switch]$InstallDeps,
    [string]$ServiceName = "CaptchaService",
    [string]$SourceDir   = "C:\Users\PC\Desktop\Source\Tax\CaptchaService",
    [string]$TargetDir   = "D:\IIS WEB\decapcha.win-tech.vn\CaptchaService",
    [string]$PythonExe   = "C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "===== CaptchaService Deploy =====" -ForegroundColor Cyan
Write-Host "Source : $SourceDir"
Write-Host "Target : $TargetDir"
Write-Host ""

if (-not (Test-Path $SourceDir)) { throw "Source khong ton tai: $SourceDir" }
if (-not (Test-Path $TargetDir)) { throw "Target khong ton tai: $TargetDir" }

$svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if (-not $svc) { throw "Khong tim thay service '$ServiceName'" }

# --- 1. Stop service ---
Write-Host "[1/4] Stopping service..." -ForegroundColor Yellow
if ($svc.Status -eq "Running") {
    Stop-Service -Name $ServiceName -Force
    Start-Sleep -Seconds 3
    $conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($conn) {
        Stop-Process -Id $conn.OwningProcess -Force
        Start-Sleep -Seconds 2
    }
}
Write-Host "      Stopped." -ForegroundColor Green

# --- 2. Copy file ---
Write-Host "[2/4] Copying files..." -ForegroundColor Yellow

# Code Python
$pyFiles = Get-ChildItem -Path $SourceDir -Filter "*.py" -File
foreach ($f in $pyFiles) {
    Copy-Item $f.FullName $TargetDir -Force
    Write-Host "      $($f.Name)" -ForegroundColor DarkGray
}

# Cac file phu (KHONG copy .env, web.config, deploy.ps1)
$extraFiles = @("labels.csv", "requirements.txt", "README.md", "SETUP.md")
foreach ($name in $extraFiles) {
    $p = Join-Path $SourceDir $name
    if (Test-Path $p) {
        Copy-Item $p $TargetDir -Force
        Write-Host "      $name" -ForegroundColor DarkGray
    }
}

# Model weights (.pth) - bat buoc co
$modelFiles = @("captcha_model.pth", "captcha_model_v6.pth")
foreach ($name in $modelFiles) {
    $p = Join-Path $SourceDir $name
    if (Test-Path $p) {
        $size = [math]::Round((Get-Item $p).Length / 1MB, 1)
        Write-Host "      $name (${size}MB)..." -ForegroundColor DarkGray
        Copy-Item $p $TargetDir -Force
    }
}

Write-Host "      Done." -ForegroundColor Green

# --- 3. pip install neu can ---
if ($InstallDeps) {
    Write-Host "[3/4] Installing pip packages..." -ForegroundColor Yellow
    if (-not (Test-Path $PythonExe)) { throw "Khong thay Python: $PythonExe" }
    & $PythonExe -m pip install -r (Join-Path $TargetDir "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    # Cai Playwright Chromium (can cho SVG renderer fallback)
    Write-Host "      Installing Playwright Chromium..." -ForegroundColor DarkGray
    & $PythonExe -m playwright install chromium
    Write-Host "      Done." -ForegroundColor Green
} else {
    Write-Host "[3/4] Skip pip install" -ForegroundColor DarkGray
}

# --- 4. Start service ---
Write-Host "[4/4] Starting service..." -ForegroundColor Yellow
Start-Service -Name $ServiceName
Start-Sleep -Seconds 8
$svc = Get-Service -Name $ServiceName
Write-Host "      Status: $($svc.Status)" -ForegroundColor Green

# --- Health check ---
Write-Host ""
Write-Host "Health check..." -ForegroundColor Cyan
$ok = $false
for ($i = 1; $i -le 6; $i++) {
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
        Write-Host "  OK -> engine: $($resp.engine)" -ForegroundColor Green
        if ($resp.engine -like "*CNN*") {
            Write-Host "  CNN Model loaded thanh cong!" -ForegroundColor Green
        } else {
            Write-Host "  CANH BAO: Van dang fallback EasyOCR, kiem tra file .pth!" -ForegroundColor Yellow
        }
        $ok = $true
        break
    } catch {
        Write-Host "  Thu $i/6, doi 5s..." -ForegroundColor DarkGray
        Start-Sleep -Seconds 5
    }
}

if (-not $ok) {
    Write-Host "  FAIL. Xem log:" -ForegroundColor Red
    Write-Host "    Get-Content '$TargetDir\service.log' -Tail 80" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "===== Done =====" -ForegroundColor Green
Write-Host ""
