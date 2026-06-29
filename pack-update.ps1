# ============================================================
# Pack Update - Gom file can deploy vao 1 file zip
# ------------------------------------------------------------
# Chay tren MAY DEV (PowerShell, khong can Administrator):
#   cd "C:\Users\PC\Desktop\Source\Tax\CaptchaService"
#   .\pack-update.ps1
#
# Output: C:\Users\PC\Desktop\update.zip
# Sau do: copy zip qua RDP -> server -> giai nen -> paste de
# ============================================================

$ErrorActionPreference = "Stop"

$src = "C:\Users\PC\Desktop\Source\Tax\CaptchaService"
$out = "C:\Users\PC\Desktop\update.zip"
$tmp = "C:\Users\PC\Desktop\_update_staging"

Write-Host ""
Write-Host "===== Pack Update =====" -ForegroundColor Cyan
Write-Host "Source: $src"
Write-Host "Output: $out"
Write-Host ""

if (-not (Test-Path $src)) { throw "Khong tim thay source: $src" }

# Xoa staging cu neu co
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
if (Test-Path $out) { Remove-Item $out -Force }
New-Item -ItemType Directory -Path $tmp | Out-Null

# Danh sach file can copy vao zip
$includeFiles = @(
    # === Code Python runtime (production) ===
    "main.py",
    "config.py",
    "ocr_engine.py",
    "captcha_model_resnet6.py",   # model def 6 ký tự (đang dùng)
    "captcha_model_v2.py",        # model def 5 ký tự
    "collector.py",
    "svg_renderer.py",
    "tct_client.py",
    "tcnnt_client.py",            # tra cứu MST (tracuunnt) - 5 ký tự
    "keystore.py",                # API key store + log (SQLite)
    # Lưu ý: train.py / labeler.py / tools/ là dev-only -> KHÔNG đóng gói lên prod
    # Lưu ý: tcnnt_keys.db KHÔNG đóng gói (giữ DB key thật trên server)

    # === Model weights ===
    "captcha_model.pth",
    "captcha_model_v6.pth",

    # === Cau hinh & du lieu ===
    "requirements.txt",
    "labels.csv",

    # === Docs (optional) ===
    "README.md",
    "SETUP.md"
)

Write-Host "[1/3] Copy files vao staging..." -ForegroundColor Yellow
$totalSize = 0
$copied = 0
$missing = @()

foreach ($name in $includeFiles) {
    $p = Join-Path $src $name
    if (Test-Path $p) {
        $size = (Get-Item $p).Length
        $totalSize += $size
        $sizeStr = if ($size -gt 1MB) { "{0:N1} MB" -f ($size/1MB) } else { "{0:N0} KB" -f ($size/1KB) }
        Write-Host ("      {0,-30} {1,10}" -f $name, $sizeStr) -ForegroundColor DarkGray
        Copy-Item $p $tmp -Force
        $copied++
    } else {
        $missing += $name
    }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "Cac file khong tim thay (bo qua):" -ForegroundColor DarkYellow
    foreach ($m in $missing) { Write-Host "      - $m" -ForegroundColor DarkYellow }
}

Write-Host ""
Write-Host "Tong: $copied files, $([math]::Round($totalSize/1MB, 1)) MB" -ForegroundColor Green

# Nen zip
Write-Host ""
Write-Host "[2/3] Tao zip..." -ForegroundColor Yellow
Compress-Archive -Path "$tmp\*" -DestinationPath $out -CompressionLevel Optimal
$zipSize = (Get-Item $out).Length
Write-Host "      Done: $([math]::Round($zipSize/1MB, 1)) MB" -ForegroundColor Green

# Cleanup staging
Write-Host ""
Write-Host "[3/3] Cleanup..." -ForegroundColor Yellow
Remove-Item $tmp -Recurse -Force
Write-Host "      Done." -ForegroundColor Green

Write-Host ""
Write-Host "===== Xong! =====" -ForegroundColor Green
Write-Host ""
Write-Host "File zip da tao tai:" -ForegroundColor Cyan
Write-Host "  $out" -ForegroundColor White
Write-Host ""
Write-Host "Buoc tiep theo:" -ForegroundColor Cyan
Write-Host "  1. Copy update.zip qua RDP vao server (Ctrl+C / Ctrl+V)" -ForegroundColor White
Write-Host "  2. Tren server: Stop-Service CaptchaService" -ForegroundColor White
Write-Host "  3. Giai nen + paste de vao D:\IIS WEB\decapcha.win-tech.vn\CaptchaService" -ForegroundColor White
Write-Host "  4. (Lan dau) Cai pip: python -m pip install -r requirements.txt" -ForegroundColor White
Write-Host "  5. Start-Service CaptchaService" -ForegroundColor White
Write-Host ""

# Mo Explorer chi vao file zip cho tien
Start-Process explorer.exe "/select,`"$out`""
