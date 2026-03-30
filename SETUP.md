# CaptchaService - Hướng dẫn Setup (dành cho .NET dev lần đầu dùng Python)

## 1. Cài Python

- Tải Python 3.11 tại: https://www.python.org/downloads/
- Khi cài **BẮT BUỘC tick "Add Python to PATH"**
- Mở terminal kiểm tra:
  ```
  python --version
  ```
  Phải hiện `Python 3.11.x`

## 2. Mở project trong VS Code

- Mở VS Code → File → Open Folder → chọn `CaptchaService`
- Cài Extension: **Python** (của Microsoft) - tìm trong Extensions (Ctrl+Shift+X)

## 3. Tạo Virtual Environment (giống NuGet packages riêng cho project)

Mở Terminal trong VS Code (Ctrl+`):

```bash
# Tạo venv (giống tạo project riêng, tránh xung đột thư viện)
python -m venv venv

# Kích hoạt venv
# Windows CMD:
venv\Scripts\activate
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Git Bash:
source venv/Scripts/activate

# Khi thành công sẽ thấy (venv) ở đầu dòng terminal
```

## 4. Cài thư viện (giống NuGet restore)

```bash
pip install -r requirements.txt
```

Lần đầu sẽ tải ~500MB (EasyOCR + PyTorch). Chờ 5-10 phút.

## 5. Cài GTK (cần cho cairosvg - chuyển SVG sang PNG)

### Windows:
- Tải GTK3 Runtime tại: https://github.com/nickvdyck/weasyprint-win/releases
- Hoặc dùng choco: `choco install gtk-runtime`
- **Khởi động lại VS Code** sau khi cài

## 6. Chạy server

```bash
python main.py
```

Hoặc:
```bash
uvicorn main:app --reload --port 8000
```

Lần đầu chạy sẽ download model EasyOCR (~100MB).

## 7. Test API

Mở trình duyệt: http://localhost:8000/docs

Đây là Swagger UI (giống Swagger trong .NET) - có thể test trực tiếp.

### Test bằng curl:

```bash
# Giải CAPTCHA từ base64 (tương thích TrueCaptcha)
curl -X POST http://localhost:8000/one/gettext \
  -H "Content-Type: application/json" \
  -d "{\"data\": \"<base64_png_here>\"}"

# Tự động lấy + giải CAPTCHA TCT
curl http://localhost:8000/captcha/solve

# Xem thống kê ảnh đã thu thập
curl http://localhost:8000/stats
```

## 8. Tích hợp với WinTax_API (C#)

Trong file `TCT_Service.cs`, chỉ cần đổi 1 dòng:

```csharp
// CŨ:
string API_URL = "https://api.apitruecaptcha.org/one/gettext";

// MỚI:
string API_URL = "http://localhost:8000/one/gettext";
```

Và bỏ `userid`/`apikey` trong JSON body (không cần nữa, nhưng gửi cũng không sao).

## So sánh với .NET

| Python              | .NET tương đương          |
|---------------------|---------------------------|
| `venv`              | NuGet packages folder     |
| `pip install`       | `dotnet restore`          |
| `requirements.txt`  | `.csproj` PackageReference|
| `uvicorn`           | `Kestrel`                 |
| `FastAPI`           | `ASP.NET Core Web API`    |
| `/docs` (Swagger)   | `/swagger`                |
| `pydantic` model    | `class` DTO               |
| `main.py`           | `Program.cs`              |
| `@app.get("/path")` | `[HttpGet("path")]`       |
