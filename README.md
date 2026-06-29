# CaptchaService

API service to solve CAPTCHA from TCT (Vietnam Tax Authority) using a custom-trained CNN model.
Replaces the paid TrueCaptcha API with a self-hosted, free solution.

## Architecture

```
Client (.NET) ──> CaptchaService (Python/FastAPI) ──> TCT CAPTCHA API
                         │
                    CNN Model (ResNet18)
                    Accuracy: ~95%
```

## Project Structure

```
CaptchaService/
  # --- Runtime (production) ---
  main.py                  # FastAPI server + API endpoints
  config.py                # Configuration (reads from .env)
  ocr_engine.py            # OCR engine (CNN model + EasyOCR + Tesseract fallback)
  captcha_model_resnet6.py # Model def 6 ký tự (ResNet18 + 6 heads) -- ĐANG DÙNG cho /captcha/solve
  captcha_model_v2.py      # Model def 5 ký tự (ResNet + SE attention)
  tct_client.py            # TCT API client (fetch CAPTCHA, authenticate)
  svg_renderer.py          # SVG to PNG renderer (Playwright subprocess)
  collector.py             # Save CAPTCHA images for training
  captcha_model.pth        # Weights 6 ký tự (not in git, see below)
  captcha_model_v6.pth     # Weights 5 ký tự (not in git)

  # --- Dev tooling (không deploy lên prod) ---
  training/
    train.py               # Train model 5 ký tự -> captcha_model_v6.pth
    labeler.py             # Web UI gán nhãn ảnh training
  tools/
    check_model.py         # Kiểm tra checkpoint .pth
    export_to_onnx.py      # Export model 5 ký tự sang ONNX
    down_version.py        # Hạ ONNX IR version
  archive/                 # File cũ/không dùng (mini-CNN lỗi thời, onnx experiments)

  # --- Config ---
  .env                     # Environment config (not in git, see below)
  .env.example             # Template for .env
  requirements.txt         # Python dependencies
```

## Quick Start (Development)

### Prerequisites

- Python 3.11 (download: https://www.python.org/downloads/release/python-3119/)
- Tesseract OCR (download: https://github.com/UB-Mannheim/tesseract/wiki)

### 1. Clone and setup

```powershell
git clone <repo-url>
cd CaptchaService

# Create virtual environment
py -3.11 -m venv venv
venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install Playwright browser
python -m playwright install chromium
```

### 2. Configure environment

```powershell
copy .env.example .env
```

Edit `.env` and fill in the values. For development, you can leave `CAPTCHA_API_KEY` empty (disables auth).

### 3. Get the trained model

The trained model file `captcha_model.pth` (~45MB) is not included in the git repository.

**Option A: Get from team**
- Contact the project maintainer to get the `captcha_model.pth` file
- Place it in the project root directory

**Option B: Train your own** (see "Training" section below)

### 4. Run

```powershell
venv\Scripts\Activate.ps1
python main.py
```

Server starts at `http://localhost:8000`

### 5. Verify

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Demo page: `http://localhost:8000/demo`

## API Endpoints

### Solve CAPTCHA (main endpoint)

```
GET /captcha/solve
```

Fetches a CAPTCHA from TCT, solves it, and returns the result.

Response:
```json
{
  "token": "A59HD6",
  "key": "69c5fb907509471a14967def"
}
```

### Solve from base64 image (TrueCaptcha-compatible)

```
POST /one/gettext
Content-Type: application/json

{"data": "<base64 PNG image>"}
```

Response:
```json
{
  "result": "A59HD6",
  "saved": true
}
```

### Health check

```
GET /health
```

### Demo page

```
GET /demo
```

Web UI to load CAPTCHA from TCT, detect, and verify results.

## Authentication

When `CAPTCHA_API_KEY` is set in `.env`, all API endpoints (except `/health`, `/docs`, `/demo`) require authentication.

**Header:**
```
X-API-Key: your-secret-key
```

**Or query parameter:**
```
GET /captcha/solve?api_key=your-secret-key
```

### Calling from .NET

```csharp
var client = new HttpClient();
client.DefaultRequestHeaders.Add("X-API-Key", "your-secret-key");

var result = await client.GetFromJsonAsync<CaptchaResult>(
    "http://your-server:8000/captcha/solve");

Console.WriteLine($"Text: {result.Token}, Key: {result.Key}");

record CaptchaResult(string Token, string Key);
```

## Training Your Own Model

If you don't have the pre-trained model or want to improve accuracy:

### Step 1: Collect CAPTCHA images

Start the server and use the crawler to collect images:

```
POST http://localhost:8000/crawler/start?target=3000&delay=0.5
```

This saves ~3000 CAPTCHA images to `training_data/`.

### Step 2: Label images

```powershell
python training/labeler.py
```

Open `http://localhost:8001` in browser. For each image:
- Look at the CAPTCHA image
- Type the correct text
- Press Enter
- Repeat (~3 seconds per image)

You need at least 1500+ labeled images for decent accuracy.

### Step 3: Train

```powershell
python training/train.py
```

Training takes ~30-60 minutes on CPU. The trained model (5 ký tự) is saved to `captcha_model_v6.pth`.

Expected results:
- 1500 labeled images: ~90-95% accuracy
- 3000 labeled images: ~95-98% accuracy

### Step 4: Verify

Restart the server. It will automatically load the new model.
Use the demo page (`/demo`) to test accuracy with real CAPTCHAs.

### Continuous improvement

On the demo page, you can input correct text when the model predicts wrong. This saves the image + correct label to training data. After collecting enough corrections, click "Retrain" to improve the model.

## Deploy on Windows Server

### 1. Setup on server

```powershell
# Copy project files (exclude venv/, training_data/, collected_captchas/)
# Copy captcha_model.pth separately

# Install Python 3.11
# Install Tesseract OCR

cd C:\CaptchaService
py -3.11 -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m playwright install chromium
```

### 2. Configure

```powershell
copy .env.example .env
notepad .env
```

Set production values:
```
CAPTCHA_API_KEY=a-strong-random-key
PORT=8000
TCT_CAPTCHA_URL=https://hoadondientu.gdt.gov.vn:30000/captcha
TCT_AUTH_URL=https://hoadondientu.gdt.gov.vn:30000/security-taxpayer/authenticate
```

### 3. Run as Windows Service (NSSM)

```powershell
# Install NSSM
winget install nssm

# Create service
nssm install CaptchaService "C:\CaptchaService\venv\Scripts\python.exe" "C:\CaptchaService\main.py"
nssm set CaptchaService AppDirectory "C:\CaptchaService"
nssm set CaptchaService Start SERVICE_AUTO_START

# Start service
nssm start CaptchaService
```

### 4. Verify

```powershell
curl http://localhost:8000/health
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `CAPTCHA_API_KEY` | No | (empty) | API key for authentication. Empty = no auth (dev mode) |
| `PORT` | No | 8000 | Server port |
| `TCT_CAPTCHA_URL` | Yes | - | TCT CAPTCHA endpoint URL |
| `TCT_AUTH_URL` | Yes | - | TCT authentication endpoint URL |

## Tech Stack

- **Python 3.11** + **FastAPI** - API server
- **PyTorch** + **ResNet18** - CNN model for CAPTCHA recognition
- **EasyOCR** + **Tesseract** - Fallback OCR engines
- **Playwright** - SVG to PNG rendering (headless Chromium)
- **OpenCV** - Image preprocessing
