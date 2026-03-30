"""
CaptchaService - API giải CAPTCHA thay thế TrueCaptcha.
Tương thích với flow cũ của WinTax_API (C#).

Chạy: uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import base64
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from ocr_engine import engine
from collector import save_captcha, get_stats
import tct_client
import config

app = FastAPI(
    title="CaptchaService",
    description="API giải CAPTCHA cho WinTax - thay thế TrueCaptcha",
    version="1.0.0",
)


# ===== API Key Middleware =====

# Paths không cần API key
PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/demo", "/"}


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Dev mode: API_KEY rỗng → bỏ qua auth
        if not config.API_KEY:
            return await call_next(request)

        # Public paths → không cần key
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Check API key từ header hoặc query param
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")

        if api_key != config.API_KEY:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or missing API Key", "hint": "Set header X-API-Key"}
            )

        return await call_next(request)


app.add_middleware(ApiKeyMiddleware)


# ===== Request/Response Models =====

class OCRRequest(BaseModel):
    """Tương thích với format TrueCaptcha cũ."""
    data: str  # base64 encoded image

    class Config:
        json_schema_extra = {
            "example": {
                "data": "<base64 PNG image>"
            }
        }


class OCRResponse(BaseModel):
    result: str  # text CAPTCHA đã giải
    saved: bool = False  # đã lưu để training chưa


class CaptchaSolveResponse(BaseModel):
    """Trả về giống captchaVal trong C#."""
    token: str  # text CAPTCHA đã giải
    key: str    # session key từ TCT


# ===== API Endpoints =====

@app.post("/one/gettext", response_model=OCRResponse)
async def ocr_solve(req: OCRRequest):
    """
    Giải CAPTCHA từ base64 image.

    Endpoint này TƯƠNG THÍCH với TrueCaptcha API cũ.
    Trong code C# chỉ cần đổi URL từ:
      https://api.apitruecaptcha.org/one/gettext
    thành:
      http://localhost:8000/one/gettext

    Và bỏ userid/apikey (không cần nữa).
    """
    try:
        img_bytes = base64.b64decode(req.data)
        text = engine.solve(img_bytes)

        if not text:
            raise HTTPException(status_code=422, detail="Không nhận dạng được CAPTCHA")

        # Lưu ảnh gốc + label để sau training
        saved_path = save_captcha(img_bytes, text)

        return OCRResponse(result=text, saved=saved_path is not None)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/captcha/solve", response_model=CaptchaSolveResponse)
async def captcha_solve():
    """
    Tự động: Lấy CAPTCHA từ TCT → Giải → Trả về {token, key}.

    Endpoint này thay thế hoàn toàn hàm getCapt() trong C#.
    Client chỉ cần gọi 1 API này là có đủ ckey + cvalue để login.
    """
    try:
        # Bước 1: Lấy CAPTCHA từ TCT
        png_bytes, captcha_key = tct_client.fetch_captcha()

        # Bước 2: Giải bằng OCR
        text = engine.solve(png_bytes)

        if not text:
            raise HTTPException(status_code=422, detail="Không giải được CAPTCHA")

        # Bước 3: Lưu để training
        save_captcha(png_bytes, text)

        return CaptchaSolveResponse(token=text, key=captcha_key)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def collection_stats():
    """Xem thống kê CAPTCHA đã thu thập."""
    return get_stats()


# ===== Background Crawler =====

import asyncio
import threading
from collector import save_captcha_raw

_crawl_task = None
_crawl_status = {"running": False, "collected": 0, "target": 0, "errors": 0}


def _crawl_worker(target: int, delay: float):
    """Worker chạy trong thread riêng: tải CAPTCHA liên tục."""
    import requests
    global _crawl_status

    _crawl_status = {"running": True, "collected": 0, "target": target, "errors": 0}

    for i in range(target):
        if not _crawl_status["running"]:
            print(f"[Crawler] Stopped by user at {_crawl_status['collected']} images.")
            break

        try:
            # Lấy CAPTCHA từ TCT
            resp = requests.get(config.TCT_CAPTCHA_URL, timeout=15, verify=False)
            resp.raise_for_status()
            data = resp.json()
            svg_content = data["content"]

            # Render SVG → PNG
            png_bytes = tct_client.svg_to_png(svg_content)

            # Lưu ảnh
            path = save_captcha_raw(png_bytes)
            _crawl_status["collected"] += 1

            if _crawl_status["collected"] % 50 == 0:
                print(f"[Crawler] Progress: {_crawl_status['collected']}/{target}")

        except Exception as e:
            _crawl_status["errors"] += 1
            print(f"[Crawler] Error #{_crawl_status['errors']}: {e}")

        # Delay giữa các request (tránh bị rate limit)
        import time
        time.sleep(delay)

    _crawl_status["running"] = False
    print(f"[Crawler] Done! Collected {_crawl_status['collected']} images.")


@app.post("/crawler/start")
async def crawler_start(target: int = 2000, delay: float = 1.0):
    """
    Bắt đầu tải CAPTCHA hàng loạt.

    - target: số ảnh cần tải (mặc định 2000)
    - delay: giây chờ giữa mỗi request (mặc định 1.0s, tránh rate limit)

    Ước tính thời gian: target × delay giây
    - 2000 ảnh × 1s = ~33 phút
    - 5000 ảnh × 0.5s = ~42 phút
    """
    global _crawl_task

    if _crawl_status["running"]:
        return {"error": "Crawler đang chạy", "status": _crawl_status}

    _crawl_task = threading.Thread(
        target=_crawl_worker,
        args=(target, delay),
        daemon=True
    )
    _crawl_task.start()

    return {
        "message": f"Crawler started! Đang tải {target} ảnh CAPTCHA...",
        "target": target,
        "delay": delay,
        "estimated_minutes": round(target * delay / 60, 1),
    }


@app.get("/crawler/status")
async def crawler_status():
    """Xem trạng thái crawler."""
    stats = get_stats()
    return {
        **_crawl_status,
        "total_in_folder": stats["total_images"],
    }


@app.post("/crawler/stop")
async def crawler_stop():
    """Dừng crawler."""
    _crawl_status["running"] = False
    return {"message": "Crawler stopping..."}


# ===== Feedback & Retrain =====

import csv
from pathlib import Path

LABELS_FILE = Path(__file__).parent / "labels.csv"

_train_status = {"running": False, "epoch": 0, "total_epochs": 0, "val_acc": 0, "message": ""}


class FeedbackRequest(BaseModel):
    image_base64: str
    correct_label: str


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """
    Nhận feedback: lưu ảnh + label đúng vào training_data/ + labels.csv.
    Gọi mỗi khi user nhập Actual text trên demo page.
    """
    if not req.correct_label.strip() or len(req.correct_label.strip()) < 3:
        raise HTTPException(400, "Label phải ít nhất 3 ký tự")

    label = req.correct_label.strip().upper()

    # Lưu ảnh
    img_bytes = base64.b64decode(req.image_base64)
    filepath = save_captcha_raw(img_bytes)
    if not filepath:
        raise HTTPException(500, "Không thể lưu ảnh")

    filename = os.path.basename(filepath)

    # Append vào labels.csv
    with open(LABELS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([filename, label])

    # Đếm tổng labels
    labeled_count = 0
    if LABELS_FILE.exists():
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            labeled_count = sum(1 for _ in csv.reader(f))

    return {"ok": True, "filename": filename, "label": label, "total_labels": labeled_count}


def _train_worker():
    """Chạy train trong thread riêng."""
    global _train_status
    import subprocess
    import sys

    _train_status = {"running": True, "epoch": 0, "total_epochs": 80, "val_acc": 0, "message": "Training..."}

    try:
        result = subprocess.run(
            [sys.executable, "train.py"],
            capture_output=True, text=True, timeout=3600,
            cwd=str(Path(__file__).parent),
        )

        # Parse kết quả
        output = result.stdout
        if "Best Val Accuracy:" in output:
            import re
            match = re.search(r"Best Val Accuracy:\s*([\d.]+)%", output)
            val_acc = float(match.group(1)) if match else 0
            _train_status["val_acc"] = val_acc
            _train_status["message"] = f"Done! Val Accuracy: {val_acc}%"

            # Reload model trong OCR engine
            engine._custom_model_loaded = False
            engine._custom_model = None
        else:
            _train_status["message"] = f"Train finished. Output: {output[-200:]}"

        if result.returncode != 0:
            _train_status["message"] = f"Error: {result.stderr[-300:]}"

    except Exception as e:
        _train_status["message"] = f"Error: {str(e)}"
    finally:
        _train_status["running"] = False


@app.post("/api/retrain")
async def retrain_model():
    """Trigger retrain model từ labels.csv hiện tại. Chạy background."""
    if _train_status["running"]:
        return {"error": "Đang train rồi!", "status": _train_status}

    t = threading.Thread(target=_train_worker, daemon=True)
    t.start()

    return {"message": "Training started! Chờ ~30-40 phút.", "status": _train_status}


@app.get("/api/train-status")
async def train_status():
    """Xem trạng thái training."""
    labeled_count = 0
    if LABELS_FILE.exists():
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            labeled_count = sum(1 for _ in csv.reader(f))

    return {**_train_status, "total_labels": labeled_count}


@app.get("/health")
async def health():
    model_info = "CNN Model" if engine._custom_model else "EasyOCR + Tesseract"
    return {"status": "ok", "engine": model_info}


@app.get("/debug/captcha")
async def debug_captcha():
    """Debug: Xem raw SVG và PNG từ TCT."""
    import base64
    import config as cfg
    resp = __import__("requests").get(cfg.TCT_CAPTCHA_URL, timeout=15, verify=False)
    data = resp.json()

    svg_content = data["content"]
    png_bytes = tct_client.svg_to_png(svg_content)
    png_b64 = base64.b64encode(png_bytes).decode()

    return {
        "key": data["key"],
        "svg_raw": svg_content[:2000],
        "png_base64": png_b64,
        "png_preview": f"data:image/png;base64,{png_b64}",
    }


# ===== Demo UI =====

from fastapi.responses import HTMLResponse

DEMO_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>CAPTCHA Detector Demo</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0f0f23; color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif;
    display: flex; flex-direction: column; align-items: center; padding: 40px 20px;
    min-height: 100vh;
  }
  h1 { color: #00d2d3; margin-bottom: 8px; font-size: 1.6rem; }
  .subtitle { color: #636e72; margin-bottom: 30px; font-size: 0.9rem; }
  .card {
    background: #16213e; border: 2px solid #2d3436; border-radius: 16px;
    padding: 30px; width: 100%; max-width: 700px; text-align: center;
  }
  .captcha-box {
    background: #0a0a1a; border-radius: 12px; padding: 20px; margin-bottom: 20px;
    min-height: 140px; display: flex; align-items: center; justify-content: center;
  }
  .captcha-box img {
    max-width: 100%; height: auto; border-radius: 8px;
    image-rendering: auto;
  }
  .captcha-box .placeholder { color: #444; font-size: 1.1rem; }
  .result-box {
    background: #0a0a1a; border: 2px solid #2d3436; border-radius: 12px;
    padding: 16px; margin: 16px 0; min-height: 60px;
    display: flex; align-items: center; justify-content: center; gap: 16px;
  }
  .result-text {
    font-size: 2.2rem; font-family: monospace; font-weight: 700;
    letter-spacing: 8px; color: #00d2d3;
  }
  .result-conf { color: #636e72; font-size: 0.85rem; }
  .btn-row { display: flex; gap: 12px; justify-content: center; margin-top: 20px; }
  button {
    padding: 12px 28px; font-size: 1rem; font-weight: 600; border: none;
    border-radius: 10px; cursor: pointer; transition: all 0.2s;
  }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-load { background: #0984e3; color: #fff; }
  .btn-load:hover:not(:disabled) { background: #0773c5; }
  .btn-detect { background: #00b894; color: #fff; }
  .btn-detect:hover:not(:disabled) { background: #00a381; }
  .btn-both { background: #6c5ce7; color: #fff; }
  .btn-both:hover:not(:disabled) { background: #5a4bd1; }
  .status { margin-top: 16px; font-size: 0.85rem; color: #636e72; min-height: 20px; }
  .history { margin-top: 24px; width: 100%; }
  .history h3 { color: #636e72; font-size: 0.85rem; margin-bottom: 8px; font-weight: 400; }
  .history table { width: 100%; border-collapse: collapse; }
  .history th { color: #636e72; font-size: 0.75rem; text-align: left; padding: 4px 8px; border-bottom: 1px solid #2d3436; }
  .history td { padding: 6px 8px; font-size: 0.85rem; border-bottom: 1px solid #1a1a2e; font-family: monospace; }
  .history .correct { color: #00b894; }
  .history .wrong { color: #d63031; }
  .match-input {
    background: #0f3460; border: 2px solid #3282b8; border-radius: 6px;
    color: #fff; padding: 4px 8px; font-family: monospace; font-size: 0.85rem;
    text-transform: uppercase; width: 85px; text-align: center;
  }
  .match-input.saved { border-color: #00b894; background: #0a3d2e; }
  .btn-fix {
    background: #e17055; color: #fff; border: none; border-radius: 6px;
    padding: 4px 10px; font-size: 0.75rem; cursor: pointer; font-weight: 600;
  }
  .btn-fix:hover { background: #d35400; }
  .btn-fix.done { background: #00b894; cursor: default; }
  .toast {
    position: fixed; top: 20px; right: 20px; padding: 10px 20px; border-radius: 8px;
    color: #fff; font-weight: 500; opacity: 0; transition: opacity 0.3s; z-index: 100;
    pointer-events: none;
  }
  .toast.show { opacity: 1; }
  .train-info {
    margin-top: 12px; padding: 8px 16px; background: #0a0a1a; border-radius: 8px;
    font-size: 0.8rem; color: #636e72;
  }
  .train-info .count { color: #feca57; font-weight: 600; }
  .stats { display: flex; gap: 20px; justify-content: center; margin-top: 12px; }
  .stat { text-align: center; }
  .stat .num { font-size: 1.4rem; font-weight: 700; color: #feca57; }
  .stat .lbl { font-size: 0.7rem; color: #636e72; }
</style>
</head>
<body>
  <h1>CAPTCHA Detector</h1>
  <p class="subtitle">Load CAPTCHA tu TCT → Detect bang CNN Model (95.4% accuracy)</p>

  <div class="card">
    <div class="captcha-box" id="captchaBox">
      <span class="placeholder">Nhan "Load" de lay CAPTCHA</span>
    </div>

    <div class="result-box" id="resultBox" style="display:none;">
      <div>
        <div class="result-text" id="resultText"></div>
        <div class="result-conf" id="resultConf"></div>
      </div>
    </div>

    <div class="btn-row">
      <button class="btn-load" id="btnLoad" onclick="loadCaptcha()">Load CAPTCHA</button>
      <button class="btn-detect" id="btnDetect" onclick="detectCaptcha()" disabled>Detect</button>
      <button class="btn-both" id="btnBoth" onclick="loadAndDetect()">Load + Detect</button>
      <button style="background:#fdcb6e;color:#000;" onclick="retrain()">Retrain</button>
    </div>

    <div class="status" id="status"></div>

    <div class="stats" id="statsBox" style="display:none;">
      <div class="stat"><div class="num" id="totalCount">0</div><div class="lbl">Total</div></div>
      <div class="stat"><div class="num" id="correctCount">0</div><div class="lbl">Correct</div></div>
      <div class="stat"><div class="num" id="accuracyPct">0%</div><div class="lbl">Accuracy</div></div>
    </div>

    <div class="train-info" id="trainInfo">
      Training data: <span class="count" id="labelCount">0</span> labels |
      New feedback: <span class="count" id="feedbackCount">0</span>
    </div>
  </div>

  <div id="toast" class="toast"></div>

  <div class="card history" id="historyCard" style="margin-top:20px; display:none;">
    <h3>History - Nhap text dung, nhan Fix de luu vao training data</h3>
    <table>
      <thead><tr><th>#</th><th>Image</th><th>Predicted</th><th>Actual</th><th></th><th>Match</th></tr></thead>
      <tbody id="historyBody"></tbody>
    </table>
  </div>

<script>
let currentImage = null;
let currentKey = null;
let history = [];
let totalChecked = 0;
let totalCorrect = 0;
let feedbackCount = 0;

function setStatus(msg) { document.getElementById('status').textContent = msg; }

function showToast(msg, color) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.background = color || '#00b894';
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2000);
}

function updateTrainInfo() {
  fetch('/api/train-status').then(r => r.json()).then(data => {
    document.getElementById('labelCount').textContent = data.total_labels || 0;
  }).catch(() => {});
  document.getElementById('feedbackCount').textContent = feedbackCount;
}

async function loadCaptcha() {
  const btn = document.getElementById('btnLoad');
  btn.disabled = true;
  setStatus('Dang lay CAPTCHA tu TCT...');
  document.getElementById('resultBox').style.display = 'none';

  try {
    const res = await fetch('/debug/captcha');
    const data = await res.json();
    currentImage = data.png_base64;
    currentKey = data.key;

    document.getElementById('captchaBox').innerHTML =
      '<img src="data:image/png;base64,' + currentImage + '" alt="CAPTCHA">';
    document.getElementById('btnDetect').disabled = false;
    setStatus('CAPTCHA loaded! Nhan "Detect" de nhan dang.');
  } catch(e) {
    setStatus('Loi: ' + e.message);
  }
  btn.disabled = false;
}

async function detectCaptcha() {
  if (!currentImage) return;
  const btn = document.getElementById('btnDetect');
  btn.disabled = true;
  setStatus('Dang nhan dang...');

  try {
    const res = await fetch('/one/gettext', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({data: currentImage})
    });
    const data = await res.json();

    document.getElementById('resultText').textContent = data.result;
    document.getElementById('resultConf').textContent = '';
    document.getElementById('resultBox').style.display = '';
    setStatus('Done!');

    addHistory(data.result);
  } catch(e) {
    setStatus('Loi: ' + e.message);
  }
  btn.disabled = false;
}

async function loadAndDetect() {
  await loadCaptcha();
  if (currentImage) await detectCaptcha();
}

function addHistory(predicted) {
  const idx = history.length + 1;
  history.push({idx, predicted, actual: '', image: currentImage, saved: false});

  const tbody = document.getElementById('historyBody');
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>${idx}</td>
    <td><img src="data:image/png;base64,${currentImage}" style="height:35px;border-radius:4px;"></td>
    <td style="color:#00d2d3;font-weight:700;letter-spacing:2px;">${predicted}</td>
    <td><input class="match-input" id="actual_${idx}" placeholder="..." onkeydown="if(event.key==='Enter'){fixLabel(${idx})}"></td>
    <td><button class="btn-fix" id="fix_${idx}" onclick="fixLabel(${idx})">Fix</button></td>
    <td id="match_${idx}">-</td>
  `;
  tbody.insertBefore(tr, tbody.firstChild);

  document.getElementById('historyCard').style.display = '';
  updateTrainInfo();
}

async function fixLabel(idx) {
  const inputEl = document.getElementById('actual_' + idx);
  const input = inputEl.value.toUpperCase().trim();
  if (!input) { inputEl.focus(); return; }

  const item = history[idx - 1];
  if (item.saved) { showToast('Da luu roi!', '#636e72'); return; }

  item.actual = input;
  const match = item.predicted === input;

  const cell = document.getElementById('match_' + idx);
  cell.textContent = match ? 'OK' : 'SAI';
  cell.className = match ? 'correct' : 'wrong';

  // Save feedback to training data
  const fixBtn = document.getElementById('fix_' + idx);
  fixBtn.textContent = '...';
  fixBtn.disabled = true;

  try {
    const res = await fetch('/api/feedback', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({image_base64: item.image, correct_label: input})
    });
    const data = await res.json();

    if (data.ok) {
      item.saved = true;
      feedbackCount++;
      inputEl.classList.add('saved');
      inputEl.readOnly = true;
      fixBtn.textContent = 'Saved';
      fixBtn.className = 'btn-fix done';
      showToast('Saved: ' + input + ' → ' + data.filename, '#00b894');
    } else {
      fixBtn.textContent = 'Fix';
      fixBtn.disabled = false;
      showToast('Error: ' + (data.detail || 'unknown'), '#d63031');
    }
  } catch(e) {
    fixBtn.textContent = 'Fix';
    fixBtn.disabled = false;
    showToast('Network error', '#d63031');
  }

  // Recalculate stats
  totalChecked = 0; totalCorrect = 0;
  history.forEach(h => {
    if (h.actual) {
      totalChecked++;
      if (h.predicted === h.actual) totalCorrect++;
    }
  });

  document.getElementById('statsBox').style.display = '';
  document.getElementById('totalCount').textContent = totalChecked;
  document.getElementById('correctCount').textContent = totalCorrect;
  document.getElementById('accuracyPct').textContent =
    totalChecked > 0 ? Math.round(totalCorrect/totalChecked*100) + '%' : '0%';

  updateTrainInfo();
}

async function retrain() {
  if (!confirm('Bat dau retrain model? (~30-40 phut)')) return;
  try {
    const res = await fetch('/api/retrain', {method: 'POST'});
    const data = await res.json();
    setStatus(data.message || data.error);
    if (!data.error) pollTrainStatus();
  } catch(e) { setStatus('Retrain error: ' + e.message); }
}

async function pollTrainStatus() {
  const poll = setInterval(async () => {
    try {
      const res = await fetch('/api/train-status');
      const data = await res.json();
      setStatus('Training... ' + data.message);
      if (!data.running) {
        clearInterval(poll);
        setStatus('Train done! ' + data.message);
      }
    } catch(e) { clearInterval(poll); }
  }, 5000);
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); loadAndDetect(); }
});

// Init
updateTrainInfo();
</script>
</body>
</html>"""


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Demo UI: Load CAPTCHA từ TCT → Detect bằng CNN Model."""
    return DEMO_HTML


# ===== Chạy trực tiếp =====

if __name__ == "__main__":
    import uvicorn
    import config
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
