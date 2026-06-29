"""
CAPTCHA Labeling Tool - Standalone FastAPI app
Run: python training/labeler.py   (từ thư mục gốc project)
Open: http://localhost:8001
"""

import base64
import csv
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Dữ liệu nằm ở thư mục gốc project (cha của training/)
ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = ROOT / "training_data_5"
LABELS_FILE = ROOT / "labels_5.csv"

app = FastAPI(title="CAPTCHA Labeler")


# --- Data helpers ---

def get_all_images() -> list[str]:
    if not TRAINING_DIR.exists():
        return []
    files = sorted(f.name for f in TRAINING_DIR.iterdir() if f.suffix.lower() == ".png")
    return files


def get_labeled() -> dict[str, str]:
    labeled = {}
    if LABELS_FILE.exists():
        with open(LABELS_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    labeled[row[0]] = row[1]
    return labeled


def save_label(filename: str, label: str):
    labeled = get_labeled()
    labeled[filename] = label
    with open(LABELS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for k, v in sorted(labeled.items()):
            writer.writerow([k, v])


def get_skipped() -> set[str]:
    if not hasattr(app.state, "skipped"):
        app.state.skipped = set()
    return app.state.skipped


def get_history() -> list[dict]:
    if not hasattr(app.state, "history"):
        app.state.history = []
    return app.state.history


def add_history(filename: str, label: str):
    h = get_history()
    h.append({"filename": filename, "label": label})
    if len(h) > 5:
        del h[:-5]


def find_next_unlabeled() -> str | None:
    all_images = get_all_images()
    labeled = get_labeled()
    skipped = get_skipped()
    for img in all_images:
        if img not in labeled and img not in skipped:
            return img
    # If all remaining are skipped, reset skips and try again
    if skipped:
        skipped.clear()
        for img in all_images:
            if img not in labeled:
                return img
    return None


# --- Models ---

class LabelRequest(BaseModel):
    filename: str
    label: str


# --- API endpoints ---

@app.get("/api/stats")
def stats():
    all_images = get_all_images()
    labeled = get_labeled()
    total = len(all_images)
    labeled_count = len(labeled)
    return {
        "total": total,
        "labeled": labeled_count,
        "remaining": total - labeled_count,
        "percent": round(labeled_count / total * 100, 1) if total > 0 else 0,
    }


@app.get("/api/next")
def next_image():
    filename = find_next_unlabeled()
    if filename is None:
        return {"done": True, "message": "All images have been labeled!"}

    img_path = TRAINING_DIR / filename
    if not img_path.exists():
        raise HTTPException(404, f"Image not found: {filename}")

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    all_images = get_all_images()
    labeled = get_labeled()
    total = len(all_images)
    labeled_count = len(labeled)

    return {
        "done": False,
        "filename": filename,
        "image": b64,
        "total": total,
        "labeled": labeled_count,
        "remaining": total - labeled_count,
        "percent": round(labeled_count / total * 100, 1) if total > 0 else 0,
        "history": get_history(),
    }


@app.post("/api/label")
def label_image(req: LabelRequest):
    if not req.label.strip():
        raise HTTPException(400, "Label cannot be empty")

    all_images = get_all_images()
    if req.filename not in all_images:
        raise HTTPException(404, f"Image not found: {req.filename}")

    label = req.label.strip().upper()
    save_label(req.filename, label)
    add_history(req.filename, label)

    # Remove from skipped if it was there
    get_skipped().discard(req.filename)

    return {"ok": True, "filename": req.filename, "label": label}


@app.get("/api/skip")
def skip_image():
    filename = find_next_unlabeled()
    if filename:
        get_skipped().add(filename)
    return {"ok": True, "skipped": filename}


@app.post("/api/undo")
def undo_last():
    """Undo the last labeled image - remove from labels.csv and reload it"""
    h = get_history()
    if not h:
        raise HTTPException(400, "No history to undo")

    last = h.pop()  # remove from history
    filename = last["filename"]
    old_label = last["label"]

    # Remove from labels.csv
    labeled = get_labeled()
    if filename in labeled:
        del labeled[filename]
        with open(LABELS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for k, v in sorted(labeled.items()):
                writer.writerow([k, v])

    # Return the image to re-label
    img_path = TRAINING_DIR / filename
    if not img_path.exists():
        raise HTTPException(404, f"Image not found: {filename}")

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    all_images = get_all_images()
    total = len(all_images)
    labeled_count = len(labeled)

    return {
        "ok": True,
        "undone": {"filename": filename, "label": old_label},
        "filename": filename,
        "image": b64,
        "total": total,
        "labeled": labeled_count,
        "remaining": total - labeled_count,
        "percent": round(labeled_count / total * 100, 1) if total > 0 else 0,
        "history": h,
    }


@app.get("/api/image/{filename}")
def serve_image(filename: str):
    img_path = TRAINING_DIR / filename
    if not img_path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(img_path, media_type="image/png")


# --- HTML UI ---

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CAPTCHA Labeler</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif;
    display: flex; flex-direction: column; align-items: center; min-height: 100vh;
    padding: 30px 20px;
  }
  h1 { font-size: 1.4rem; color: #7f8fa6; margin-bottom: 8px; font-weight: 400; }
  .progress {
    font-size: 1.1rem; color: #00d2d3; margin-bottom: 24px; font-variant-numeric: tabular-nums;
  }
  .progress .pct { color: #feca57; font-weight: 600; }
  .filename { color: #636e72; font-size: 0.85rem; margin-bottom: 10px; font-family: monospace; }
  .image-box {
    background: #16213e; border: 2px solid #2d3436; border-radius: 12px;
    padding: 20px; margin-bottom: 20px; display: flex; align-items: center;
    justify-content: center; min-height: 180px;
  }
  .image-box img {
    max-width: 800px; width: 100%; height: auto;
    image-rendering: auto; border-radius: 6px;
  }
  .input-row { display: flex; gap: 10px; margin-bottom: 16px; width: 100%; max-width: 500px; }
  #labelInput {
    flex: 1; padding: 12px 16px; font-size: 1.4rem; font-family: monospace;
    background: #0f3460; border: 2px solid #3282b8; border-radius: 8px;
    color: #fff; text-transform: uppercase; letter-spacing: 3px; text-align: center;
    outline: none; transition: border-color 0.2s;
  }
  #labelInput:focus { border-color: #00d2d3; }
  .hint { color: #636e72; font-size: 0.8rem; margin-bottom: 24px; }
  .hint kbd {
    background: #2d3436; padding: 2px 7px; border-radius: 4px;
    font-family: monospace; font-size: 0.8rem; border: 1px solid #444;
  }
  .feedback {
    position: fixed; top: 20px; right: 20px; padding: 10px 20px; border-radius: 8px;
    font-size: 0.95rem; font-weight: 500; opacity: 0; transition: opacity 0.3s;
    pointer-events: none; z-index: 100;
  }
  .feedback.success { background: #00b894; color: #fff; }
  .feedback.error { background: #d63031; color: #fff; }
  .feedback.show { opacity: 1; }
  .history { margin-top: 10px; width: 100%; max-width: 500px; }
  .history h3 { font-size: 0.85rem; color: #636e72; margin-bottom: 6px; font-weight: 400; }
  .history-list { list-style: none; }
  .history-list li {
    display: flex; justify-content: space-between; padding: 4px 10px;
    font-size: 0.82rem; font-family: monospace; color: #7f8fa6;
    border-bottom: 1px solid #2d3436;
  }
  .history-list li .lbl { color: #00d2d3; font-weight: 600; }
  .done-msg {
    font-size: 1.6rem; color: #00b894; margin-top: 60px; text-align: center;
  }
  .done-msg p { font-size: 1rem; color: #7f8fa6; margin-top: 10px; }
</style>
</head>
<body>
  <h1>CAPTCHA Labeler</h1>
  <div class="progress" id="progress">Loading...</div>
  <div class="filename" id="filename"></div>
  <div class="image-box" id="imageBox">
    <img id="captchaImg" src="" alt="CAPTCHA" style="display:none;">
  </div>
  <div class="input-row">
    <input type="text" id="labelInput" placeholder="Type CAPTCHA text..." autocomplete="off" spellcheck="false">
  </div>
  <div class="hint">
    <kbd>Enter</kbd> submit &nbsp; <kbd>Ctrl+Z</kbd> undo last &nbsp; <kbd>Ctrl+S</kbd> skip
  </div>
  <div class="feedback" id="feedback"></div>
  <div class="history" id="historySection" style="display:none;">
    <h3>Recent labels</h3>
    <ul class="history-list" id="historyList"></ul>
  </div>

<script>
const img = document.getElementById('captchaImg');
const input = document.getElementById('labelInput');
const progress = document.getElementById('progress');
const filenameEl = document.getElementById('filename');
const feedback = document.getElementById('feedback');
const historySection = document.getElementById('historySection');
const historyList = document.getElementById('historyList');
const imageBox = document.getElementById('imageBox');

let currentFilename = null;
let feedbackTimer = null;

function showFeedback(msg, type) {
  feedback.textContent = msg;
  feedback.className = 'feedback ' + type + ' show';
  clearTimeout(feedbackTimer);
  feedbackTimer = setTimeout(() => { feedback.classList.remove('show'); }, 1500);
}

function updateProgress(data) {
  if (data.total !== undefined) {
    progress.innerHTML = `Labeled: <strong>${data.labeled}</strong> / ${data.total} (<span class="pct">${data.percent}%</span>)`;
  }
}

function updateHistory(history) {
  if (!history || history.length === 0) {
    historySection.style.display = 'none';
    return;
  }
  historySection.style.display = '';
  historyList.innerHTML = '';
  for (let i = history.length - 1; i >= 0; i--) {
    const li = document.createElement('li');
    li.innerHTML = `<span>${history[i].filename}</span><span class="lbl">${history[i].label}</span>`;
    historyList.appendChild(li);
  }
}

async function loadNext() {
  try {
    const res = await fetch('/api/next');
    const data = await res.json();
    if (data.done) {
      imageBox.innerHTML = '';
      filenameEl.textContent = '';
      input.style.display = 'none';
      const doneDiv = document.createElement('div');
      doneDiv.className = 'done-msg';
      doneDiv.innerHTML = '<div>All done!</div><p>All images have been labeled.</p>';
      imageBox.parentNode.insertBefore(doneDiv, imageBox.nextSibling);
      updateProgress({total: data.total || 0, labeled: data.total || 0, percent: 100});
      return;
    }
    currentFilename = data.filename;
    img.src = 'data:image/png;base64,' + data.image;
    img.style.display = '';
    filenameEl.textContent = data.filename;
    updateProgress(data);
    updateHistory(data.history);
    input.value = '';
    input.focus();
  } catch (e) {
    showFeedback('Failed to load image', 'error');
  }
}

async function submitLabel() {
  const label = input.value.trim();
  if (!label || !currentFilename) return;
  try {
    const res = await fetch('/api/label', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filename: currentFilename, label: label})
    });
    const data = await res.json();
    if (data.ok) {
      showFeedback(`${data.label} saved`, 'success');
      loadNext();
    } else {
      showFeedback('Error saving label', 'error');
    }
  } catch (e) {
    showFeedback('Network error', 'error');
  }
}

async function skipImage() {
  try {
    await fetch('/api/skip');
    showFeedback('Skipped', 'success');
    loadNext();
  } catch (e) {
    showFeedback('Skip failed', 'error');
  }
}

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    submitLabel();
  }
});

async function undoLast() {
  try {
    const res = await fetch('/api/undo', {method: 'POST'});
    if (!res.ok) { showFeedback('Nothing to undo', 'error'); return; }
    const data = await res.json();
    if (data.ok) {
      currentFilename = data.filename;
      img.src = 'data:image/png;base64,' + data.image;
      img.style.display = '';
      filenameEl.textContent = data.filename;
      updateProgress(data);
      updateHistory(data.history);
      input.value = data.undone.label;  // pre-fill old label for easy correction
      input.focus();
      input.select();
      showFeedback('Undo: ' + data.undone.filename, 'success');
    }
  } catch (e) {
    showFeedback('Undo failed', 'error');
  }
}

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === 'z') {
    e.preventDefault();
    undoLast();
  }
  if (e.ctrlKey && e.key === 's') {
    e.preventDefault();
    skipImage();
  }
});

// Auto-uppercase
input.addEventListener('input', () => {
  const pos = input.selectionStart;
  input.value = input.value.toUpperCase();
  input.setSelectionRange(pos, pos);
});

loadNext();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


if __name__ == "__main__":
    print("CAPTCHA Labeler running at http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
