"""
TCT Client - Gọi API Tổng cục Thuế lấy CAPTCHA và đăng nhập.
SVG → PNG bằng Playwright (ưu tiên) → cairosvg → PIL fallback.
"""
import os
import re
import ssl
import sys
import subprocess
import tempfile
import warnings

import requests
# pyrefly: ignore [missing-import]
import urllib3

import config

# Tat SSL warning + tat verify cho urllib stdlib (phong cac thu vien khac dung urllib)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=Warning)
try:
    _ctx = ssl._create_unverified_context
    ssl._create_default_https_context = _ctx
except AttributeError:
    pass

# Đường dẫn python trong venv
PYTHON_EXE = sys.executable
RENDERER_SCRIPT = os.path.join(os.path.dirname(__file__), "svg_renderer.py")


def _svg_to_png_playwright(svg_content: str) -> bytes:
    """
    Render SVG → PNG bằng Playwright subprocess.
    Throws nếu Playwright/Chromium không available.
    """
    tmp_dir = tempfile.gettempdir()
    html_path = os.path.join(tmp_dir, "captcha_render.html")
    png_path  = os.path.join(tmp_dir, "captcha_render.png")

    html = f"""<!DOCTYPE html>
<html><head><style>
    * {{ margin: 0; padding: 0; }}
    body {{ background: white; }}
    svg {{ transform: scale(3); transform-origin: top left; }}
</style></head>
<body>{svg_content}</body></html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    try:
        env = os.environ.copy()
        env["PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            [PYTHON_EXE, RENDERER_SCRIPT, html_path, png_path],
            capture_output=True, text=True, timeout=30, env=env,
        )

        if result.returncode != 0:
            err = (result.stderr or "").strip()
            out = (result.stdout or "").strip()
            raise RuntimeError(
                f"Playwright failed (code={result.returncode}) "
                f"stderr={err[:400] or '(empty)'} stdout={out[:200] or '(empty)'}"
            )

        if not os.path.exists(png_path):
            raise RuntimeError("PNG file not created by Playwright")

        with open(png_path, "rb") as f:
            return f.read()
    finally:
        for p in [html_path, png_path]:
            try: os.unlink(p)
            except OSError: pass


def _svg_to_png_cairosvg(svg_content: str) -> bytes:
    """
    Fallback 1: Render SVG → PNG bằng cairosvg (pure Python, không cần browser).
    pip install cairosvg
    """
    import cairosvg  # type: ignore
    return cairosvg.svg2png(
        bytestring=svg_content.encode("utf-8"),
        scale=3.0,
        background_color="white",
    )


def _extract_svg_text(svg_content: str) -> bytes:
    """
    Fallback 2: Dự phòng cuối - tạo ảnh trắng placeholder (224x80) nếu không render được.
    OCR engine vẫn sẽ chạy nhưng kết quả kém. Tốt hơn crash.
    """
    try:
        # pyrefly: ignore [missing-import]
        from PIL import Image, ImageDraw, ImageFont
        import io

        # Tìm text trong SVG (nếu có thẻ <text>)
        texts = re.findall(r'<text[^>]*>([^<]+)</text>', svg_content)
        label = "".join(texts).strip() or "??????"

        img = Image.new("RGB", (390, 150), "white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 40), label, fill="black")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        # Trả về ảnh trắng tối giản 1x1
        return (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )


def svg_to_png(svg_content: str) -> bytes:
    """
    Render SVG → PNG với multi-level fallback:
      1. Playwright (subprocess) - chất lượng tốt nhất
      2. cairosvg               - không cần browser
      3. PIL extract text        - last resort
    """
    # Level 1: Playwright
    try:
        return _svg_to_png_playwright(svg_content)
    except Exception as e:
        print(f"[svg_to_png] Playwright failed: {e}. Trying cairosvg...")

    # Level 2: cairosvg
    try:
        return _svg_to_png_cairosvg(svg_content)
    except Exception as e:
        print(f"[svg_to_png] cairosvg failed: {e}. Using PIL fallback...")

    # Level 3: PIL placeholder
    return _extract_svg_text(svg_content)


def fetch_captcha() -> tuple[bytes, str]:
    """Lấy CAPTCHA từ TCT → convert SVG sang PNG."""
    resp = requests.get(config.TCT_CAPTCHA_URL, timeout=15, verify=False)
    resp.raise_for_status()
    data = resp.json()

    captcha_key = data["key"]
    svg_content = data["content"]
    png_bytes = svg_to_png(svg_content)

    return png_bytes, captcha_key


def authenticate(username: str, password: str, ckey: str, cvalue: str) -> dict:
    """Đăng nhập TCT với CAPTCHA đã giải."""
    payload = {
        "username": username,
        "password": password,
        "ckey": ckey,
        "cvalue": cvalue,
    }

    resp = requests.post(
        config.TCT_AUTH_URL,
        json=payload,
        timeout=15,
        verify=False,
    )

    if resp.status_code == 200 and "token" in resp.text:
        return resp.json()
    else:
        return {"error": "Login Fail", "detail": resp.text}
