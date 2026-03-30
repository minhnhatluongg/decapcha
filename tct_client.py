"""
TCT Client - Gọi API Tổng cục Thuế lấy CAPTCHA và đăng nhập.
SVG → PNG bằng Playwright chạy trong subprocess riêng (tránh lỗi Windows asyncio).
"""
import os
import sys
import subprocess
import tempfile
import requests

import config

# Đường dẫn python trong venv
PYTHON_EXE = sys.executable
RENDERER_SCRIPT = os.path.join(os.path.dirname(__file__), "svg_renderer.py")


def svg_to_png(svg_content: str) -> bytes:
    """
    Render SVG → PNG bằng Playwright (chạy subprocess riêng).
    """
    tmp_dir = tempfile.gettempdir()
    html_path = os.path.join(tmp_dir, "captcha_render.html")
    png_path = os.path.join(tmp_dir, "captcha_render.png")

    # Tạo HTML chứa SVG
    html = f"""<!DOCTYPE html>
<html>
<head><style>
    * {{ margin: 0; padding: 0; }}
    body {{ background: white; }}
    svg {{ transform: scale(3); transform-origin: top left; }}
</style></head>
<body>{svg_content}</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    try:
        # Gọi renderer subprocess
        result = subprocess.run(
            [PYTHON_EXE, RENDERER_SCRIPT, html_path, png_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Renderer failed: {result.stderr}")

        if not os.path.exists(png_path):
            raise RuntimeError("PNG file not created")

        with open(png_path, "rb") as f:
            return f.read()

    finally:
        for p in [html_path, png_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


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
