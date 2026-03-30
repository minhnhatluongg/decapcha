"""
Script độc lập: Render SVG → PNG bằng Playwright.
Được gọi từ main process qua subprocess.

Usage: python svg_renderer.py <input.html> <output.png>
"""
import sys
from playwright.sync_api import sync_playwright


def render(html_path: str, png_path: str):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 600, "height": 120})

        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        page.set_content(html, wait_until="networkidle")
        page.screenshot(path=png_path, type="png")
        page.close()
        browser.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python svg_renderer.py <input.html> <output.png>")
        sys.exit(1)

    render(sys.argv[1], sys.argv[2])
