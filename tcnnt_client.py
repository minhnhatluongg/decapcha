"""
TCNNT Client - Tra cứu thông tin người nộp thuế từ tracuunnt.gdt.gov.vn.

Flow (mstdn.jsp):
  1. GET trang form  -> lấy cookie phiên (JSESSIONID + F5 TS...) + scrape field ẩn (cm,...)
  2. GET captcha.png -> giải bằng model 5 ký tự (ocr_engine.solve5)
  3. POST mstdn.jsp  -> {cm, mst, fullname, address, cmt, captcha} cùng Session
  4. Parse bảng HTML kết quả

Lưu ý: Tổng cục Thuế có anti-bot (F5). Captcha có thể bị từ chối vài lần đầu dù
giải đúng -> retry trong 1 request, có backoff, đếm count_Try.
"""
import re
import html
import time
import uuid
import threading
import warnings

import requests
# pyrefly: ignore [missing-import]
import urllib3

import config
from ocr_engine import engine

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=Warning)

# Header giả lập browser thật để giảm khả năng bị anti-bot chặn
_BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def _text(raw: str) -> str:
    """Bỏ tag HTML + giải mã entity + gọn khoảng trắng."""
    return html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()


def _scrape_form_fields(page_html: str) -> dict:
    """Lấy mọi <input> trong form (giữ value mặc định, gồm field ẩn 'cm').

    Bỏ qua button/submit/reset. mst + captcha sẽ được ghi đè khi POST.
    """
    m = re.search(r'<form[^>]*name="myform".*?</form>', page_html, re.S | re.I)
    scope = m.group(0) if m else page_html

    fields: dict = {}
    for tag in re.findall(r"<input[^>]*>", scope, re.I):
        name = re.search(r'name="([^"]*)"', tag)
        if not name:
            continue
        typ = re.search(r'type="([^"]*)"', tag)
        if typ and typ.group(1).lower() in ("button", "submit", "reset", "image"):
            continue
        val = re.search(r'value="([^"]*)"', tag)
        fields[name.group(1)] = html.unescape(val.group(1)) if val else ""
    return fields


def parse_results(page_html: str) -> list[dict]:
    """Parse bảng kết quả (class ta_border) -> list dict mỗi dòng NNT."""
    rows: list[dict] = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page_html, re.S | re.I):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S | re.I)
        # Dòng dữ liệu: >=6 cột, cột đầu là STT (số)
        if len(tds) >= 6 and re.fullmatch(r"\d+", _text(tds[0])):
            rows.append({
                "stt": _text(tds[0]),
                "mst": _text(tds[1]),
                "name": _text(tds[2]),
                "address": _text(tds[3]),
                "tax_authority": _text(tds[4]),
                "status": _text(tds[5]),
            })
    return rows


# Các trạng thái phản hồi từ server
_CAPTCHA_ERR = ("nhập đúng mã", "nh&#7853;p &#273;&#250;ng m&#227;")   # captcha sai -> retry
_NO_RESULT   = ("không tìm thấy", "kh&#244;ng t&#236;m th&#7845;y")     # MST hợp lệ nhưng không có dữ liệu


def _classify(page_html: str) -> str:
    low = page_html.lower()
    if any(s in low for s in _CAPTCHA_ERR):
        return "captcha_error"
    if any(s in low for s in _NO_RESULT):
        return "no_result"
    return "blocked"   # form trả về nhưng không kết quả, không báo lỗi captcha -> nghi anti-bot


def _extract_notice(page_html: str) -> str:
    """Lấy đúng câu thông báo 'Không tìm thấy...' mà TCT trả về (bỏ tag + giải entity)."""
    m = re.search(r"Không tìm thấy[^.<]*\.?", _text(page_html))
    return m.group(0).strip() if m else ""


# ===== CIRCUIT BREAKER =====
# Khi TCT chặn IP: sau nhiều lần fail liên tiếp -> NGẮT hẳn (không gọi TCT nữa) trong 1 khoảng,
# để lưu lượng về ~0 giúp F5 tự gỡ chặn nhanh. Tránh vòng khuếch đại retry giữ block sống mãi.
_cb_lock = threading.Lock()
_cb_fail_streak = 0
_cb_open_until = 0.0
_CB_FAIL_THRESHOLD = 5     # 5 lần tra fail liên tiếp -> mở mạch
_CB_COOLDOWN = 120.0       # ngắt 120s: trả lỗi ngay, KHÔNG chạm TCT


def _cb_remaining() -> float:
    return max(0.0, _cb_open_until - time.time())


def _cb_record(success: bool) -> None:
    global _cb_fail_streak, _cb_open_until
    with _cb_lock:
        if success:
            _cb_fail_streak = 0
            _cb_open_until = 0.0
        else:
            _cb_fail_streak += 1
            if _cb_fail_streak >= _CB_FAIL_THRESHOLD:
                _cb_open_until = time.time() + _CB_COOLDOWN


def lookup_mst(mst: str, max_tries: int = 12, delay: float = 1.5,
               min_conf: float = 0.0, timeout: int = 20) -> dict:
    """Tra cứu MST, retry tới khi ra kết quả hoặc hết lượt.

    Returns dict:
      { mst, address, count_Try, found, results[], status, message }
    """
    mst = (mst or "").strip()
    if not mst:
        return {"mst": mst, "found": False, "count_Try": 0,
                "status": "error", "message": "MST trống", "results": []}

    # Circuit breaker: nếu đang bị chặn (mạch mở) -> trả lỗi NGAY, KHÔNG chạm TCT
    rem = _cb_remaining()
    if rem > 0:
        return {"mst": mst, "found": False, "count_Try": 0, "status": "circuit_open",
                "message": f"Tạm ngắt tra cứu {int(rem)}s do TCT đang chặn IP (tự thử lại sau).",
                "results": []}

    session = requests.Session()
    session.headers.update(_BASE_HEADERS)

    count_try = 0
    last_status = "init"
    block_streak = 0   # số lần liên tiếp bị chặn/lỗi mạng -> thoát sớm

    # Bước 1: warm-up trang form (lấy cookie + field ẩn)
    try:
        page = session.get(config.TCNNT_LOOKUP_URL, timeout=timeout, verify=False)
        form_fields = _scrape_form_fields(page.text)
    except requests.RequestException as e:
        _cb_record(False)
        return {"mst": mst, "found": False, "count_Try": 0, "status": "network_error",
                "message": f"Không tải được trang TCNNT: {type(e).__name__}", "results": []}

    post_headers = {
        "Referer": config.TCNNT_LOOKUP_URL,
        "Origin": config.TCNNT_ORIGIN,
        "Content-Type": "application/x-www-form-urlencoded",
    }

    for attempt in range(1, max_tries + 1):
        try:
            # Bước 2: lấy + giải captcha (uuid chống cache)
            cap = session.get(f"{config.TCNNT_PNG_URL}{uuid.uuid4()}",
                              timeout=timeout, verify=False)
            text, conf = engine.solve5(cap.content)
            count_try += 1

            if not text or conf < min_conf:
                last_status = "low_conf"
                time.sleep(delay)
                continue

            # Bước 3: submit (giữ field ẩn 'cm'..., ghi đè mst + captcha)
            # Captcha TCNNT là chữ thường -> lower() cho khớp (model xuất uppercase)
            data = {**form_fields, "mst": mst, "fullname": "", "address": "",
                    "cmt": "", "captcha": text.lower()}
            resp = session.post(config.TCNNT_LOOKUP_URL, data=data,
                                headers=post_headers, timeout=timeout, verify=False)

            # Bước 4: parse
            results = parse_results(resp.text)
            if results:
                primary = next((r for r in results if r["mst"] == mst), results[0])
                _cb_record(True)
                return {
                    "mst": mst,
                    "address": primary["address"],
                    "count_Try": count_try,
                    "found": True,
                    "status": "ok",
                    "message": f"Thành công sau {count_try} lần giải captcha",
                    "results": results,
                }

            last_status = _classify(resp.text)
            if last_status == "no_result":
                _cb_record(True)   # TCT phản hồi bình thường (không phải bị chặn)
                notice = _extract_notice(resp.text) or "Không tìm thấy người nộp thuế nào phù hợp."
                return {"mst": mst, "address": "", "count_Try": count_try,
                        "found": False, "status": "no_result",
                        "message": notice, "results": []}
            if last_status == "blocked":
                block_streak += 1
                if block_streak >= 3:   # bị chặn liên tục -> thoát sớm, đừng dội thêm
                    break
            else:
                block_streak = 0   # captcha_error: TCT vẫn phản hồi -> không tính là chặn
            # captcha_error / blocked -> retry với backoff nhẹ
            time.sleep(delay * (1 + 0.15 * attempt))

        except requests.RequestException as e:
            last_status = f"network:{type(e).__name__}"
            block_streak += 1
            if block_streak >= 3:
                break
            time.sleep(delay)
        except Exception as e:
            # Khi TCT chặn, captcha.png trả HTML/lỗi -> solve5 decode ảnh ném exception.
            # Bắt hết để KHÔNG bao giờ ném ra endpoint (tránh lỗi 500), coi như 1 lần thử hỏng.
            last_status = f"error:{type(e).__name__}"
            block_streak += 1
            if block_streak >= 3:
                break
            time.sleep(delay)

    _cb_record(False)
    return {
        "mst": mst, "address": "", "count_Try": count_try, "found": False,
        "status": "exhausted",
        "message": f"Chưa lấy được (lý do cuối: {last_status}). "
                   "Có thể bị anti-bot chặn IP - hệ thống tự lùi, thử lại sau.",
        "results": [],
    }
