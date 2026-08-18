"""
Proxy pool — rải request tới TCT qua nhiều IP proxy để tránh bị F5 chặn IP server.

Nguyên tắc:
  - Xoay vòng proxy theo TỪNG LƯỢT tra (1 lookup = 1 proxy cho cả phiên), vì cookie/anti-bot
    của TCT gắn với IP — đổi IP giữa chừng sẽ hỏng phiên.
  - Mỗi proxy có "sức khỏe": bị chặn -> nghỉ (cooldown) -> tự dùng lại sau.
  - REVERT: tắt qua config.USE_PROXY (env) hoặc set_enabled(False) lúc chạy -> gọi thẳng như cũ.
  - Tất cả proxy đang nghỉ -> trả None -> gọi thẳng (circuit breaker ở tcnnt_client sẽ đỡ).

File proxies.txt (mỗi dòng 1 proxy), hỗ trợ:
  http://user:pass@host:port
  host:port
  host:port:user:pass
"""
import os
import time
import threading
import itertools

import config

_lock = threading.Lock()
_proxies: list = []          # [(key, {"http":url,"https":url}), ...]
_blocked_until: dict = {}    # key -> timestamp hết cooldown
_rr = None                   # con trỏ xoay vòng
_enabled_override = None     # None = theo config; True/False = ép runtime (admin toggle)


def _parse_line(line: str):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "://" in line:
        return line
    parts = line.split(":")
    if len(parts) == 2:
        return f"http://{parts[0]}:{parts[1]}"
    if len(parts) == 4:
        host, port, user, pw = parts
        return f"http://{user}:{pw}@{host}:{port}"
    return None


def load() -> int:
    """Đọc proxies từ config.PROXY_FILE. Gọi lúc start + khi reload."""
    global _proxies, _rr, _blocked_until
    items = []
    path = config.PROXY_FILE
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for raw in f:
                url = _parse_line(raw)
                if url:
                    items.append((url, {"http": url, "https": url}))
    with _lock:
        _proxies = items
        _blocked_until = {}
        _rr = itertools.cycle(range(len(items))) if items else None
    print(f"[proxy_pool] Loaded {len(items)} proxies from {path}")
    return len(items)


def enabled() -> bool:
    if _enabled_override is not None:
        return _enabled_override
    return config.USE_PROXY


def set_enabled(val):
    """Bật/tắt lúc chạy. None = theo config (env)."""
    global _enabled_override
    _enabled_override = val


def acquire():
    """Lấy 1 proxy khỏe cho 1 lượt tra. Trả (proxies_dict, key) hoặc (None, 'direct')."""
    if not enabled():
        return None, "direct"
    now = time.time()
    with _lock:
        if not _proxies or _rr is None:
            return None, "direct"
        for _ in range(len(_proxies)):
            i = next(_rr)
            key, pd = _proxies[i]
            if _blocked_until.get(key, 0) <= now:
                return pd, key
        # tất cả đang cooldown -> gọi thẳng (circuit breaker sẽ đỡ)
        return None, "direct"


def report(key, ok: bool):
    """Báo kết quả 1 lượt: ok=True -> proxy khỏe; ok=False -> cho proxy nghỉ cooldown."""
    if not key or key == "direct":
        return
    with _lock:
        if ok:
            _blocked_until.pop(key, None)
        else:
            _blocked_until[key] = time.time() + config.PROXY_COOLDOWN


def status() -> dict:
    now = time.time()
    with _lock:
        blocked = sorted(k for k, t in _blocked_until.items() if t > now)
        return {
            "enabled": enabled(),
            "total": len(_proxies),
            "healthy": len(_proxies) - len(blocked),
            "blocked_count": len(blocked),
        }


# Nạp proxy ngay khi import (nếu có file)
load()
