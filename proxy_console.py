"""
App console quản lý PROXY của CaptchaService.

Chạy TRÊN SERVER (cạnh config.py):  python proxy_console.py
Nó gọi vào admin API /admin/proxy của service ĐANG CHẠY -> bật/tắt/reload ăn ngay,
KHÔNG cần restart service.

Đọc port + mật khẩu admin từ .env (qua config.py). Nếu thiếu sẽ hỏi.
"""
import sys
import time

import requests

try:
    import config
    _PORT = config.PORT
    _TOKEN = config.ADMIN_PASSWORD
except Exception:
    _PORT, _TOKEN = 8000, ""

BASE = f"http://127.0.0.1:{_PORT}"


def _token() -> str:
    global _TOKEN
    if not _TOKEN:
        _TOKEN = input("Nhập ADMIN token (mật khẩu admin): ").strip()
    return _TOKEN


def call(action: str) -> dict:
    r = requests.get(f"{BASE}/admin/proxy", params={"action": action},
                     headers={"X-Admin-Token": _token()}, timeout=20)
    if r.status_code == 401 or r.status_code == 403:
        raise RuntimeError("Sai ADMIN token (401/403).")
    r.raise_for_status()
    return r.json()


def line(st: dict) -> str:
    on = "🟢 BẬT" if st.get("enabled") else "🔴 TẮT (gọi thẳng)"
    return (f"Proxy: {on}  |  Tổng: {st.get('total', 0)}  |  "
            f"Khỏe: {st.get('healthy', 0)}  |  Đang nghỉ: {st.get('blocked_count', 0)}")


def monitor(interval: int = 5):
    print(f"\n--- Theo dõi mỗi {interval}s (Ctrl+C để dừng) ---")
    try:
        while True:
            try:
                print(f"[{time.strftime('%H:%M:%S')}] {line(call('status'))}")
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Lỗi: {e}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n--- Dừng theo dõi ---")


def main():
    print("=" * 56)
    print("   QUẢN LÝ PROXY — CaptchaService")
    print(f"   Server: {BASE}")
    print("=" * 56)
    while True:
        try:
            print("\n" + line(call("status")))
        except Exception as e:
            print(f"\n[!] Không gọi được server: {e}")
            print("    - Service CaptchaService đang chạy chưa? Port đúng chưa? Token đúng chưa?")

        print("""
  1. BẬT proxy
  2. TẮT proxy (revert -> gọi thẳng IP server)
  3. Reload danh sách proxies.txt
  4. Theo dõi realtime
  0. Thoát
""")
        c = input("Chọn: ").strip()
        try:
            if c == "1":
                print("  => " + line(call("on")));      print("  ✅ Đã BẬT proxy.")
            elif c == "2":
                print("  => " + line(call("off")));     print("  ✅ Đã TẮT (đang gọi thẳng).")
            elif c == "3":
                print("  => " + line(call("reload")));  print("  ✅ Đã reload proxies.txt.")
            elif c == "4":
                monitor()
            elif c == "0":
                print("Tạm biệt!"); sys.exit(0)
            else:
                print("  (Chọn 0-4)")
        except Exception as e:
            print(f"  [!] Lỗi: {e}")


if __name__ == "__main__":
    main()
