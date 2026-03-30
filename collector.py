"""
Collector - Lưu ảnh CAPTCHA vào 1 thư mục duy nhất để sau labeling + training.

Cấu trúc thư mục:
  training_data/
    00001.png
    00002.png
    ...
"""
import os
import time
import threading

import config

# Thư mục lưu ảnh training (1 folder duy nhất)
TRAINING_DIR = os.path.join(os.path.dirname(__file__), "training_data")

# Đếm số ảnh đã lưu
_counter_lock = threading.Lock()


def _get_next_id() -> int:
    """Lấy ID tiếp theo dựa trên số file đã có."""
    os.makedirs(TRAINING_DIR, exist_ok=True)
    existing = [f for f in os.listdir(TRAINING_DIR) if f.endswith(".png")]
    return len(existing) + 1


def save_captcha_raw(img_bytes: bytes) -> str | None:
    """
    Lưu ảnh CAPTCHA gốc vào training_data/ (KHÔNG phân folder theo label).
    Tên file: 00001.png, 00002.png, ...
    Returns: đường dẫn file đã lưu.
    """
    try:
        os.makedirs(TRAINING_DIR, exist_ok=True)

        with _counter_lock:
            next_id = _get_next_id()
            filename = f"{next_id:05d}.png"
            filepath = os.path.join(TRAINING_DIR, filename)

            with open(filepath, "wb") as f:
                f.write(img_bytes)

        return filepath

    except Exception as e:
        print(f"[Collector] Lỗi lưu captcha: {e}")
        return None


def save_captcha(img_bytes: bytes, solved_text: str) -> str | None:
    """Backward compatible - lưu vào training_data/."""
    return save_captcha_raw(img_bytes)


def get_stats() -> dict:
    """Thống kê số lượng CAPTCHA đã thu thập."""
    if not os.path.exists(TRAINING_DIR):
        return {"total_images": 0, "directory": TRAINING_DIR}

    images = [f for f in os.listdir(TRAINING_DIR) if f.endswith(".png")]
    return {
        "total_images": len(images),
        "directory": TRAINING_DIR,
    }
