import os
from dotenv import load_dotenv

# Load .env file nếu có
load_dotenv()

# Server
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))

# Thư mục lưu ảnh CAPTCHA đã giải (để sau training)
COLLECTED_DIR = os.path.join(os.path.dirname(__file__), "collected_captchas")

# TCT endpoints
TCT_CAPTCHA_URL = os.environ.get("TCT_CAPTCHA_URL", "")
TCT_AUTH_URL = os.environ.get("TCT_AUTH_URL", "")

# EasyOCR
OCR_LANGUAGES = ["en"]  # CAPTCHA TCT dùng ký tự latin + số
OCR_GPU = False          # Đổi True nếu có GPU NVIDIA

# Thư mục lưu ảnh CAPTCHA bộ 5 chữ số (TCT mới)
TRAINING_DATA_5 = os.path.join(os.path.dirname(__file__), "training_data_5")

# Tự động tạo folder nếu chưa có
for folder in [COLLECTED_DIR, TRAINING_DATA_5]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"[*] Created folder: {folder}")

# Endpoint trực tiếp lấy ảnh PNG của TCNNT
# Link bạn đưa: https://tracuunnt.gdt.gov.vn/tcnnt/captcha.png?uid=
TCNNT_PNG_URL = "https://tracuunnt.gdt.gov.vn/tcnnt/captcha.png?uid="

# ===== TCNNT - Tra cứu thông tin người nộp thuế (mstdn.jsp) =====
# Trang form tra cứu (GET để lấy cookie + scrape field, POST để submit)
TCNNT_LOOKUP_URL = "https://tracuunnt.gdt.gov.vn/tcnnt/mstdn.jsp"
TCNNT_ORIGIN     = "https://tracuunnt.gdt.gov.vn"

# ===== API key store + Admin dashboard (SQLite) =====
# Mật khẩu admin để vào /dashboard. BẮT BUỘC đặt trong .env khi chạy prod.
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "").strip()

# File SQLite lưu key + log (không commit, không deploy đè - giữ trên server)
KEYS_DB_PATH = os.environ.get(
    "KEYS_DB_PATH", os.path.join(os.path.dirname(__file__), "tcnnt_keys.db")
)

# Tự dọn log gọi API cũ hơn số ngày này
LOG_RETENTION_DAYS = int(os.environ.get("LOG_RETENTION_DAYS", 2))

# ===== Cache MST (giảm gọi TCT: cùng 1 MST tra lại trong TTL -> trả cache, KHÔNG chạm TCT) =====
# Thông tin NNT ít đổi -> TTL dài. Mặc định 7 ngày. 0 = tắt cache.
MST_CACHE_TTL = int(os.environ.get("MST_CACHE_TTL", 7 * 24 * 3600))