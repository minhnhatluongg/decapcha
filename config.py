import os
from dotenv import load_dotenv

# Load .env file nếu có
load_dotenv()

# Server
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))

# API Key authentication (rỗng = dev mode, không check)
API_KEY = os.environ.get("CAPTCHA_API_KEY", "")

# Thư mục lưu ảnh CAPTCHA đã giải (để sau training)
COLLECTED_DIR = os.path.join(os.path.dirname(__file__), "collected_captchas")

# TCT endpoints
TCT_CAPTCHA_URL = os.environ.get("TCT_CAPTCHA_URL", "")
TCT_AUTH_URL = os.environ.get("TCT_AUTH_URL", "")

# EasyOCR
OCR_LANGUAGES = ["en"]  # CAPTCHA TCT dùng ký tự latin + số
OCR_GPU = False          # Đổi True nếu có GPU NVIDIA
