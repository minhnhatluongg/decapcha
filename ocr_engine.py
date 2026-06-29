"""
OCR Engine - Giải CAPTCHA bằng Custom CNN Model (ưu tiên) + EasyOCR + Tesseract fallback.

Model 6 ký tự (TCT - hoadondientu) : captcha_model.pth    + captcha_model_resnet6.py (ResNet18)   <- ĐANG DÙNG
Model 5 ký tự (TCNNT)              : captcha_model_v6.pth + captcha_model_v2.py       (ResNet+SE)
"""
# pyrefly: ignore [missing-import]
import easyocr
# pyrefly: ignore [missing-import]
import pytesseract
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image
import base64
import re
import os
import time
from collections import Counter
from pathlib import Path

# pyrefly: ignore [missing-import]
import torch
import config

# Tesseract path trên Windows
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    print(f"Tesseract found: {TESSERACT_PATH}")
else:
    print(f"WARNING: Tesseract not found at {TESSERACT_PATH}")

# ===== Chọn model đang dùng =====

# --- Model 6 ký tự (TCT - hoadondientu.gdt.gov.vn) - ResNet18 ---
MODEL_PATH      = Path(__file__).parent / "captcha_model.pth"
MODEL_MODULE    = "captcha_model_resnet6"   # captcha_model_resnet6.py

# --- Model 5 ký tự (TCNNT) - ResNet18 v6 ---
# MODEL_PATH    = Path(__file__).parent / "captcha_model_v6.pth"
# MODEL_MODULE  = "captcha_model_v2"


class OCREngine:
    """Multi-engine OCR: Custom CNN (ưu tiên) + EasyOCR + Tesseract fallback."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._reader          = None
            cls._instance._custom_model    = None
            cls._instance._custom_model_loaded = False
            # Model 5 ký tự (TCNNT) - nạp riêng, độc lập với model active ở trên
            cls._instance._model5          = None
            cls._instance._model5_loaded   = False
        return cls._instance

    # ===== Load Model =====

    def _load_custom_model(self):
        """Load trained CNN model từ MODEL_PATH + MODEL_MODULE."""
        if self._custom_model_loaded:
            return self._custom_model

        if not MODEL_PATH.exists():
            print(f"[INFO] No custom model found at {MODEL_PATH}. Using EasyOCR + Tesseract.")
            self._custom_model_loaded = True
            return None

        try:
            module      = __import__(MODEL_MODULE, fromlist=["CaptchaCNN"])
            CaptchaCNN  = module.CaptchaCNN
            device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint  = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)

            model = CaptchaCNN().to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            self._custom_model = {
                "model"  : model,
                "device" : device,
                "module" : module,
                "val_acc": checkpoint.get("val_acc", 0),
            }
            self._custom_model_loaded = True
            print(f"[OK] Custom CNN loaded! Module={MODEL_MODULE} | Val acc={checkpoint.get('val_acc', 0):.1f}%")
            return self._custom_model

        except Exception as e:
            print(f"[WARN] Failed to load custom model: {e}")
            self._custom_model_loaded = True
            return None

    # ===== Load Model 5 ký tự (TCNNT) - độc lập với model active =====

    def _load_model5(self):
        """Nạp model 5 ký tự (captcha_model_v6.pth + captcha_model_v2) cho TCNNT.

        Tách riêng khỏi _load_custom_model() để KHÔNG ảnh hưởng model 6 ký tự
        đang phục vụ /captcha/solve. Cache lại sau lần nạp đầu.
        """
        if self._model5_loaded:
            return self._model5

        path = Path(__file__).parent / "captcha_model_v6.pth"
        if not path.exists():
            print(f"[INFO] Khong tim thay model 5 ky tu tai {path}.")
            self._model5_loaded = True
            return None

        try:
            module     = __import__("captcha_model_v2", fromlist=["CaptchaCNN"])
            device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(str(path), map_location=device, weights_only=False)

            model = module.CaptchaCNN().to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            self._model5 = {"model": model, "device": device, "module": module}
            self._model5_loaded = True
            acc = checkpoint.get("val_full_acc", checkpoint.get("val_acc", 0)) or 0
            print(f"[OK] Model 5 ky tu loaded! captcha_model_v6.pth | Acc={acc:.1f}%")
            return self._model5
        except Exception as e:
            print(f"[WARN] Failed to load model 5 ky tu: {e}")
            self._model5_loaded = True
            return None

    def solve5(self, img_bytes: bytes) -> tuple[str, float]:
        """Giải CAPTCHA 5 ký tự (TCNNT) -> (text, avg_confidence).

        Trả về ("", 0.0) nếu model chưa nạp được.
        """
        info = self._load_model5()
        if info is None:
            return "", 0.0

        module = info["module"]
        model  = info["model"]
        device = info["device"]

        img    = self._preprocess_for_cnn_5char(img_bytes).astype(np.float32) / 255.0
        tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

        with torch.no_grad():
            outputs = model(tensor)

        result, total_conf = "", 0.0
        for i in range(module.CAPTCHA_LEN):
            probs    = torch.softmax(outputs[i][0], dim=0)
            conf, idx = probs.max(0)
            result    += module.CHARS[idx.item()]
            total_conf += conf.item()

        return result, total_conf / module.CAPTCHA_LEN

    # ===== EasyOCR reader (lazy) =====

    @property
    def reader(self) -> easyocr.Reader:
        if self._reader is None:
            print("Loading EasyOCR model...")
            self._reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=config.OCR_GPU)
            print("EasyOCR model loaded.")
        return self._reader

    # ===== Image Utils =====

    def _auto_crop(self, img: np.ndarray, padding: int = 10) -> np.ndarray:
        """Crop bỏ viền trắng thừa, chỉ giữ vùng có nội dung."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return img
        x, y, w, h = cv2.boundingRect(coords)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + padding * 2)
        h = min(img.shape[0] - y, h + padding * 2)
        return img[y:y+h, x:x+w]

    def _to_grayscale(self, img_bytes: bytes) -> np.ndarray:
        """Bytes → grayscale ndarray."""
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Không thể đọc ảnh")
        return img

    # ===== Preprocessing: 6 ký tự (TCT - SVG có đường kẻ nhiễu) =====

    def _preprocess_for_cnn_6char(self, img_bytes: bytes) -> np.ndarray:
        """
        Preprocessing cho model 6 ký tự (captcha_model_resnet6.py).
        Pipeline: BGR → Gray → Invert → Xóa đường kẻ → Crop → Resize
        """
        from captcha_model_resnet6 import IMG_WIDTH, IMG_HEIGHT

        nparr     = np.frombuffer(img_bytes, np.uint8)
        img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_color is None:
            raise ValueError("Cannot read image")

        gray     = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        _, binary = cv2.threshold(inverted, 80, 255, cv2.THRESH_BINARY)

        # Xóa đường kẻ ngang, chéo
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        h_lines  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        d_kernel1 = np.eye(30, dtype=np.uint8)
        d_lines1  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, d_kernel1)
        d_kernel2 = np.fliplr(np.eye(30, dtype=np.uint8))
        d_lines2  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, d_kernel2)

        all_lines = cv2.add(h_lines, d_lines1)
        all_lines = cv2.add(all_lines, d_lines2)
        all_lines = cv2.dilate(all_lines,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                               iterations=1)

        cleaned = cv2.subtract(binary, all_lines)
        cleaned = cv2.dilate(cleaned,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
                             iterations=1)

        img = cv2.bitwise_not(cleaned)   # chữ đen trên nền trắng

        # Crop
        _, thresh = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            pad = 5
            x = max(0, x - pad);  y = max(0, y - pad)
            w = min(img.shape[1] - x, w + pad * 2)
            h = min(img.shape[0] - y, h + pad * 2)
            img = img[y:y+h, x:x+w]

        return cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    # ===== Preprocessing: 5 ký tự (TCNNT - PNG sạch) =====

    def _preprocess_for_cnn_5char(self, img_bytes: bytes) -> np.ndarray:
        """
        Preprocessing cho model 5 ký tự (captcha_model_v2.py).
        Pipeline: Alpha → Gray → Invert → Resize  (giống CaptchaDataset._preprocess)
        """
        from captcha_model_v2 import IMG_WIDTH, IMG_HEIGHT

        nparr   = np.frombuffer(img_bytes, np.uint8)
        img_raw = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img_raw is None:
            raise ValueError("Cannot read image")

        # Xử lý alpha channel
        if img_raw.ndim == 3 and img_raw.shape[2] == 4:
            alpha     = img_raw[:, :, 3] / 255.0
            color     = img_raw[:, :, :3].astype(np.float32)
            white_bg  = np.ones_like(color) * 255.0
            img_color = (color * alpha[:, :, None] + white_bg * (1 - alpha[:, :, None])).astype(np.uint8)
        else:
            img_color = img_raw if img_raw.ndim == 3 else cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)

        gray     = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        return cv2.resize(inverted, (IMG_WIDTH, IMG_HEIGHT))

    # ===== Chọn preprocessing theo MODEL_MODULE =====

    def _preprocess_for_cnn(self, img_bytes: bytes) -> np.ndarray:
        """Dispatch đúng preprocessing theo model đang dùng."""
        if MODEL_MODULE == "captcha_model_v2":
            return self._preprocess_for_cnn_5char(img_bytes)
        else:
            return self._preprocess_for_cnn_6char(img_bytes)

    # ===== OCR: Custom CNN =====

    def _ocr_custom_cnn(self, img_bytes: bytes) -> tuple[str, float]:
        """Nhận dạng bằng custom trained CNN model."""
        model_info = self._load_custom_model()
        if model_info is None:
            return "", 0

        module = model_info["module"]
        model  = model_info["model"]
        device = model_info["device"]

        CAPTCHA_LEN = module.CAPTCHA_LEN
        CHARS       = module.CHARS

        img    = self._preprocess_for_cnn(img_bytes)
        img    = img.astype(np.float32) / 255.0
        img    = np.expand_dims(img, axis=0)          # (1, H, W)
        tensor = torch.tensor(img).unsqueeze(0).to(device)  # (1, 1, H, W)

        with torch.no_grad():
            outputs = model(tensor)

        result     = ""
        total_conf = 0.0
        for i in range(CAPTCHA_LEN):
            probs = torch.softmax(outputs[i][0], dim=0)
            conf, idx = probs.max(0)
            result     += CHARS[idx.item()]
            total_conf += conf.item()

        avg_conf = total_conf / CAPTCHA_LEN
        return result, avg_conf

    # ===== OCR: EasyOCR =====

    def _clean_text(self, text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]", "", text).upper()

    def _ocr_easyocr(self, img: np.ndarray) -> tuple[str, float]:
        results = self.reader.readtext(
            img, detail=1,
            allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            paragraph=False, text_threshold=0.3, low_text=0.3,
            width_ths=1.5, mag_ratio=1.5,
        )
        text = "".join(t for (_, t, _) in results)
        confs = [c for (_, _, c) in results]
        return self._clean_text(text), (sum(confs) / len(confs) if confs else 0)

    # ===== OCR: Tesseract =====

    def _ocr_tesseract(self, img: np.ndarray) -> tuple[str, float]:
        try:
            cfg  = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            data = pytesseract.image_to_data(
                Image.fromarray(img), config=cfg,
                output_type=pytesseract.Output.DICT
            )
            text  = ""
            confs = []
            for i, word in enumerate(data["text"]):
                word = word.strip()
                if word:
                    text += word
                    c = int(data["conf"][i])
                    if c > 0:
                        confs.append(c / 100.0)
            return self._clean_text(text), (sum(confs) / len(confs) if confs else 0)
        except Exception as e:
            print(f"[Tesseract] Error: {e}")
            return "", 0

    # ===== Voting =====

    def _normalize_candidate(self, text: str, target_len: int) -> str:
        """
        Normalize candidate thành đúng độ dài target_len.
        - Nếu quá ngắn: pad thêm ký tự phổ biến ở cuối
        - Nếu quá dài: crop từ giữa
        """
        if len(text) == target_len:
            return text
        
        if len(text) < target_len:
            # Pad với ký tự phổ biến nhất từ text
            if text:
                most_common_char = Counter(text).most_common(1)[0][0]
                return text + most_common_char * (target_len - len(text))
            else:
                return "A" * target_len
        
        # Quá dài: crop từ giữa
        start = (len(text) - target_len) // 2
        return text[start:start + target_len]

    def _vote_per_char(self, candidates: list[tuple[str, float]]) -> str:
        """
        Voting per character với độ dài cố định từ model.
        Normalize mọi candidate thành cùng độ dài trước khi voting.
        """
        model_info = self._load_custom_model()
        expected_len = 6  # Default fallback
        if model_info:
            expected_len = model_info["module"].CAPTCHA_LEN
        
        if not candidates:
            return "A" * expected_len
        
        # Normalize mọi candidate thành expected_len
        normalized = [(self._normalize_candidate(t, expected_len), c) for t, c in candidates]
        
        # Voting per position
        result = ""
        for pos in range(expected_len):
            votes = {}
            for text, conf in normalized:
                ch = text[pos]
                votes[ch] = votes.get(ch, 0) + conf
            # Lấy ký tự có vote cao nhất
            result += max(votes, key=votes.get) if votes else "A"
        
        return result

    # ===== Preprocessing fallback (EasyOCR/Tesseract) =====

    def preprocess_for_lines_removal(self, img_bytes: bytes) -> np.ndarray:
        """Preprocessing mạnh cho EasyOCR/Tesseract: xóa đường kẻ + crop."""
        nparr = np.frombuffer(img_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Không thể đọc ảnh")

        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        _, binary = cv2.threshold(inverted, 80, 255, cv2.THRESH_BINARY)

        for kernel in [
            cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40)),
            np.eye(30, dtype=np.uint8),
            np.fliplr(np.eye(30, dtype=np.uint8)),
        ]:
            lines   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary  = cv2.subtract(binary, cv2.dilate(lines,
                      cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1))

        cleaned = cv2.dilate(binary,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
                             iterations=1)
        return self._auto_crop(cv2.bitwise_not(cleaned))

    def preprocess_light(self, img_bytes: bytes) -> np.ndarray:
        """Preprocessing nhẹ cho EasyOCR/Tesseract: adaptive threshold + crop."""
        gray   = self._to_grayscale(img_bytes)
        binary = cv2.adaptiveThreshold(gray, 255,
                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
        return self._auto_crop(binary)

    # ===== Main Solve =====

    def solve(self, img_bytes: bytes) -> str:
        """
        Giải CAPTCHA bằng Custom CNN Model (6 ký tự).
        Luôn trả về 6 ký tự.
        """
        cnn_text, cnn_conf = self._ocr_custom_cnn(img_bytes)
        print(f"[OCR] CNN ({MODEL_MODULE}): '{cnn_text}' (conf: {cnn_conf:.2f})")
        
        if cnn_text:
            return cnn_text
        
        # Fallback: trả về 6 ký tự "A" nếu model không thể detect
        return "AAAAAA"

    def solve_base64(self, b64_data: str) -> str:
        img_bytes = base64.b64decode(b64_data)
        return self.solve(img_bytes)


# Global instance
engine = OCREngine()