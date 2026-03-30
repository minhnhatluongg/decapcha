"""
OCR Engine - Giải CAPTCHA bằng Custom CNN Model (ưu tiên) + EasyOCR + Tesseract fallback.
"""
import easyocr
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
import base64
import re
import os
import time
from collections import Counter
from pathlib import Path

import torch

import config

# Tesseract path trên Windows
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    print(f"Tesseract found: {TESSERACT_PATH}")
else:
    print(f"WARNING: Tesseract not found at {TESSERACT_PATH}")


MODEL_PATH = Path(__file__).parent / "captcha_model.pth"


class OCREngine:
    """Multi-engine OCR: Custom CNN (ưu tiên) + EasyOCR + Tesseract fallback."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._reader = None
            cls._instance._custom_model = None
            cls._instance._custom_model_loaded = False
        return cls._instance

    def _load_custom_model(self):
        """Load trained CNN model nếu có."""
        if self._custom_model_loaded:
            return self._custom_model

        if MODEL_PATH.exists():
            try:
                from captcha_model import CaptchaCNN, IMG_WIDTH, IMG_HEIGHT
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)

                model = CaptchaCNN().to(device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()

                self._custom_model = {
                    "model": model,
                    "device": device,
                    "val_acc": checkpoint.get("val_acc", 0),
                }
                self._custom_model_loaded = True
                print(f"✅ Custom CNN model loaded! (Val accuracy: {checkpoint.get('val_acc', 0):.1f}%)")
                return self._custom_model
            except Exception as e:
                print(f"⚠ Failed to load custom model: {e}")
                self._custom_model_loaded = True
                return None
        else:
            print("ℹ No custom model found (captcha_model.pth). Using EasyOCR + Tesseract.")
            self._custom_model_loaded = True
            return None

    @property
    def reader(self) -> easyocr.Reader:
        if self._reader is None:
            print("Loading EasyOCR model...")
            self._reader = easyocr.Reader(
                config.OCR_LANGUAGES,
                gpu=config.OCR_GPU
            )
            print("EasyOCR model loaded.")
        return self._reader

    # ===== Image Utils =====

    def _auto_crop(self, img: np.ndarray, padding: int = 10) -> np.ndarray:
        """
        Tự động crop bỏ viền trắng thừa, chỉ giữ vùng có nội dung.
        """
        # Nếu ảnh trắng trên đen (text đen trên nền trắng)
        # Invert để tìm contour
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

        # Tìm vùng có nội dung
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return img

        x, y, w, h = cv2.boundingRect(coords)

        # Thêm padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + padding * 2)
        h = min(img.shape[0] - y, h + padding * 2)

        return img[y:y+h, x:x+w]

    def _to_grayscale(self, img_bytes: bytes) -> np.ndarray:
        """Đọc ảnh từ bytes → grayscale."""
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Không thể đọc ảnh")
        return img

    # ===== Preprocessing =====

    def preprocess_for_lines_removal(self, img_bytes: bytes) -> np.ndarray:
        """Preprocessing mạnh: loại bỏ đường kẻ nhiễu + auto crop."""
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Không thể đọc ảnh")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        _, binary = cv2.threshold(inverted, 80, 255, cv2.THRESH_BINARY)

        # Detect lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        d_kernel1 = np.eye(30, dtype=np.uint8)
        d_lines1 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, d_kernel1)

        d_kernel2 = np.fliplr(np.eye(30, dtype=np.uint8))
        d_lines2 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, d_kernel2)

        all_lines = cv2.add(h_lines, v_lines)
        all_lines = cv2.add(all_lines, d_lines1)
        all_lines = cv2.add(all_lines, d_lines2)

        line_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        all_lines = cv2.dilate(all_lines, line_dilate, iterations=1)

        cleaned = cv2.subtract(binary, all_lines)

        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.dilate(cleaned, repair_kernel, iterations=1)

        # Invert lại: chữ đen trên nền trắng
        result = cv2.bitwise_not(cleaned)

        # Auto crop bỏ viền trắng thừa
        result = self._auto_crop(result)

        return result

    def preprocess_light(self, img_bytes: bytes) -> np.ndarray:
        """Preprocessing nhẹ: grayscale + adaptive threshold + crop."""
        gray = self._to_grayscale(img_bytes)

        # Adaptive threshold tốt hơn Otsu cho ảnh có nhiễu
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=5
        )

        # Auto crop
        binary = self._auto_crop(binary)

        return binary

    # ===== OCR Engines =====

    def _clean_text(self, text: str) -> str:
        """Loại bỏ ký tự đặc biệt, chỉ giữ chữ + số, uppercase."""
        text = re.sub(r"[^a-zA-Z0-9]", "", text)
        return text.upper()

    def _ocr_easyocr(self, img: np.ndarray) -> tuple[str, float]:
        """Nhận dạng bằng EasyOCR."""
        results = self.reader.readtext(
            img,
            detail=1,
            allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            paragraph=False,
            text_threshold=0.3,
            low_text=0.3,
            width_ths=1.5,       # Gộp text gần nhau
            mag_ratio=1.5,       # Phóng to thêm để nhận dạng tốt hơn
        )
        text = ""
        total_conf = 0
        for (_, t, conf) in results:
            text += t
            total_conf += conf
        text = self._clean_text(text)
        avg_conf = total_conf / len(results) if results else 0
        return text, avg_conf

    def _ocr_tesseract(self, img: np.ndarray) -> tuple[str, float]:
        """Nhận dạng bằng Tesseract."""
        try:
            # PSM 8 = single word (CAPTCHA thường là 1 word liền)
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

            pil_img = Image.fromarray(img)

            data = pytesseract.image_to_data(
                pil_img,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )

            text = ""
            confs = []
            for i, word in enumerate(data["text"]):
                word = word.strip()
                if word:
                    text += word
                    conf = int(data["conf"][i])
                    if conf > 0:
                        confs.append(conf / 100.0)

            text = self._clean_text(text)
            avg_conf = sum(confs) / len(confs) if confs else 0
            return text, avg_conf

        except Exception as e:
            print(f"[Tesseract] Error: {e}")
            return "", 0

    # ===== Voting =====

    def _vote_per_char(self, candidates: list[tuple[str, float]]) -> str:
        """
        Vote theo từng vị trí ký tự, dùng confidence làm trọng số.
        """
        # Chỉ lấy candidates có ít nhất 4 ký tự
        valid_candidates = [(t, c) for t, c in candidates if len(t) >= 4]
        if not valid_candidates:
            # Fallback: trả kết quả confidence cao nhất
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0] if candidates else ""

        # Tìm độ dài phổ biến nhất
        lengths = [len(t) for t, c in valid_candidates]
        target_len = Counter(lengths).most_common(1)[0][0]

        # Lấy candidates có đúng độ dài
        matched = [(t, c) for t, c in valid_candidates if len(t) == target_len]
        if not matched:
            valid_candidates.sort(key=lambda x: x[1], reverse=True)
            return valid_candidates[0][0]

        # Vote từng vị trí
        result = ""
        for pos in range(target_len):
            char_votes = {}
            for text, conf in matched:
                ch = text[pos]
                char_votes[ch] = char_votes.get(ch, 0) + conf
            best_char = max(char_votes, key=char_votes.get)
            result += best_char

        return result

    # ===== Custom CNN =====

    def _preprocess_for_cnn(self, img_bytes: bytes) -> np.ndarray:
        """
        Preprocessing GIỐNG HỆT train.py:
        1. Remove noise lines
        2. Auto crop
        3. Resize về IMG_WIDTH x IMG_HEIGHT
        """
        from captcha_model import IMG_WIDTH, IMG_HEIGHT

        nparr = np.frombuffer(img_bytes, np.uint8)
        img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_color is None:
            raise ValueError("Cannot read image")

        # === Remove noise lines (giống train.py) ===
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        _, binary = cv2.threshold(inverted, 80, 255, cv2.THRESH_BINARY)

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        d_kernel1 = np.eye(30, dtype=np.uint8)
        d_lines1 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, d_kernel1)
        d_kernel2 = np.fliplr(np.eye(30, dtype=np.uint8))
        d_lines2 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, d_kernel2)

        all_lines = cv2.add(h_lines, d_lines1)
        all_lines = cv2.add(all_lines, d_lines2)

        line_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        all_lines = cv2.dilate(all_lines, line_dilate, iterations=1)

        cleaned = cv2.subtract(binary, all_lines)

        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.dilate(cleaned, repair_kernel, iterations=1)

        img = cv2.bitwise_not(cleaned)  # text đen trên nền trắng

        # === Auto crop (giống train.py) ===
        _, thresh = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            pad = 5
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(img.shape[1] - x, w + pad * 2)
            h = min(img.shape[0] - y, h + pad * 2)
            img = img[y:y+h, x:x+w]

        # === Resize ===
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        return img

    def _ocr_custom_cnn(self, img_bytes: bytes) -> tuple[str, float]:
        """Nhận dạng bằng custom trained CNN model."""
        model_info = self._load_custom_model()
        if model_info is None:
            return "", 0

        from captcha_model import IMG_WIDTH, IMG_HEIGHT, CAPTCHA_LEN, CHARS, decode_prediction

        model = model_info["model"]
        device = model_info["device"]

        # Preprocessing giống hệt training
        img = self._preprocess_for_cnn(img_bytes)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # (1, H, W)

        # Predict
        tensor = torch.tensor(img).unsqueeze(0).to(device)  # (1, 1, H, W)

        with torch.no_grad():
            outputs = model(tensor)

        # Decode + confidence
        result = ""
        total_conf = 0
        for i in range(CAPTCHA_LEN):
            probs = torch.softmax(outputs[i][0], dim=0)
            conf, idx = probs.max(0)
            result += CHARS[idx.item()]
            total_conf += conf.item()

        avg_conf = total_conf / CAPTCHA_LEN
        return result, avg_conf

    # ===== Main Solve =====

    def solve(self, img_bytes: bytes) -> str:
        """
        Giải CAPTCHA:
        1. Ưu tiên Custom CNN model (nếu đã train)
        2. Fallback: EasyOCR + Tesseract voting
        """
        # === Thử Custom CNN trước ===
        cnn_text, cnn_conf = self._ocr_custom_cnn(img_bytes)
        if cnn_text and cnn_conf > 0.5:
            print(f"[OCR] 🧠 CNN Model: '{cnn_text}' (conf: {cnn_conf:.2f})")
            return cnn_text

        # === Fallback: EasyOCR + Tesseract ===
        print("[OCR] CNN not available or low confidence, using EasyOCR + Tesseract...")

        ts = int(time.time() * 1000)
        debug_dir = os.path.join(os.path.dirname(__file__), "debug_images")
        os.makedirs(debug_dir, exist_ok=True)

        # Preprocessing
        processed = self.preprocess_for_lines_removal(img_bytes)
        light = self.preprocess_light(img_bytes)

        cv2.imwrite(os.path.join(debug_dir, f"{ts}_processed.png"), processed)
        cv2.imwrite(os.path.join(debug_dir, f"{ts}_light.png"), light)

        # Chạy 4 lần OCR
        easy_processed, easy_p_conf = self._ocr_easyocr(processed)
        easy_raw, easy_r_conf = self._ocr_easyocr(light)
        tess_processed, tess_p_conf = self._ocr_tesseract(processed)
        tess_raw, tess_r_conf = self._ocr_tesseract(light)

        print(f"[OCR] EasyOCR+processed:  '{easy_processed}' ({easy_p_conf:.2f})")
        print(f"[OCR] EasyOCR+light:      '{easy_raw}' ({easy_r_conf:.2f})")
        print(f"[OCR] Tesseract+processed:'{tess_processed}' ({tess_p_conf:.2f})")
        print(f"[OCR] Tesseract+light:    '{tess_raw}' ({tess_r_conf:.2f})")

        candidates = []
        if cnn_text:
            candidates.append((cnn_text, cnn_conf + 0.3))  # Boost CNN score
        if easy_processed:
            candidates.append((easy_processed, easy_p_conf))
        if easy_raw:
            candidates.append((easy_raw, easy_r_conf))
        if tess_processed:
            candidates.append((tess_processed, tess_p_conf))
        if tess_raw:
            candidates.append((tess_raw, tess_r_conf))

        if not candidates:
            return ""

        result = self._vote_per_char(candidates)
        print(f"[OCR] >>> VOTED RESULT: '{result}'")
        return result

    def solve_base64(self, b64_data: str) -> str:
        """Giải CAPTCHA từ base64 string."""
        img_bytes = base64.b64decode(b64_data)
        return self.solve(img_bytes)


# Global instance
engine = OCREngine()
