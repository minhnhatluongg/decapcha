import os
import csv
import time
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import từ file captcha_model.py
from captcha_model import (
    CaptchaCNN, CHARS, NUM_CHARS, CAPTCHA_LEN,
    IMG_WIDTH, IMG_HEIGHT, char_to_idx, decode_prediction
)

BASE_DIR = Path(__file__).parent
TRAINING_DIR = BASE_DIR / "training_data_5"
LABELS_FILE = BASE_DIR / "labels_5.csv"
MODEL_FILE = BASE_DIR / "captcha_model_v5.pth"

# ===== Hyperparameters =====
BATCH_SIZE = 64
EPOCHS = 100            # Tăng lên 100 vì có Early Stopping lo rồi
LEARNING_RATE = 0.001   # Mức vàng cho Mini-CNN
TRAIN_SPLIT = 0.85 
EARLY_STOP_PATIENCE = 30 

# ===== Dataset =====

class CaptchaDataset(Dataset):
    def __init__(self, samples: list[tuple[str, str]], augment: bool = False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        # Xoay nhẹ ảnh
        if random.random() < 0.5:
            angle = random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            # Quan trọng: Nền đã đảo thành đen nên borderValue=0
            img = cv2.warpAffine(img, M, (w, h), borderValue=0)
        return img

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 1. Đọc ảnh kèm kênh Alpha (IMREAD_UNCHANGED)
        img_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img_raw is None: raise ValueError(f"Lỗi đọc ảnh: {img_path}")

        # 2. Xử lý kênh Alpha (Biến vùng trong suốt thành nền trắng)
        if img_raw.shape[-1] == 4: 
            alpha = img_raw[:, :, 3] / 255.0
            color = img_raw[:, :, :3]
            white_bg = np.ones_like(color, dtype=np.uint8) * 255
            img_color = (color * alpha[:, :, np.newaxis] + white_bg * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
        else:
            img_color = img_raw

        # 3. Chuyển xám và Đảo màu (Biến Chữ đen nền trắng -> Chữ trắng nền đen)
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        img_final = cv2.bitwise_not(gray) 

        # 4. Resize về đúng kích thước config
        img_final = cv2.resize(img_final, (IMG_WIDTH, IMG_HEIGHT))
        
        # Debug file đầu tiên của mỗi epoch để kiểm tra AI thấy gì
        if idx == 0: cv2.imwrite("ai_debug_view.png", img_final)

        if self.augment:
            img_final = self._augment_image(img_final)

        # 5. Normalize & Tensor
        img_out = img_final.astype(np.float32) / 255.0
        img_out = np.expand_dims(img_out, axis=0) # Shape: (1, H, W)

        label_indices = [char_to_idx(c) for c in label.upper()]
        return torch.tensor(img_out), torch.tensor(label_indices, dtype=torch.long)

# ===== Data Loading =====

def load_data():
    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy {LABELS_FILE}")

    samples = []
    skipped = 0
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2: continue
            filename, label = row[0].strip(), row[1].strip().upper()
            
            # Chỉ lấy nhãn có đúng độ dài quy định (5 ký tự)
            if len(label) != CAPTCHA_LEN or not all(c in CHARS for c in label):
                skipped += 1
                continue

            img_path = str(TRAINING_DIR / filename)
            if os.path.exists(img_path):
                samples.append((img_path, label))
            else:
                skipped += 1

    if skipped > 0:
        print(f"⚠ Đã bỏ qua {skipped} mẫu không hợp lệ.")
    return samples

def split_data(samples):
    random.shuffle(samples)
    idx = int(len(samples) * TRAIN_SPLIT)
    return samples[:idx], samples[idx:]

# ===== Training =====

def train():
    print("=" * 60)
    print("   CAPTCHA TRAINING - MINI CNN v2")
    print("=" * 60)

    samples = load_data()
    print(f"📂 Tổng mẫu: {len(samples)}")
    if len(samples) < 100: return

    train_set, val_set = split_data(samples)
    train_loader = DataLoader(CaptchaDataset(train_set, augment=False), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CaptchaDataset(val_set, augment=False), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥 Device: {device}")

    model = CaptchaCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    # Khởi tạo kỷ lục
    best_char_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n🚀 Bắt đầu train {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        
        # --- Train ---
        model.train()
        train_loss, train_correct_chars, train_total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = sum(criterion(outputs[i], labels[:, i]) for i in range(CAPTCHA_LEN))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += images.size(0)
            for i in range(CAPTCHA_LEN):
                train_correct_chars += (outputs[i].argmax(1) == labels[:, i]).sum().item()

        # --- Validation ---
        model.eval()
        val_loss, val_correct_chars, val_correct_full, val_total = 0, 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = sum(criterion(outputs[i], labels[:, i]) for i in range(CAPTCHA_LEN))
                val_loss += loss.item()
                
                batch_size = images.size(0)
                val_total += batch_size
                
                # Check từng ký tự
                char_matches = torch.stack([outputs[i].argmax(1) == labels[:, i] for i in range(CAPTCHA_LEN)])
                val_correct_chars += char_matches.sum().item()
                # Check cả chuỗi
                val_correct_full += char_matches.all(dim=0).sum().item()

        # Tính toán chỉ số
        avg_val_loss = val_loss / len(val_loader)
        val_char_acc = (val_correct_chars / (val_total * CAPTCHA_LEN)) * 100
        val_full_acc = (val_correct_full / val_total) * 100
        
        scheduler.step(avg_val_loss)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:2d} | Loss: {avg_val_loss:.3f} | Char Acc: {val_char_acc:5.1f}% | Full Acc: {val_full_acc:5.1f}% | {elapsed:.1f}s")

        # --- Logic lưu Model (Ưu tiên Char Acc) ---
        if val_char_acc > best_char_acc or (abs(val_char_acc - best_char_acc) < 1e-7 and avg_val_loss < best_val_loss):
            best_char_acc = val_char_acc
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_char_acc": val_char_acc,
                "val_acc": val_full_acc,
                "epoch": epoch,
                "chars": CHARS,
                "captcha_len": CAPTCHA_LEN,
                "img_size": (IMG_HEIGHT, IMG_WIDTH),
            }, str(MODEL_FILE))
            print(f"   ✅ Đã lưu model tốt nhất!")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n⏹ Dừng sớm do không tiến bộ.")
                break

    # Quick test cuối cùng
    print("\n🧪 Quick test 10 mẫu ngẫu nhiên...")
    model.load_state_dict(torch.load(str(MODEL_FILE))["model_state_dict"])
    model.eval()
    with torch.no_grad():
        for i in range(min(10, len(val_set))):
            img, label = CaptchaDataset([val_set[i]])[0]
            out = model(img.unsqueeze(0).to(device))
            pred = decode_prediction(out)
            match = "✅" if pred == val_set[i][1] else "❌"
            print(f"   {match} Pred: {pred} | Real: {val_set[i][1]}")

if __name__ == "__main__":
    train()