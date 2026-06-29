"""
train.py - Trainer cho model 5 ký tự (TCNNT).

Kiến trúc: captcha_model_v2.py (ResNet + SE attention) -> captcha_model_v6.pth
Đặc điểm:
1. Augmentation BẬT cho train set (rotation, noise, blur, shift)
2. Label Smoothing thay CrossEntropyLoss thuần
3. OneCycleLR scheduler thay ReduceLROnPlateau
4. Gradient clipping chống exploding gradients
5. Early stopping dựa trên FULL accuracy thay Char accuracy
6. Mixup augmentation

Chạy:  python training/train.py   (từ thư mục gốc project)
"""

import os
import sys
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

# Cho phép import module model ở thư mục gốc project, bất kể CWD
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from captcha_model_v2 import (
    CaptchaCNN, CHARS, NUM_CHARS, CAPTCHA_LEN,
    IMG_WIDTH, IMG_HEIGHT, char_to_idx, decode_prediction
)

# Dữ liệu + output model nằm ở thư mục gốc project
TRAINING_DIR = ROOT / "training_data_5"
LABELS_FILE  = ROOT / "labels_5.csv"
MODEL_FILE   = ROOT / "captcha_model_v6.pth"

# ===== Hyperparameters =====
BATCH_SIZE          = 32    # Nhỏ hơn -> gradient noise tốt hơn cho ResNet
EPOCHS              = 150
LEARNING_RATE       = 3e-4  # OneCycleLR sẽ tự điều chỉnh
TRAIN_SPLIT         = 0.85
EARLY_STOP_PATIENCE = 40    # Kiên nhẫn hơn vì LR schedule phức tạp hơn
LABEL_SMOOTHING     = 0.1   # Tránh model quá tự tin với nhãn nhiễu
GRAD_CLIP           = 1.0   # Gradient clipping


# ===== Dataset =====

class CaptchaDataset(Dataset):
    def __init__(self, samples: list, augment: bool = False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _preprocess(self, img_path: str) -> np.ndarray:
        """Đọc + xử lý alpha channel + grayscale + invert"""
        img_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img_raw is None:
            raise ValueError(f"Không đọc được: {img_path}")

        # Xử lý alpha channel
        if img_raw.ndim == 3 and img_raw.shape[2] == 4:
            alpha    = img_raw[:, :, 3] / 255.0
            color    = img_raw[:, :, :3].astype(np.float32)
            white_bg = np.ones_like(color) * 255.0
            img_color = (color * alpha[:, :, None] + white_bg * (1 - alpha[:, :, None])).astype(np.uint8)
        else:
            img_color = img_raw if img_raw.ndim == 3 else cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)

        gray      = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        inverted  = cv2.bitwise_not(gray)
        resized   = cv2.resize(inverted, (IMG_WIDTH, IMG_HEIGHT))
        return resized

    def _augment(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape

        # 1. Xoay nhẹ
        if random.random() < 0.6:
            angle = random.uniform(-8, 8)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=0)

        # 2. Dịch chuyển ngang/dọc nhỏ
        if random.random() < 0.5:
            tx = random.randint(-4, 4)
            ty = random.randint(-2, 2)
            M  = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w, h), borderValue=0)

        # 3. Blur nhẹ (mô phỏng nét mờ)
        if random.random() < 0.3:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        # 4. Thêm noise Gaussian
        if random.random() < 0.4:
            noise = np.random.normal(0, random.uniform(5, 20), img.shape).astype(np.float32)
            img   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 5. Điều chỉnh độ sáng/tương phản
        if random.random() < 0.4:
            alpha_c = random.uniform(0.8, 1.2)  # contrast
            beta_c  = random.randint(-20, 20)    # brightness
            img = np.clip(img.astype(np.float32) * alpha_c + beta_c, 0, 255).astype(np.uint8)

        return img

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = self._preprocess(img_path)

        if self.augment:
            img = self._augment(img)

        img_tensor = torch.tensor(img.astype(np.float32) / 255.0, dtype=torch.float32).unsqueeze(0)
        lbl_tensor = torch.tensor([char_to_idx(c) for c in label.upper()], dtype=torch.long)
        return img_tensor, lbl_tensor


# ===== Mixup Augmentation =====

def mixup_batch(images, labels, alpha=0.2):
    """
    Mixup: trộn 2 ảnh với tỉ lệ lambda.
    Giúp model học boundary mượt hơn.
    """
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam    = np.random.beta(alpha, alpha)
    bs     = images.size(0)
    idx    = torch.randperm(bs)
    mixed  = lam * images + (1 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


def mixup_criterion(criterion, outputs, y_a, y_b, lam):
    loss = 0
    for i in range(CAPTCHA_LEN):
        loss += lam * criterion(outputs[i], y_a[:, i]) + (1 - lam) * criterion(outputs[i], y_b[:, i])
    return loss


# ===== Data Loading =====

def load_data():
    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy {LABELS_FILE}")

    samples, skipped = [], 0
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            filename = row[0].strip()
            label    = row[1].strip().upper()

            if len(label) != CAPTCHA_LEN or not all(c in CHARS for c in label):
                skipped += 1
                continue

            img_path = str(TRAINING_DIR / filename)
            if os.path.exists(img_path):
                samples.append((img_path, label))
            else:
                skipped += 1

    if skipped:
        print(f"⚠  Bỏ qua {skipped} mẫu không hợp lệ.")
    return samples


# ===== Training Loop =====

def train():
    print("=" * 60)
    print("   CAPTCHA TRAINING v2 - ResNet + SE + OneCycleLR")
    print("=" * 60)

    samples = load_data()
    print(f"📂 Tổng mẫu hợp lệ: {len(samples)}")
    if len(samples) < 100:
        print("❌ Không đủ dữ liệu để train!")
        return

    random.shuffle(samples)
    split      = int(len(samples) * TRAIN_SPLIT)
    train_set  = samples[:split]
    val_set    = samples[split:]
    print(f"   Train: {len(train_set)} | Val: {len(val_set)}")

    train_loader = DataLoader(
        CaptchaDataset(train_set, augment=True),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        CaptchaDataset(val_set, augment=False),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0
    )

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    model     = CaptchaCNN().to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # OneCycleLR: LR tăng rồi giảm dần — rất hiệu quả
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE * 10,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_char_acc  = 0.0
    best_full_acc  = 0.0
    patience_counter = 0
    use_mixup      = True  # Tắt nếu không hiệu quả

    print(f"\n🚀 Bắt đầu train {EPOCHS} epochs...\n")
    print(f"{'Epoch':>6} | {'Loss':>7} | {'Char%':>7} | {'Full%':>7} | {'LR':>8} | {'Time':>5}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ---------- Train ----------
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            if use_mixup and random.random() < 0.5:
                mixed, y_a, y_b, lam = mixup_batch(images, labels, alpha=0.2)
                outputs = model(mixed)
                loss    = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(images)
                loss    = sum(criterion(outputs[i], labels[:, i]) for i in range(CAPTCHA_LEN))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        # ---------- Validate ----------
        model.eval()
        val_correct_chars = 0
        val_correct_full  = 0
        val_total         = 0
        val_loss          = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = sum(criterion(outputs[i], labels[:, i]) for i in range(CAPTCHA_LEN))
                val_loss += loss.item()

                bs = images.size(0)
                val_total += bs

                matches = torch.stack([outputs[i].argmax(1) == labels[:, i] for i in range(CAPTCHA_LEN)])
                val_correct_chars += matches.sum().item()
                val_correct_full  += matches.all(dim=0).sum().item()

        avg_loss  = val_loss / len(val_loader)
        char_acc  = val_correct_chars / (val_total * CAPTCHA_LEN) * 100
        full_acc  = val_correct_full  / val_total * 100
        cur_lr    = scheduler.get_last_lr()[0]
        elapsed   = time.time() - t0

        saved_mark = ""
        # Ưu tiên full_acc, dùng char_acc làm tiebreaker
        if full_acc > best_full_acc or (abs(full_acc - best_full_acc) < 1e-7 and char_acc > best_char_acc):
            best_full_acc  = full_acc
            best_char_acc  = char_acc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_char_acc": char_acc,
                "val_full_acc": full_acc,
                "epoch": epoch,
                "chars": CHARS,
                "captcha_len": CAPTCHA_LEN,
                "img_size": (IMG_HEIGHT, IMG_WIDTH),
            }, str(MODEL_FILE))
            saved_mark = " ✅"
        else:
            patience_counter += 1

        print(f"{epoch:6d} | {avg_loss:7.3f} | {char_acc:6.1f}% | {full_acc:6.1f}% | {cur_lr:.2e} | {elapsed:.1f}s{saved_mark}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⏹  Early stopping tại epoch {epoch} (patience={EARLY_STOP_PATIENCE})")
            break

    # ---------- Quick Test ----------
    print(f"\n🏆 Best: Char {best_char_acc:.1f}% | Full {best_full_acc:.1f}%")
    print("\n🧪 Quick test 10 mẫu từ validation set...")

    checkpoint = torch.load(str(MODEL_FILE), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    correct = 0
    with torch.no_grad():
        test_samples = random.sample(val_set, min(10, len(val_set)))
        for img_path, true_label in test_samples:
            img, _ = CaptchaDataset([(img_path, true_label)])[0]
            out    = model(img.unsqueeze(0).to(device))
            pred   = decode_prediction(out)
            ok     = pred == true_label
            correct += ok
            mark   = "✅" if ok else "❌"
            print(f"   {mark} Pred: {pred} | Real: {true_label}")

    print(f"\n   Kết quả: {correct}/10 đúng hoàn toàn")
    print(f"\n💾 Model lưu tại: {MODEL_FILE}")


if __name__ == "__main__":
    train()