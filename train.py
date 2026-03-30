"""
CAPTCHA Model Training Script
Chạy: python train.py
Output: captcha_model.pth (file model đã train)

Yêu cầu:
  - Folder training_data/ chứa ảnh PNG
  - File labels.csv chứa labels (filename,label)
"""
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

from captcha_model import (
    CaptchaCNN, CHARS, NUM_CHARS, CAPTCHA_LEN,
    IMG_WIDTH, IMG_HEIGHT, char_to_idx, decode_prediction
)

BASE_DIR = Path(__file__).parent
TRAINING_DIR = BASE_DIR / "training_data"
LABELS_FILE = BASE_DIR / "labels.csv"
MODEL_FILE = BASE_DIR / "captcha_model.pth"

# ===== Hyperparameters =====
BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 0.003
TRAIN_SPLIT = 0.85  # 85% train, 15% val
EARLY_STOP_PATIENCE = 20  # Dừng nếu val accuracy không tăng sau N epochs


# ===== Dataset =====

class CaptchaDataset(Dataset):
    def __init__(self, samples: list[tuple[str, str]], augment: bool = False):
        """
        samples: list of (image_path, label_text)
        augment: True cho train set, False cho val set
        """
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """Data augmentation nhẹ: xoay, dịch, nhiễu."""
        h, w = img.shape

        # Random rotation nhẹ (-5 đến +5 độ)
        if random.random() < 0.5:
            angle = random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=255)

        # Random brightness
        if random.random() < 0.5:
            delta = random.randint(-30, 30)
            img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)

        # Random Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 10, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    @staticmethod
    def _remove_noise_lines(img_color: np.ndarray) -> np.ndarray:
        """Loại bỏ đường kẻ nhiễu từ ảnh CAPTCHA."""
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        _, binary = cv2.threshold(inverted, 80, 255, cv2.THRESH_BINARY)

        # Detect horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        # Detect diagonal lines
        d_kernel1 = np.eye(30, dtype=np.uint8)
        d_lines1 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, d_kernel1)
        d_kernel2 = np.fliplr(np.eye(30, dtype=np.uint8))
        d_lines2 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, d_kernel2)

        all_lines = cv2.add(h_lines, d_lines1)
        all_lines = cv2.add(all_lines, d_lines2)

        # Dilate lines slightly
        line_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        all_lines = cv2.dilate(all_lines, line_dilate, iterations=1)

        # Remove lines
        cleaned = cv2.subtract(binary, all_lines)

        # Repair
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.dilate(cleaned, repair_kernel, iterations=1)

        # Invert back: text đen trên nền trắng
        return cv2.bitwise_not(cleaned)

    @staticmethod
    def _auto_crop(img: np.ndarray, padding: int = 5) -> np.ndarray:
        """Crop bỏ viền trắng thừa."""
        _, thresh = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return img
        x, y, w, h = cv2.boundingRect(coords)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + padding * 2)
        h = min(img.shape[0] - y, h + padding * 2)
        return img[y:y+h, x:x+w]

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Đọc ảnh color (để remove noise lines)
        img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_color is None:
            raise ValueError(f"Cannot read image: {img_path}")

        # Bước 1: Loại bỏ noise lines
        img = self._remove_noise_lines(img_color)

        # Bước 2: Auto crop
        img = self._auto_crop(img)

        # Bước 3: Resize về kích thước chuẩn
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        # Augmentation
        if self.augment:
            img = self._augment_image(img)

        # Normalize [0, 255] -> [0, 1]
        img = img.astype(np.float32) / 255.0

        # Add channel dim: (H, W) -> (1, H, W)
        img = np.expand_dims(img, axis=0)

        # Label -> tensor of indices
        label_indices = [char_to_idx(c) for c in label.upper()]

        return torch.tensor(img), torch.tensor(label_indices, dtype=torch.long)


# ===== Data Loading =====

def load_data():
    """Đọc labels.csv và trả về list (image_path, label)."""
    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy {LABELS_FILE}")

    samples = []
    skipped = 0

    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            filename, label = row[0].strip(), row[1].strip().upper()

            # Validate
            if len(label) != CAPTCHA_LEN:
                skipped += 1
                continue

            if not all(c in CHARS for c in label):
                skipped += 1
                continue

            img_path = str(TRAINING_DIR / filename)
            if not os.path.exists(img_path):
                skipped += 1
                continue

            samples.append((img_path, label))

    if skipped > 0:
        print(f"⚠ Bỏ qua {skipped} mẫu không hợp lệ (label sai hoặc thiếu ảnh)")

    return samples


def split_data(samples):
    """Chia train/val set."""
    random.shuffle(samples)
    split_idx = int(len(samples) * TRAIN_SPLIT)
    return samples[:split_idx], samples[split_idx:]


# ===== Training =====

def train():
    print("=" * 60)
    print("  CAPTCHA Model Training")
    print("=" * 60)

    # Load data
    print("\n📂 Loading data...")
    samples = load_data()
    print(f"   Tổng mẫu hợp lệ: {len(samples)}")

    if len(samples) < 100:
        print("❌ Cần ít nhất 100 mẫu để train!")
        return

    train_set, val_set = split_data(samples)
    print(f"   Train: {len(train_set)} | Val: {len(val_set)}")

    train_loader = DataLoader(
        CaptchaDataset(train_set, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        CaptchaDataset(val_set, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥 Device: {device}")

    model = CaptchaCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Training loop
    print(f"\n🚀 Training {EPOCHS} epochs...")
    print("-" * 60)

    best_val_acc = 0
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # === Train ===
        model.train()
        train_loss = 0
        train_correct_chars = 0
        train_correct_full = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Loss = sum of 6 heads
            loss = sum(
                criterion(outputs[i], labels[:, i])
                for i in range(CAPTCHA_LEN)
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_size = images.size(0)
            train_total += batch_size

            # Accuracy
            for i in range(CAPTCHA_LEN):
                preds = outputs[i].argmax(dim=1)
                train_correct_chars += (preds == labels[:, i]).sum().item()

            # Full string accuracy
            for b in range(batch_size):
                predicted = "".join(
                    CHARS[outputs[i][b].argmax().item()]
                    for i in range(CAPTCHA_LEN)
                )
                actual = "".join(
                    CHARS[labels[b, i].item()]
                    for i in range(CAPTCHA_LEN)
                )
                if predicted == actual:
                    train_correct_full += 1

        train_char_acc = train_correct_chars / (train_total * CAPTCHA_LEN) * 100
        train_full_acc = train_correct_full / train_total * 100
        avg_train_loss = train_loss / len(train_loader)

        # === Validation ===
        model.eval()
        val_loss = 0
        val_correct_chars = 0
        val_correct_full = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = sum(
                    criterion(outputs[i], labels[:, i])
                    for i in range(CAPTCHA_LEN)
                )

                val_loss += loss.item()
                batch_size = images.size(0)
                val_total += batch_size

                for i in range(CAPTCHA_LEN):
                    preds = outputs[i].argmax(dim=1)
                    val_correct_chars += (preds == labels[:, i]).sum().item()

                for b in range(batch_size):
                    predicted = "".join(
                        CHARS[outputs[i][b].argmax().item()]
                        for i in range(CAPTCHA_LEN)
                    )
                    actual = "".join(
                        CHARS[labels[b, i].item()]
                        for i in range(CAPTCHA_LEN)
                    )
                    if predicted == actual:
                        val_correct_full += 1

        val_char_acc = val_correct_chars / (val_total * CAPTCHA_LEN) * 100
        val_full_acc = val_correct_full / val_total * 100
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        elapsed = time.time() - t0

        # Print progress
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_full_acc:5.1f}% (char {train_char_acc:.1f}%) | "
            f"Val Acc: {val_full_acc:5.1f}% (char {val_char_acc:.1f}%) | "
            f"{elapsed:.1f}s"
        )

        # Save best model (cũng save theo val_char_acc nếu full_acc vẫn 0)
        save_condition = (
            val_full_acc > best_val_acc
            or (val_full_acc == best_val_acc == 0 and avg_val_loss < best_val_loss)
        )
        if save_condition:
            best_val_acc = val_full_acc
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_full_acc,
                "val_char_acc": val_char_acc,
                "epoch": epoch,
                "chars": CHARS,
                "captcha_len": CAPTCHA_LEN,
                "img_size": (IMG_HEIGHT, IMG_WIDTH),
            }, str(MODEL_FILE))
            print(f"   ✅ Best model saved! Val accuracy: {val_full_acc:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n⏹ Early stopping! No improvement for {EARLY_STOP_PATIENCE} epochs.")
                break

    # Summary
    print("\n" + "=" * 60)
    print(f"  Training Complete!")
    print(f"  Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"  Model saved to: {MODEL_FILE}")
    print("=" * 60)

    # Quick test
    print("\n🧪 Quick test trên 10 mẫu validation...")
    if not MODEL_FILE.exists():
        print("   ⚠ Model file not saved (accuracy too low). Skipping test.")
        return
    model.load_state_dict(torch.load(str(MODEL_FILE), map_location=device, weights_only=False)["model_state_dict"])
    model.eval()

    test_samples = val_set[:10]
    test_ds = CaptchaDataset(test_samples)

    correct = 0
    with torch.no_grad():
        for i in range(min(10, len(test_ds))):
            img, label = test_ds[i]
            img = img.unsqueeze(0).to(device)
            outputs = model(img)
            predicted = decode_prediction([o[0] for o in outputs])
            actual = test_samples[i][1]
            match = "✅" if predicted == actual else "❌"
            if predicted == actual:
                correct += 1
            print(f"   {match} {test_samples[i][0].split(os.sep)[-1]}: "
                  f"Predicted={predicted} | Actual={actual}")

    print(f"\n   Test result: {correct}/10 correct")


if __name__ == "__main__":
    train()
