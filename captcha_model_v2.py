"""
captcha_model_v2.py - Kiến trúc mạnh hơn cho CAPTCHA recognition
Thay thế Mini CNN bằng ResNet-style CNN + Squeeze-Excitation blocks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== CONFIG (giữ nguyên để tương thích) =====
CHARS      = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CHARS  = len(CHARS)   # 36
CAPTCHA_LEN = 5
IMG_WIDTH  = 140
IMG_HEIGHT = 50

def char_to_idx(c: str) -> int:
    return CHARS.index(c.upper())

def decode_prediction(outputs) -> str:
    return "".join(CHARS[o.argmax(1).item()] for o in outputs)


# ===== Building blocks =====

class SEBlock(nn.Module):
    """Squeeze-Excitation: giúp model tập trung vào feature quan trọng"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResBlock(nn.Module):
    """Residual block với SE attention"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.se = SEBlock(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.se(self.conv(x)))


class CaptchaCNN(nn.Module):
    """
    Kiến trúc cải tiến:
    - 4 stages convolution với residual connections
    - SE attention blocks
    - Dropout mạnh hơn để tránh overfit
    - 5 output heads độc lập (mỗi head = 1 ký tự)
    """
    def __init__(self):
        super().__init__()

        # Stage 1: 1 -> 32 channels
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32),
            nn.MaxPool2d(2),   # 50x140 -> 25x70
            nn.Dropout2d(0.1),
        )

        # Stage 2: 32 -> 64 channels
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
            nn.MaxPool2d(2),   # 25x70 -> 12x35
            nn.Dropout2d(0.15),
        )

        # Stage 3: 64 -> 128 channels
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128),
            nn.MaxPool2d(2),   # 12x35 -> 6x17
            nn.Dropout2d(0.2),
        )

        # Stage 4: 128 -> 256 channels
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResBlock(256),
            nn.AdaptiveAvgPool2d((2, 8)),  # -> 2x8 cố định
            nn.Dropout2d(0.25),
        )

        feature_dim = 256 * 2 * 8  # = 4096

        # 5 classifier heads độc lập
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, NUM_CHARS),
            )
            for _ in range(CAPTCHA_LEN)
        ])

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.view(x.size(0), -1)
        return [head(x) for head in self.heads]