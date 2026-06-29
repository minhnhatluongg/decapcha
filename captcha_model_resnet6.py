"""
CAPTCHA CNN Model - ResNet18 backbone + 6 classification heads.
Dùng cho CAPTCHA 6 ký tự từ hoadondientu.gdt.gov.vn
"""
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torchvision.models as models

# 36 ký tự: 0-9 + A-Z
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CHARS = len(CHARS)  # 36
CAPTCHA_LEN = 6
IMG_WIDTH  = 160   # img_size từ checkpoint: (H=60, W=160)
IMG_HEIGHT = 60


def char_to_idx(c: str) -> int:
    return CHARS.index(c)


def idx_to_char(i: int) -> str:
    return CHARS[i]


def decode_prediction(outputs: list) -> str:
    """Decode model output (list of 6 tensors) thành text 6 ký tự."""
    result = ""
    for i in range(CAPTCHA_LEN):
        idx = outputs[i].argmax().item()
        result += idx_to_char(idx)
    return result


class CaptchaCNN(nn.Module):
    """
    ResNet18 backbone + 6 heads cho 6 ký tự CAPTCHA.
    Input: grayscale (1 channel)
    """

    def __init__(self):
        super().__init__()

        # ResNet18 pre-trained
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Sửa conv1: nhận 1 channel (grayscale) thay vì 3 (RGB)
        old_conv = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(old_conv.weight.mean(dim=1, keepdim=True))

        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ResNet18 output = 512 features
        self.dropout = nn.Dropout(0.5)

        # 6 output heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, NUM_CHARS),
            )
            for _ in range(CAPTCHA_LEN)
        ])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (batch, 512)
        x = self.dropout(x)
        return [head(x) for head in self.heads]
