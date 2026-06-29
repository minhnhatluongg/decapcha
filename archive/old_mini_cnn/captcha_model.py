import torch
import torch.nn as nn

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" # Viết hoa theo labels của Nhật
NUM_CHARS = len(CHARS)
CAPTCHA_LEN = 6
IMG_WIDTH = 130
IMG_HEIGHT = 50

class CaptchaCNN(nn.Module):
    def __init__(self):
        super(CaptchaCNN, self).__init__()
        # Layer 1: Nhận ảnh 130x50
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # Còn 65x25
        )
        # Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) # Còn 32x12
        )
        # Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2) # Còn 16x6
        )
        
        # Fully Connected
        # 128 channels * 16 width * 6 height = 12288
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 6 đầu ra cho 6 ký tự
        self.heads = nn.ModuleList([
            nn.Linear(256, NUM_CHARS) for _ in range(CAPTCHA_LEN)
        ])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return [head(x) for head in self.heads]

def char_to_idx(c): return CHARS.index(c)
def decode_prediction(outputs):
    res = ""
    for out in outputs:
        res += CHARS[out.argmax().item()]
    return res