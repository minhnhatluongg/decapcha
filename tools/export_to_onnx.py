# export_to_onnx.py - export model 5 ký tự (captcha_model_v2) sang ONNX.
# Chạy: python tools/export_to_onnx.py   (từ thư mục gốc project)
import sys
from pathlib import Path

import torch

# Cho phép import module model + neo đường dẫn về thư mục gốc project
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from captcha_model_v2 import CaptchaCNN, IMG_WIDTH, IMG_HEIGHT, CAPTCHA_LEN, CHARS, char_to_idx

device = torch.device("cpu")
checkpoint = torch.load(str(ROOT / "captcha_model_v6.pth"), map_location=device)

model = CaptchaCNN()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dummy = torch.zeros(1, 1, IMG_HEIGHT, IMG_WIDTH)  # batch=1, channel=1 (grayscale)

# Export với dynamic batch size
torch.onnx.export(
    model, dummy,
    str(ROOT / "captcha_model_v6.onnx"),
    input_names=["input"],
    output_names=[f"output_{i}" for i in range(CAPTCHA_LEN)],  # 5 outputs
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=17,
    do_constant_folding=True,
)

# Lưu metadata để C# biết CHARS và preprocessing
import json
meta = {
    "chars": CHARS,
    "captcha_len": CAPTCHA_LEN,
    "img_width": IMG_WIDTH,
    "img_height": IMG_HEIGHT,
}
with open(str(ROOT / "captcha_model_v6.onnx.meta.json"), "w") as f:
    json.dump(meta, f)

print(f"✅ Exported! CHARS={CHARS}, LEN={CAPTCHA_LEN}, SIZE={IMG_WIDTH}x{IMG_HEIGHT}")