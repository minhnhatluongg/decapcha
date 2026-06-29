# down_version.py - hạ ONNX IR version xuống 8 (cho ONNX Runtime cũ).
# Chạy: python tools/down_version.py   (từ thư mục gốc project)
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parent.parent

model = onnx.load(str(ROOT / "captcha_model_v6.onnx"))
onnx.checker.check_model(model)
# Convert xuống IR version 8
model.ir_version = 8
onnx.save(model, str(ROOT / "captcha_model_v6_ir8.onnx"))