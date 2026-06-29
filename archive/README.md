# archive/

File được dọn khỏi thư mục gốc ngày 2026-06-26. **Không** file nào ở đây được
runtime (`/captcha/solve`) hay script deploy sử dụng — có thể xóa hẳn nếu chắc chắn không cần.

- `onnx_experiments/` — bản export ONNX thử nghiệm (runtime dùng `.pth` qua torch, không dùng `.onnx`).
  Regenerate bằng `export_to_onnx.py` + `down_version.py` nếu cần.
- `dead_code/captcha_model_6.py` — file đặt tên sai (tên "6" nhưng bên trong `CAPTCHA_LEN = 5`),
  không được module nào import.
- `debug_images/` — ảnh debug mồ côi; `ai_debug_view.png` được trainer tự tạo lại khi train.
- `old_mini_cnn/` — kiến trúc Mini-CNN cũ đã bị thay thế:
  `captcha_model.py` (model def) + `train_minicnn.py` (trainer ra `captcha_model_v5.pth`).
  Production hiện dùng ResNet18 (`captcha_model_resnet6.py`), **không** liên quan các file này.

## Model đang chạy production (giữ ở thư mục gốc)
- 6 ký tự (TCT): `captcha_model.pth` + `captcha_model_resnet6.py`  ← API đang dùng
- 5 ký tự (TCNNT): `captcha_model_v6.pth` + `captcha_model_v2.py`   (đổi config trong `ocr_engine.py` để bật)
