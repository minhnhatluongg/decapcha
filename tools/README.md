# tools/

Tiện ích chạy 1 lần. **Không** deploy lên production.
Chạy từ **thư mục gốc project** (script tự neo đường dẫn về root):

```powershell
python tools/check_model.py        # in thông tin checkpoint captcha_model.pth + captcha_model_v6.pth
python tools/export_to_onnx.py     # captcha_model_v6.pth -> captcha_model_v6.onnx (+ .meta.json)
python tools/down_version.py       # captcha_model_v6.onnx -> captcha_model_v6_ir8.onnx (IR v8)
```

Lưu ý: runtime giải CAPTCHA dùng `.pth` qua PyTorch, **không** dùng `.onnx`.
Các tool ONNX ở đây chỉ phục vụ export cho client khác (vd C# ONNX Runtime).
