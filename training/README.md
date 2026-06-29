# training/

Công cụ chuẩn bị dữ liệu & huấn luyện model. **Không** deploy lên production.
Luôn chạy từ **thư mục gốc project** (các script tự neo đường dẫn về root):

```powershell
# 1. Gán nhãn ảnh trong training_data_5/  -> ghi labels_5.csv
python training/labeler.py        # mở http://localhost:8001

# 2. Train model 5 ký tự (ResNet + SE)  -> xuất captcha_model_v6.pth ở thư mục gốc
python training/train.py
```

- `train.py` dùng `captcha_model_v2.py` (ở thư mục gốc), đọc `training_data_5/` + `labels_5.csv`.
- Model 6 ký tự (`captcha_model_resnet6.py` / `captcha_model.pth`) hiện không có script train kèm repo.
- Bản trainer Mini-CNN cũ đã chuyển sang `archive/old_mini_cnn/`.
