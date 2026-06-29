import torch
from pathlib import Path

# Model weights nằm ở thư mục gốc project (cha của tools/)
ROOT = Path(__file__).resolve().parent.parent

print('=== captcha_model.pth ===')
try:
    ckpt = torch.load(str(ROOT / 'captcha_model.pth'), map_location='cpu', weights_only=False)
    print(f'Keys: {list(ckpt.keys())}')
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        head_keys = [k for k in state.keys() if 'heads' in k]
        num_heads = len(set([k.split('.')[1] for k in head_keys]))
        print(f'Number of heads: {num_heads}')
        print(f'Val acc: {ckpt.get("val_acc", "N/A")}')
except Exception as e:
    print(f'Error: {e}')

print('\n=== captcha_model_v6.pth ===')
try:
    ckpt = torch.load(str(ROOT / 'captcha_model_v6.pth'), map_location='cpu', weights_only=False)
    print(f'Keys: {list(ckpt.keys())}')
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        head_keys = [k for k in state.keys() if 'heads' in k]
        num_heads = len(set([k.split('.')[1] for k in head_keys]))
        print(f'Number of heads: {num_heads}')
        print(f'Val acc: {ckpt.get("val_acc", "N/A")}')
except Exception as e:
    print(f'Error: {e}')
