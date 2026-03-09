import torch
from model1 import PointCloudVAE

# 路径换成你的
ckpt_path = '/home/yuwenshi/B737/checkpoint_all_1/vae_epoch_8400.pth'
device = torch.device('cpu')

model = PointCloudVAE(latent_dim=128,
       plane_resolution=128,
       plane_features=32,
       num_fourier_freqs=8,
       num_points_uniform=4000,
       num_points_curvature=4000, # 传入
       num_points_importance=4000)
ckpt = torch.load(ckpt_path, map_location=device,weights_only=False)

# 处理 module. 前缀
state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# === 关键调试代码 ===
model_keys = set(model.state_dict().keys())
ckpt_keys = set(state_dict.keys())

# 计算交集
loaded_keys = model_keys.intersection(ckpt_keys)
missing_in_model = ckpt_keys - model_keys
missing_in_ckpt = model_keys - ckpt_keys

print(f"模型总参数量: {len(model_keys)}")
print(f"成功匹配参数: {len(loaded_keys)}")
print("-" * 30)

# 检查 Encoder 关键层是否加载
encoder_loaded = any('sa1.mlp_convs.0.weight' in k for k in loaded_keys)
print(f"Encoder (SA1) 是否加载成功? : {'✅ YES' if encoder_loaded else '❌ NO (严重错误!)'}")

if not encoder_loaded:
    print("\n⚠️ 你的 ckpt 中的键名可能是:")
    print(list(ckpt_keys)[:5])
    print("\n⚠️ 你的 model 中的键名是:")
    print(list(model_keys)[:5])