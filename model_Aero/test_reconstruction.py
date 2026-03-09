import os
import torch
import numpy as np
from model import PointCloudVAE

def parse_polar(polar_path):
    """从 polar 文件中提取真实的 CL 值"""
    try:
        with open(polar_path, 'r') as f:
            lines = f.readlines()
            data_line = lines[-1].strip()
            if not data_line: 
                data_line = lines[-2].strip()
            parts = data_line.split()
            cl = float(parts[4])
            return cl
    except Exception as e:
        print(f"解析错误: {e}")
        return None

def predict_single_pointcloud(pc_path, polar_path, ckpt_path):
    # 1. 训练时计算出的标准化参数 (请确保这和你训练日志里打印的完全一致！)
    # 根据你之前的日志，Mean=0.4902, Std=0.0887
    CL_MEAN = 0.4902
    CL_STD = 0.0887

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # 2. 初始化模型并加载权重
    print("🛠️ 正在加载模型...")
    model = PointCloudVAE(
        latent_dim=128,
        plane_resolution=128,
        plane_features=32,
        num_fourier_freqs=8,
        num_points_uniform=4000,
        num_points_curvature=4000,
        num_points_importance=4000
    ).to(device)

    # 加载 Stage 2 微调后的最佳权重
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval() # 切换到推理模式

    # 3. 处理输入的点云数据
    print(f"📥 正在加载点云: {os.path.basename(pc_path)}")
    pc_raw = np.load(pc_path)
    pc_uni = pc_raw['uniform'][:4000]
    pc_cur = pc_raw['curvature'][:4000]
    pc_imp = pc_raw['importance'][:4000]
    pc_input = np.concatenate([pc_uni, pc_cur, pc_imp], axis=0)
    
    # 转换为 Tensor，并增加 batch 维度 (1, 12000, 3)
    pc_tensor = torch.from_numpy(pc_input).float().unsqueeze(0).to(device)

    # 4. 获取真实的 CL 用于对比
    real_cl = parse_polar(polar_path)

    # 5. 模型推理
    print("🚀 正在预测气动性能...")
    with torch.no_grad():
        # 模型返回的第四个参数是气动预测值
        _, _, _, pred_cl_norm = model(pc_tensor)
        
        # 提取标量值
        pred_cl_norm = pred_cl_norm.item()

    # 6. 反标准化 (还原为真实物理量级)
    # 公式: 预测值 = 归一化值 * 标准差 + 均值
    pred_cl_phys = pred_cl_norm * CL_STD + CL_MEAN

    # 7. 打印对比结果
    print("-" * 40)
    print("📊 预测结果对比")
    print("-" * 40)
    if real_cl is not None:
        error_abs = abs(pred_cl_phys - real_cl)
        error_rel = (error_abs / abs(real_cl)) * 100 if real_cl != 0 else 0
        
        print(f"✅ 真实升力系数 (Ground Truth) : {real_cl:.5f}")
        print(f"🤖 模型预测系数 (Predicted)    : {pred_cl_phys:.5f}")
        print(f"📉 绝对误差 (Absolute Error)   : {error_abs:.5f}")
        print(f"📉 相对误差 (Relative Error)   : {error_rel:.2f}%")
    else:
        print(f"🤖 模型预测系数 (Predicted)    : {pred_cl_phys:.5f}")
        print("⚠️ 无法获取真实 CL 进行对比。")
    print("-" * 40)

if __name__ == "__main__":
    # --- 请替换为你想测试的具体文件路径 ---
    TEST_ID = "G58_2" # 假设测试第100个样本
    
    pc_file = f"/home/yuwenshi/B737/G58_pc_1299/pointcloud/{TEST_ID}_pc.npz"
    polar_file = f"/home/yuwenshi/B737/G58_aero_1299/G58_aero_1299/{TEST_ID}/{TEST_ID}_VSPGeom.polar"
    checkpoint = "checkpoints_stage2_6/best_cl_model.pth"

    predict_single_pointcloud(pc_file, polar_file, checkpoint)