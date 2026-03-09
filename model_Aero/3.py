import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import r2_score, mean_absolute_error

# 导入你的模块
from model2 import PointCloudVAE
from dataset import SDFDataset

def run_evaluation_and_plot():
    # --- 1. 配置参数 (请确保路径与 train.py 一致) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = '/home/yuwenshi/B737/checkpoints_stage1+2一起训练_2/best_cl_model.pth' # 指向你新训练的模型
    
    # 初始化数据集
    full_dataset = SDFDataset(
        pc_root_dir='/home/yuwenshi/B737/B737_1299/G58_pc_1299/pointcloud',
        aero_root_dir='/home/yuwenshi/B737/B737_1299/G58_aero_1299/G58_aero_1299',
        sdf_dir='/home/yuwenshi/B737/B737_1299/sdf_data'
    )
    
    cl_mean = full_dataset.cl_mean
    cl_std = full_dataset.cl_std

    # --- 2. 划分训练集和测试集 (必须和 train1.py 保持一致的 seed) ---
    torch.manual_seed(42)  # 使用和 train1.py 相同的 seed
    np.random.seed(42)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # shuffle=False 保证顺序读取，方便后续评估
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=4)

    # --- 3. 加载模型 ---
    model = PointCloudVAE(
       latent_dim=128,
       plane_resolution=128,
       plane_features=32,
       num_fourier_freqs=8,
       num_points_uniform=4000,
       num_points_curvature=4000, 
       num_points_importance=4000).to(device)
       
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # --- 定义一个辅助函数来提取数据 ---
    def get_preds_and_gts(dataloader, desc="Processing"):
        all_preds = []
        all_gts = []
        print(f"🚀 开始提取 {desc} 结果...")
        with torch.no_grad():
            for data in dataloader:
                pc = data['point_cloud'].to(device)
                
                # 模型输出归一化预测值
                _, _, _, pred_cl_norm = model(pc)
                
                # 反归一化：预测值转换回物理量级
                pred_cl_phys = (pred_cl_norm.cpu().numpy() * cl_std) + cl_mean
                
                # 真实值获取逻辑
                if 'raw_cl' in data:
                    gt_cl_phys = data['raw_cl'].numpy()
                else:
                    gt_cl_norm = data['aero_label'][:, 0].numpy()
                    gt_cl_phys = (gt_cl_norm * cl_std) + cl_mean

                all_preds.extend(pred_cl_phys.flatten())
                all_gts.extend(gt_cl_phys.flatten())
        return np.array(all_preds), np.array(all_gts)

    # 获取训练集和验证集的预测值与真实值
    train_preds, train_gts = get_preds_and_gts(train_loader, "训练集")
    val_preds, val_gts = get_preds_and_gts(val_loader, "测试集")

    # --- 4. 绘图部分 ---
    plt.figure(figsize=(9, 9), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid') 

    # 绘制散点 (区分训练集和测试集)
    plt.scatter(train_gts, train_preds, alpha=0.4, edgecolors='white', color="#0099ff", s=50, label='Train Samples')
    plt.scatter(val_gts, val_preds, alpha=0.8, edgecolors='white', color="#e6605192", marker='^', s=60, label='Test Samples')

    # 动态计算对角线范围 (包含训练和测试集的最值)
    v_min = min(train_gts.min(), train_preds.min(), val_gts.min(), val_preds.min())
    v_max = max(train_gts.max(), train_preds.max(), val_gts.max(), val_preds.max())
    pad = (v_max - v_min) * 0.1
    lims = [v_min - pad, v_max + pad]

    # 绘制 y=x 对角线 (黑色虚线)
    plt.plot(lims, lims, color="#0000003e", linestyle='--', linewidth=2, label='Identity Line ($y=x$)', zorder=1)

    # 计算指标
    train_r2 = r2_score(train_gts, train_preds)
    train_mae = mean_absolute_error(train_gts, train_preds)
    val_r2 = r2_score(val_gts, val_preds)
    val_mae = mean_absolute_error(val_gts, val_preds)

    # 装饰
    plt.title('Forward Prediction Consistency ($C_L$)', fontsize=16, fontweight='bold')
    plt.xlabel('Ground Truth $C_L$ (CFD)', fontsize=14)
    plt.ylabel('NN Predicted $C_L$', fontsize=14)
    
    # 强制 1:1 比例
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(lims)
    plt.ylim(lims)
    
    # 指标框 (分别显示 Train 和 Test 的指标)
    stats_text = (
        f"Train Set:\n$R^2 = {train_r2:.4f}$\n$MAE = {train_mae:.4f}$\n\n"
        f"Test Set:\n$R^2 = {val_r2:.4f}$\n$MAE = {val_mae:.4f}$"
    )
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    
    # 保存结果
    save_dir = '/home/yuwenshi/B737/verification_results'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'cl.png')
    plt.savefig(save_path)
    plt.show()
    
    print(f"✅ 结果已保存至: {save_path}")
    print(f"📊 Train R2={train_r2:.4f}, Train MAE={train_mae:.4f}")
    print(f"📊 Test R2={val_r2:.4f}, Test MAE={val_mae:.4f}")

if __name__ == "__main__":
    run_evaluation_and_plot()