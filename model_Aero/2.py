import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import PointCloudVAE

def run_latent_inverse_search():
    # --- 1. 配置参数 ---
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    ckpt_path = '/home/yuwenshi/B737/checkpoints_stage2_6/best_cl_model.pth' # 替换为你的最新模型权重
    
    # 必须与训练时完全一致的统计信息
    CL_MEAN = 0.4902  # 请替换为你数据集的真实 mean
    CL_STD = 0.0887   # 请替换为你数据集的真实 std

    # 设定的目标 CL 列表 {0.1 - 0.8}
    target_cls = [ 0.2,  0.4,  0.6,  0.8]
    samples_per_target = 100  # 每个目标 CL 生成 20 个不同的构型
    optimization_steps = 300 # 梯度下降步数
    lr_z = 0.05              # 潜空间搜索的学习率

    # --- 2. 加载模型 ---
    model = PointCloudVAE(
       latent_dim=128,
       plane_resolution=128,
       plane_features=32,
       num_fourier_freqs=8,
       num_points_uniform=4000,
       num_points_curvature=4000, # 传入
       num_points_importance=4000).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval() # 开启评估模式
    
    # 冻结模型所有参数（在逆向搜索中，模型权重固定，只更新潜向量 z）
    for param in model.parameters():
        param.requires_grad = False

    # 存储结果用于绘图
    all_targets = []
    all_preds = []

    print("🚀 开始在潜空间执行逆向搜索...")
    
    # --- 3. 逆向搜索核心逻辑 ---
    for target_cl in target_cls:
        print(f"🎯 正在搜索目标 C_L = {target_cl} 的构型...")
        
        for i in range(samples_per_target):
            # a. 随机初始化潜向量 z (服从标准正态分布)
            z = torch.randn(1, 128, device=device, requires_grad=True)
            
            # b. 为 z 设置优化器
            optimizer_z = optim.Adam([z], lr=lr_z)
            criterion = nn.MSELoss()
            
            # 目标值转换为 Tensor
            target_tensor = torch.tensor([[target_cl]], dtype=torch.float32, device=device)
            
            # c. 梯度下降迭代更新 z
            for step in range(optimization_steps):
                optimizer_z.zero_grad()
                
                # 直接通过 aero_decoder 预测
                pred_cl_norm = model.aero_decoder(z)
                
                # 反归一化到物理量级计算 Loss
                pred_cl_phys = pred_cl_norm * CL_STD + CL_MEAN
                
                loss = criterion(pred_cl_phys, target_tensor)
                loss.backward()
                optimizer_z.step()
                
            # d. 记录优化后的最终预测值
            final_pred_norm = model.aero_decoder(z).detach()
            final_pred_phys = (final_pred_norm * CL_STD + CL_MEAN).item()
            
            all_targets.append(target_cl)
            all_preds.append(final_pred_phys)

    # --- 4. 论文级可视化 ---
    print("\n📊 正在生成分布散点图...")
    plt.figure(figsize=(9, 8), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid')

    # 使用调色板区分不同的目标组
    unique_targets = np.unique(all_targets)
    colors = sns.color_palette("husl", len(unique_targets))

    # 绘制散点 (加入微小的 x 轴抖动(jitter)以防点重叠，不影响 y 轴真实预测值)
    for idx, target in enumerate(unique_targets):
        mask = np.array(all_targets) == target
        x_jitter = np.random.normal(0, 0.005, size=np.sum(mask)) # 仅视觉抖动
        plt.scatter(np.array(all_targets)[mask] + x_jitter, 
                    np.array(all_preds)[mask], 
                    color=colors[idx], alpha=0.8, edgecolor='w', s=80, 
                    label=f'Target $C_L$={target}')

    # 绘制 y=x 对角线 (理想情况)
    plt.plot([0.05, 0.85], [0.05, 0.85], color='red', linestyle='--', linewidth=2.5, label='Ideal ($y=x$)')

    # 图表修饰
    plt.title('Latent Space Inverse Search: Target vs. Predicted $C_L$', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Target $C_L$ (Inverse Design Goal)', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted $C_L$ of Generated Configurations', fontsize=14, fontweight='bold')
    
    plt.xlim(0.05, 0.85)
    plt.ylim(0.05, 0.85)
    plt.xticks(np.arange(0.1, 0.9, 0.1), fontsize=12)
    plt.yticks(np.arange(0.1, 0.9, 0.1), fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    plt.savefig('/home/yuwenshi/B737/verification_results/Inverse_Search_CL_Distribution.png', dpi=300, bbox_inches='tight')
    #plt.savefig('Inverse_Search_CL_Distribution.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    print("✅ 图像已保存为 Inverse_Search_CL_Distribution.png 和 .pdf")

if __name__ == "__main__":
    run_latent_inverse_search()