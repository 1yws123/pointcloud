import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from schedulers import WarmupCosineScheduler 

# 导入你的模块
from model1 import PointCloudVAE
from ..dataset import SDFDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

def get_args():
    parser = argparse.ArgumentParser(description='End-to-End Aero Training from Scratch')
    parser.add_argument('--pc_root', type=str, default='/home/yuwenshi/B737/B737_1299/G58_pc_1299/pointcloud')
    parser.add_argument('--aero_root', type=str, default='/home/yuwenshi/B737/B737_1299/G58_aero_1299/G58_aero_1299')
    parser.add_argument('--sdf_dir', type=str, default='/home/yuwenshi/B737/B737_1299/sdf_data')
    parser.add_argument('--save_dir', type=str, default='/home/yuwenshi/B737/model_Aero/encoder_Aerodecoder/checkpoints_stage2_测试900个数据(从头训2000epoch)')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--lr', type=float, default=2e-4) # 推荐初始学习率
    return parser.parse_args()

def validate(model, val_loader, device, cl_mean, cl_std):
    model.eval()
    val_mae_phys = 0.0
    val_mse_norm = 0.0
    
    criterion_mse = nn.MSELoss()
    
    with torch.no_grad():
        for data in val_loader:
            # 确保是 float32 类型
            pc = data['point_cloud'].to(device).float()
            # 确保形状是 [B, N, 3] 而不是 [B, 3, N]
            if pc.size(-1) != 3 and pc.size(1) == 3:
                pc = pc.transpose(1, 2)
            pc = pc.contiguous()

            gt_cl_norm = data['aero_label'][:, 0].unsqueeze(1).to(device).float()
            
            # 推理 (使用索引获取输出，避免解包错误)
            outputs = model(pc)
            pred_cl_norm = outputs[3]
            
            val_mse_norm += criterion_mse(pred_cl_norm, gt_cl_norm).item()
            pred_cl_phys = pred_cl_norm * cl_std + cl_mean
            gt_cl_phys = gt_cl_norm * cl_std + cl_mean
            val_mae_phys += torch.mean(torch.abs(pred_cl_phys - gt_cl_phys)).item()
            
    return val_mse_norm / len(val_loader), val_mae_phys / len(val_loader)

def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 数据准备
    full_dataset = SDFDataset(args.pc_root, args.aero_root, args.sdf_dir)
    CL_MEAN = full_dataset.cl_mean
    CL_STD = full_dataset.cl_std
    print(f"Dataset Statistics: Mean={CL_MEAN:.4f}, Std={CL_STD:.4f}")

    torch.manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. 模型初始化 (随机初始化，不加载 Stage 1 权重)
    model = PointCloudVAE(
       latent_dim=128,
       plane_resolution=128,
       plane_features=32,
       num_fourier_freqs=8,
       num_points_uniform=4000,
       num_points_curvature=4000,
       num_points_importance=4000).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"�� 检测到 {torch.cuda.device_count()} 块 GPU，已启用 DataParallel")
        model = nn.DataParallel(model)

    # 3. 优化器 (全量训练)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=args.epochs//10, total_epochs=args.epochs)

    # 4. 损失函数
    criterion_mse = nn.MSELoss() 

    log_file = os.path.join(args.save_dir, 'train_log.csv')
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'Train_Loss', 'Val_MSE_Norm', 'Val_Real_MAE', 'LR'])

    best_mae = float('inf')

    # 5. 训练循环
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        total_samples = 0
        
        for data in train_loader:
            # ================== 核心修复 1: 数据格式与清理 ==================
            pc = data['point_cloud'].to(device).float()
            
            # 清理极端异常值（如果有 NaN，替换为 0，防止 C++ 算子崩溃）
            if torch.isnan(pc).any() or torch.isinf(pc).any():
                pc = torch.nan_to_num(pc, nan=0.0, posinf=0.0, neginf=0.0)

            # 纠正形状：如果进来的是 [B, 3, 12000]，转置为 [B, 12000, 3]
            if pc.size(-1) != 3 and pc.size(1) == 3:
                pc = pc.transpose(1, 2)
            pc = pc.contiguous()
            # ================================================================

            gt_cl_norm = data['aero_label'][:, 0].unsqueeze(1).to(device).float()
            current_batch_size = pc.size(0) 
            
            optimizer.zero_grad()
            
            # ================== 核心修复 2: 解包参数数量对齐 ==================
            # model 返回 5 个参数，这里用 5 个接收，避免 ValueError 崩溃
            _, _, _, pred_cl_norm = model(pc)
            # ================================================================
            
            loss = 1 * criterion_mse(pred_cl_norm, gt_cl_norm) 
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        avg_train_loss = epoch_loss / total_samples

        if (epoch + 1) % 1 == 0:
            val_mse, val_mae = validate(model, val_loader, device, CL_MEAN, CL_STD)
            print(f"E{epoch+1:04d} | Train Loss:{avg_train_loss:.5f} | Val MSE:{val_mse:.5f} | Val Real MAE:{val_mae:.5f} | LR:{current_lr:.2e}")
            
            if val_mae < best_mae:
                best_mae = val_mae
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_cl_model.pth'))
                print(f"  --> Best Model Saved (MAE: {val_mae:.5f})")
        elif (epoch + 1) % 5 == 0:
            print(f"E{epoch+1:04d} | Train Loss:{avg_train_loss:.5f} | (Skipped Val) | LR:{current_lr:.2e}")

        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_train_loss, val_mse, val_mae, current_lr])

    print(f"\n✅ 训练已完成 {args.epochs} 个 Epoch！正在保存最终模型...")
    final_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(final_state_dict, os.path.join(args.save_dir, 'final_model.pth'))
    print(f"💾 最终模型已保存至: {os.path.join(args.save_dir, 'final_model.pth')}")
    
if __name__ == '__main__':
    main()