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
from model2 import PointCloudVAE
from dataset import SDFDataset

def get_args():
    parser = argparse.ArgumentParser(description='End-to-End Aero Training from Scratch')
    parser.add_argument('--pc_root', type=str, default='/home/yuwenshi/B737/测试1/pc_2890/pointcloud')
    parser.add_argument('--aero_root', type=str, default='/home/yuwenshi/B737/测试1/merged_aero')
    parser.add_argument('--sdf_dir', type=str, default='/home/yuwenshi/B737/测试1/sdf_2890/sdf_data')
    parser.add_argument('--save_dir', type=str, default='checkpoints_stage2_测试20个数据(从头训_latent_dim=1024)')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4) # 推荐初始学习率
    return parser.parse_args()

def validate(model, val_loader, device, cl_mean, cl_std):
    model.eval()
    val_mae_phys = 0.0
    val_mse_norm = 0.0
    
    criterion_mse = nn.MSELoss()
    
    with torch.no_grad():
        for data in val_loader:
            pc = data['point_cloud'].to(device)
            # 取归一化标签用于算 Loss
            gt_cl_norm = data['aero_label'][:, 0].unsqueeze(1).to(device)
            
            # 推理
            _, _, _, pred_cl_norm = model(pc)
            
            # 1. 计算归一化 MSE (监控训练收敛)
            val_mse_norm += criterion_mse(pred_cl_norm, gt_cl_norm).item()
            
            # 2. 计算真实物理 MAE (监控工程精度)
            pred_cl_phys = pred_cl_norm * cl_std + cl_mean
            gt_cl_phys = gt_cl_norm * cl_std + cl_mean
            val_mae_phys += torch.mean(torch.abs(pred_cl_phys - gt_cl_phys)).item()
            
    return val_mse_norm / len(val_loader), val_mae_phys / len(val_loader)

def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

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
       num_points_curvature=4000, # 传入
       num_points_importance=4000).to(device)

    # 3. 优化器 (全量训练)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=args.epochs//10, total_epochs=args.epochs)

    # 4. 损失函数：针对长尾分布使用混合 Loss
    criterion_mse = nn.MSELoss() # 对极端值更敏感
    criterion_mae = nn.L1Loss()  # 对中间值更稳定

    log_file = os.path.join(args.save_dir, 'train_log.csv')
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'Train_Loss', 'Val_MSE_Norm', 'Val_Real_MAE', 'LR'])

    best_mae = float('inf')

    val_mse, val_mae = 0.0, 0.0 
    # 5. 训练循环
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        total_samples = 0
        
        for data in train_loader:
            pc = data['point_cloud'].to(device)
            gt_cl_norm = data['aero_label'][:, 0].unsqueeze(1).to(device)

            current_batch_size = pc.size(0) 
            optimizer.zero_grad()
            _, _, _, pred_cl_norm = model(pc)
            
            # 混合 Loss：MSE 强制模型去学习离群点 (0.09, 0.8)
            loss = 1 * criterion_mse(pred_cl_norm, gt_cl_norm) 
            #+ 0.3 * criterion_mae(pred_cl_norm, gt_cl_norm)
            
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 防止梯度爆炸
            optimizer.step()
            epoch_loss += loss.item() * current_batch_size
            total_samples += current_batch_size # 累加总样本数

        # 验证
        #val_mse, val_mae = validate(model, val_loader, device, CL_MEAN, CL_STD)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        avg_train_loss = epoch_loss / total_samples

        if (epoch + 1) % 1 == 0:
            val_mse, val_mae = validate(model, val_loader, device, CL_MEAN, CL_STD)
            
            print(f"E{epoch+1:04d} | Train Loss:{avg_train_loss:.5f} | Val MSE:{val_mse:.5f} | Val Real MAE:{val_mae:.5f} | LR:{current_lr:.2e}")
            
            # 只有在验证时才判断是否保存最佳模型
            if val_mae < best_mae:
                best_mae = val_mae
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_cl_model.pth'))
                print(f"  --> Best Model Saved (MAE: {val_mae:.5f})")
        # 如果你希望平时也能看到训练没有卡死，可以加个 elif 打印
        elif (epoch + 1) % 5 == 0:
            print(f"E{epoch+1:04d} | Train Loss:{avg_train_loss:.5f} | (Skipped Val) | LR:{current_lr:.2e}")

        # 每个 epoch 写入日志
        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_train_loss, val_mse, val_mae, current_lr])

if __name__ == '__main__':
    main()