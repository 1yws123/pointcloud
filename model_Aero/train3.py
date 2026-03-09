import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from schedulers import WarmupCosineScheduler 

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" 

# 导入你的模型 (必须是带有 1024维 x 输出的 model2)
from model2 import PointCloudVAE
from dataset import SDFDataset

def get_args():
    parser = argparse.ArgumentParser(description='End-to-End Joint Aero-Geometry Training')
    parser.add_argument('--pc_root', type=str, default='/home/yuwenshi/B737/B737_1299/G58_pc_1299/pointcloud')
    parser.add_argument('--aero_root', type=str, default='/home/yuwenshi/B737/B737_1299/G58_aero_1299/G58_aero_1299')
    parser.add_argument('--sdf_dir', type=str, default='/home/yuwenshi/B737/B737_1299/sdf_data')
    
    parser.add_argument('--stage1_ckpt', type=str, default='/home/yuwenshi/B737/checkpoint_all_2/vae_epoch_16200.pth')
    parser.add_argument('--save_dir', type=str, default='/home/yuwenshi/B737/checkpoints_stage1+2一起训练_2')
    
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=48)
    # 联合训练，给予 Encoder 足够动力，同时用 SDF Loss 拉住它
    parser.add_argument('--lr_aero', type=float, default=1e-3) 
    parser.add_argument('--lr_encoder', type=float, default=1e-4) 
    
    # 继承你 Stage 1 的精髓参数
    parser.add_argument('--beta_kl', type=float, default=1e-6)
    parser.add_argument('--surface_threshold', type=float, default=0.02)
    return parser.parse_args()

def validate(model, val_loader, device, cl_mean, cl_std):
    model.eval()
    val_mae_phys = 0.0
    val_mse_norm = 0.0
    criterion_mse = nn.MSELoss()
    
    with torch.no_grad():
        for data in val_loader:
            pc = data['point_cloud'].to(device)
            gt_cl_norm = data['aero_label'][:, 0].unsqueeze(1).to(device)
            
            # 验证集只测气动，不需要传 SDF points
            _, _, _, pred_cl_norm = model(pc)
            
            val_mse_norm += criterion_mse(pred_cl_norm, gt_cl_norm).item()
            pred_cl_phys = pred_cl_norm * cl_std + cl_mean
            gt_cl_phys = gt_cl_norm * cl_std + cl_mean
            val_mae_phys += torch.mean(torch.abs(pred_cl_phys - gt_cl_phys)).item()
            
    return val_mse_norm / len(val_loader), val_mae_phys / len(val_loader)

def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 数据准备 (因为要算 SDF Loss，Dataset 会返回 SDF 采样点)
    full_dataset = SDFDataset(args.pc_root, args.aero_root, args.sdf_dir, 
                              num_points_sdf=16384) # 联合训练时，1.6万点足够稳住几何了
    CL_MEAN = full_dataset.cl_mean
    CL_STD = full_dataset.cl_std

    torch.manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              drop_last=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # 2. 模型初始化 
    model = PointCloudVAE(
       latent_dim=128, plane_resolution=128, plane_features=32, num_fourier_freqs=8,
       num_points_uniform=4000, num_points_curvature=4000, num_points_importance=4000
    ).to(device)

    # 3. 加载 Stage 1 预训练权重
    if os.path.exists(args.stage1_ckpt):
        print(f"🛠️ 正在加载 Stage 1 预训练权重: {args.stage1_ckpt}")
        checkpoint = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ 权重加载成功！")

     # ================= 🌟 新增：启用多 GPU 并行 =================
    if torch.cuda.device_count() > 1:
        print(f"🚀 检测到 {torch.cuda.device_count()} 块 GPU，已启用 DataParallel 并行训练！")
        model = nn.DataParallel(model)

    # ================= 🌟 终极修改：解冻 Encoder，开启差分联合微调 =================
    print("🔓 Encoder 已解冻！SDF定海神针已就位，开始物理-几何联合微调...")
    
    # ⚠️ 注意：使用了 DataParallel 后，模型被包在了一层 module 里，所以获取参数要加 .module
    model_to_optim = model.module if hasattr(model, 'module') else model

    optimizer = optim.AdamW([
        {'params': model_to_optim.encoder.parameters(), 'lr': args.lr_encoder}, 
        {'params': model_to_optim.decoder.parameters(), 'lr': args.lr_encoder}, 
        {'params': model_to_optim.sdf_head.parameters(), 'lr': args.lr_encoder},
        {'params': model_to_optim.aero_decoder.parameters(), 'lr': args.lr_aero} 
    ], weight_decay=1e-4) 
    # =======================================================================
    
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=args.epochs//10, total_epochs=args.epochs)

    # Loss 声明 (复刻你 Stage 1 的 L1 设定)
    criterion_aero = nn.MSELoss() 
    criterion_sdf = nn.L1Loss(reduction='mean') 

    log_file = os.path.join(args.save_dir, 'train_log.csv')
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'Aero_Loss', 'SDF_Recon', 'KL_Loss', 'Val_Real_MAE', 'LR'])

    best_mae = float('inf')

    # 5. 训练循环
    for epoch in range(args.epochs):
        model.train()
        epoch_aero, epoch_sdf, epoch_kl = 0.0, 0.0, 0.0
        total_samples = 0
        
        for data in train_loader:
            pc = data['point_cloud'].to(device)
            gt_cl_norm = data['aero_label'][:, 0].unsqueeze(1).to(device)
            
            # 取出 SDF 数据用于计算几何约束
            sdf_points_gt = data['sdf_points'].to(device)
            sdf_values_gt = data['sdf_values'].to(device)

            current_batch_size = pc.size(0) 
            optimizer.zero_grad()
            
            # 前向传播 (传入 query_points 激活 SDF 分支)
            sdf_values_pred, mu, logvar, pred_cl_norm = model(pc, query_points=sdf_points_gt)
            
            # ================= 🌟 联合 Loss 计算区 =================
            
            # 1. 气动预测 Loss
            loss_aero = criterion_aero(pred_cl_norm, gt_cl_norm)
            
            # 2. 几何重构 Loss (完美复刻你 Stage 1 的 Mask 逻辑)
            surface_mask = torch.abs(sdf_values_gt) < args.surface_threshold
            loss_surface = criterion_sdf(sdf_values_pred[surface_mask], sdf_values_gt[surface_mask])
            loss_non_surface = criterion_sdf(sdf_values_pred[~surface_mask], sdf_values_gt[~surface_mask])
            
            if torch.isnan(loss_surface): loss_surface = 0.0
            if torch.isnan(loss_non_surface): loss_non_surface = 0.0

            loss_recon = loss_surface * 3.0 + loss_non_surface
            
            # 3. KL Divergence (复刻 Stage 1)
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            
            # 🌟 联合盘算：用气动引导微调 (权重10)，用几何死死锁住形状 (权重1)
            loss = 1.0 * loss_aero + 1000.0 * loss_recon + args.beta_kl * loss_kl
            # =======================================================

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step()
            
            epoch_aero += loss_aero.item() * current_batch_size
            epoch_sdf += loss_recon.item() * current_batch_size # 记录纯 Recon
            epoch_kl += loss_kl.item() * current_batch_size
            total_samples += current_batch_size 

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        avg_aero = epoch_aero / total_samples
        avg_sdf = epoch_sdf / total_samples
        avg_kl = epoch_kl / total_samples

        if (epoch + 1) % 1 == 0:
            val_mse, val_mae = validate(model, val_loader, device, CL_MEAN, CL_STD)
            print(f"E{epoch+1:03d} | Aero:{avg_aero:.4f} | Recon:{avg_sdf:.4f} | KL:{avg_kl:.4f} | Val MAE:{val_mae:.5f} | LR:{current_lr:.2e}")
            
            if val_mae < best_mae:
                best_mae = val_mae
                state_dict_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(state_dict_to_save, os.path.join(args.save_dir, 'best_cl_model.pth'))
                print(f"  🎉 --> 新的 Best Model 已保存! (MAE: {val_mae:.5f})")

        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_aero, avg_sdf, avg_kl, val_mae, current_lr])

if __name__ == '__main__':
    main()