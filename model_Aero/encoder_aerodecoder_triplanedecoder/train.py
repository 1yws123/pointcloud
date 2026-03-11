import os
import csv
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
from schedulers import WarmupCosineScheduler 

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

# 导入带有物理分支的模型 (确保 model4 中有 AeroDecoder)
from model import PointCloudVAE
from dataset import SDFDataset

def get_args():
    parser = argparse.ArgumentParser(description='From Scratch Joint Geometry & Aero Training')
    parser.add_argument('--pc_root', type=str, default='/home/yuwenshi/B737/B737_1299/G58_pc_1299/pointcloud')
    parser.add_argument('--aero_root', type=str, default='/home/yuwenshi/B737/B737_1299/G58_aero_1299/G58_aero_1299')
    parser.add_argument('--sdf_dir', type=str, default='/home/yuwenshi/B737/B737_1299/sdf_data')
    
    parser.add_argument('--save_dir', type=str, default='/home/yuwenshi/B737/model_Aero/encoder_aerodecoder_triplanedecoder/checkpoints_stage1+2从头训')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=24)
    
    # 学习率控制
    parser.add_argument('--lr_base', type=float, default=1e-3, help='Encoder 和 Decoder 的基础学习率')
    parser.add_argument('--lr_aero', type=float, default=2e-3, help='气动预测头使用略高的学习率')
    
    # 损失权重
    parser.add_argument('--beta_kl', type=float, default=1e-6)
    parser.add_argument('--surface_threshold', type=float, default=0.02)
    parser.add_argument('--num_points_sdf', type=int, default=250000, help='训练初期保证几何所需的大采样点')
    
    parser.add_argument('--resume', type=str, default=None, help='checkpoint 路径，用于恢复训练')
    parser.add_argument('--save_interval', type=int, default=200, help='每隔多少个 epoch 保存一次完整状态')

    return parser.parse_args()

def calculate_r2(preds, targets):
    target_mean = torch.mean(targets)
    ss_tot = torch.sum((targets - target_mean) ** 2)
    ss_res = torch.sum((targets - preds) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return r2.item()

def validate(model, val_loader, device, cl_mean, cl_std):
    model.eval()
    val_mae_phys = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in val_loader:
            pc = data['point_cloud'].to(device)
            gt_cl_norm = data['aero_label'][:, 0].unsqueeze(1).to(device)
            
            # 推理 (只需点云，无需 query points)
            _, _, _, pred_cl_norm, _ = model(pc)
            
            # 记录数据用于算 R2
            all_preds.append(pred_cl_norm)
            all_targets.append(gt_cl_norm)
            
            pred_cl_phys = pred_cl_norm * cl_std + cl_mean
            gt_cl_phys = gt_cl_norm * cl_std + cl_mean
            val_mae_phys += torch.mean(torch.abs(pred_cl_phys - gt_cl_phys)).item()
            
    # 计算全局 R2
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    r2_score = calculate_r2(all_preds, all_targets)
            
    return val_mae_phys / len(val_loader), r2_score

def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==================================================")
    print("🚀 从头开始：几何与物理联合感知训练 (Scratch)")
    print("==================================================")

    # 1. 数据集准备
    full_dataset = SDFDataset(
        args.pc_root, args.aero_root, args.sdf_dir, 
        num_points_sdf=args.num_points_sdf,
        surface_threshold=args.surface_threshold
    )
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

    # 2. 模型初始化 (From Scratch)
    model = PointCloudVAE(
       latent_dim=128, plane_resolution=128, plane_features=32, num_fourier_freqs=8,
       num_points_uniform=4000, num_points_curvature=4000, num_points_importance=4000
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"�� 检测到 {torch.cuda.device_count()} 块 GPU，已启用 DataParallel")
        model = nn.DataParallel(model)

    model_to_optim = model.module if hasattr(model, 'module') else model

    # 3. 差异化优化器：给 AeroDecoder 更强的正则化防止死记硬背
    optimizer = optim.AdamW([
        {'params': model_to_optim.encoder.parameters(), 'lr': args.lr_base, 'weight_decay': 1e-4}, 
        {'params': model_to_optim.decoder.parameters(), 'lr': args.lr_base, 'weight_decay': 1e-4}, 
        {'params': model_to_optim.sdf_head.parameters(), 'lr': args.lr_base, 'weight_decay': 1e-4},
        # 气动头：更高的学习率，更强的 weight_decay，强迫它找泛化规律
        {'params': model_to_optim.aero_decoder.parameters(), 'lr': args.lr_aero, 'weight_decay': 1e-2} 
    ]) 
    
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=20, total_epochs=args.epochs)
    # =================断点续训 (Resume) 逻辑 =================
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> 正在从 '{args.resume}' 加载 Checkpoint...")
            checkpoint = torch.load(args.resume, map_location=device)
            model_to_optim.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"=> Checkpoint 加载完毕。将从 Epoch {start_epoch + 1} 继续训练。")
        else:
            print(f"!! 警告：找不到 Checkpoint 文件 '{args.resume}'，将从头开始训练。")
    # =========================================================

    # 损失函数替换：使用 SmoothL1Loss 替代 MSE，防止气动离群点引发梯度爆炸

    criterion_aero = nn.MSELoss()
    criterion_sdf = nn.L1Loss(reduction='mean') 

    log_file = os.path.join(args.save_dir, 'train_log.csv')
    # 如果是续训，且日志文件存在，则使用追加模式 'a'；否则使用写入模式 'w' 并写入表头
    file_mode = 'a' if args.resume and os.path.exists(log_file) else 'w'
    with open(log_file, file_mode, newline='') as f:
        writer = csv.writer(f)
        if file_mode == 'w':
            writer.writerow(['Epoch', 'Recon', 'KL_Loss', 'Aero_MSE', 'Reg_Loss', 'Val_MAE', 'Val_R2', 'LR'])
    best_r2 = -float('inf')

    # 4. 训练循环
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_aero, epoch_sdf, epoch_reg ,epoch_kl = 0.0, 0.0, 0.0 ,0.0
        
        # 动态权重调度 (Dynamic Weighting)
        # 前 30 个 epoch 纯做几何，让 Encoder 形成基本三维认知，不被气动梯度干扰
        if epoch < 50:
            w_aero = 0.0
            w_reg = 0.0
        else:
            # 30 之后平滑引入气动损失
            progress = min(1.0, (epoch - 50) / 100.0) 
            w_aero = progress * 0.5      # 主气动 Loss
            w_reg = progress * 0.1        # 物理拓扑约束 Loss

        for data in train_loader:
            pc = data['point_cloud'].to(device)
            gt_cl_norm = data['aero_label'][:, 0].unsqueeze(1).to(device)
            sdf_points_gt = data['sdf_points'].to(device)
            sdf_values_gt = data['sdf_values'].to(device)

            current_batch_size = pc.size(0) 
            optimizer.zero_grad()
            
            # 数据增强：对抗局部过拟合
            scale = torch.empty(current_batch_size, 1, 1, device=device).uniform_(0.995, 1.005)
            pc_aug = pc * scale + torch.randn_like(pc) * 0.0001
            sdf_points_aug = sdf_points_gt * scale
            sdf_values_aug = sdf_values_gt * scale.view(current_batch_size, 1, 1)

            # 前向传播 (确保 model4 返回 phys_feature 用于正则化)
            sdf_pred, mu, logvar, aero_pred, phys_feat = model(pc_aug, query_points=sdf_points_aug)

            # ---------------- 损失计算 ----------------
            # 1. 几何 SDF Loss
            surface_mask = torch.abs(sdf_values_aug) < args.surface_threshold
            loss_surface = criterion_sdf(sdf_pred[surface_mask], sdf_values_aug[surface_mask])
            loss_non_surface = criterion_sdf(sdf_pred[~surface_mask], sdf_values_aug[~surface_mask])
            loss_recon = (loss_surface * 3.0 + loss_non_surface) if not torch.isnan(loss_surface) else loss_non_surface
            
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

            # 2. 气动预测 Loss
            loss_aero = criterion_aero(aero_pred, gt_cl_norm)
            
            # 3. 🌟 物理拓扑对比正则化 (Physics-Aware Feature Regularization) 🌟
            # 强制：特征在潜空间的距离，必须与它们的 CL 差异正相关
            # 这能彻底杜绝模型“死记硬背” ID，因为死记硬背的特征空间是杂乱无章的
            
            dist_cl = torch.cdist(gt_cl_norm, gt_cl_norm)  # [B, B]
            target_sim = torch.exp(-dist_cl)
            norm_feat = F.normalize(phys_feat, p=2, dim=1) # 归一化特征
            pred_sim = torch.mm(norm_feat, norm_feat.t())
            # 将 CL 差距缩放到和特征距离近似的量级
            loss_phys_reg = F.mse_loss(pred_sim, target_sim)

            # 总 Loss
            loss = loss_recon + args.beta_kl * loss_kl + w_aero * loss_aero + w_reg * loss_phys_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step()
            
            epoch_sdf += loss_recon.item()
            epoch_aero += loss_aero.item()
            epoch_reg += loss_phys_reg.item()
            epoch_kl += loss_kl.item()

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        avg_sdf = epoch_sdf / len(train_loader)
        avg_aero = epoch_aero / len(train_loader)
        avg_reg = epoch_reg / len(train_loader)
        avg_kl = epoch_kl / len(train_loader)

        # 验证与打印
        val_mae, val_r2 = validate(model, val_loader, device, CL_MEAN, CL_STD)
        
        print(f"E{epoch+1:03d} | Recon:{avg_sdf:.6f} | KL:{avg_kl:.4f} | Aero:{avg_aero:.4f} | Reg:{avg_reg:.4f} | Val MAE:{val_mae:.4f} | Val R2:{val_r2:.4f} | W_Aero:{w_aero:.1f}")
        
        if val_r2 > best_r2 and epoch > 100: # 只有引入气动后才存模型
            best_r2 = val_r2
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state_dict, os.path.join(args.save_dir, 'best_joint_model.pth'))
            print(f"  🎉 --> 新的 Best Model 已保存! (R2: {val_r2:.4f})")

        # ================= 新增：定期保存完整 Checkpoint (间隔保存) =================
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            save_path = os.path.join(args.save_dir, f"joint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_to_optim.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'args': vars(args)
            }, save_path)
            print(f"  💾 --> 定期 Checkpoint 已保存至: {save_path}")

        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_sdf, avg_kl, avg_aero, avg_reg, val_mae, val_r2, current_lr])

    print(f"\n✅ 训练已完成！最终模型已保存。")
    final_path = os.path.join(args.save_dir, 'final_model.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model_to_optim.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': vars(args)
    }, final_path)

if __name__ == '__main__':
    main()