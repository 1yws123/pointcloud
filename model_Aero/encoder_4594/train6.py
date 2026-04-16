import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split,Subset
import numpy as np
from schedulers import WarmupCosineScheduler
from torch import autocast
from torch.amp import GradScaler
import time 
from accelerate import Accelerator
import random

# 导入你的模块
from model2 import PointCloudVAE
from dataset import SDFDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

def get_args():
    parser = argparse.ArgumentParser(description='End-to-End Aero Training from Scratch')
    parser.add_argument('--pc_root', type=str, default='/home/yuwenshi/B737/B737_4594/pc1')
    parser.add_argument('--aero_root', type=str, default='/home/yuwenshi/B737/G58_4594_aero')
    parser.add_argument('--sdf_dir', type=str, default='/home/yuwenshi/B737/B737_4594/sdf')
    parser.add_argument('--save_dir', type=str, default='/home/yuwenshi/B737/model_Aero/encoder_4594/checkpoints_5')
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-4) # 推荐初始学习率
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例') 
    return parser.parse_args()

def validate(model, val_loader, device, cl_mean, cl_std, full_dataset):
    """验证集评估"""
    model.eval()
    total_mse = 0
    total_mae = 0
    samples = 0
    with torch.no_grad():
        for batch in val_loader:
            # .to(device) 
            points = batch['point_cloud'].to(device)
            cl_real_gt = batch['raw_cl'].to(device).float().view(-1, 1)
            
            #shift_norm = batch['shift'].to(device)
            #scale_norm = batch['scale'].to(device)
            #aux_vec = torch.cat([shift_norm, scale_norm], dim=1)    # [B, 4]
            aux_vec = None 

            # 我们只需要物理解码器的预测，query_points 可以传 None 节省显存
            _, _, _, cl_pred_norm = model(points, aux_vec=aux_vec, query_points=None, aero_only=True)
            cl_pred_real = cl_pred_norm * cl_std + cl_mean
            mse = F.mse_loss(cl_pred_real, cl_real_gt)
            mae = F.l1_loss(cl_pred_real, cl_real_gt)
            
            batch_size = points.size(0)
            total_mse += mse.item() * batch_size
            total_mae += mae.item() * batch_size
            samples += batch_size
            
    return total_mse / (samples + 1e-8), total_mae / (samples + 1e-8)

def main():
    args = get_args()
    accelerator = Accelerator()
    if accelerator.is_main_process:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        print("Pre-loading dataset into memory...")
    
    full_dataset = SDFDataset(
        pc_root_dir=args.pc_root, 
        aero_root_dir=args.aero_root, 
        sdf_dir=args.sdf_dir,
        mode='aero' 
    )
    cl_mean = full_dataset.cl_mean
    cl_std = full_dataset.cl_std
    #torch.manual_seed(42)
    #train_size = int((1 - args.val_split) * len(full_dataset))
    #val_size = len(full_dataset) - train_size
    #train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    family_to_indices = {}
    for idx, data_item in enumerate(full_dataset.memory_cache):
        # 这里的 file_id 就是几何族群的唯一标识 (如 'G58_4594')
        family_id = data_item['file_id'] 
        if family_id not in family_to_indices:
            family_to_indices[family_id] = []
        family_to_indices[family_id].append(idx)
        
    unique_families = list(family_to_indices.keys())
    print(f"   --> 共有 {len(unique_families)} 个独立的几何族群。")
    random.seed(42)
    random.shuffle(unique_families)

    # 3. 按比例将族群分配到训练集和验证集
    target_train_size = int((1 - args.val_split) * len(full_dataset))
    
    train_indices = []
    val_indices = []
    
    for family_id in unique_families:
        # 如果训练集还没装够，就把当前族群的所有样本放入训练集
        if len(train_indices) < target_train_size:
            train_indices.extend(family_to_indices[family_id])
        else:
            val_indices.extend(family_to_indices[family_id])

    # 4. 根据分好的索引生成 Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    print(f"✅ 划分完成: 训练集样本 {len(train_dataset)} 个, 验证集样本 {len(val_dataset)} 个。")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. 初始化模型并加载权重
    model = PointCloudVAE(
       latent_dim=128,
       plane_resolution=128,
       plane_features=32,
       num_fourier_freqs=8,
       num_points_uniform=4000,
       num_points_curvature=4000,
       num_points_importance=4000)
    
    for name, param in model.named_parameters():
        if "decoder" in name and "aero" not in name:  # 冻结 Triplane 几何解码器
            param.requires_grad = False
        elif "sdf_head" in name:                      # 冻结 SDF 预测头
            param.requires_grad = False
        else:
            param.requires_grad = True 
            
    # 3. 优化器 (全量训练)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-2)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=20, total_epochs=args.epochs)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    log_file = os.path.join(args.save_dir, 'train_aero_log.csv')
    if accelerator.is_main_process:
        with open(log_file, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_mse', 'val_mae', 'LR'])
        print("Start training physics branch...")

    best_mae = float('inf')

    # 5. 训练循环
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        train_start_time = time.time()
        for i, batch in enumerate(train_loader):
            points = batch['point_cloud']
            if model.training:
                # 加万分之一量级的高斯噪声
                points = points + torch.randn_like(points) * 1e-4 
            cl_gt_norm = batch['aero_label'][:, 0].float().view(-1, 1)
            
            #shift_norm = batch['shift']         # [B, 3]
            #scale_norm = batch['scale']         # [B, 1]
            #aux_vec = torch.cat([shift_norm, scale_norm], dim=1)    # [B, 4]
            aux_vec = None 

            optimizer.zero_grad()
            
            # --- 手动拆解 Forward 过程，以便在潜空间注入噪声 ---
            # 1. 过 Encoder (此时 sa3 和 fc 层会根据物理 Loss 更新!)
            _, _, _, cl_pred_norm = model(points, aux_vec=aux_vec, query_points=None, aero_only=True)
            
            # 复合物理 Loss (MSE + MAE)
            loss_mse = F.mse_loss(cl_pred_norm, cl_gt_norm)
            loss_mae = F.l1_loss(cl_pred_norm, cl_gt_norm)
            loss = loss_mse + loss_mae

            accelerator.backward(loss)
            optimizer.step()

            epoch_loss += loss.item()  #135-160

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 验证
        avg_train_loss = epoch_loss / len(train_loader)
        train_time = time.time() - train_start_time  # 训练总耗时
        if accelerator.is_main_process:
            val_and_save_start = time.time()
            if (epoch + 1) % 20 == 0:
                # 用 accelerator.unwrap_model 提取原始模型供验证使用
                raw_model = accelerator.unwrap_model(model)
                val_mse, val_mae = validate(raw_model, val_loader, accelerator.device, cl_mean, cl_std, full_dataset)
                
                if val_mae < best_mae:
                    best_mae = val_mae
                    torch.save(raw_model.state_dict(), os.path.join(args.save_dir, 'best_cl_model.pth'))
                    print(f"  --> 🌟 Best Model Saved! Val Real MAE: {val_mae:.4f}")     
                
                val_and_save_time = time.time() - val_and_save_start
                print(f"E{epoch+1:04d} | Train Loss: {avg_train_loss:.4f} ({train_time:.1f}s) | Val MAE: {val_mae:.4f} | LR: {current_lr:.2e} | 验证耗时: {val_and_save_time:.1f}s ")
       
                with open(log_file, 'a', newline='') as f:
                    csv.writer(f).writerow([epoch+1, avg_train_loss, val_mse, val_mae, current_lr])

            else:
                val_and_save_time = time.time() - val_and_save_start
                print(f"E{epoch+1:04d} | Train Loss: {avg_train_loss:.4f} ({train_time:.1f}s) | Val MAE: ------ | LR: {current_lr:.2e}")
                
                with open(log_file, 'a', newline='') as f:
                    csv.writer(f).writerow([epoch+1, avg_train_loss, "", "", current_lr])

            if (epoch + 1) % 200 == 0:
                torch.save(accelerator.unwrap_model(model).state_dict(), 
                           os.path.join(args.save_dir, f'cl_model_epoch_{epoch+1}.pth'))
        accelerator.wait_for_everyone()

    
if __name__ == '__main__':
    main()