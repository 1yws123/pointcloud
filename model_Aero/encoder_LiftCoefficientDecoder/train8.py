import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from schedulers import WarmupCosineScheduler

# 导入你的模块
from model10 import PointCloudVAE
from dataset import SDFDataset

import random
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# --- 环境配置 ---
# 建议在命令行中指定显卡，或者在这里统一管理
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser(description='PhysGen Style: Lift Coefficient Decoder Training')
    # 路径配置
    parser.add_argument('--pc_root', type=str, default='/home/yuwenshi/B737/B737_1299/G58_pc_1299/pointcloud')
    parser.add_argument('--aero_root', type=str, default='/home/yuwenshi/B737/B737_1299/G58_aero_1299/G58_aero_1299')
    parser.add_argument('--sdf_dir', type=str, default='/home/yuwenshi/B737/B737_1299/sdf_data')
    parser.add_argument('--save_dir', type=str, default='/home/yuwenshi/B737/model_Aero/encoder_LiftCoefficientDecoder/checkpoints_10')

    parser.add_argument('--pretrained_path', type=str, default='/home/yuwenshi/B737/checkpoint_all_2/vae_epoch_16400.pth', help='预训练好的几何VAE权重路径')
    
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例')
    
    return parser.parse_args()

def validate(model, val_loader, device, cl_mean, cl_std):
    """验证集评估"""
    model.eval()
    total_mse = 0
    total_mae = 0
    samples = 0
    with torch.no_grad():
        for batch in val_loader:
            # 适配你的 dataset.py 返回的字典格式
            points = batch['point_cloud'].to(device)
            cl_real_gt = batch['raw_cl'].to(device).float().view(-1, 1)
            
            # 我们只需要物理解码器的预测，query_points 可以传 None 节省显存
            _, _, _, cl_pred_norm = model(points, query_points=None)
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
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 1. 加载数据集 (根据你的 dataset.py 构造函数名适配)
    print("Pre-loading dataset into memory...")
    full_dataset = SDFDataset(
        pc_root_dir=args.pc_root, 
        aero_root_dir=args.aero_root, 
        sdf_dir=args.sdf_dir
    )
    cl_mean = full_dataset.cl_mean
    cl_std = full_dataset.cl_std
    torch.manual_seed(42)
    train_size = int((1 - args.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # 2. 初始化模型并加载权重
    model = PointCloudVAE(
       latent_dim=128,
       plane_resolution=128,
       plane_features=32,
       num_fourier_freqs=8,
       num_points_uniform=4000,
       num_points_curvature=4000,
       num_points_importance=4000).to(DEVICE)

    
    print(f"Loading pretrained weights from {args.pretrained_path}...")
    # map_location='cpu' 是一种稳健的加载方式
    state_dict = torch.load(args.pretrained_path, map_location='cpu',weights_only=False)
    # strict=False 因为我们要训练新增的 aero_decoder，而权重文件里没有它
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(DEVICE)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)
    
    # 3. 冻结 Encoder，但锁定几何重建分支（不浪费算力）
    for name, param in model.named_parameters():
        if "aero_decoder" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False 

    # 4. 优化器 (只传入 requires_grad=True 的参数)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,weight_decay=1e-3)
    
    # 5. 调度器
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=args.epochs // 10,
        total_epochs=args.epochs
    )
        
    # 6. 日志记录
    log_file = os.path.join(args.save_dir, 'train_aero_log.csv')
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_mse', 'val_mae', 'LR'])

    best_mae = float('inf')
    print("Start training physics branch...")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for i, batch in enumerate(train_loader):
            points = batch['point_cloud'].to(DEVICE)
            cl_gt_norm = batch['aero_label'][:, 0].to(DEVICE).float().view(-1, 1)

            optimizer.zero_grad()
            
            # Forward: 前三个返回值 (sdf, mu, var) 在冻结训练时通常不使用
            # 如果你的模型定义中第四个返回值是 aero_pred:
            _, _, _, cl_pred_norm = model(points, query_points=None)

            # 复合物理 Loss (MSE + MAE)
            loss_mse = F.mse_loss(cl_pred_norm, cl_gt_norm)
            loss_mae = F.l1_loss(cl_pred_norm, cl_gt_norm)
            loss = loss_mse + loss_mae

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 更新调度器并记录 LR
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 验证
        val_mse, val_mae = validate(model, val_loader, DEVICE, cl_mean, cl_std)
        avg_train_loss = epoch_loss / len(train_loader)
        
        print(f"E{epoch+1:04d} | Train Norm Loss: {avg_train_loss:.4f} | Val MAE: {val_mae:.4f} | LR: {current_lr:.2e}")

        # 保存策略
        is_parallel = isinstance(model, nn.DataParallel)
        
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.module.state_dict() if is_parallel else model.state_dict(), 
                       os.path.join(args.save_dir, 'best_cl_model.pth'))
            print(f"  --> Best Model Saved Val Real MAE: {val_mae:.4f}.")

        if (epoch + 1) % 500 == 0:
            torch.save(model.module.state_dict() if is_parallel else model.state_dict(), 
                       os.path.join(args.save_dir, f'cl_model_epoch_{epoch+1}.pth'))

        # 写入日志
        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_train_loss, val_mse, val_mae, current_lr])

if __name__ == "__main__":
    main()