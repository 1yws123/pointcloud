import os
import csv
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from schedulers import WarmupCosineScheduler

# 导入你的模块（确保 dataset.py 存在于同级目录）
from dataset import SDFDataset
from model import PointCloudVAE

# --- 环境配置 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===============================================================
# 2. 权重组装加载器
# ===============================================================
def load_stage1_weights(model, shape_ckpt, cl_ckpt, cd_ckpt):
    """分别加载 Stage 1 中训练好的 Shape、CL、CD 权重"""
    print("--- 正在融合 Stage 1 预训练权重 ---")
    
    # 1. 加载 Shape (包含 Encoder, Decoder, SDF Head)
    if os.path.exists(shape_ckpt):
        state = torch.load(shape_ckpt, map_location='cpu', weights_only=False)
        if 'model_state_dict' in state: state = state['model_state_dict']
        # 剔除可能残留的单体 aero_decoder 权重
        shape_state = {k: v for k, v in state.items() if 'aero_decoder' not in k}
        model.load_state_dict(shape_state, strict=False)
        print(f"✅ 加载 Shape 权重: {shape_ckpt}")
    else:
        print(f"⚠️ 未找到 Shape 权重: {shape_ckpt}")

    # 2. 加载 CL Decoder
    if os.path.exists(cl_ckpt):
        state = torch.load(cl_ckpt, map_location='cpu')
        if 'model_state_dict' in state: state = state['model_state_dict']
        cl_state = {k.replace('aero_decoder.', 'cl_decoder.'): v for k, v in state.items() if 'aero_decoder' in k}
        model.load_state_dict(cl_state, strict=False)
        print(f"✅ 加载 CL 权重: {cl_ckpt}")

    # 3. 加载 CD Decoder
    if os.path.exists(cd_ckpt):
        state = torch.load(cd_ckpt, map_location='cpu')
        if 'model_state_dict' in state: state = state['model_state_dict']
        cd_state = {k.replace('aero_decoder.', 'cd_decoder.'): v for k, v in state.items() if 'aero_decoder' in k}
        model.load_state_dict(cd_state, strict=False)
        print(f"✅ 加载 CD 权重: {cd_ckpt}")
        
    return model

# ===============================================================
# 3. 参数与验证逻辑
# ===============================================================
def get_args():
    parser = argparse.ArgumentParser(description='PhysGen Stage 2: Joint Fine-Tuning')
    
    # 数据路径
    parser.add_argument('--pc_root', type=str, default='/home/yuwenshi/B737/B737_1299/G58_pc_1299/pointcloud')
    parser.add_argument('--aero_root', type=str, default='/home/yuwenshi/B737/B737_1299/G58_aero_1299/G58_aero_1299')
    parser.add_argument('--sdf_dir', type=str, default='/home/yuwenshi/B737/B737_1299/sdf_data')
    parser.add_argument('--save_dir', type=str, default='/home/yuwenshi/B737/model_Aero/encoder_3decoder/checkpoints_1')

    # Stage 1 预训练权重路径
    parser.add_argument('--ckpt_shape', type=str, default='/home/yuwenshi/B737/checkpoint_all_2/vae_epoch_16400.pth')
    parser.add_argument('--ckpt_cl', type=str, default='/home/yuwenshi/B737/model_Aero/encoder_LiftCoefficientDecoder/checkpoints_10/best_cl_model.pth')
    parser.add_argument('--ckpt_cd', type=str, default='/home/yuwenshi/B737/model_Aero/encoder_cdDecoder/checkpoints_4/best_cd_model.pth')
    
    # 超参数 (对应论文设定)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=30)  # 如果显存不够可以调小
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--val_split', type=float, default=0.2)
    
    # 损失权重 (PhysGen paper: lambda_shape=10, lambda_physics=0.1, lambda_drag=10)
    parser.add_argument('--lambda_shape', type=float, default=100)
    parser.add_argument('--lambda_cl', type=float, default=1.0)
    parser.add_argument('--lambda_cd', type=float, default=0.5)
    parser.add_argument('--beta_kl', type=float, default=1e-6)
    parser.add_argument('--surface_threshold', type=float, default=0.02)
    parser.add_argument('--num_points_sdf', type=int, default=100000)
    
    return parser.parse_args()

def validate(model, val_loader, device, cl_stats, cd_stats, threshold):
    """验证集评估，返回 Shape、CL、CD 的真实指标"""
    model.eval()
    total_shape_l1 = 0
    total_cl_mae, total_cd_mae = 0, 0
    samples = 0
    
    cl_mean, cl_std = cl_stats
    cd_mean, cd_std = cd_stats
    
    recon_loss_fn = nn.L1Loss(reduction='mean')

    with torch.no_grad():
        for batch in val_loader:
            points = batch['point_cloud'].to(device)
            sdf_pts = batch['sdf_points'].to(device)
            sdf_gt = batch['sdf_values'].to(device)
            
            cl_gt = batch['raw_cl'].to(device).float().view(-1, 1)
            cd_gt = batch['raw_cd'].to(device).float().view(-1, 1)
            
            sdf_pred, _, _, cl_pred_norm, cd_pred_norm = model(points, query_points=sdf_pts)
            
            # --- Shape Metric ---
            surf_mask = torch.abs(sdf_gt) < threshold
            if surf_mask.sum() > 0:
                shape_l1 = recon_loss_fn(sdf_pred[surf_mask], sdf_gt[surf_mask]).item()
            else:
                shape_l1 = recon_loss_fn(sdf_pred, sdf_gt).item()
                
            # --- Aero Metric (逆标准化计算真实 MAE) ---
            cl_pred_real = cl_pred_norm * cl_std + cl_mean
            cd_pred_real = cd_pred_norm * cd_std + cd_mean
            
            cl_mae = F.l1_loss(cl_pred_real, cl_gt).item()
            cd_mae = F.l1_loss(cd_pred_real, cd_gt).item()
            
            batch_size = points.size(0)
            total_shape_l1 += shape_l1 * batch_size
            total_cl_mae += cl_mae * batch_size
            total_cd_mae += cd_mae * batch_size
            samples += batch_size
            
    return total_shape_l1 / samples, total_cl_mae / samples, total_cd_mae / samples

# ===============================================================
# 4. 主训练流程
# ===============================================================
def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("Pre-loading dataset into memory...")
    full_dataset = SDFDataset(
        pc_root_dir=args.pc_root, 
        aero_root_dir=args.aero_root, 
        sdf_dir=args.sdf_dir,
        num_points_sdf=args.num_points_sdf
    )
    
    cl_stats = (full_dataset.cl_mean, full_dataset.cl_std)
    cd_stats = (full_dataset.cd_mean, full_dataset.cd_std)
    
    train_size = int((1 - args.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # 初始化模型
    model = PointCloudVAE(
       latent_dim=128, plane_resolution=128, plane_features=32,
       num_fourier_freqs=8, num_points_uniform=4000, num_points_curvature=4000, num_points_importance=4000
    ).to(DEVICE)

    # 加载分别训练好的权重
    model = load_stage1_weights(model, args.ckpt_shape, args.ckpt_cl, args.ckpt_cd)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for Joint fine-tuning.")
        model = nn.DataParallel(model)

    # 联合微调：解冻所有参数
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=args.epochs // 10, total_epochs=args.epochs)
    reconstruction_loss_fn = nn.L1Loss(reduction='mean')
        
    # 日志记录
    log_file = os.path.join(args.save_dir, 'train_joint_log.csv')
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'Loss_Total', 'Loss_Shape', 'Loss_CL', 'Loss_CD', 'Val_Shape_L1', 'Val_CL_MAE', 'Val_CD_MAE', 'LR'])

    best_score = float('inf') # 综合判断标准，可以使用 CL_MAE + CD_MAE 作为核心保存指标
    
    print("\n🚀 Start Stage 2 Joint Fine-tuning...")
    for epoch in range(args.epochs):
        model.train()
        ep_loss, ep_l_shape, ep_l_cl, ep_l_cd = 0, 0, 0, 0
        
        start_time = time.time()
        for i, batch in enumerate(train_loader):
            points = batch['point_cloud'].to(DEVICE)
            sdf_pts = batch['sdf_points'].to(DEVICE)
            sdf_gt = batch['sdf_values'].to(DEVICE)
            
            cl_gt_norm = batch['aero_label'][:, 0].to(DEVICE).float().view(-1, 1)
            cd_gt_norm = batch['aero_label'][:, 1].to(DEVICE).float().view(-1, 1)

            optimizer.zero_grad()
            
            # Forward
            sdf_pred, mu, logvar, cl_pred_norm, cd_pred_norm = model(points, query_points=sdf_pts)

            # --- 1. Shape Loss ---
            surface_mask = torch.abs(sdf_gt) < args.surface_threshold
            non_surf_mask = ~surface_mask
            if surface_mask.sum() > 0:
                loss_surf = reconstruction_loss_fn(sdf_pred[surface_mask], sdf_gt[surface_mask])
            else:
                loss_surf = torch.tensor(0.0, device=DEVICE)  # 保持为 Tensor 类型
            # 安全计算非表面 Loss
            if non_surf_mask.sum() > 0:
                loss_non_surf = reconstruction_loss_fn(sdf_pred[non_surf_mask], sdf_gt[non_surf_mask])
            else:
                loss_non_surf = torch.tensor(0.0, device=DEVICE)
            
            recon_loss = loss_surf * 3.0 + loss_non_surf
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            shape_loss = recon_loss + args.beta_kl * kl_loss

            # --- 2. Physics (CL) Loss ---
            cl_loss = F.mse_loss(cl_pred_norm, cl_gt_norm) + F.l1_loss(cl_pred_norm, cl_gt_norm)

            # --- 3. Drag (CD) Loss ---
            cd_loss = F.mse_loss(cd_pred_norm, cd_gt_norm) + F.l1_loss(cd_pred_norm, cd_gt_norm)

            # --- 4. Total Loss ---
            total_loss = (args.lambda_shape * shape_loss) + (args.lambda_cl * cl_loss) + (args.lambda_cd * cd_loss)

            total_loss.backward()
            optimizer.step()

            ep_loss += total_loss.item()
            ep_l_shape += shape_loss.item()
            ep_l_cl += cl_loss.item()
            ep_l_cd += cd_loss.item()

        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        n_batches = len(train_loader)
        
        # 验证集评估
        val_shape_l1, val_cl_mae, val_cd_mae = validate(model, val_loader, DEVICE, cl_stats, cd_stats, args.surface_threshold)
        
        print(f"E{epoch+1:03d} | Total: {ep_loss/n_batches:.4f} (Shp:{ep_l_shape/n_batches:.4f} Cl:{ep_l_cl/n_batches:.4f} Cd:{ep_l_cd/n_batches:.4f}) "
              f"| Val Shp_L1: {val_shape_l1:.4f} Cl_MAE: {val_cl_mae:.4f} Cd_MAE: {val_cd_mae:.4f} | LR: {current_lr:.2e}")

        # 保存策略 (依据气动指标总和)
        current_score = val_cl_mae + val_cd_mae
        is_parallel = isinstance(model, nn.DataParallel)
        
        if current_score < best_score:
            best_score = current_score
            torch.save(model.module.state_dict() if is_parallel else model.state_dict(), 
                       os.path.join(args.save_dir, 'best_joint_model.pth'))
            print(f"  🌟 [Best Model Saved] Combined Aero MAE Score: {best_score:.4f}")

        if (epoch + 1) % 500 == 0:
            torch.save(model.module.state_dict() if is_parallel else model.state_dict(), 
                       os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pth'))
        # 写入日志
        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, ep_loss/n_batches, ep_l_shape/n_batches, ep_l_cl/n_batches, ep_l_cd/n_batches, 
                                    val_shape_l1, val_cl_mae, val_cd_mae, current_lr])

if __name__ == "__main__":
    main()