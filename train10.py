import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import argparse
import re 
from schedulers import WarmupCosineScheduler

from model9 import PointCloudVAE 
from dataset4 import SDFDataset   

# --- 1. 命令行參數 ---
parser = argparse.ArgumentParser(description='Triplane-VAE 訓練腳本')
parser.add_argument('--resume', type=str, default=None,help='checkpoint 檔案恢復訓練')
parser.add_argument('--epochs', type=int, default=2000, help='總共要訓練到的 epoch 數量')
parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
parser.add_argument('--lr', type=float, default=1e-3, help='最大學習率(1e-2:7_2,1e-3:7_3,1e-1:7_4)')
parser.add_argument('--beta_kl', type=float, default=1e-6, help='KL 損失的權重')
parser.add_argument('--latent_dim', type=int, default=128, help='潛在空間維度')
parser.add_argument('--plane_res', type=int, default=16, help='特徵平面的分辨率')
parser.add_argument('--plane_feat', type=int, default=4, help='特徵平面的通道數')
parser.add_argument('--num_points_pc', type=int, default=8192, help='点云采样点数')
parser.add_argument('--num_points_sdf', type=int, default=8192, help='SDF采样点数')
parser.add_argument('--fourier_dim', type=int, default=8, help='傅里葉特徵的頻率數量')
parser.add_argument('--surface_ratio', type=float, default=0.8, help='表面采样比例')
parser.add_argument('--surface_threshold', type=float, default=0.02, help='表面点SDF阈值')

args = parser.parse_args()

# --- 2. 路徑與設備設定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PC_ROOT_DIR = "F:/Code/base_dataset"
SDF_DIR = "F:/Code/pointnet2/DeepSDF/Data2/SdfSamples/DataSource/heart1"
CHECKPOINT_DIR = "./checkpoint10"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- 3. 準備數據 ---
print("正在初始化數據集...")
train_dataset = SDFDataset(pc_root_dir=PC_ROOT_DIR, sdf_dir=SDF_DIR,num_points_pc=args.num_points_pc, 
    num_points_sdf=args.num_points_sdf,surface_ratio=args.surface_ratio,
    surface_threshold=args.surface_threshold)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

# --- 4. 初始化模型與優化器 ---
print("正在初始化完整的 VAE 模型...")
model = PointCloudVAE(
    latent_dim=args.latent_dim,
    plane_resolution=args.plane_res,
    plane_features=args.plane_feat,
    num_fourier_freqs=args.fourier_dim
).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = WarmupCosineScheduler(optimizer,warmup_epochs=args.epochs // 10,total_epochs=args.epochs)
reconstruction_loss_fn = torch.nn.L1Loss(reduction='mean')


# --- 5.checkpoint 加載 ---
start_epoch = 0
if args.resume:
    checkpoint_path = os.path.join(CHECKPOINT_DIR, args.resume)
    if os.path.isfile(checkpoint_path):
       print(f"=> 正在從 checkpoints '{args.resume}' 加載...")
       checkpoint = torch.load(args.resume, map_location=DEVICE)
       model.load_state_dict(checkpoint['model_state_dict'])
       optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       start_epoch = checkpoint['epoch']
       print(f"=> Checkpoint 加載完畢。將從 Epoch {start_epoch + 1} 繼續。")
    else:
        print(f"!! 错误：找不到指定的 checkpoint 文件 '{checkpoint_path}'")
        exit()

# --- 6. 訓練 ---
print(f"======================")
print(f"開始在 {DEVICE} 上訓練...")
print(f"======================")

# === 调度器参数 ===

log_path = os.path.join(CHECKPOINT_DIR, "log_fixed.csv")
if not os.path.isfile(log_path):
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Recon_Loss', 'KL_Loss', 'Beta', 'LR'])
        
start_time = time.time()
for epoch in range(start_epoch, args.epochs):
    model.train()


    # --- 训练循环 ---
    epoch_recon_loss, epoch_kl_loss = 0.0, 0.0
    for i, batch in enumerate(train_dataloader):
        point_clouds = batch['point_cloud'].to(DEVICE)
        sdf_points_gt = batch['sdf_points'].to(DEVICE)
        sdf_values_gt = batch['sdf_values'].to(DEVICE)
        
        centroid = torch.mean(point_clouds, dim=1, keepdim=True)
        point_clouds_normalized = point_clouds - centroid
        furthest_distance = torch.max(torch.sqrt(torch.sum(point_clouds_normalized**2, dim=2)), dim=1, keepdim=True)[0]
        point_clouds_normalized = point_clouds_normalized / furthest_distance.unsqueeze(-1)
        
        optimizer.zero_grad()
        triplanes, mu, log_var = model(point_clouds_normalized)
        sdf_values_pred = model.query_sdf(triplanes, sdf_points_gt)
        
        surface_mask = torch.abs(sdf_values_gt) < args.surface_threshold
        reconstruction_loss = reconstruction_loss_fn(sdf_values_pred[surface_mask], sdf_values_gt[surface_mask]) * 2.0 + \
                              reconstruction_loss_fn(sdf_values_pred[~surface_mask], sdf_values_gt[~surface_mask])
        
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        total_loss = reconstruction_loss + args.beta_kl * kl_loss
        
        total_loss.backward()
        optimizer.step()
        
        epoch_recon_loss += reconstruction_loss.item()
        epoch_kl_loss += kl_loss.item()

    scheduler.step()
    # --- 日志记录 ---
    avg_recon_loss = epoch_recon_loss / len(train_dataloader)
    avg_kl_loss = epoch_kl_loss / len(train_dataloader)
    elapsed_time = time.time() - start_time
    
    print(f"Epoch [{epoch+1}/{args.epochs}] | Recon Loss: {avg_recon_loss:.6f} | KL Loss: {avg_kl_loss:.4f} | Beta: {args.beta_kl:.8f} | LR: {scheduler.get_last_lr()[0]:.8f} | Time: {elapsed_time/60:.2f} min")

    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_recon_loss, avg_kl_loss, args.beta_kl, scheduler.get_last_lr()[0]])

    # --- 保存 Checkpoint ---
    if (epoch + 1) % 50 == 0:
        save_path = os.path.join(CHECKPOINT_DIR, f"vae_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        print(f"Checkpoint 已保存至: {save_path}")

print("訓練完成！")
