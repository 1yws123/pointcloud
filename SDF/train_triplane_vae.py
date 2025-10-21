import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from tqdm import tqdm

# ---------------------------------------------------------------------------- #
#                                   元件 1: 資料集                               #
# ---------------------------------------------------------------------------- #

class DeepSDFDataset(Dataset):
    def __init__(self, latent_vectors, npz_files, samples_per_scene=16384):
        self.latent_vectors = latent_vectors
        self.npz_files = npz_files
        self.samples_per_scene = samples_per_scene

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        latent_vec = self.latent_vectors[idx]
        
        data = np.load(self.npz_files[idx])
        pos_samples = data["pos"]
        neg_samples = data["neg"]
        
        all_samples = np.concatenate([pos_samples, neg_samples], axis=0)
        
        num_samples = all_samples.shape[0]
        replace = num_samples < self.samples_per_scene
        random_indices = np.random.choice(num_samples, self.samples_per_scene, replace=replace)
        selected_samples = all_samples[random_indices]

        points = torch.from_numpy(selected_samples[:, :3]).float()
        sdf_gt = torch.from_numpy(selected_samples[:, 3]).float().unsqueeze(-1)
        
        return latent_vec, points, sdf_gt

# ---------------------------------------------------------------------------- #
#                                   元件 2: 模型定義                               #
# ---------------------------------------------------------------------------- #

class LatentToTriplane(nn.Module):
    def __init__(self, latent_dim, feature_dim, resolution):
        super().__init__()
        self.feature_dim = feature_dim
        self.resolution = resolution
        output_size = 3 * feature_dim * resolution * resolution
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, output_size)
        )

    def forward(self, z):
        x = self.net(z)
        return x.view(-1, 3, self.feature_dim, self.resolution, self.resolution)

class VAE_Encoder(nn.Module):
    def __init__(self, feature_dim, latent_dim_vae, resolution):
        super().__init__()
        in_channels = 3 * feature_dim
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        final_res = resolution // 8
        self.fc_mu = nn.Linear(256 * final_res * final_res, latent_dim_vae)
        self.fc_logvar = nn.Linear(256 * final_res * final_res, latent_dim_vae)

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, -1, x.shape[3], x.shape[4])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(b, -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim_vae, feature_dim, resolution):
        super().__init__()
        out_channels = 3 * feature_dim
        self.feature_dim = feature_dim
        self.final_res = resolution // 8
        
        self.fc = nn.Linear(latent_dim_vae, 256 * self.final_res * self.final_res)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        b = z.shape[0]
        # 修正了這裡的 view 計算
        x = F.relu(self.fc(z)).view(b, 256,self.final_res,self.final_res )
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))
        return x.view(b, 3, self.feature_dim, x.shape[2], x.shape[3])

class SharedSDFDecoder(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        input_dim = 3 * feature_dim + 3
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, points, triplanes):
        B, N, _ = points.shape
        xy_plane_coords = points[:, :, [0, 1]].unsqueeze(1)
        yz_plane_coords = points[:, :, [1, 2]].unsqueeze(1)
        xz_plane_coords = points[:, :, [0, 2]].unsqueeze(1)
        
        # --- 核心修正：移除 .unsqueeze(1) ---
        # 輸入給 grid_sample 的特徵圖應為 4D: [B, C, H, W]
        feat_xy = F.grid_sample(triplanes[:, 0, ...], xy_plane_coords, mode='bilinear', padding_mode='border', align_corners=True).squeeze(2)
        feat_yz = F.grid_sample(triplanes[:, 1, ...], yz_plane_coords, mode='bilinear', padding_mode='border', align_corners=True).squeeze(2)
        feat_xz = F.grid_sample(triplanes[:, 2, ...], xz_plane_coords, mode='bilinear', padding_mode='border', align_corners=True).squeeze(2)
        
        feat_xy, feat_yz, feat_xz = feat_xy.permute(0, 2, 1), feat_yz.permute(0, 2, 1), feat_xz.permute(0, 2, 1)
        features = torch.cat([feat_xy, feat_yz, feat_xz, points], dim=2)
        
        sdf_pred = self.net(features.view(B * N, -1))
        return sdf_pred.view(B, N, 1)

# ---------------------------------------------------------------------------- #
#                                   元件 3: 訓練循環                                #
# ---------------------------------------------------------------------------- #
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    experiment_dir = r"F:\Code\pointnet2\DeepSDF\MyTrainingExperiment"
    npz_dir = r"F:\Code\pointnet2\DeepSDF\Data2\SdfSamples\DataSource\heart"
    
    LATENT_DIM_DEEPSDF = 256
    LATENT_DIM_VAE = 64
    TRIPLANE_FEATURE_DIM = 16
    TRIPLANE_RESOLUTION = 32
    SAMPLES_PER_SCENE = 4096
    BATCH_SIZE = 2
    
    EPOCHS = 1000
    LR = 1e-4
    BETA = 0.001

    print("Loading pre-trained latent vectors...")
    latent_codes_path = os.path.join(experiment_dir, "LatentCodes", "latest.pth")
    latent_data = torch.load(latent_codes_path)
    latent_vectors_tensor = latent_data['latent_codes']['weight'].detach()
    
    npz_files_pattern = os.path.join(npz_dir, "*.npz")
    npz_files_list = glob.glob(npz_files_pattern)
    print(f"Found {len(npz_files_list)} npz files.")

    dataset = DeepSDFDataset(latent_vectors_tensor, npz_files_list, samples_per_scene=SAMPLES_PER_SCENE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    latent_to_triplane = LatentToTriplane(LATENT_DIM_DEEPSDF, TRIPLANE_FEATURE_DIM, TRIPLANE_RESOLUTION).to(device)
    vae_encoder = VAE_Encoder(TRIPLANE_FEATURE_DIM, LATENT_DIM_VAE, TRIPLANE_RESOLUTION).to(device)
    vae_decoder = VAE_Decoder(LATENT_DIM_VAE, TRIPLANE_FEATURE_DIM, TRIPLANE_RESOLUTION).to(device)
    sdf_decoder = SharedSDFDecoder(TRIPLANE_FEATURE_DIM).to(device)
    
    params = list(latent_to_triplane.parameters()) + list(vae_encoder.parameters()) + list(vae_decoder.parameters()) + list(sdf_decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)
    
    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for latent_vec, points, sdf_gt in pbar:
            latent_vec, points, sdf_gt = latent_vec.to(device), points.to(device), sdf_gt.to(device)

            optimizer.zero_grad()
            
            original_triplanes = latent_to_triplane(latent_vec)
            mu, logvar = vae_encoder(original_triplanes)
            z = reparameterize(mu, logvar)
            reconstructed_triplanes = vae_decoder(z)
            sdf_pred = sdf_decoder(points, reconstructed_triplanes)
            
            recon_loss = F.l1_loss(sdf_pred, sdf_gt)
            kl_loss = kl_divergence(mu, logvar) / (BATCH_SIZE * SAMPLES_PER_SCENE)
            
            total_loss = recon_loss + BETA * kl_loss
            
            total_loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=total_loss.item(), recon=recon_loss.item(), kl=kl_loss.item())

    print("\n訓練完成！正在儲存模型權重...")
    
    # 創建一個資料夾來存放 VAE 模型的權重
    vae_model_dir = os.path.join(experiment_dir, "TriplaneVAE_Models")
    os.makedirs(vae_model_dir, exist_ok=True)
    
    # 儲存 VAE 解碼器和共享 SDF 解碼器
    torch.save(vae_decoder.state_dict(), os.path.join(vae_model_dir, "vae_decoder.pth"))
    torch.save(sdf_decoder.state_dict(), os.path.join(vae_model_dir, "sdf_decoder.pth"))
    
    print(f"模型已成功儲存到: {vae_model_dir}")
    
if __name__ == '__main__':
    train()