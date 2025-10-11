import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================================================
# 1. PointNet++ 辅助函数 (来自 model.py)
# <--- 修正：使用 model.py 中更标准、更清晰的辅助函数实现
# ===============================================================
def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = torch.sum((xyz.unsqueeze(1) - new_xyz.unsqueeze(2)) ** 2, -1)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn(xyz, k):
    # xyz: [B, N, 3]
    B, N, _ = xyz.shape
    # 计算 pairwise 距离: [B, N, N]
    dist = torch.cdist(xyz, xyz)   # [B, N, N]
    idx = dist.topk(k=k, largest=False)[1]  # [B, N, k]  # 距离最小的 k 个
    return idx

# ===============================================================
# 2. PointNetSetAbstraction (来自 model.py)
# <--- 修正：替换为 model.py 中计算正确的模块
# ===============================================================
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint, self.radius, self.nsample, self.group_all = npoint, radius, nsample, group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        # 这里的 last_channel 计算是正确的，包含了坐标信息
        last_channel = in_channel + 3 
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        if self.group_all:
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_points = torch.cat([xyz, points], dim=2) if points is not None else xyz
            grouped_points = grouped_points.permute(0, 2, 1).unsqueeze(2)
        else:
            new_xyz_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, new_xyz_idx)
            group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, self.npoint, 1, 3)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz
            grouped_points = grouped_points.permute(0, 3, 2, 1)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))

        if self.group_all:
            new_points = torch.max(grouped_points, 3)[0]
        else:
            new_points = torch.max(grouped_points, 2)[0]

        return new_xyz, new_points.permute(0, 2, 1)

class GeometryAwareAttentionBlock(nn.Module):
    def __init__(self, in_channels, k=16):
        super(GeometryAwareAttentionBlock, self).__init__()
        self.k = k
        self.in_channels = in_channels

        # Multi-Head Self-Attention 分支
        self.mhsa = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        self.linear_mhsa = nn.Linear(in_channels, in_channels)

        # KNN Query 分支
        self.linear_knn1 = nn.Linear(in_channels, in_channels)
        self.linear_knn2 = nn.Linear(in_channels, in_channels)
        
        # 融合层
        self.linear_concat = nn.Linear(in_channels * 2, in_channels)
        
        # 激活和归一化
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(in_channels)

    def forward(self, xyz, features):
        """
        xyz: 点的坐标 [B, N, 3]
        features: 点的特征 [B, N, C]
        """
        B, N, C = features.shape
        
        # 1. Multi-Head Self-Attention 分支
        attn_output, _ = self.mhsa(features, features, features)
        global_features = self.linear_mhsa(attn_output)

        # 2. KNN Query 分支
        knn_idx = knn(xyz, k=self.k)
        knn_features = index_points(features, knn_idx)
        
        processed_knn_features = self.relu(self.linear_knn1(knn_features))
        local_features = torch.max(processed_knn_features, dim=2)[0]
        local_features = self.linear_knn2(local_features)

        # 3. 融合
        concatenated_features = torch.cat([global_features, local_features], dim=-1)
        fused_features = self.relu(self.linear_concat(concatenated_features))

        # 4. 残差连接
        output_features = self.norm1(fused_features + features)
        
        return output_features    
# ===============================================================
# 3. Encoder (保持不变，但依赖于修正后的 PointNetSetAbstraction)
# ===============================================================
class Encoder(nn.Module):
    def __init__(self, latent_dim=128,num_fourier_freqs=6):
        super(Encoder, self).__init__()
        self.input_embedder = FourierEmbedder(num_freqs=num_fourier_freqs, input_dim=3)
        sa1_in_channel = self.input_embedder.out_dim
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=sa1_in_channel, mlp=[64, 64, 128], group_all=False)
        self.geo_attn = GeometryAwareAttentionBlock(in_channels=128, k=16)
        
        # 增加一个融合层来处理残差连接
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256, mlp=[256, 512, 1024], group_all=True)
        
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

    def forward(self, xyz):
        initial_features = self.input_embedder(xyz)
        l1_xyz, l1_points = self.sa1(xyz, initial_features) # l1_points: [B, 512, 128]
        
        l1_points_attn = self.geo_attn(l1_xyz, l1_points)
        
        # 将原始特征和注意力增强后的特征进行融合
        l1_points_fused = self.fusion_conv(l1_points.transpose(1, 2) + l1_points_attn.transpose(1, 2)).transpose(1, 2)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points_fused) # 将融合后的特征送入下一层
        
        _, global_feature = self.sa3(l2_xyz, l2_points)
        
        x = global_feature.squeeze(1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
# ===============================================================
# 4. Triplane Decoder & Fourier Embedder
# ===============================================================
class TriplaneDecoder(nn.Module):
    def __init__(self, latent_dim=128, plane_resolution=64, plane_features=8):
        super(TriplaneDecoder, self).__init__()
        
        # --- 动态构建部分 ---
        self.start_res = 4
        self.target_res = plane_resolution

        # 检查分辨率是否有效 (必须是2的幂，且大于等于起始分辨率)
        assert self.target_res >= self.start_res and (self.target_res & (self.target_res - 1) == 0), \
            f"目标分辨率 (plane_resolution) 必须是4或更高的2的幂, 但得到的是 {self.target_res}"

        # 计算需要多少次上采样 (每次 stride=2 的上采样使分辨率翻倍)
        num_upsamples = int(math.log2(self.target_res / self.start_res))

        # --- 网络层定义 ---
        self.fc_start = nn.Linear(latent_dim, 256 * self.start_res * self.start_res)
        
        # 动态创建上采样层
        upsample_layers = []
        in_channels = 256
        for i in range(num_upsamples):
            # 最后一层通道数减半，其他层通道数也相应调整
            out_channels = 16 if i == num_upsamples - 1 else in_channels // 2
            upsample_layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ])
            in_channels = out_channels
        
        self.upsample_layers = nn.Sequential(*upsample_layers)
        
        # 头部卷积层现在接收最后一层上采样的输出通道数
        self.head_xy = nn.Conv2d(in_channels, plane_features, kernel_size=3, stride=1, padding=1)
        self.head_yz = nn.Conv2d(in_channels, plane_features, kernel_size=3, stride=1, padding=1)
        self.head_xz = nn.Conv2d(in_channels, plane_features, kernel_size=3, stride=1, padding=1)
        
    def forward(self, z):
        x = self.fc_start(z)
        x = x.view(x.shape[0], 256, self.start_res, self.start_res)
        shared_features = self.upsample_layers(x)
        
        plane_xy = self.head_xy(shared_features)
        plane_yz = self.head_yz(shared_features)
        plane_xz = self.head_xz(shared_features)
        
        return plane_xy, plane_yz, plane_xz
    
class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=6, input_dim=3):
        super().__init__()
        freq = 2.0 ** torch.arange(num_freqs)
        self.register_buffer("freq", freq, persistent=False)
        self.out_dim = input_dim * (num_freqs * 2 + 1)

    def forward(self, x: torch.Tensor):
        embed = (x[..., None].contiguous() * self.freq).view(*x.shape[:-1], -1)
        return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
    
# ===============================================================
# 5. PointCloudVAE 主模型
# ===============================================================
class PointCloudVAE(nn.Module):
    def __init__(self, latent_dim, plane_resolution, plane_features, num_fourier_freqs=6):
        super(PointCloudVAE, self).__init__()
        self.encoder = Encoder(latent_dim, num_fourier_freqs=num_fourier_freqs)
        
        self.decoder = TriplaneDecoder(
            latent_dim=latent_dim,
            plane_resolution=plane_resolution,
            plane_features=plane_features
        )
        
        self.fourier_embedder = FourierEmbedder(
            num_freqs=num_fourier_freqs,
            input_dim=3
        )
        
        input_dim_sdf_head = (plane_features * 3) + self.fourier_embedder.out_dim
        self.sdf_head = nn.Sequential(
             nn.Linear(input_dim_sdf_head, 512),
             nn.ReLU(),
             nn.Linear(512, 512),
             nn.ReLU(),
             nn.Linear(512, 256),
             nn.ReLU(),
             nn.Linear(256, 1)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        triplanes = self.decoder(z)
        return triplanes, mu, log_var

    def query_sdf(self, triplanes, query_points):
        # <--- 修正：接收元组并正确解包
        plane_xy, plane_yz, plane_xz = triplanes
        batch_size, num_query_points, _ = query_points.shape
        
        # 1. 从三个平面采样特征
        # F.grid_sample 需要 grid 的 shape 是 [B, H_out, W_out, 2]
        # 我们要查询 N 个点，所以可以看作是 1xN 的网格
        grid_xy = query_points[:, :, [0, 1]].view(batch_size, num_query_points, 1, 2)
        features_xy = F.grid_sample(plane_xy, grid_xy, align_corners=True, padding_mode="border", mode='bilinear').squeeze(-1)

        grid_yz = query_points[:, :, [1, 2]].view(batch_size, num_query_points, 1, 2)
        features_yz = F.grid_sample(plane_yz, grid_yz, align_corners=True, padding_mode="border", mode='bilinear').squeeze(-1)

        grid_xz = query_points[:, :, [0, 2]].view(batch_size, num_query_points, 1, 2)
        features_xz = F.grid_sample(plane_xz, grid_xz, align_corners=True, padding_mode="border", mode='bilinear').squeeze(-1)
        
        # 采样后的 shape 都是 [B, C, M]，需要变为 [B, M, C]
        features_xy = features_xy.transpose(1, 2)
        features_yz = features_yz.transpose(1, 2)
        features_xz = features_xz.transpose(1, 2)

        # 2. 计算傅里叶特征
        fourier_features = self.fourier_embedder(query_points)

        # 3. 聚合所有特征
        aggregated_features = torch.cat([features_xy, features_yz, features_xz, fourier_features], dim=-1)

        # 4. 预测 SDF
        predicted_sdf = self.sdf_head(aggregated_features)
        
        return predicted_sdf
