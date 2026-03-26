import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================================================
# CUDA 加速：自动检测 pointnet2_ops / torch_cluster，没有则退回 Python 实现
# ===============================================================
try:
    from pointnet2_ops import pointnet2_utils as p2_utils
    _HAS_POINTNET2_OPS = True
    print("[INFO] pointnet2_ops 已加载，使用 CUDA 加速 FPS & Ball Query")
except ImportError:
    _HAS_POINTNET2_OPS = False

try:
    import torch_cluster
    _HAS_TORCH_CLUSTER = True
    print("[INFO] torch_cluster 已加载，使用 CUDA 加速 KNN")
except ImportError:
    _HAS_TORCH_CLUSTER = False


def _fps_python(xyz, npoint):
    """纯 Python 的 FPS（慢，仅作退路）"""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def farthest_point_sample(xyz, npoint):
    """FPS：有 pointnet2_ops 就用 CUDA，否则退回 Python"""
    if _HAS_POINTNET2_OPS and xyz.is_cuda:
        # 1. 确保传入的是 float32 类型并连续
        idx = p2_utils.furthest_point_sample(xyz.float().contiguous(), npoint).long()
        # [核心修复 1]：C++ 后端在异常数据下可能返回乱码索引，强制 Clamp 彻底杜绝越界
        idx = torch.clamp(idx, min=0, max=xyz.shape[1] - 1)
        return idx
    return _fps_python(xyz, npoint)


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    
    # [核心修复 2]：无论上游传来什么索引，在这里做最后一道防线，确保不越界
    idx = torch.clamp(idx, min=0, max=points.shape[1] - 1)
    
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
    
    # [核心修复 3]：用 torch.cdist 替代 unsqueeze 广播！
    # 原写法在 BS=120 时会瞬间占用 ~6GB 显存极易引发崩溃，cdist 高度优化且不产生中间巨型张量
    #sqrdists = torch.cdist(new_xyz, xyz) ** 2  # Shape: [B, S, N]
    sqrdists = torch.cdist(new_xyz.contiguous(), xyz.contiguous()) ** 2
    
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    
    # --- 确保 group_first 不包含越界值 N ---
    group_first[group_first == N] = 0 
    
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn(xyz, k):
    """KNN：有 torch_cluster 就用 CUDA，否则用 cdist（内存优化版）"""
    B, N, _ = xyz.shape

    if _HAS_TORCH_CLUSTER and xyz.is_cuda:
        # torch_cluster.knn 要求 2D 输入 + batch 向量
        device = xyz.device
        xyz_flat = xyz.reshape(B * N, -1)                          # [B*N, 3]
        batch_vec = torch.arange(B, device=device).repeat_interleave(N)  # [B*N]
        # 返回 [2, B*N*k]，第 0 行是 query idx，第 1 行是 neighbor idx
        assign = torch_cluster.knn(xyz_flat, xyz_flat, k, batch_vec, batch_vec)
        idx = assign[1].reshape(B, N, k)
        # torch_cluster 返回的是全局索引，转成 batch 内局部索引
        offset = torch.arange(B, device=device).view(B, 1, 1) * N
        idx = idx - offset
        return idx

    # 退路：纯 PyTorch，分 chunk 降低峰值内存（原版一次性算 [B, N, N]）
    all_idx = []
    for b in range(B):
        d = torch.cdist(xyz[b:b+1], xyz[b:b+1]).squeeze(0)  # [N, N]
        all_idx.append(d.topk(k=k, largest=False)[1])        # [N, k]
    return torch.stack(all_idx, dim=0)  # [B, N, k]

# ===============================================================
# 模块定义 (PointNet++, Attention)
# ===============================================================
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint, self.radius, self.nsample, self.group_all = npoint, radius, nsample, group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
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
            if self.npoint is None or self.npoint == 0:
                new_xyz = xyz
            else:
                new_xyz_idx = farthest_point_sample(xyz, self.npoint)
                new_xyz = index_points(xyz, new_xyz_idx)
            group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, self.npoint or N, 1, 3)
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
        self.mhsa = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        self.linear_mhsa = nn.Linear(in_channels, in_channels)
        self.linear_knn1 = nn.Linear(in_channels, in_channels)
        self.linear_knn2 = nn.Linear(in_channels, in_channels)
        self.linear_concat = nn.Linear(in_channels * 2, in_channels)
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(in_channels)

    def forward(self, xyz, features):
        B, N, C = features.shape
        attn_output, _ = self.mhsa(features, features, features)
        global_features = self.linear_mhsa(attn_output)
        knn_idx = knn(xyz, k=self.k)
        knn_features = index_points(features, knn_idx)
        processed_knn_features = self.relu(self.linear_knn1(knn_features))
        local_features = torch.max(processed_knn_features, dim=2)[0]
        local_features = self.linear_knn2(local_features)
        concatenated_features = torch.cat([global_features, local_features], dim=-1)
        fused_features = self.relu(self.linear_concat(concatenated_features))
        output_features = self.norm1(fused_features + features)
        return output_features    

# ===============================================================
# 3. Encoder (三路并行修改版)
class Encoder(nn.Module):
    # 更新初始化参数，接收三个数量
    def __init__(self, latent_dim=128, num_fourier_freqs=8, 
                 num_points_uniform=4000, 
                 num_points_curvature=4000, # 新增
                 num_points_importance=4000):
        super(Encoder, self).__init__()
        self.num_points_uniform = num_points_uniform
        self.num_points_curvature = num_points_curvature # 新增
        self.num_points_importance = num_points_importance

        self.input_embedder = FourierEmbedder(num_freqs=num_fourier_freqs, input_dim=3)
        sa1_in_channel = self.input_embedder.out_dim
        
        # --- 核心修改：三路并行 SA1 ---
        # 1. Uniform 分支
        self.sa1_uniform = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=32, in_channel=sa1_in_channel, mlp=[64, 64, 128], group_all=False)
        # 2. Curvature 分支 (新增)
        self.sa1_curvature = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=32, in_channel=sa1_in_channel, mlp=[64, 64, 128], group_all=False)
        # 3. Importance 分支
        self.sa1_importance = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=32, in_channel=sa1_in_channel, mlp=[64, 64, 128], group_all=False)
        # 后续层
        # 🌟 Attention 移到 SA2 之后（512 点），而非 SA1 之后（3072 点）
        # 复杂度从 O(3072^2) 降到 O(512^2)，约 36 倍加速

        self.fusion_conv = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        # sa2: 3072 -> 512 点
        self.sa2 = PointNetSetAbstraction(npoint=512, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256], group_all=False)

        # Attention 放在 SA2 之后，处理 512 点 x 256 维
        self.geo_attn = GeometryAwareAttentionBlock(in_channels=256, k=16)

        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256, mlp=[256, 512, 1024], group_all=True)
        
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

    def forward(self, xyz):
        # xyz shape: [B, Total_Points, 3]
        # Total_Points = U + C + I

        # --- 核心修改：根据 Dataset 中的拼接顺序进行切分 ---
        # 索引计算
        idx_u_end = self.num_points_uniform
        idx_c_end = self.num_points_uniform + self.num_points_curvature
        
        # 1. 切分坐标
        xyz_uniform = xyz[:, :idx_u_end, :]
        xyz_curvature = xyz[:, idx_u_end:idx_c_end, :] # 中间段是 Curvature
        xyz_importance = xyz[:, idx_c_end:, :]

        # 2. Embedding 和切分特征
        initial_features = self.input_embedder(xyz)
        features_uniform = initial_features[:, :idx_u_end, :]
        features_curvature = initial_features[:, idx_u_end:idx_c_end, :]
        features_importance = initial_features[:, idx_c_end:, :]

        # 3. 三路分别通过 SA1
        l1_xyz_u, l1_points_u = self.sa1_uniform(xyz_uniform, features_uniform)
        l1_xyz_c, l1_points_c = self.sa1_curvature(xyz_curvature, features_curvature)
        l1_xyz_i, l1_points_i = self.sa1_importance(xyz_importance, features_importance)

        # 4. 融合 (Concatenate)
        # 将三路特征拼在一起，组成一个新的点云特征集
        l1_xyz = torch.cat([l1_xyz_u, l1_xyz_c, l1_xyz_i], dim=1)       # [B, 768, 3]
        l1_points = torch.cat([l1_points_u, l1_points_c, l1_points_i], dim=1) # [B, 768, 128]
        
        # 5. 后续处理 (Fusion -> SA2 -> Attention -> SA3)
        # 先用简单的 Conv1d 融合三路特征
        l1_points_fused = self.fusion_conv(l1_points.transpose(1, 2)).transpose(1, 2)

        # SA2: 3072 点 -> 512 点（大幅下采样）
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points_fused)

        # Attention 在 512 点上做，而非 3072 点，速度快 ~36 倍
        l2_points = self.geo_attn(l2_xyz, l2_points)

        _, global_feature = self.sa3(l2_xyz, l2_points)
        
        x = global_feature.squeeze(1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar,l2_points

# ===============================================================
# Decoder & Embedder (保持不变)
# ===============================================================
class TriplaneDecoder(nn.Module):
    def __init__(self, latent_dim=128, plane_resolution=64, plane_features=8):
        super(TriplaneDecoder, self).__init__()
        self.start_res = 4
        self.target_res = plane_resolution
        assert self.target_res >= self.start_res and (self.target_res & (self.target_res - 1) == 0), \
            f"目标分辨率 (plane_resolution) 必须是4或更高的2的幂, 但得到的是 {self.target_res}"
        num_upsamples = int(math.log2(self.target_res / self.start_res))
        self.fc_start = nn.Linear(latent_dim, 256 * self.start_res * self.start_res)
        upsample_layers = []
        in_channels = 256
        for i in range(num_upsamples):
            out_channels = 16 if i == num_upsamples - 1 else in_channels // 2
            upsample_layers.extend([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ])
            in_channels = out_channels
        self.upsample_layers = nn.Sequential(*upsample_layers)
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
# 轻量级气动参数解码器 (SimpleAeroDecoder)
# 替代原有 ~530K 参数的 LiftCoefficientDecoder，仅 ~25K 参数
# 只从潜在向量 z 预测气动参数，避免在小数据集上过拟合
# ===============================================================
class SimpleAeroDecoder(nn.Module):
    def __init__(self, latent_dim=128, output_dim=1):
        super(SimpleAeroDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, output_dim)
        )

    def forward(self, z):
        return self.net(z)

# ===============================================================
# 核心模型：PointCloudVAE (物理感知修改版)
# ===============================================================
class PointCloudVAE(nn.Module):
    def __init__(self, latent_dim, plane_resolution, plane_features, num_fourier_freqs=6, 
                 num_points_uniform=4000, 
                 num_points_curvature=4000, # 新增
                 num_points_importance=4000):
        super(PointCloudVAE, self).__init__()
        
        # 传递参数给 Encoder
        self.encoder = Encoder(
            latent_dim, 
            num_fourier_freqs=num_fourier_freqs,
            num_points_uniform=num_points_uniform,
            num_points_curvature=num_points_curvature, # 新增
            num_points_importance=num_points_importance
        )
        
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

        # 3. Physics Decoder (Aero Branch - 简化版)
        self.aero_decoder = SimpleAeroDecoder(latent_dim=latent_dim, output_dim=1)  # 输出 CL

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, xyz, query_points=None):
        # --- Encoder 阶段 ---
        # xyz: [B, N, 3]
        mu, logvar,l2_points = self.encoder(xyz)
        
        # --- 潜在空间采样 ---
        # 用于几何重构的采样 z
        z_geo = self.reparameterize(mu, logvar)
        
        # --- 物理预测阶段 (核心修改) ---
        # 按照惯例，物理预测通常直接使用 mu，因为它是最稳定的特征表示
        # 这样可以减少采样带来的随机扰动
        aero_pred = self.aero_decoder(mu)
        
        # --- 几何重构阶段 (Triplane) ---
        triplanes = self.decoder(z_geo)
        
        sdf_pred = None
        if query_points is not None:
            # query_points: [B, M, 3]
            sdf_pred = self.query_sdf(triplanes, query_points)
            
        # 返回所有需要的结果：SDF预测、均值、方差、以及物理预测值
        return sdf_pred, mu, logvar, aero_pred

    def query_sdf(self, triplanes, query_points):
        plane_xy, plane_yz, plane_xz = triplanes
        batch_size, num_query_points, _ = query_points.shape
        grid_xy = query_points[:, :, [0, 1]].view(batch_size, num_query_points, 1, 2)
        features_xy = F.grid_sample(plane_xy, grid_xy, align_corners=True, padding_mode="border", mode='bilinear').squeeze(-1)
        grid_yz = query_points[:, :, [1, 2]].view(batch_size, num_query_points, 1, 2)
        features_yz = F.grid_sample(plane_yz, grid_yz, align_corners=True, padding_mode="border", mode='bilinear').squeeze(-1)
        grid_xz = query_points[:, :, [0, 2]].view(batch_size, num_query_points, 1, 2)
        features_xz = F.grid_sample(plane_xz, grid_xz, align_corners=True, padding_mode="border", mode='bilinear').squeeze(-1)
        features_xy = features_xy.transpose(1, 2)
        features_yz = features_yz.transpose(1, 2)
        features_xz = features_xz.transpose(1, 2)
        fourier_features = self.fourier_embedder(query_points)
        aggregated_features = torch.cat([features_xy, features_yz, features_xz, fourier_features], dim=-1)
        predicted_sdf = self.sdf_head(aggregated_features)
        return predicted_sdf