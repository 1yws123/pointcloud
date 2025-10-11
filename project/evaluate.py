# evaluate.py

import torch
import numpy as np
import os
import pandas as pd
import mcubes
import trimesh
import argparse

from model9 import PointCloudVAE 

def evaluate(args):
    """
    主评估函数，加载模型并从单个点云重建网格。
    """
    # --- 1. 设备设定 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 2. 加载 checkpoint 并获取模型参数 ---
    print(f"正在加载 checkpoint: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            # 新格式 checkpoint
            config = checkpoint['config']
            LATENT_DIM = config['latent_dim']
            PLANE_RESOLUTION = config['plane_res']
            PLANE_FEATURES = config['plane_feat']
            NUM_FOURIER_FREQS = config.get('num_fourier_freqs', 8) # 兼容舊模型
        else:
            # 旧格式 checkpoint，使用默认参数
            LATENT_DIM = args.latent_dim
            PLANE_RESOLUTION = args.plane_res
            PLANE_FEATURES = args.plane_feat
            NUM_FOURIER_FREQS = args.num_fourier_freqs
            print("警告：使用旧格式 checkpoint，将使用命令行参数作为模型配置")
            
    except Exception as e:
        print(f"加载 checkpoint 失败！错误：{e}")
        return
    
    print("正在初始化 VAE 模型...")
    model = PointCloudVAE(
        latent_dim=LATENT_DIM,
        plane_resolution=PLANE_RESOLUTION,
        plane_features=PLANE_FEATURES,
        num_fourier_freqs=NUM_FOURIER_FREQS
    ).to(DEVICE)
    
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 旧格式 checkpoint
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"加载模型权重失败！错误：{e}")
        print("请确保模型参数与训练时一致。")
        return

    model.eval()
    print(f"模型权重 '{args.checkpoint}' 加载成功。")

    #加载并预处理输入点云 
    print(f"正在加载点云: {args.input_pc}")
    try:
        df = pd.read_csv(args.input_pc)
        point_cloud = df[['x', 'y', 'z']].values.astype(np.float32)
    except Exception as e:
        print(f"读取点云文件失败！错误：{e}")
        return

    # 随机采样固定数量的点
    replace_pc = point_cloud.shape[0] < args.num_points
    pc_indices = np.random.choice(point_cloud.shape[0], args.num_points, replace=replace_pc)
    point_cloud_sampled = point_cloud[pc_indices, :]

    # 数据归一化
    centroid = np.mean(point_cloud_sampled, axis=0)
    point_cloud_normalized = point_cloud_sampled - centroid
    furthest_distance = np.max(np.sqrt(np.sum(point_cloud_normalized**2, axis=1)))
    point_cloud_normalized = point_cloud_normalized / furthest_distance
    
    pc_tensor = torch.from_numpy(point_cloud_normalized).float().unsqueeze(0).to(DEVICE)

    print("正在通过模型生成 Triplane...")
    with torch.no_grad():
        mu, log_var = model.encoder(pc_tensor)
        z = mu  # 使用均值作为潜在向量
        triplanes = model.decoder(z)

    # 生成 SDF 网格 
    print(f"正在生成 {args.grid_res}x{args.grid_res}x{args.grid_res} 的 SDF 网格...")
    grid_coords = np.linspace(-1.0, 1.0, args.grid_res)
    x, y, z = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')
    query_points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    query_points_tensor = torch.from_numpy(query_points).float().unsqueeze(0).to(DEVICE)

    sdf_values = []
    # 动态调整批次大小
    query_batch_size = min(args.max_query_batch_size, args.grid_res ** 3)
    
    for i in range(0, query_points_tensor.shape[1], query_batch_size):
        batch_points = query_points_tensor[:, i:i+query_batch_size, :]
        with torch.no_grad():
            batch_sdf = model.query_sdf(triplanes, batch_points)
        sdf_values.append(batch_sdf.cpu().numpy())
    
    sdf_grid = np.concatenate(sdf_values, axis=1).reshape(args.grid_res, args.grid_res, args.grid_res)

    print("正在使用移动立方体算法提取网格...")
    try:
        vertices, faces = mcubes.marching_cubes(sdf_grid, 0.0)
    except Exception as e:
        print(f"移动立方体算法失败！错误：{e}")
        return

    if vertices.size == 0:
        print("警告：未提取到任何网格表面。SDF 场可能全为正或全为负。")
        return

    # 将顶点坐标从 [0, res-1] 转换回 [-1, 1] 空间
    vertices = vertices / (args.grid_res - 1.0) * 2.0 - 1.0
    # 使用之前保存的归一化参数进行逆操作
    vertices = vertices * furthest_distance + centroid

    # 保存为 .obj 文件 
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(args.output_mesh)
    print(f"\n✅ 重建成功！网格已保存至: {args.output_mesh}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从单个点云重建3D网格')
    parser.add_argument('--checkpoint', type=str, default=r"./checkpoint10\vae_epoch_2000.pth",  help='训练好的模型权重 (.pth) 文件路径')
    parser.add_argument('--input_pc', type=str, default=r"F:\Code\base_dataset\11\pointcloud\000011_pc.csv",  help='输入的点云 (.csv) 文件路径')
    parser.add_argument('--output_mesh', type=str, default='./file1/reconstruction11.obj', help='输出的网格 (.obj) 文件路径')
    parser.add_argument('--num_points', type=int, default=8192, help='输入点云的采样点数，应与训练时一致')
    parser.add_argument('--grid_res', type=int, default=128, help='SDF 网格的分辨率')
    parser.add_argument('--max_query_batch_size', type=int, default=8192, help='查询SDF时的最大批次大小')
    parser.add_argument('--latent_dim', type=int, default=128, help='潜在空间维度（旧格式checkpoint）')
    parser.add_argument('--plane_res', type=int, default=16, help='特征平面的分辨率（旧格式checkpoint）')
    parser.add_argument('--plane_feat', type=int, default=4, help='特征平面的通道数（旧格式checkpoint）')
    parser.add_argument('--num_fourier_freqs', type=int, default=8, help='傅里葉特徵的頻率數量')
    args = parser.parse_args()
    
    evaluate(args)