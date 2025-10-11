import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import re

class SDFDataset(Dataset):
    def __init__(self, pc_root_dir, sdf_dir, num_points_pc=2048, num_points_sdf=8192, surface_ratio=0.8, surface_threshold=0.02):
        self.num_points_pc = num_points_pc
        self.num_points_sdf = num_points_sdf
        self.surface_ratio = surface_ratio         # 表面密集采样比例
        self.surface_threshold = surface_threshold # 表面判定阈值

        self.file_pairs = self._make_dataset(pc_root_dir, sdf_dir)

        if not self.file_pairs:
            raise RuntimeError("無法配對任何點雲和SDF文件。請檢查路徑和文件名格式。")

        print(f"成功配對 {len(self.file_pairs)} 個樣本。")

    def _make_dataset(self, pc_root_dir, sdf_dir):
        print("正在掃描並配對點雲和SDF文件...")
        pc_map = {}
        pc_files = glob.glob(os.path.join(pc_root_dir, '*', 'pointcloud', '*.csv'))
        for pc_path in pc_files:
            match = re.search(r'(\d{6})_pc\.csv', os.path.basename(pc_path))
            if match:
                file_id = match.group(1)
                pc_map[file_id] = pc_path

        print(f"找到 {len(pc_map)} 個點雲文件。")

        file_pairs = []
        sdf_files = glob.glob(os.path.join(sdf_dir, '*.npz'))
        for sdf_path in sdf_files:
            match = re.search(r'(\d{6})_mesh\.npz', os.path.basename(sdf_path))
            if match:
                file_id = match.group(1)
                if file_id in pc_map:
                    pc_path = pc_map[file_id]
                    file_pairs.append((pc_path, sdf_path))

        return file_pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        pc_path, sdf_path = self.file_pairs[idx]

        # --- 1. 加載和處理 SDF 數據 (來自 .npz 文件) ---
        sdf_data = np.load(sdf_path)
        pos_samples = sdf_data['pos']
        neg_samples = sdf_data['neg']
        all_sdf_samples = np.concatenate([pos_samples, neg_samples], axis=0)
        sdf_points = all_sdf_samples[:, :3]
        sdf_values = all_sdf_samples[:, 3:]

        # 表面采样加密
        surface_mask = np.abs(sdf_values[:, 0]) < self.surface_threshold
        surface_points = sdf_points[surface_mask]
        surface_values = sdf_values[surface_mask]
        non_surface_points = sdf_points[~surface_mask]
        non_surface_values = sdf_values[~surface_mask]

        # 计算两类采样点数
        num_surface = int(self.num_points_sdf * self.surface_ratio)
        num_non_surface = self.num_points_sdf - num_surface

        # 防止样本不足，允许重复采样
        idx_surface = np.random.choice(surface_points.shape[0], num_surface, replace=surface_points.shape[0]<num_surface)
        idx_non_surface = np.random.choice(non_surface_points.shape[0], num_non_surface, replace=non_surface_points.shape[0]<num_non_surface)

        sampled_points = np.concatenate([surface_points[idx_surface], non_surface_points[idx_non_surface]], axis=0)
        sampled_values = np.concatenate([surface_values[idx_surface], non_surface_values[idx_non_surface]], axis=0)

        # --- 2. 点云采样 ---
        point_cloud = np.loadtxt(pc_path, delimiter=',', dtype=np.float32, skiprows=1)
        if point_cloud.shape[1] > 3:
            point_cloud = point_cloud[:, :3]

        replace_pc = point_cloud.shape[0] < self.num_points_pc
        pc_indices = np.random.choice(point_cloud.shape[0], self.num_points_pc, replace=replace_pc)
        point_cloud_sampled = point_cloud[pc_indices, :]

        jitter = np.clip(0.01 * np.random.randn(*point_cloud_sampled.shape), -0.02, 0.02)
        point_cloud_sampled += jitter

        # --- 3. 轉換為 PyTorch Tensors ---
        return {
            'point_cloud': torch.from_numpy(point_cloud_sampled).float(),
            'sdf_points': torch.from_numpy(sampled_points).float(),
            'sdf_values': torch.from_numpy(sampled_values).float()
        }

# ===============================================================
# 測試
'''
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='测试数据集')
    parser.add_argument('--pc_root_dir', type=str, default=r"F:/Code/base_dataset", required=True, help='点云根目录路径')
    parser.add_argument('--sdf_dir', type=str, default=r"F:/Code/pointnet2/DeepSDF/Data2/SdfSamples/DataSource/heart", required=True, help='SDF数据目录路径')
    parser.add_argument('--num_points_pc', type=int, default=2048, help='点云采样点数')
    parser.add_argument('--num_points_sdf', type=int, default=4096, help='SDF采样点数')
    parser.add_argument('--surface_ratio', type=float, default=0.6, help='表面采样比例')
    parser.add_argument('--surface_threshold', type=float, default=0.02, help='表面点SDF阈值')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    print("正在初始化數據集...")
    try:
        dataset = SDFDataset(
            pc_root_dir=args.pc_root_dir,
            sdf_dir=args.sdf_dir,
            num_points_pc=args.num_points_pc,
            num_points_sdf=args.num_points_sdf,
            surface_ratio=args.surface_ratio,
            surface_threshold=args.surface_threshold
        )
        if len(dataset) > 0:
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            print("\n正在測試 DataLoader...")
            first_batch = next(iter(dataloader))

            pc = first_batch['point_cloud']
            sdf_p = first_batch['sdf_points']
            sdf_v = first_batch['sdf_values']

            print("成功從 DataLoader 取出一個批次！")
            print(f"點雲維度: {pc.shape}")
            print(f"SDF 採樣點維度: {sdf_p.shape}")
            print(f"SDF 真實值維度: {sdf_v.shape}")

            print("\n✅ 測試通過！您的數據管道已根據您的文件結構準備就緒。")
        else:
            print("\n⚠️ 警告：數據集為空，沒有配對到任何文件。請仔細檢查您的路徑和文件名格式。")
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        print("請仔細檢查您的文件路徑和文件內容是否正確。")
'''