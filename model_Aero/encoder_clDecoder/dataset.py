import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import re

class SDFDataset(Dataset):
    def __init__(self, pc_root_dir, aero_root_dir, sdf_dir, 
                 num_points_uniform=4000, 
                 num_points_curvature=4000, 
                 num_points_importance=4000,
                 num_points_sdf=16384, 
                 surface_ratio=0.8,surface_threshold=0.02):
        """
        全内存预加载版 Dataset (包含 SDF):
        在初始化时读取所有硬盘文件。
        __getitem__ 只负责极速的内存级动态采样，不涉及任何硬盘读写。
        """
        self.num_points_uniform = num_points_uniform
        self.num_points_curvature = num_points_curvature
        self.num_points_importance = num_points_importance
        
        self.num_points_sdf = num_points_sdf
        self.surface_ratio = surface_ratio
        
        self.pc_root_dir = pc_root_dir
        self.aero_root_dir = aero_root_dir
        self.sdf_dir = sdf_dir

        # 1. 自动匹配所有文件路径
        self.file_pairs = self._make_dataset()
        if not self.file_pairs:
            raise RuntimeError(f"无法匹配数据！请检查路径：\nPC: {pc_root_dir}\nAero: {aero_root_dir}\nSDF: {sdf_dir}")
        
        print(f"✅ 找到匹配样本总数: {len(self.file_pairs)}")

        # 2. 预计算气动标签的标准化参数
        print("🔍 正在预计算气动标签的标准化参数...")
        all_cls = []
        for pair in self.file_pairs:
            cl, _ = self._parse_polar(pair['polar_path'])
            all_cls.append(cl)
            
        all_cls = np.array(all_cls)
        self.cl_mean = np.mean(all_cls)
        self.cl_std = np.std(all_cls)
        print(f"📊 CL 统计: Mean={self.cl_mean:.4f}, Std={self.cl_std:.4f}")

        # ==========================================================
        # 🌟 核心修改：在初始化阶段提取“所有”数据进内存缓存
        # ==========================================================
        self.memory_cache = [] 

        print("🚀 正在将所有点云和 SDF 数据载入内存 (可能需要一两分钟)...")
        for i, pair in enumerate(self.file_pairs):
            
            # --- A. 提前处理好点云 ---
            with np.load(pair['pc_path']) as pc_raw:
                pc_uni = pc_raw['uniform'][:self.num_points_uniform]
                pc_cur = pc_raw['curvature'][:self.num_points_curvature]
                pc_imp = pc_raw['importance'][:self.num_points_importance]
                pc_input = np.concatenate([pc_uni, pc_cur, pc_imp], axis=0).astype(np.float32)

            # --- B. 提前读出所有的 SDF 点池 (不采样，等 getitem 时再采样) ---
            with np.load(pair['sdf_path']) as sdf_raw:
                vol_points = sdf_raw['vol_points'].astype(np.float32)
                vol_sdf = sdf_raw['vol_sdf'].astype(np.float32)
                near_points = sdf_raw['near_points'].astype(np.float32)
                near_sdf = sdf_raw['near_sdf'].astype(np.float32)
                
                # 预处理 detail_points (合并 near 和 surface)
                if 'surface_points' in sdf_raw:
                    surface_points = sdf_raw['surface_points'].astype(np.float32)
                    surface_sdf = np.zeros((surface_points.shape[0], 1), dtype=np.float32)
                    if near_sdf.ndim == 1: near_sdf = near_sdf[:, None]
                    if vol_sdf.ndim == 1: vol_sdf = vol_sdf[:, None]
                    detail_points = np.concatenate([near_points, surface_points], axis=0)
                    detail_sdf = np.concatenate([near_sdf, surface_sdf], axis=0)
                else:
                    detail_points = near_points
                    detail_sdf = near_sdf
                    if detail_sdf.ndim == 1: detail_sdf = detail_sdf[:, None]
                    if vol_sdf.ndim == 1: vol_sdf = vol_sdf[:, None]

            # --- C. 提前标准化好气动标签 ---
            cl_raw, cd_raw = self._parse_polar(pair['polar_path'])
            cl_norm = (cl_raw - self.cl_mean) / (self.cl_std + 1e-8)
            aero_label = np.array([cl_norm, cd_raw], dtype=np.float32)

            # 打包成一个字典存进列表
            self.memory_cache.append({
                'pc_input': pc_input,                  # np.ndarray
                'vol_points': vol_points,              # np.ndarray
                'vol_sdf': vol_sdf,                    # np.ndarray
                'detail_points': detail_points,        # np.ndarray
                'detail_sdf': detail_sdf,              # np.ndarray
                'aero_label': aero_label,              # np.ndarray
                'raw_cl': cl_raw,                      # float
                'file_id': pair['file_id']             # str
            })
            
            if (i + 1) % 500 == 0:
                print(f"   --> 已加载 {i + 1} / {len(self.file_pairs)} 个数据...")

        print("✅ 数据预加载完毕！")

    def _make_dataset(self):
        pc_files = glob.glob(os.path.join(self.pc_root_dir, 'G58_*_pc.npz'))
        file_pairs = []
        
        for pc_path in pc_files:
            file_name = os.path.basename(pc_path)
            match = re.search(r'(G58_\d+)', file_name)
            if not match: continue
                
            file_id = match.group(1)
            sdf_path = os.path.join(self.sdf_dir, f"{file_id}.npz")
            polar_dir = os.path.join(self.aero_root_dir, file_id)
            polar_files = glob.glob(os.path.join(polar_dir, "*.polar"))
            
            polar_path = polar_files[0] if polar_files else None
            
            if os.path.exists(sdf_path) and polar_path and os.path.exists(polar_path):
                file_pairs.append({
                    'pc_path': pc_path,
                    'sdf_path': sdf_path,
                    'polar_path': polar_path,
                    'file_id': file_id
                })
                
        file_pairs.sort(key=lambda x: int(x['file_id'].split('_')[1]))
        return file_pairs

    def _parse_polar(self, polar_path):
        try:
            with open(polar_path, 'r') as f:
                lines = f.readlines()
                data_line = lines[-1].strip()
                if not data_line: data_line = lines[-2].strip()
                parts = data_line.split()
                return float(parts[4]), float(parts[9])
        except Exception:
            return 0.0, 0.0

    # ==========================================================
    # 🌟 极速版 __getitem__：只进行纯内存计算（动态采样）和转 Tensor
    # ==========================================================
    def __getitem__(self, index):
        data = self.memory_cache[index]

        num_surface = int(self.num_points_sdf * self.surface_ratio)
        num_volume = self.num_points_sdf - num_surface

        # 动态采样细节点 (内存操作，极快)
        if data['detail_points'].shape[0] > 0:
            idx_detail = np.random.choice(data['detail_points'].shape[0], num_surface, replace=True)
            sampled_detail_points = data['detail_points'][idx_detail]
            sampled_detail_values = data['detail_sdf'][idx_detail]
        else:
            sampled_detail_points = np.zeros((num_surface, 3), dtype=np.float32)
            sampled_detail_values = np.zeros((num_surface, 1), dtype=np.float32)

        # 动态采样体积点 (内存操作，极快)
        if data['vol_points'].shape[0] > 0:
            idx_vol = np.random.choice(data['vol_points'].shape[0], num_volume, replace=True)
            sampled_vol_points = data['vol_points'][idx_vol]
            sampled_vol_values = data['vol_sdf'][idx_vol]
        else:
            sampled_vol_points = np.zeros((num_volume, 3), dtype=np.float32)
            sampled_vol_values = np.zeros((num_volume, 1), dtype=np.float32)

        # 拼接本次采样的结果
        sdf_points_sampled = np.concatenate([sampled_detail_points, sampled_vol_points], axis=0)
        sdf_values_sampled = np.concatenate([sampled_detail_values, sampled_vol_values], axis=0)

        # 返回转化好的 Tensor
        return {
            'point_cloud': torch.from_numpy(data['pc_input']),
            'sdf_points': torch.from_numpy(sdf_points_sampled),
            'sdf_values': torch.from_numpy(sdf_values_sampled),
            'aero_label': torch.from_numpy(data['aero_label']),
            'raw_cl': torch.tensor([data['raw_cl']], dtype=torch.float32),
            'file_id': data['file_id']
        }

    def __len__(self):
        return len(self.memory_cache)