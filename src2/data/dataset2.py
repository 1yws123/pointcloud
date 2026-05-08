import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import re


class SDFDataset(Dataset):
    """
    全内存预加载 Dataset。关键升级：
      1. 若 pc1/*.npz 里有 *_pool 字段，则训练时**动态**从大 pool 里随机子采样
         （Hunyuan3D-style），打破 encoder 对固定点云的记忆。
      2. 训练时启用物理无损增强：点顺序 permutation、左右镜像(y→-y)、
         小幅 jitter、point dropout。
      3. 推理时（train=False）采样**确定**，关掉所有随机增强，保证复现。
    """

    def __init__(self, pc_root_dir, aero_root_dir, sdf_dir,
                 num_points_uniform=2048,
                 num_points_curvature=2048,
                 num_points_importance=4096,
                 num_points_sdf=16384,
                 surface_ratio=0.8, surface_threshold=0.02,
                 mode='aero',
                 train=False,
                 mirror_prob=0.5,
                 jitter_std=1e-3,
                 point_dropout=0.0):
        self.mode = mode
        self.train = train
        self.mirror_prob = mirror_prob
        self.jitter_std = jitter_std
        self.point_dropout = point_dropout

        self.num_points_uniform = num_points_uniform
        self.num_points_curvature = num_points_curvature
        self.num_points_importance = num_points_importance

        self.num_points_sdf = num_points_sdf
        self.surface_ratio = surface_ratio

        self.pc_root_dir = pc_root_dir
        self.aero_root_dir = aero_root_dir
        self.sdf_dir = sdf_dir

        self.file_pairs = self._make_dataset()
        if not self.file_pairs:
            raise RuntimeError(f"无法匹配数据！\nPC: {pc_root_dir}\nAero: {aero_root_dir}\nSDF: {sdf_dir}")
        print(f"✅ 找到匹配样本总数: {len(self.file_pairs)}")

        print("🔍 预计算气动标签标准化参数...")
        all_cls, all_cds = [], []
        for pair in self.file_pairs:
            cl, cd = self._parse_polar(pair['polar_path'])
            all_cls.append(cl)
            all_cds.append(cd)
        all_cls = np.array(all_cls)
        all_cds = np.array(all_cds)
        self.cl_mean, self.cl_std = float(np.mean(all_cls)), float(np.std(all_cls))
        self.cd_mean, self.cd_std = float(np.mean(all_cds)), float(np.std(all_cds))
        print(f"📊 CL: mean={self.cl_mean:.4f} std={self.cl_std:.4f}")
        print(f"📊 CD: mean={self.cd_mean:.4f} std={self.cd_std:.4f}")

        self.memory_cache = []
        #all_shifts, all_scales = [], []

        print("🚀 载入所有样本到内存 (含大 pool)...")
        for i, pair in enumerate(self.file_pairs):
            item = self._load_one(pair)
            self.memory_cache.append(item)
            #all_shifts.append(item['shift'].reshape(1, 3))
            #all_scales.append(np.array(item['scale']).reshape(1, 1))
            if (i + 1) % 100 == 0:
                print(f"   --> {i + 1}/{len(self.file_pairs)}")

        #all_shifts = np.concatenate(all_shifts, axis=0)
        #all_scales = np.concatenate(all_scales, axis=0)
        #self.shift_mean = all_shifts.mean(axis=0)
        #self.shift_std = all_shifts.std(axis=0) + 1e-8
        #self.scale_mean = float(all_scales.mean())
        #self.scale_std = float(all_scales.std() + 1e-8)
        #print(f"Shift mean/std: {self.shift_mean} / {self.shift_std}")
        #print(f"Scale mean/std: {self.scale_mean} / {self.scale_std}")
        print("✅ 数据预加载完毕。")

    # -----------------------------------------------------------------
    def _load_one(self, pair):
        with np.load(pair['pc_path']) as pc_raw:
            #keys = set(pc_raw.files)
            #has_pool = 'uniform_pool' in keys

            #if has_pool:
        
            uniform_pool = pc_raw['uniform'].astype(np.float32)
            curvature_pool = pc_raw['curvature'].astype(np.float32)
            importance_pool = pc_raw['importance'].astype(np.float32)

            #normalization_shift = pc_raw['shift'].astype(np.float32)
            #normalization_scale = pc_raw['scale'].astype(np.float32)

        # SDF 数据（aero_only 时跳过，省内存）
        if self.mode == 'aero':
            vol_points = np.zeros((0, 3), dtype=np.float32)
            vol_sdf = np.zeros((0, 1), dtype=np.float32)
            detail_points = np.zeros((0, 3), dtype=np.float32)
            detail_sdf = np.zeros((0, 1), dtype=np.float32)
        else:
            with np.load(pair['sdf_path']) as sdf_raw:
                vol_points = sdf_raw['vol_points'].astype(np.float32)
                vol_sdf = sdf_raw['vol_sdf'].astype(np.float32)
                near_points = sdf_raw['near_points'].astype(np.float32)
                near_sdf = sdf_raw['near_sdf'].astype(np.float32)
                if near_sdf.ndim == 1: near_sdf = near_sdf[:, None]
                if vol_sdf.ndim == 1: vol_sdf = vol_sdf[:, None]
                if 'surface_points' in sdf_raw:
                    surface_points = sdf_raw['surface_points'].astype(np.float32)
                    surface_sdf = np.zeros((surface_points.shape[0], 1), dtype=np.float32)
                    detail_points = np.concatenate([near_points, surface_points], axis=0)
                    detail_sdf = np.concatenate([near_sdf, surface_sdf], axis=0)
                else:
                    detail_points, detail_sdf = near_points, near_sdf

        cl_raw, cd_raw = self._parse_polar(pair['polar_path'])
        cl_norm = (cl_raw - self.cl_mean) / (self.cl_std + 1e-8)
        cd_norm = (cd_raw - self.cd_mean) / (self.cd_std + 1e-8)
        aero_label = np.array([cl_norm, cd_norm], dtype=np.float32)

        return {
            'uniform_pool': uniform_pool,
            'curvature_pool': curvature_pool,
            'importance_pool': importance_pool,
            'vol_points': vol_points,
            'vol_sdf': vol_sdf,
            'detail_points': detail_points,
            'detail_sdf': detail_sdf,
            'aero_label': aero_label,
            'raw_cl': cl_raw,
            'raw_cd': cd_raw,
            'file_id': pair['file_id'],
        }

    def _make_dataset(self):
        pc_files = glob.glob(os.path.join(self.pc_root_dir, 'G58_*_pc.npz'))
        file_pairs = []
        for pc_path in pc_files:
            file_name = os.path.basename(pc_path)
            match = re.search(r'(G58_\d+)', file_name)
            if not match:
                continue
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
                if not data_line:
                    data_line = lines[-2].strip()
                parts = data_line.split()
                return float(parts[4]), float(parts[9])
        except Exception:
            return 0.0, 0.0

    # -----------------------------------------------------------------
    # 动态采样 + 物理无损增强
    # -----------------------------------------------------------------
    def _sample_from_pool(self, pool, n, rng):
        if pool.shape[0] == 0:
            return np.zeros((n, 3), dtype=np.float32)
        replace = pool.shape[0] < n
        idx = rng.choice(pool.shape[0], n, replace=replace)
        return pool[idx]

    def _augment(self, pts, rng):
        """物理无损增强（不改变 CL/CD）：
           1. 小幅 jitter（std 远小于特征尺度）
           2. 左右镜像 (y→-y)：机体 XZ 对称面，β=0 直飞 CL/CD 不变
           3. 点顺序 permutation
           4. point dropout（用原点替补，pad 回原长度）
        """
        if self.jitter_std > 0:
            pts = pts + rng.normal(0, self.jitter_std, size=pts.shape).astype(np.float32)
        if self.mirror_prob > 0 and rng.random() < self.mirror_prob:
            pts[:, 1] = -pts[:, 1]
        if self.point_dropout > 0:
            keep = rng.random(size=pts.shape[0]) > self.point_dropout
            if keep.sum() > 0:
                kept = pts[keep]
                # pad 回原长度：随机复制已保留的点
                n_pad = pts.shape[0] - kept.shape[0]
                if n_pad > 0:
                    idx_pad = rng.choice(kept.shape[0], n_pad, replace=True)
                    pts = np.concatenate([kept, kept[idx_pad]], axis=0)
                else:
                    pts = kept
        #perm = rng.permutation(pts.shape[0])
        #pts = pts[perm]
        return pts

    def __getitem__(self, index):
        data = self.memory_cache[index]

        # 确定 RNG：训练时用全局随机，eval 时用固定种子以保证复现
        if self.train:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed=index)

        # 从大 pool 里动态采样
        pc_u = self._sample_from_pool(data['uniform_pool'], self.num_points_uniform, rng)
        pc_c = self._sample_from_pool(data['curvature_pool'], self.num_points_curvature, rng)
        pc_i = self._sample_from_pool(data['importance_pool'], self.num_points_importance, rng)
        pc_input = np.concatenate([pc_u, pc_c, pc_i], axis=0).astype(np.float32)

        #if self.train:
        #pc_input = self._augment(pc_input, rng)
        if self.train:
            pc_input = self._augment(pc_input, rng)
            # 分块内部打乱（不跨 uniform/curvature/importance）
            u, c, i = self.num_points_uniform, self.num_points_curvature, self.num_points_importance
            idx_u = rng.permutation(u)
            idx_c = rng.permutation(c)
            idx_i = rng.permutation(i)
            pc_input = np.concatenate(
                [pc_input[:u][idx_u],
                 pc_input[u:u+c][idx_c],
                 pc_input[u+c:u+c+i][idx_i]],
                axis=0
            )

        # 标准化后的 shift/scale
        #raw_shift = np.asarray(data['shift'], dtype=np.float32).reshape(3)
        #raw_scale = np.asarray(data['scale'], dtype=np.float32).reshape(1)
        #shift_norm = (raw_shift - self.shift_mean) / (self.shift_std + 1e-8)
        #scale_norm = (raw_scale - self.scale_mean) / (self.scale_std + 1e-8)

        # SDF 动态采样（aero_only 模式返回空）
        num_surface = int(self.num_points_sdf * self.surface_ratio)
        num_volume = self.num_points_sdf - num_surface
        if self.mode == 'aero':
            sdf_points_sampled = np.zeros((0, 3), dtype=np.float32)
            sdf_values_sampled = np.zeros((0, 1), dtype=np.float32)
        else:
            if data['detail_points'].shape[0] > 0:
                ids = rng.choice(data['detail_points'].shape[0], num_surface, replace=True)
                det_pts = data['detail_points'][ids]
                det_val = data['detail_sdf'][ids]
            else:
                det_pts = np.zeros((num_surface, 3), dtype=np.float32)
                det_val = np.zeros((num_surface, 1), dtype=np.float32)
            if data['vol_points'].shape[0] > 0:
                ids = rng.choice(data['vol_points'].shape[0], num_volume, replace=True)
                vol_pts = data['vol_points'][ids]
                vol_val = data['vol_sdf'][ids]
            else:
                vol_pts = np.zeros((num_volume, 3), dtype=np.float32)
                vol_val = np.zeros((num_volume, 1), dtype=np.float32)
            sdf_points_sampled = np.concatenate([det_pts, vol_pts], axis=0)
            sdf_values_sampled = np.concatenate([det_val, vol_val], axis=0)

        return {
            'point_cloud': torch.from_numpy(pc_input),
            'sdf_points': torch.from_numpy(sdf_points_sampled),
            'sdf_values': torch.from_numpy(sdf_values_sampled),
            'aero_label': torch.from_numpy(data['aero_label']),
            'raw_cl': torch.tensor([data['raw_cl']], dtype=torch.float32),
            'raw_cd': torch.tensor([data['raw_cd']], dtype=torch.float32),
            'file_id': data['file_id']
        }

    def __len__(self):
        return len(self.memory_cache)

    # 允许训练/评估切换同一个 dataset 对象的增强开关
    def set_train_mode(self, flag: bool):
        self.train = bool(flag)
