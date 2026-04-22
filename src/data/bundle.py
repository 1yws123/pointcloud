from dataclasses import dataclass
import random

from torch.utils.data import DataLoader, Subset

from .dataset2 import SDFDataset


@dataclass
class AeroDataBundle:
    full_dataset: object
    train_dataset: object
    test_dataset: object
    train_indices: list
    test_indices: list
    cl_mean: float
    cl_std: float


@dataclass
class AeroDataLoaders:
    train_loader: object
    train_eval_loader: object
    test_loader: object


class ModeSubset(Subset):
    """Subset 包装，每次 __getitem__ 前把底层 dataset 的 train flag 置为期望值。
    num_workers=0 时安全；num_workers>0 时每个 worker 有自己的 dataset 副本，也安全。
    """
    def __init__(self, dataset, indices, train):
        super().__init__(dataset, indices)
        self.train = bool(train)

    def __getitem__(self, idx):
        if hasattr(self.dataset, 'set_train_mode'):
            self.dataset.set_train_mode(self.train)
        return super().__getitem__(idx)
    
    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]


def build_group_split_indices(memory_cache, val_split=0.2, seed=42, group_key="file_id"):
    group_to_indices = {}
    for idx, data_item in enumerate(memory_cache):
        group_id = data_item[group_key]
        group_to_indices.setdefault(group_id, []).append(idx)

    unique_groups = list(group_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_groups)

    target_train_size = int((1.0 - val_split) * len(memory_cache))
    train_indices, test_indices = [], []
    for group_id in unique_groups:
        target = train_indices if len(train_indices) < target_train_size else test_indices
        target.extend(group_to_indices[group_id])
    return train_indices, test_indices


def build_aero_data_bundle(
    pc_root_dir,
    aero_root_dir,
    sdf_dir,
    val_split=0.2,
    seed=42,
    mode="aero",
    train_indices=None,
    test_indices=None,
    num_points_uniform=2048,
    num_points_curvature=2048,
    num_points_importance=4096,
    mirror_prob=0.5,
    jitter_std=1e-3,
    point_dropout=0.0,
):
    full_dataset = SDFDataset(
        pc_root_dir=pc_root_dir,
        aero_root_dir=aero_root_dir,
        sdf_dir=sdf_dir,
        mode=mode,
        num_points_uniform=num_points_uniform,
        num_points_curvature=num_points_curvature,
        num_points_importance=num_points_importance,
        mirror_prob=mirror_prob,
        jitter_std=jitter_std,
        point_dropout=point_dropout,
        train=False,
    )
    if train_indices is None or test_indices is None:
        train_indices, test_indices = build_group_split_indices(
            full_dataset.memory_cache,
            val_split=val_split,
            seed=seed,
            group_key="file_id",
        )

    train_dataset = ModeSubset(full_dataset, list(train_indices), train=True)
    test_dataset = ModeSubset(full_dataset, list(test_indices), train=False)

    return AeroDataBundle(
        full_dataset=full_dataset,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_indices=list(train_indices),
        test_indices=list(test_indices),
        cl_mean=full_dataset.cl_mean,
        cl_std=full_dataset.cl_std,
    )


def create_aero_dataloaders(bundle, batch_size, num_workers=0, pin_memory=True, shuffle_train=True):
    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    # 评估 train 时，使用 train=False 的视图（关掉增强、固定 seed）
    train_eval_view = ModeSubset(bundle.full_dataset, bundle.train_indices, train=False)
    train_eval_loader = DataLoader(
        train_eval_view,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        bundle.test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return AeroDataLoaders(
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        test_loader=test_loader,
    )
