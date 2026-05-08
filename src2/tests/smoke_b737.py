"""Smoke test: 用 B737/ 下的 10 个样本验证整条管线。

运行（在 encoder_4594 根目录）:
    python -m src.tests.smoke_b737

会做：
    1. SDFDataset 能否正确读入老格式 pc1 (无 *_pool 字段 → 自动回退)
    2. group split + DataLoader 能否跑通
    3. 训练/评估两种 train flag 下 __getitem__ 返回的 shape
    4. 物理无损增强（镜像 / jitter）是否生效
    5. PointCloudVAE forward（CPU，减小点数以保证可接受速度）
    6. 一步反向传播 + AdamW 更新
"""

import os
import sys
import time

import numpy as np
import torch

# 允许 `python src/tests/smoke_b737.py` 直接跑
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.dataset import SDFDataset
from src.data.bundle import build_aero_data_bundle, create_aero_dataloaders
from src.models import PointCloudVAE


DATA_ROOT = os.path.join(ROOT, "src", "B737")
PC_ROOT = os.path.join(DATA_ROOT, "pc")
AERO_ROOT = os.path.join(DATA_ROOT, "aero")
SDF_ROOT = os.path.join(DATA_ROOT, "sdf")


def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    assert os.path.isdir(PC_ROOT), f"找不到 {PC_ROOT}"
    assert os.path.isdir(AERO_ROOT), f"找不到 {AERO_ROOT}"
    assert os.path.isdir(SDF_ROOT), f"找不到 {SDF_ROOT}"

    # CPU 上跑 PointNet++ 很慢，缩小点数以保证单次前向 <1 分钟
    N_U, N_C, N_I = 512, 512, 1024
    TOTAL = N_U + N_C + N_I

    # ---------------- 1. Dataset 加载（aero 模式，吃老格式 npz） ----------------
    section("1. SDFDataset load (aero mode, legacy pc npz)")
    t0 = time.time()
    ds = SDFDataset(
        pc_root_dir=PC_ROOT,
        aero_root_dir=AERO_ROOT,
        sdf_dir=SDF_ROOT,
        num_points_uniform=N_U,
        num_points_curvature=N_C,
        num_points_importance=N_I,
        mode="aero",
        train=False,
        mirror_prob=0.0,
        jitter_std=0.0,
    )
    print(f"[OK] 加载 {len(ds)} 个样本，用时 {time.time() - t0:.2f}s")
    assert len(ds) == 10, f"期望 10 个样本，实际 {len(ds)}"

    # ---------------- 2. __getitem__ 基本形状 ----------------
    section("2. __getitem__ shape check (train=False)")
    item = ds[0]
    assert item["point_cloud"].shape == (TOTAL, 3), f"pc shape {item['point_cloud'].shape}"
    assert item["aero_label"].shape == (2,)
    assert item["raw_cl"].shape == (1,)
    print(f"  point_cloud: {tuple(item['point_cloud'].shape)}  dtype={item['point_cloud'].dtype}")
    print(f"  aero_label : {tuple(item['aero_label'].shape)}   {item['aero_label'].tolist()}")
    print(f"  raw_cl     : {item['raw_cl'].item():.6f}")
    print(f"  file_id    : {item['file_id']}")

    # 推理确定性：同一 index 两次读应当完全相同（eval 使用 seed=index 的 rng）
    item_again = ds[0]
    assert torch.allclose(item["point_cloud"], item_again["point_cloud"]), \
        "eval 模式下同 index 两次读点云不一致"
    print("  [OK] eval 模式确定性：同 index 两次采样完全一致")

    # ---------------- 3. 直接测试 _augment 以避免 pool 随机性干扰 ----------------
    section("3. Augmentation sanity (direct _augment call)")
    base = item["point_cloud"].numpy().copy()

    # (a) 镜像：强制 mirror_prob=1.0, jitter=0, dropout=0
    ds.mirror_prob = 1.0
    ds.jitter_std = 0.0
    ds.point_dropout = 0.0
    pts_mirror = ds._augment(base.copy(), np.random.default_rng(seed=1))
    y_before_sorted = np.sort(base[:, 1])
    y_mirror_sorted = np.sort(-pts_mirror[:, 1])
    assert np.allclose(y_before_sorted, y_mirror_sorted, atol=1e-5), "镜像未生效"
    print("  [OK] mirror_prob=1.0 → y 精确翻号")

    # (b) jitter：关掉镜像，只开 jitter
    ds.mirror_prob = 0.0
    ds.jitter_std = 1e-3
    pts_jit = ds._augment(base.copy(), np.random.default_rng(seed=2))
    diff = np.abs(pts_jit - base[np.argsort(np.arange(len(base)))])  # jitter 后有 permutation
    # 排序后比较总分布的相近性
    d_sorted = np.abs(np.sort(pts_jit.flatten()) - np.sort(base.flatten())).max()
    assert 1e-6 < d_sorted < 1e-1, f"jitter 幅度异常: {d_sorted}"
    print(f"  [OK] jitter_std=1e-3 → 排序后最大分布差 {d_sorted:.4e}")

    # (c) permutation：关掉镜像/jitter，点集应当一致但顺序不同
    ds.mirror_prob = 0.0
    ds.jitter_std = 0.0
    pts_perm = ds._augment(base.copy(), np.random.default_rng(seed=3))
    assert not np.array_equal(pts_perm, base), "permutation 未生效（顺序相同）"
    assert np.array_equal(np.sort(pts_perm.flatten()), np.sort(base.flatten())), "permutation 改变了点集"
    print("  [OK] permutation → 顺序改变但点集相同")

    # (d) 动态采样（每次 __getitem__ 从 pool 抽不同子集）
    ds.set_train_mode(True)
    a = ds[0]["point_cloud"].numpy()
    b = ds[0]["point_cloud"].numpy()
    sa = np.sort(a.flatten())
    sb = np.sort(b.flatten())
    same_set = np.array_equal(sa, sb)
    print(f"  pool 动态采样两次点集{'相同（老格式 pool 只有 4000 点，采样数 ≤ pool）' if same_set else '不同 ✅'}")

    # ---------------- 4. bundle + DataLoader ----------------
    section("4. Bundle + DataLoader")
    ds.set_train_mode(False)
    bundle = build_aero_data_bundle(
        pc_root_dir=PC_ROOT,
        aero_root_dir=AERO_ROOT,
        sdf_dir=SDF_ROOT,
        val_split=0.3,
        seed=42,
        mode="aero",
        num_points_uniform=N_U,
        num_points_curvature=N_C,
        num_points_importance=N_I,
        mirror_prob=0.5,
        jitter_std=1e-3,
    )
    print(f"  train samples: {len(bundle.train_dataset)} | test samples: {len(bundle.test_dataset)}")
    assert len(bundle.train_dataset) + len(bundle.test_dataset) == 10

    loaders = create_aero_dataloaders(bundle, batch_size=4, num_workers=0, pin_memory=False)
    batch = next(iter(loaders.train_loader))
    assert batch["point_cloud"].shape == (4, TOTAL, 3)
    print(f"  train batch pc shape: {tuple(batch['point_cloud'].shape)}")
    print(f"  test loader len={len(loaders.test_loader)}, "
          f"train_eval loader len={len(loaders.train_eval_loader)}")

    # ---------------- 5. PointCloudVAE forward (CPU) ----------------
    section("5. PointCloudVAE forward (CPU, slow)")
    model = PointCloudVAE(
        latent_dim=64,
        plane_resolution=32,
        plane_features=8,
        num_fourier_freqs=4,
        num_points_uniform=N_U,
        num_points_curvature=N_C,
        num_points_importance=N_I,
        dropout=0.1,
    )
    model.train()
    print(f"  model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    t0 = time.time()
    _, mu, logvar, cl_pred = model(
        batch["point_cloud"], aux_vec=None, query_points=None, aero_only=True
    )
    dt = time.time() - t0
    print(f"  forward: {dt:.2f}s  mu={tuple(mu.shape)}  logvar={tuple(logvar.shape)}  cl_pred={tuple(cl_pred.shape)}")
    assert cl_pred.shape == (4, 1)
    assert mu.shape == (4, 64)

    # ---------------- 6. KL + MSE backward + 一步 AdamW ----------------
    section("6. One optimizer step (KL + MSE + MAE)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    cl_gt = batch["aero_label"][:, 0].float().view(-1, 1)
    optimizer.zero_grad()
    _, mu, logvar, cl_pred = model(batch["point_cloud"], aux_vec=None, query_points=None, aero_only=True)
    loss_mse = torch.nn.functional.mse_loss(cl_pred, cl_gt)
    loss_mae = torch.nn.functional.l1_loss(cl_pred, cl_gt)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    loss = loss_mse + loss_mae + 1e-4 * kl
    loss.backward()
    optimizer.step()
    print(f"  loss_mse={loss_mse.item():.4f}  loss_mae={loss_mae.item():.4f}  "
          f"kl={kl.item():.4f}  total={loss.item():.4f}")
    print("  [OK] 反向传播 + AdamW 更新成功")

    # ---------------- 7. eval 模式下 aero_decoder 走 mu ----------------
    section("7. eval mode (aero_decoder uses mu, no reparameterize noise)")
    model.eval()
    # FPS 在 CPU 回退路径下起点是 torch.randint，会带来轻微抖动；
    # 为验证 reparameterize 被关闭，先固定 torch seed 再对比两次 forward。
    with torch.no_grad():
        torch.manual_seed(0)
        _, _, _, p1 = model(batch["point_cloud"], aux_vec=None, query_points=None, aero_only=True)
        torch.manual_seed(0)
        _, _, _, p2 = model(batch["point_cloud"], aux_vec=None, query_points=None, aero_only=True)
    assert torch.allclose(p1, p2), "即便固定 seed，eval 仍不一致，说明还有其他随机源"
    print(f"  [OK] 固定 seed 下两次 forward 完全一致  p1[0]={p1[0, 0].item():+.4f}  p2[0]={p2[0, 0].item():+.4f}")

    # 不固定 seed 时看抖动量级（CPU FPS 起点随机 → 会有小幅差异，GPU CUDA FPS 是 deterministic）
    with torch.no_grad():
        _, _, _, q1 = model(batch["point_cloud"], aux_vec=None, query_points=None, aero_only=True)
        _, _, _, q2 = model(batch["point_cloud"], aux_vec=None, query_points=None, aero_only=True)
    drift = (q1 - q2).abs().max().item()
    print(f"  eval 模式下不固定 seed 两次 forward 最大差异: {drift:.4e}  "
          f"(CPU 回退 FPS 起点随机所致；GPU 上为 0)")

    section("ALL PASS")


if __name__ == "__main__":
    main()
