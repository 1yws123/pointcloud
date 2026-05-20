import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from src.config import build_train_arg_parser, load_yaml_model_overrides, model_config_dict
from src.data import build_aero_data_bundle, create_aero_dataloaders
from src.engine import AeroTrainer
from src.models import PointNetPP_CL
from src.optim import WarmupCosineScheduler
from src.utils import ensure_output_dirs


cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_cfg=None):
    if model_cfg is None:
        model_cfg = model_config_dict()
    model = PointNetPP_CL(
        num_points_uniform=model_cfg["num_points_uniform"],
        num_points_curvature=model_cfg["num_points_curvature"],
        num_points_importance=model_cfg["num_points_importance"],
        out_dim=1,
    )
    return model


def main():
    from accelerate import Accelerator

    args = build_train_arg_parser().parse_args()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.train_gpu
    set_random_seed(args.seed)

    accelerator = Accelerator()
    output_dirs = ensure_output_dirs(args.save_dir) if accelerator.is_main_process else None

    if accelerator.is_main_process:
        print("[Setup] Loading dataset into memory...")

    mcfg = model_config_dict()
    mcfg.update(load_yaml_model_overrides(getattr(args, "config", None)))
    data_bundle = build_aero_data_bundle(
        pc_root_dir=args.pc_root,
        aero_root_dir=args.aero_root,
        sdf_dir=args.sdf_dir,
        val_split=args.val_split,
        seed=args.seed,
        mode="aero",
        num_points_uniform=mcfg["num_points_uniform"],
        num_points_curvature=mcfg["num_points_curvature"],
        num_points_importance=mcfg["num_points_importance"],
        mirror_prob=args.mirror_prob,
        jitter_std=args.jitter_std,   # jitter 在 trainer 里按 args.jitter_std 统一加
        point_dropout=args.point_dropout,
    )
    dataloaders = create_aero_dataloaders(
        data_bundle,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle_train=True,
    )

    if accelerator.is_main_process:
        print(f"[Setup] Total samples: {len(data_bundle.full_dataset)}")
        print(
            f"[Setup] Train samples: {len(data_bundle.train_dataset)} | "
            f"Test samples: {len(data_bundle.test_dataset)}"
        )
        print(f"[Setup] CL mean/std: {data_bundle.cl_mean:.6f} / {data_bundle.cl_std:.6f}")

    model = build_model(mcfg)
    optimizer = optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=args.epochs // 10, total_epochs=args.epochs)

    model, optimizer, dataloaders.train_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        dataloaders.train_loader,
        scheduler,
    )

    trainer = AeroTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        args=args,
        data_bundle=data_bundle,
        dataloaders=dataloaders,
        output_dirs=output_dirs,
        model_config=mcfg,
    )
    trainer.train()


if __name__ == "__main__":
    main()
