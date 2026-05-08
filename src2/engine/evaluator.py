import os

import torch

from src1.models import PointCloudVAE,PointNetPP_CL

from src1.data import build_aero_data_bundle, create_aero_dataloaders
from src1.utils import evaluate_aero_regression, format_metric_line, load_history_rows, parse_checkpoint_payload, save_json
from src1.visualization import plot_force_scatter, plot_loss_convergence


class AeroEvaluator:
    def __init__(self, args, output_dirs):
        self.args = args
        self.output_dirs = output_dirs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_checkpoint(self):
        print(f"[Eval] Loading checkpoint: {self.args.ckpt_path}")
        checkpoint = torch.load(self.args.ckpt_path, map_location=self.device)
        state_dict, metadata = parse_checkpoint_payload(checkpoint)

        model_cfg = metadata.get(
            "model_config",
            {
                "latent_dim": 128,
                "plane_resolution": 128,
                "plane_features": 32,
                "num_fourier_freqs": 8,
                "num_points_uniform": 4000,
                "num_points_curvature": 4000,
                "num_points_importance": 4000,
            },
        )

        seed = metadata.get("args", {}).get("seed", self.args.seed)
        val_split = metadata.get("args", {}).get("val_split", self.args.val_split)
        data_bundle = build_aero_data_bundle(
            pc_root_dir=self.args.pc_root,
            aero_root_dir=self.args.aero_root,
            sdf_dir=self.args.sdf_dir,
            val_split=val_split,
            seed=seed,
            mode="aero",
            train_indices=metadata.get("train_indices"),
            test_indices=metadata.get("val_indices"),
            num_points_uniform=model_cfg.get("num_points_uniform", 2048),
            num_points_curvature=model_cfg.get("num_points_curvature", 2048),
            num_points_importance=model_cfg.get("num_points_importance", 4096),
            mirror_prob=0.0,     # 评估时关掉所有随机
            jitter_std=0.0,
            point_dropout=0.0,
        )
        cl_mean = metadata.get("cl_mean", data_bundle.cl_mean)
        cl_std = metadata.get("cl_std", data_bundle.cl_std)
        dataloaders = create_aero_dataloaders(
            data_bundle,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            shuffle_train=False,
        )

        #model = PointCloudVAE(**model_cfg).to(self.device)
        model = PointNetPP_CL(
            num_points_uniform=model_cfg.get("num_points_uniform", 2048),
            num_points_curvature=model_cfg.get("num_points_curvature", 2048),
            num_points_importance=model_cfg.get("num_points_importance", 4096),
            out_dim=1,
        ).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()

        train_metrics, train_preds, train_gts, _ = evaluate_aero_regression(
            model, dataloaders.train_eval_loader, self.device, cl_mean, cl_std
        )
        test_metrics, test_preds, test_gts, _ = evaluate_aero_regression(
            model, dataloaders.test_loader, self.device, cl_mean, cl_std
        )

        summary = {
            "checkpoint": self.args.ckpt_path,
            "checkpoint_epoch": int(metadata.get("epoch", -1)),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_samples": len(data_bundle.train_dataset),
            "test_samples": len(data_bundle.test_dataset),
            "cl_mean": float(cl_mean),
            "cl_std": float(cl_std),
        }
        save_json(summary, os.path.join(self.output_dirs.metrics, "offline_eval_summary.json"))

        plot_force_scatter(
            train_gts=train_gts,
            train_preds=train_preds,
            train_metrics=train_metrics,
            val_gts=test_gts,
            val_preds=test_preds,
            val_metrics=test_metrics,
            save_path=os.path.join(self.output_dirs.plots, "offline_force_compare.png"),
            target_name="C_L",
            epoch=metadata.get("epoch"),
        )

        history_rows = load_history_rows(os.path.join(self.output_dirs.root, "train_aero_history.csv"))
        if history_rows:
            plot_loss_convergence(history_rows, os.path.join(self.output_dirs.plots, "loss_convergence_latest.png"))

        print("[Eval] Summary")
        print("  " + format_metric_line("Train", train_metrics))
        print("  " + format_metric_line("Test ", test_metrics))
        print(f"[Eval] Force comparison plot: {os.path.join(self.output_dirs.plots, 'offline_force_compare.png')}")
        if history_rows:
            print(f"[Eval] Loss curve plot: {os.path.join(self.output_dirs.plots, 'loss_convergence_latest.png')}")

        return summary
