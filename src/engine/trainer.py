import os
import time

import torch
import torch.nn.functional as F

from src.utils import (
    HISTORY_FIELDS,
    append_history_row,
    attach_metrics,
    build_checkpoint_payload,
    create_history_row,
    evaluate_aero_regression,
    format_metric_line,
    save_json,
)
from src.visualization import plot_force_scatter, plot_loss_convergence


class AeroTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        accelerator,
        args,
        data_bundle,
        dataloaders,
        output_dirs,
        model_config,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.args = args
        self.data_bundle = data_bundle
        self.dataloaders = dataloaders
        self.output_dirs = output_dirs
        self.model_config = model_config

        self.history_rows = []
        self.best_val_mae = float("inf")
        self.history_csv_path = os.path.join(output_dirs.root, "train_aero_history.csv")
        self.latest_json_path = os.path.join(output_dirs.root, "latest_metrics.json")
        self.best_json_path = os.path.join(output_dirs.root, "best_metrics.json")
        self.best_ckpt_path = os.path.join(output_dirs.root, "best_cl_model.pth")
        self.latest_ckpt_path = os.path.join(output_dirs.root, "latest_cl_model.pth")

        # 正则与早停超参（可从 args 读）
        self.kl_beta = float(getattr(args, "kl_beta", 1e-4))
        self.jitter_std = float(getattr(args, "jitter_std", 1e-3))
        self.early_stop_patience = int(getattr(args, "early_stop_patience", 0))
        self.patience_counter = 0
        self.should_stop = False

    def train(self):
        if self.accelerator.is_main_process:
            print("[Train] Start training physics branch...")

        for epoch in range(1, self.args.epochs + 1):
            avg_train_loss, current_lr, train_time = self._train_one_epoch()
            history_row = create_history_row(epoch=epoch, lr=current_lr, train_loss=avg_train_loss)

            should_eval = epoch % self.args.eval_interval == 0 or epoch == 1 or epoch == self.args.epochs
            should_save_ckpt = epoch % self.args.checkpoint_interval == 0 or epoch == self.args.epochs

            if self.accelerator.is_main_process and should_eval:
                self._evaluate_epoch(epoch, history_row, avg_train_loss, current_lr, train_time, should_save_ckpt)
            else:
                self.history_rows.append(history_row)
                if self.accelerator.is_main_process:
                    append_history_row(self.history_csv_path, history_row, HISTORY_FIELDS)
                    if should_save_ckpt:
                        self._save_checkpoint(
                            path=os.path.join(self.output_dirs.checkpoints, f"cl_model_epoch_{epoch:04d}.pth"),
                            epoch=epoch,
                        )
                        self._save_checkpoint(path=self.latest_ckpt_path, epoch=epoch)
                    print(
                        f"[Epoch {epoch:04d}/{self.args.epochs:04d}] "
                        f"train_loss={avg_train_loss:.4f} | lr={current_lr:.2e} | train_time={train_time:.1f}s"
                    )

            self.accelerator.wait_for_everyone()

            if self.should_stop:
                if self.accelerator.is_main_process:
                    print(f"[EarlyStop] val_mae 连续 {self.early_stop_patience} 轮不下降，终止训练。")
                break

    def _train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        running_reg = 0.0
        train_start_time = time.time()

        for batch in self.dataloaders.train_loader:
            points = batch["point_cloud"]
            if self.jitter_std > 0:
                points = points + torch.randn_like(points) * self.jitter_std
            cl_gt_norm = batch["aero_label"][:, 0].float().view(-1, 1)

            self.optimizer.zero_grad(set_to_none=True)
            _, mu, logvar, cl_pred_norm = self.model(points, aux_vec=None, query_points=None, aero_only=True)

            loss_mse = F.mse_loss(cl_pred_norm, cl_gt_norm)
            loss_mae = F.l1_loss(cl_pred_norm, cl_gt_norm)
            # KL 正则：让 q(z|x) 靠近 N(0, I)，同时让 logvar 分支真正"活"起来
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = loss_mse + loss_mae + self.kl_beta * kl

            self.accelerator.backward(loss)
            self.optimizer.step()
            running_loss += (loss_mse + loss_mae).item()
            running_reg += kl.item()

        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]["lr"]
        n_batches = max(len(self.dataloaders.train_loader), 1)
        avg_train_loss = running_loss / n_batches
        avg_kl = running_reg / n_batches
        train_time = time.time() - train_start_time
        if self.accelerator.is_main_process:
            self._last_kl = avg_kl
        return avg_train_loss, current_lr, train_time

    def _evaluate_epoch(self, epoch, history_row, avg_train_loss, current_lr, train_time, should_save_ckpt):
        raw_model = self.accelerator.unwrap_model(self.model)
        eval_start = time.time()

        train_metrics, train_preds, train_gts, _ = evaluate_aero_regression(
            raw_model,
            self.dataloaders.train_eval_loader,
            self.accelerator.device,
            self.data_bundle.cl_mean,
            self.data_bundle.cl_std,
        )
        val_metrics, val_preds, val_gts, _ = evaluate_aero_regression(
            raw_model,
            self.dataloaders.test_loader,
            self.accelerator.device,
            self.data_bundle.cl_mean,
            self.data_bundle.cl_std,
        )

        attach_metrics(history_row, "train", train_metrics)
        attach_metrics(history_row, "val", val_metrics)
        self.history_rows.append(history_row)
        append_history_row(self.history_csv_path, history_row, HISTORY_FIELDS)

        latest_summary = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": avg_train_loss,
            "train_metrics": train_metrics,
            "test_metrics": val_metrics,
        }
        save_json(latest_summary, self.latest_json_path)

        improved = val_metrics["mae"] < self.best_val_mae
        if improved:
            self.best_val_mae = val_metrics["mae"]
            self.patience_counter = 0
            self._save_checkpoint(self.best_ckpt_path, epoch)
            save_json(latest_summary, self.best_json_path)
        else:
            self.patience_counter += 1
            if self.early_stop_patience > 0 and self.patience_counter >= self.early_stop_patience:
                self.should_stop = True

        if should_save_ckpt:
            self._save_checkpoint(os.path.join(self.output_dirs.checkpoints, f"cl_model_epoch_{epoch:04d}.pth"), epoch)
        self._save_checkpoint(self.latest_ckpt_path, epoch)

        if epoch % self.args.plot_interval == 0 or epoch == 1 or epoch == self.args.epochs or improved:
            plot_loss_convergence(self.history_rows, os.path.join(self.output_dirs.plots, "loss_convergence_latest.png"))
            plot_force_scatter(
                train_gts=train_gts,
                train_preds=train_preds,
                train_metrics=train_metrics,
                val_gts=val_gts,
                val_preds=val_preds,
                val_metrics=val_metrics,
                save_path=os.path.join(self.output_dirs.plots, f"force_compare_epoch_{epoch:04d}.png"),
                target_name="C_L",
                epoch=epoch,
            )
            plot_force_scatter(
                train_gts=train_gts,
                train_preds=train_preds,
                train_metrics=train_metrics,
                val_gts=val_gts,
                val_preds=val_preds,
                val_metrics=val_metrics,
                save_path=os.path.join(self.output_dirs.plots, "force_compare_latest.png"),
                target_name="C_L",
                epoch=epoch,
            )
            if improved:
                plot_force_scatter(
                    train_gts=train_gts,
                    train_preds=train_preds,
                    train_metrics=train_metrics,
                    val_gts=val_gts,
                    val_preds=val_preds,
                    val_metrics=val_metrics,
                    save_path=os.path.join(self.output_dirs.plots, "force_compare_best.png"),
                    target_name="C_L",
                    epoch=epoch,
                )

        eval_time = time.time() - eval_start
        print(
            f"[Epoch {epoch:04d}/{self.args.epochs:04d}] "
            f"train_loss={avg_train_loss:.4f} | lr={current_lr:.2e} | "
            f"train_time={train_time:.1f}s | eval_time={eval_time:.1f}s"
        )
        print("  " + format_metric_line("Train", train_metrics))
        print("  " + format_metric_line("Test ", val_metrics))
        if improved:
            print(f"  [Best] Updated best checkpoint: {self.best_ckpt_path}")

    def _save_checkpoint(self, path, epoch):
        raw_model = self.accelerator.unwrap_model(self.model)
        torch.save(
            build_checkpoint_payload(
                raw_model=raw_model,
                args=self.args,
                epoch=epoch,
                best_val_mae=self.best_val_mae,
                data_bundle=self.data_bundle,
                model_config=self.model_config,
            ),
            path,
        )
