import numpy as np
import torch


def denormalize(values, mean, std):
    return (np.asarray(values, dtype=np.float32) * float(std)) + float(mean)


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    count = int(y_true.size)
    if count == 0:
        nan = float("nan")
        return {
            "count": 0,
            "mae": nan,
            "maxe": nan,
            "mse": nan,
            "rmse": nan,
            "bias": nan,
            "r2": nan,
        }

    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    mse = float(np.mean(errors ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(errors ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "count": count,
        "mae": float(np.mean(abs_errors)),
        "maxe": float(np.max(abs_errors)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "bias": float(np.mean(errors)),
        "r2": r2,
    }


def evaluate_aero_regression(model, dataloader, device, cl_mean, cl_std):
    was_training = model.training
    model.eval()

    all_preds = []
    all_gts = []
    all_file_ids = []

    with torch.no_grad():
        for batch in dataloader:
            points = batch["point_cloud"].to(device)
            _, _, _, pred_cl_norm = model(points, aux_vec=None, query_points=None, aero_only=True)

            pred_cl_phys = denormalize(pred_cl_norm.detach().cpu().numpy(), cl_mean, cl_std).reshape(-1)
            gt_cl_phys = batch["raw_cl"].detach().cpu().numpy().reshape(-1)

            all_preds.append(pred_cl_phys)
            all_gts.append(gt_cl_phys)
            all_file_ids.extend(batch["file_id"])

    if was_training:
        model.train()

    preds = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0,), dtype=np.float32)
    gts = np.concatenate(all_gts, axis=0) if all_gts else np.zeros((0,), dtype=np.float32)
    metrics = compute_regression_metrics(gts, preds)
    return metrics, preds, gts, all_file_ids


def format_metric_line(split_name, metrics):
    return (
        f"{split_name}: "
        f"MAE={metrics['mae']:.4f} | "
        f"MAXE={metrics['maxe']:.4f} | "
        f"RMSE={metrics['rmse']:.4f} | "
        f"R2={metrics['r2']:.4f} | "
        f"Bias={metrics['bias']:.4f} | "
        f"N={metrics['count']}"
    )
