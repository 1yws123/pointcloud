import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_encoder_4594")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_style():
    """兼容新老 matplotlib：新版本用 seaborn-v0_8-whitegrid，老版本用 seaborn-whitegrid，
    都没有就退回 default。"""
    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot", "default"):
        try:
            plt.style.use(name)
            return
        except (OSError, ValueError):
            continue


def _history_series(history_rows, key):
    xs = []
    ys = []
    for row in history_rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isnan(value):
            continue
        xs.append(int(row["epoch"]))
        ys.append(value)
    return xs, ys


def plot_loss_convergence(history_rows, save_path):
    if not history_rows:
        return

    _safe_style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 13), dpi=220, sharex=True)

    epochs, train_loss = _history_series(history_rows, "train_loss")
    axes[0].plot(epochs, train_loss, color="#1f77b4", linewidth=2, label="Train Loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Convergence")
    axes[0].legend(loc="best")

    for key, label, color in [
        ("train_mae", "Train MAE", "#1f77b4"),
        ("val_mae", "Test MAE", "#d62728"),
        ("train_maxe", "Train MAXE", "#17becf"),
        ("val_maxe", "Test MAXE", "#ff7f0e"),
    ]:
        xs, ys = _history_series(history_rows, key)
        if xs:
            axes[1].plot(xs, ys, marker="o", linewidth=1.6, label=label, color=color)
    axes[1].set_ylabel("Error")
    axes[1].set_title("Evaluation Error")
    axes[1].legend(loc="best")

    for key, label, color in [
        ("train_r2", "Train R2", "#1f77b4"),
        ("val_r2", "Test R2", "#d62728"),
    ]:
        xs, ys = _history_series(history_rows, key)
        if xs:
            axes[2].plot(xs, ys, marker="o", linewidth=1.6, label=label, color=color)
    axes[2].axhline(0.0, linestyle="--", linewidth=1.0, color="#555555")
    axes[2].set_ylabel("R2")
    axes[2].set_xlabel("Epoch")
    axes[2].set_title("Generalization Trend")
    axes[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _scatter_limits(arrays):
    finite_arrays = [np.asarray(arr, dtype=np.float32).reshape(-1) for arr in arrays if len(arr) > 0]
    if not finite_arrays:
        return [-1.0, 1.0]

    concatenated = np.concatenate(finite_arrays, axis=0)
    v_min = float(np.min(concatenated))
    v_max = float(np.max(concatenated))
    if v_min == v_max:
        delta = 1.0 if v_min == 0.0 else abs(v_min) * 0.1
        return [v_min - delta, v_max + delta]
    pad = (v_max - v_min) * 0.1
    return [v_min - pad, v_max + pad]


def plot_force_scatter(
    train_gts,
    train_preds,
    train_metrics,
    val_gts,
    val_preds,
    val_metrics,
    save_path,
    target_name="C_L",
    epoch=None,
):
    _safe_style()
    fig, ax = plt.subplots(figsize=(9, 9), dpi=260)

    if len(train_gts) > 0:
        ax.scatter(
            train_gts,
            train_preds,
            alpha=0.45,
            edgecolors="white",
            linewidths=0.5,
            color="#0b84f3",
            s=42,
            label="Train Samples",
        )
    if len(val_gts) > 0:
        ax.scatter(
            val_gts,
            val_preds,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
            color="#f26419",
            marker="^",
            s=52,
            label="Test Samples",
        )

    lims = _scatter_limits([train_gts, train_preds, val_gts, val_preds])
    ax.plot(lims, lims, color="#333333", linestyle="--", linewidth=1.8, label="Identity Line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")

    title_suffix = f" @ Epoch {epoch}" if epoch is not None else ""
    ax.set_title(f"Force Comparison ({target_name}){title_suffix}", fontsize=16, fontweight="bold")
    ax.set_xlabel(f"Ground Truth {target_name}", fontsize=13)
    ax.set_ylabel(f"Predicted {target_name}", fontsize=13)

    stats_text = (
        f"Train:\n"
        f"MAE={train_metrics['mae']:.4f}\n"
        f"MAXE={train_metrics['maxe']:.4f}\n"
        f"R2={train_metrics['r2']:.4f}\n\n"
        f"Test:\n"
        f"MAE={val_metrics['mae']:.4f}\n"
        f"MAXE={val_metrics['maxe']:.4f}\n"
        f"R2={val_metrics['r2']:.4f}"
    )
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
    )
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
