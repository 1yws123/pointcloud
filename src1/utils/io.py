import json
import os
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class AeroOutputDirs:
    root: str
    checkpoints: str
    plots: str
    metrics: str


def ensure_output_dirs(save_dir):
    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    plots_dir = os.path.join(save_dir, "plots")
    metrics_dir = os.path.join(save_dir, "metrics")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    return AeroOutputDirs(
        root=save_dir,
        checkpoints=checkpoints_dir,
        plots=plots_dir,
        metrics=metrics_dir,
    )


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def strip_module_prefix(state_dict):
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        cleaned[key.replace("module.", "", 1) if key.startswith("module.") else key] = value
    return cleaned


def parse_checkpoint_payload(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return strip_module_prefix(payload["model_state_dict"]), payload
    return strip_module_prefix(payload), {}


def _namespace_to_dict(args):
    if isinstance(args, dict):
        return dict(args)
    if hasattr(args, "__dict__"):
        return dict(vars(args))
    return {}


def build_checkpoint_payload(raw_model, args, epoch, best_val_mae, data_bundle, model_config):
    return {
        "model_state_dict": raw_model.state_dict(),
        "epoch": int(epoch),
        "best_val_mae": float(best_val_mae),
        "cl_mean": float(data_bundle.cl_mean),
        "cl_std": float(data_bundle.cl_std),
        "train_indices": list(data_bundle.train_indices),
        "val_indices": list(data_bundle.test_indices),
        "args": _namespace_to_dict(args),
        "model_config": dict(model_config),
    }
