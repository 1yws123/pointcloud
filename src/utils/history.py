import csv
import os

import numpy as np


HISTORY_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "train_mae",
    "train_maxe",
    "train_mse",
    "train_rmse",
    "train_bias",
    "train_r2",
    "train_count",
    "val_mae",
    "val_maxe",
    "val_mse",
    "val_rmse",
    "val_bias",
    "val_r2",
    "val_count",
]


def create_history_row(epoch, lr, train_loss):
    row = {
        "epoch": epoch,
        "lr": lr,
        "train_loss": train_loss,
    }
    for split in ("train", "val"):
        row.update(
            {
                f"{split}_mae": np.nan,
                f"{split}_maxe": np.nan,
                f"{split}_mse": np.nan,
                f"{split}_rmse": np.nan,
                f"{split}_bias": np.nan,
                f"{split}_r2": np.nan,
                f"{split}_count": 0,
            }
        )
    return row


def attach_metrics(row, split_name, metrics):
    row[f"{split_name}_mae"] = metrics["mae"]
    row[f"{split_name}_maxe"] = metrics["maxe"]
    row[f"{split_name}_mse"] = metrics["mse"]
    row[f"{split_name}_rmse"] = metrics["rmse"]
    row[f"{split_name}_bias"] = metrics["bias"]
    row[f"{split_name}_r2"] = metrics["r2"]
    row[f"{split_name}_count"] = metrics["count"]


def append_history_row(csv_path, row, fieldnames=None):
    fieldnames = fieldnames or HISTORY_FIELDS
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_history_rows(csv_path):
    if not os.path.exists(csv_path):
        return []

    rows = []
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {"epoch": int(row["epoch"])}
            for key, value in row.items():
                if key == "epoch":
                    continue
                if value is None or value == "":
                    parsed[key] = np.nan
                else:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
            rows.append(parsed)
    return rows
