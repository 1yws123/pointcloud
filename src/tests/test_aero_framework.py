import unittest

import numpy as np
import importlib

from src.data import build_group_split_indices
from src.engine import AeroEvaluator, AeroTrainer
from src.utils import attach_metrics, compute_regression_metrics, create_history_row, format_metric_line


class GroupSplitTests(unittest.TestCase):
    def test_group_split_keeps_groups_intact_and_is_deterministic(self):
        memory_cache = [
            {"file_id": "G58_1"},
            {"file_id": "G58_1"},
            {"file_id": "G58_2"},
            {"file_id": "G58_2"},
            {"file_id": "G58_3"},
            {"file_id": "G58_4"},
        ]

        train_a, val_a = build_group_split_indices(memory_cache, val_split=0.34, seed=42)
        train_b, val_b = build_group_split_indices(memory_cache, val_split=0.34, seed=42)

        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertTrue(set(train_a).isdisjoint(val_a))
        self.assertEqual(sorted(train_a + val_a), list(range(len(memory_cache))))

        groups_in_train = {memory_cache[idx]["file_id"] for idx in train_a}
        groups_in_val = {memory_cache[idx]["file_id"] for idx in val_a}
        self.assertTrue(groups_in_train.isdisjoint(groups_in_val))


class RegressionMetricTests(unittest.TestCase):
    def test_compute_regression_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y_pred = np.array([1.0, 2.5, 2.0], dtype=np.float32)

        metrics = compute_regression_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["mae"], 0.5, places=6)
        self.assertAlmostEqual(metrics["maxe"], 1.0, places=6)
        self.assertAlmostEqual(metrics["mse"], 0.4166666667, places=6)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(0.4166666667), places=6)
        self.assertAlmostEqual(metrics["bias"], -0.1666666667, places=6)
        self.assertAlmostEqual(metrics["r2"], 0.375, places=6)
        self.assertEqual(metrics["count"], 3)


class HistoryTests(unittest.TestCase):
    def test_history_row_accepts_metric_attachment(self):
        row = create_history_row(epoch=3, lr=1e-3, train_loss=0.12)
        attach_metrics(
            row,
            "train",
            {
                "mae": 0.1,
                "maxe": 0.3,
                "mse": 0.02,
                "rmse": 0.1414,
                "bias": -0.01,
                "r2": 0.8,
                "count": 16,
            },
        )

        self.assertEqual(row["epoch"], 3)
        self.assertAlmostEqual(row["train_mae"], 0.1, places=6)
        self.assertAlmostEqual(row["train_r2"], 0.8, places=6)
        self.assertEqual(row["train_count"], 16)
        self.assertTrue(np.isnan(row["val_mae"]))


class PackageLayoutTests(unittest.TestCase):
    def test_subpackages_export_primary_interfaces(self):
        self.assertEqual(AeroTrainer.__name__, "AeroTrainer")
        self.assertEqual(AeroEvaluator.__name__, "AeroEvaluator")
        text = format_metric_line(
            "Train",
            {"mae": 0.1, "maxe": 0.2, "rmse": 0.15, "r2": 0.7, "bias": 0.01, "count": 8},
        )
        self.assertIn("Train:", text)
        self.assertIn("MAE=0.1000", text)

    def test_named_entrypoints_exist(self):
        train_module = importlib.import_module("src.train")
        eval_module = importlib.import_module("src.evaluate")
        self.assertTrue(callable(train_module.main))
        self.assertTrue(callable(eval_module.main))


if __name__ == "__main__":
    unittest.main()
