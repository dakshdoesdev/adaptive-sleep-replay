import random
import tempfile
import unittest
from pathlib import Path

import torch

from split_mnist_baseline import (
    _reservoir_add,
    aggregate_runs,
    make_plot,
    summarize_accuracy,
)


class BaselineMetricTests(unittest.TestCase):
    def test_summary_uses_final_accuracy_and_forgetting(self) -> None:
        matrix = [
            [0.80, None, None, None, None],
            [0.70, 0.90, None, None, None],
            [0.60, 0.80, 0.90, None, None],
            [0.50, 0.70, 0.80, 0.90, None],
            [0.40, 0.60, 0.70, 0.80, 0.90],
        ]
        result = summarize_accuracy(matrix)
        self.assertAlmostEqual(result["final_average_accuracy"], 0.68)
        self.assertAlmostEqual(result["average_forgetting"], 0.25)

    def test_reservoir_respects_fixed_capacity(self) -> None:
        memory: list[tuple[torch.Tensor, int]] = []
        images = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        labels = torch.arange(10)
        seen = _reservoir_add(
            memory, images, labels, 0, 3, random.Random(42)
        )
        self.assertEqual(seen, 10)
        self.assertEqual(len(memory), 3)
        self.assertTrue(all(0 <= label < 10 for _, label in memory))

    def test_paired_aggregation_keeps_per_seed_differences(self) -> None:
        def entry(final_accuracy: float, forgetting: float) -> dict:
            return {
                "accuracy_matrix": [[0.5] * 5 for _ in range(5)],
                "final_average_accuracy": final_accuracy,
                "average_forgetting": forgetting,
                "new_task_accuracy": [0.8] * 5,
                "incoming_training_examples": 60_000,
                "replay_examples": 15_040,
                "optimizer_steps": 471,
                "replay_buffer_capacity": 500,
                "training_and_evaluation_seconds": 10.0,
            }

        runs = {
            "1": {
                "methods": {
                    "sequential": entry(0.4, 0.3),
                    "uniform_replay": entry(0.6, 0.1),
                }
            },
            "2": {
                "methods": {
                    "sequential": entry(0.5, 0.4),
                    "uniform_replay": entry(0.7, 0.2),
                }
            },
        }
        result = aggregate_runs(runs)
        self.assertAlmostEqual(
            result["paired_differences"]["final_average_accuracy"]["mean"], 0.2
        )
        self.assertAlmostEqual(
            result["paired_differences"]["average_forgetting"]["mean"], -0.2
        )
        self.assertAlmostEqual(
            result["methods"]["uniform_replay"]["final_average_accuracy"], 0.65
        )
        result["configuration"] = {"seeds": [1, 2]}
        with tempfile.TemporaryDirectory() as directory:
            plot = Path(directory) / "pilot.png"
            make_plot(result, plot)
            self.assertTrue(plot.is_file())


if __name__ == "__main__":
    unittest.main()
