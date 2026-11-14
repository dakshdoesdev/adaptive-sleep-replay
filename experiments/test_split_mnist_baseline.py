import random
import unittest

import torch

from split_mnist_baseline import _reservoir_add, summarize_accuracy


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


if __name__ == "__main__":
    unittest.main()
