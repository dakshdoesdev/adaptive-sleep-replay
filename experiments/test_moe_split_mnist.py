import unittest

import torch

from moe_split_mnist import (
    DIAGNOSTIC_WINDOWS,
    NUM_EXPERTS,
    SparseMoEClassifier,
    _diagnostic_windows,
    _model_and_profile,
)


class SparseMoETests(unittest.TestCase):
    def test_top_one_forward_returns_logits_auxiliary_and_one_route_per_example(self) -> None:
        model = SparseMoEClassifier()
        images = torch.randn(7, 1, 28, 28)
        logits, auxiliary_loss, routes, probabilities = model.forward_with_auxiliary(
            images
        )
        self.assertEqual(tuple(logits.shape), (7, 10))
        self.assertEqual(tuple(routes.shape), (7,))
        self.assertEqual(tuple(probabilities.shape), (7, model.num_experts))
        self.assertTrue(torch.isfinite(auxiliary_loss))
        self.assertTrue(torch.all((0 <= routes) & (routes < model.num_experts)))

    def test_auxiliary_and_task_losses_backpropagate_to_router_and_expert(self) -> None:
        model = SparseMoEClassifier()
        images = torch.randn(8, 1, 28, 28)
        logits, auxiliary_loss, router_z_loss, _, _ = model.forward_with_regularizers(
            images
        )
        loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, 10, (8,)))
        (loss + 0.01 * auxiliary_loss + 0.001 * router_z_loss).backward()
        self.assertIsNotNone(model.router.weight.grad)
        self.assertTrue(any(expert[0].weight.grad is not None for expert in model.experts))
        self.assertTrue(torch.isfinite(router_z_loss))

    def test_router_noise_is_training_only(self) -> None:
        model = SparseMoEClassifier(router_noise_std=0.5)
        images = torch.randn(16, 1, 28, 28)
        model.train()
        torch.manual_seed(1)
        train_probabilities_a = model.forward_with_regularizers(images)[-1]
        torch.manual_seed(2)
        train_probabilities_b = model.forward_with_regularizers(images)[-1]
        self.assertFalse(torch.allclose(train_probabilities_a, train_probabilities_b))

        model.eval()
        eval_probabilities_a = model.forward_with_regularizers(images)[-1]
        eval_probabilities_b = model.forward_with_regularizers(images)[-1]
        self.assertTrue(torch.equal(eval_probabilities_a, eval_probabilities_b))

    def test_sparse_moe_matches_dense_active_macs_but_expands_total_capacity(self) -> None:
        dense, dense_profile = _model_and_profile("dense")
        moe, moe_profile = _model_and_profile("sparse_moe")
        self.assertLessEqual(
            abs(
                moe_profile["estimated_forward_macs_per_example"]
                - dense_profile["estimated_forward_macs_per_example"]
            )
            / dense_profile["estimated_forward_macs_per_example"],
            0.01,
        )
        self.assertGreater(moe_profile["total_parameters"], dense_profile["total_parameters"])
        self.assertEqual(moe_profile["top_k"], 1)

    def test_training_diagnostics_capture_route_probabilities_and_loss_ratio(self) -> None:
        counts = [
            [[0] * NUM_EXPERTS for _ in range(DIAGNOSTIC_WINDOWS)]
            for _ in range(5)
        ]
        probabilities = [
            [[0.0] * NUM_EXPERTS for _ in range(DIAGNOSTIC_WINDOWS)]
            for _ in range(5)
        ]
        classification = [[0.0] * DIAGNOSTIC_WINDOWS for _ in range(5)]
        auxiliary = [[0.0] * DIAGNOSTIC_WINDOWS for _ in range(5)]
        examples = [[0] * DIAGNOSTIC_WINDOWS for _ in range(5)]
        counts[0][0] = [2, 0, 0, 0]
        probabilities[0][0] = [1.0, 0.5, 0.3, 0.2]
        classification[0][0] = 2.0
        auxiliary[0][0] = 1.0
        examples[0][0] = 2

        result = _diagnostic_windows(
            counts, probabilities, classification, auxiliary, examples
        )
        first_window = result[0]["windows"][0]
        self.assertEqual(first_window["route_fraction"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(first_window["normalized_route_entropy"], 0.0)
        self.assertEqual(first_window["mean_classification_loss"], 1.0)
        self.assertEqual(first_window["mean_unweighted_auxiliary_loss"], 0.5)
        self.assertAlmostEqual(
            first_window["weighted_auxiliary_to_classification_loss_ratio"],
            0.005,
        )
        self.assertEqual(result[0]["windows"][1]["incoming_examples"], 0)


if __name__ == "__main__":
    unittest.main()
