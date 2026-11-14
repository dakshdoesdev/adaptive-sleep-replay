import unittest

import torch

from moe_split_mnist import SparseMoEClassifier, _model_and_profile


class SparseMoETests(unittest.TestCase):
    def test_top_one_forward_returns_logits_auxiliary_and_one_route_per_example(self) -> None:
        model = SparseMoEClassifier()
        images = torch.randn(7, 1, 28, 28)
        logits, auxiliary_loss, routes = model.forward_with_auxiliary(images)
        self.assertEqual(tuple(logits.shape), (7, 10))
        self.assertEqual(tuple(routes.shape), (7,))
        self.assertTrue(torch.isfinite(auxiliary_loss))
        self.assertTrue(torch.all((0 <= routes) & (routes < model.num_experts)))

    def test_auxiliary_and_task_losses_backpropagate_to_router_and_expert(self) -> None:
        model = SparseMoEClassifier()
        images = torch.randn(8, 1, 28, 28)
        logits, auxiliary_loss, _ = model.forward_with_auxiliary(images)
        loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, 10, (8,)))
        (loss + 0.01 * auxiliary_loss).backward()
        self.assertIsNotNone(model.router.weight.grad)
        self.assertTrue(any(expert[0].weight.grad is not None for expert in model.experts))

    def test_sparse_moe_keeps_total_parameters_near_dense_with_lower_estimated_macs(self) -> None:
        dense, dense_profile = _model_and_profile("dense")
        moe, moe_profile = _model_and_profile("sparse_moe")
        self.assertLess(
            abs(moe_profile["total_parameters"] - dense_profile["total_parameters"])
            / dense_profile["total_parameters"],
            0.02,
        )
        self.assertLess(
            moe_profile["estimated_forward_macs_per_example"],
            dense_profile["estimated_forward_macs_per_example"],
        )
        self.assertEqual(moe_profile["top_k"], 1)


if __name__ == "__main__":
    unittest.main()
