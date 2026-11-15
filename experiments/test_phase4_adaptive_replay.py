import random
import unittest

import torch

from phase4_adaptive_replay import (
    accept_adaptive_trigger,
    gradient_cosine,
    loss_spike,
    reserve_probe,
)


class AdaptiveReplayTriggerTests(unittest.TestCase):
    def test_loss_spike_uses_relative_threshold(self) -> None:
        self.assertTrue(loss_spike(1.11, 1.0, 0.10))
        self.assertFalse(loss_spike(1.10, 1.0, 0.10))
        with self.assertRaises(ValueError):
            loss_spike(1.0, 1.0, -0.1)

    def test_gradient_cosine_detects_conflict_and_zero_norm(self) -> None:
        current = (torch.tensor([1.0, 0.0]),)
        opposing = (torch.tensor([-2.0, 0.0]),)
        aligned = (torch.tensor([2.0, 0.0]),)
        zero = (torch.zeros(2),)
        self.assertAlmostEqual(gradient_cosine(current, opposing), -1.0)
        self.assertAlmostEqual(gradient_cosine(current, aligned), 1.0)
        self.assertIsNone(gradient_cosine(current, zero))
        self.assertIsNone(gradient_cosine((None,), (None,)))

    def test_cooldown_and_hard_budget_reject_triggers(self) -> None:
        self.assertTrue(accept_adaptive_trigger(True, 20, None, 0, 43, 25))
        self.assertFalse(accept_adaptive_trigger(True, 44, 20, 1, 43, 25))
        self.assertTrue(accept_adaptive_trigger(True, 45, 20, 1, 43, 25))
        self.assertFalse(accept_adaptive_trigger(True, 100, 20, 43, 43, 25))
        self.assertFalse(accept_adaptive_trigger(False, 100, None, 0, 43, 25))

    def test_probe_is_reserved_without_exceeding_total_memory_capacity(self) -> None:
        memory = [(torch.tensor([float(index)]), index) for index in range(500)]
        replay_memory, probe = reserve_probe(memory, 32, random.Random(42))
        self.assertEqual(len(replay_memory), 468)
        self.assertEqual(len(probe), 32)
        self.assertEqual(len(replay_memory) + len(probe), 500)
        self.assertFalse(set(id(item) for item in replay_memory) & set(id(item) for item in probe))


if __name__ == "__main__":
    unittest.main()
