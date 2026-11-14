import random
import unittest

from replay_burst_scheduling import make_burst_schedule


class ReplayBurstScheduleTests(unittest.TestCase):
    def test_periodic_schedule_uses_fixed_interval(self) -> None:
        self.assertEqual(
            make_burst_schedule("periodic_replay", 11, 3, random.Random(42)),
            [3, 6, 9],
        )

    def test_random_schedule_matches_periodic_burst_count_and_is_reproducible(self) -> None:
        first = make_burst_schedule("random_replay", 11, 3, random.Random(42))
        second = make_burst_schedule("random_replay", 11, 3, random.Random(42))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(1 <= position <= 11 for position in first))

    def test_schedule_handles_short_stream_and_invalid_inputs(self) -> None:
        self.assertEqual(
            make_burst_schedule("random_replay", 2, 3, random.Random(42)), []
        )
        with self.assertRaises(ValueError):
            make_burst_schedule("periodic_replay", 3, 0, random.Random(42))


if __name__ == "__main__":
    unittest.main()
