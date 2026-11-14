# Phase 2: periodic versus random replay-burst timing

**Status: exploratory pilot, not a final thesis result.** Five paired seeds
(42–46), three epochs per task, on a Colab Tesla T4. The comparison holds
incoming exposure, replay buffer, replay batch size, and replay-burst budget
fixed; only burst placement differs.

| Method | Final average accuracy | Average forgetting | Replay bursts / seed | Replay examples / seed | Total optimizer steps / seed | Mean train + eval time |
|---|---:|---:|---:|---:|---:|---:|
| Periodic bursts | 43.94 ± 1.71% | 68.18 ± 2.25 pp | 43 | 1,376 | 1,456 | 23.58 s |
| Random bursts | 29.60 ± 5.54% | 85.85 ± 7.33 pp | 43 | 1,376 | 1,456 | 23.44 s |

Values are mean ± sample standard deviation across seeds. The paired
random-minus-periodic difference in final accuracy was **−14.35 pp**; periodic
placement had higher final accuracy on all five seeds. The paired difference
in forgetting was **+17.68 pp** (positive means random forgot more).

| Seed | Periodic accuracy | Random accuracy | Random − periodic |
|---:|---:|---:|---:|
| 42 | 43.92% | 35.15% | −8.77 pp |
| 43 | 41.03% | 25.05% | −15.98 pp |
| 44 | 44.61% | 36.08% | −8.53 pp |
| 45 | 44.77% | 26.72% | −18.05 pp |
| 46 | 45.39% | 25.00% | −20.39 pp |

Each method received 180,000 incoming training examples and 1,413 incoming
optimizer updates per seed. The 43 single-step replay bursts add 1,376 replay
examples and 43 updates. The run used one Split-MNIST task order and a single
periodic interval (`k = 25`); the interval was not tuned. Stage-end test
accuracy was not used to schedule replay. The 0.14-second mean runtime
difference is descriptive only; both methods perform the same update counts,
but this does not establish a matched-FLOP or compute-saving result.

This pilot suggests that periodic placement is better than random placement
for this particular small replay budget and configuration. It does **not**
show that periodic replay is generally optimal, that adaptive scheduling is
unnecessary, or that replay saves compute. Five seeds and one task order are
not enough for a thesis conclusion.

## Artifacts

- [`phase_02_replay_bursts_t4.json`](phase_02_replay_bursts_t4.json): per-seed
  matrices, schedules, metrics, environment, and paired differences.
- [`phase_02_replay_bursts_t4.png`](phase_02_replay_bursts_t4.png): summary
  plot generated from the recorded results.
- Exact source revision used: `cef2941`.

Next: tune `k` using development streams only, then compare periodic replay
with a fixed-burst forgetting-triggered method under the protocol's monitoring
and total-compute accounting.
