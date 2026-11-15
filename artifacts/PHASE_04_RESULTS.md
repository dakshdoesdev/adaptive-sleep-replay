# Phase 4: periodic versus adaptive replay triggers

**Status: exploratory five-seed pilot, not a final hypothesis test.** Five
paired Split-MNIST seeds (42–46), three epochs per task, ran on the local
NVIDIA RTX 3050 Laptop GPU. Source revision: `17be479`.

## Results

| Method | Final average accuracy | Average forgetting | Replay bursts / seed | Replay examples / seed | Trigger checks / seed | Mean total runtime |
|---|---:|---:|---:|---:|---:|---:|
| Periodic (`k=25`) | 45.25 ± 3.23% | 66.58 ± 4.15 pp | 43.0 | 1,376 | 0 | 14.81 s |
| Probe-loss trigger | 26.29 ± 4.59% | 90.38 ± 5.74 pp | 29.6 ± 1.14 | 947.2 ± 36.49 | 109 | 14.76 s |
| Gradient-conflict trigger | 35.69 ± 3.33% | 78.57 ± 4.20 pp | 34.6 ± 1.67 | 1,107.2 ± 53.55 | 109 | 15.09 s |

Values are mean ± sample standard deviation over seeds, except the fixed
periodic count and trigger-check count. Paired adaptive-minus-periodic
accuracy differences were **−18.95 ± 3.74 pp** for probe loss and
**−9.55 ± 2.88 pp** for gradient conflict. Paired forgetting differences
were **+23.80 ± 4.71 pp** and **+11.99 ± 3.62 pp**, respectively; positive
means more forgetting than periodic replay.

All methods received 180,000 incoming examples per seed. The 500-slot buffer
reserved the same fixed 32-example probe for every method, leaving 468 mutable
reservoir examples. Periodic replay used 43 bursts / 1,376 examples per seed.
Adaptive methods had a **maximum**, not a required count, of 43 bursts; they
used fewer replay examples and had fewer optimizer updates as a result.

The mean measured trigger-monitoring time was 0.022 s per seed for probe loss
and 0.112 s for gradient conflict, versus 14.76–15.09 s total run time. The
loss method performed 3,520 probe forward-example evaluations per seed,
including the initial 32-example EMA baseline. The gradient method performed
3,488 probe forward and 3,488 probe backward example evaluations per seed;
the incoming gradient was reused by its normal training update. These
timings are descriptive for this hardware and workload, not FLOP measurements.
There was no material wall-clock saving, and fewer replay updates came with
substantially worse retention in this configuration.

The 500-example memory payload is approximately 1.57 MB when counted as
float32 28×28 image values plus int64 labels (Python container overhead is not
included); the fixed probe accounts for 32 of those 500 examples.

## Interpretation

This pilot does **not** support the claim that either tested trigger improves
retention or sample efficiency over periodic placement. It is not a matched
realized-replay comparison, and the trigger thresholds were prespecified pilot
values rather than tuned on held-out development streams. It therefore does
not settle H1 or H2. Threshold/check-frequency tuning on development streams,
followed by matched-replay and matched-total-compute evaluations on held-out
streams, remains necessary.

The loss trigger used a stable 32-example probe sampled from the initial
reservoir, a lagged EMA with beta 0.90, and a 10% relative-spike threshold.
The gradient trigger fired when incoming-versus-probe cosine was below zero.
Both checked every 10 incoming steps, warmed up during Task 1, used a 25-step
cooldown, and stopped at 43 bursts. Signals were computed before the incoming
update; any replay update followed it. Test accuracy did not control triggers.

## Artifacts

- [`phase_04_adaptive_replay_rtx3050.json`](phase_04_adaptive_replay_rtx3050.json):
  per-seed accuracy matrices, trigger logs, replay schedules, cost counts,
  timing breakdowns, source revision, and environment.
- [`phase_04_adaptive_replay_rtx3050.png`](phase_04_adaptive_replay_rtx3050.png):
  aggregate accuracy and burst-count plot.
- Protocol and reproduction: [`../experiments/phase-04-adaptive-replay.md`](../experiments/phase-04-adaptive-replay.md).
