# Phase 3 diagnostic follow-up: active-compute-matched MoE

**Status: exploratory five-seed run complete.** This follow-up corrects the
original Phase 3 pilot's active-capacity mismatch and records router behavior
during training. It ran on the local RTX 3050 Laptop GPU using source commit
`56b2701`, seeds 42–46, three epochs per task.

## Results

| Model | Final average accuracy | Average forgetting | Total parameters | Active parameters / example | Estimated forward MACs / example | Mean train + eval time |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 43.95 ± 1.71% | 68.17 ± 2.25 pp | 269,322 | 269,322 | 268,800 | 15.06 s |
| Sparse MoE | 29.08 ± 2.24% | 86.82 ± 2.82 pp | 468,238 | 270,478 | 269,824 | 17.01 s |

Values are mean ± sample standard deviation across seeds. The paired
MoE-minus-dense accuracy difference was **−14.87 pp**; MoE was lower on each
seed (−11.76, −12.46, −17.80, −14.30, and −18.03 pp). The paired forgetting
difference was **+18.65 pp** (positive means MoE forgot more).

The MoE's estimated active forward MACs are **0.38% above** dense, meeting the
predeclared 1% compute-match tolerance. Its total parameter count is 73.9%
higher because all four experts are stored while only one expert is active
per example. The MoE ran about 13.0% slower (17.01 s vs. 15.06 s) in this
small-model setup; estimated MACs are analytical, not measured FLOPs.

Both methods received 180,000 incoming examples, 1,413 incoming optimizer
steps, 43 replay bursts, 1,376 replay examples, and 1,456 total optimizer
steps per seed.

## When router collapse occurred

The MoE was configured with four top-1 experts. For each task, training was
split into four chronological windows. Averaged over seeds, on the second
task the maximum expert load rose from **97.3%** in the first window to
effectively **100%** in the remaining windows. Mean normalized routing
entropy fell from **0.0667** to approximately zero. By tasks 3–5, routing
was already at or above 99.8% to a single expert in the first window.

The mean maximum router probability was about **0.935** in the first window
of the second task and about **0.987** afterward, indicating increasingly saturated
gates alongside hard-route collapse. The weighted auxiliary/classification
loss ratio grew during training (for example, on the fifth task it averaged 0.06,
0.28, 0.37, and 0.42 over successive windows). That ratio compares scalar
loss magnitudes, **not gradient magnitudes**; it does not prove that the
balance gradient is strong enough to reverse saturated routing.

The diagnostic narrows the finding: collapse is visible early, despite
active-compute matching, and is accompanied by increasingly confident gates.
It still does not isolate a unique cause. The next controlled diagnosis
should inspect router-gradient norms and compare pre-registered balance-loss
settings or router noise on development streams; do not tune against final
test accuracy.

This is not a general verdict on MoE. It is a small Split-MNIST MLP pilot with
one task order and one router configuration, not an LLM-sleep reproduction.

## Artifacts

- [`phase_03_moe_compute_matched_rtx3050.json`](phase_03_moe_compute_matched_rtx3050.json):
  per-seed metrics, four-window route/loss diagnostics, paired differences,
  runtime, model profiles, and environment.
- [`phase_03_moe_compute_matched_rtx3050.png`](phase_03_moe_compute_matched_rtx3050.png):
  summary plot.
- Run this from the
  [version-controlled Colab lab notebook](../notebooks/adaptive_sleep_replay_lab.ipynb).
