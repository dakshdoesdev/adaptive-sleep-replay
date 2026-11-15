# Phase 3c: router noise and z-loss intervention pilot

**Status: one-seed development pilot; not a controlled multi-seed ablation.**
The Split-MNIST run used seed 42, three epochs per task, and the RTX 3050
Laptop GPU. Source revision: `17be479`.

## Configuration and outcome

The MoE used four top-1 experts, 128-wide expert bottlenecks, the existing
Switch-style load-balancing loss (`alpha = 0.01`), Gaussian router-logit noise
with standard deviation `0.1` during training, and a z-loss coefficient of
`0.001`. The z-loss was the mean squared log-sum-exp of the raw router logits;
the noise was not applied during evaluation. Replay remained periodic at
`k = 25`, with 43 bursts / 1,376 replay examples.

| Model / configuration | Final average accuracy | Average forgetting |
|---|---:|---:|
| Dense compute-matched control | 43.95% | 68.21 pp |
| Sparse MoE, original router settings | 32.19% | 82.99 pp |
| Sparse MoE, noise + z-loss | 33.85% | 80.81 pp |

The original and intervention MoE numbers above are the seed-42 entries from
separate runs. The intervention was **+1.66 pp** in accuracy and **−2.18 pp**
in forgetting versus the original seed-42 MoE run, but one seed cannot show a
reliable effect.

## Router behavior

The intervention did not prevent collapse. Maximum expert load for the
intervention was 80.6% on Task 1, 99.997% on Task 2, and 100% on Tasks 3–5.
Task-2 normalized route entropy was 0.00023. In the first four training
windows of Task 2, maximum load was 99.989%, then 100% in the remaining
windows. Mean router probability on the dominant expert ranged from 0.887 to
0.972 across those windows.

The z-loss and its weighted ratio to classification loss were logged, but
these scalar values do not establish that the router gradient was effective.
The z-loss form follows the router-logit stabilization objective described in
[ST-MoE](https://arxiv.org/abs/2202.08906).
This pilot does not test noise-only or z-loss-only conditions, does not
isolate a causal mechanism, and does not support a general claim about MoE.
The next step is a predeclared multi-seed development ablation with separate
noise-only, z-loss-only, combined, and unregularized controls, plus router
gradient-norm diagnostics.

## Artifacts

- [`phase_03c_router_regularization_rtx3050.json`](phase_03c_router_regularization_rtx3050.json):
  seed-level metrics, router windows, z-loss diagnostics, environment, and
  source revision.
- [`phase_03c_router_regularization_rtx3050.png`](phase_03c_router_regularization_rtx3050.png):
  dense versus intervention-MoE plot.
