# Phase 3: dense versus sparse MoE under periodic replay

**Status: exploratory architecture pilot, not a final thesis result.** Five
paired seeds (42–46), three epochs per task, on a Colab Tesla T4. Both models
used the same Split-MNIST stream, periodic replay positions, buffer contents,
replay examples, and optimizer-step counts.

| Model | Final average accuracy | Average forgetting | Total parameters | Active parameters / example | Estimated forward MACs / example | Mean train + eval time |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 43.94 ± 1.71% | 68.18 ± 2.25 pp | 269,322 | 269,322 | 268,800 | 23.66 s |
| Sparse MoE | 34.02 ± 5.19% | 80.68 ± 6.45 pp | 271,246 | 221,230 | 220,672 | 26.30 s |

Values are mean ± sample standard deviation across seeds. The paired
MoE-minus-dense final-accuracy difference was **−9.92 pp**; MoE was lower on
all five seeds. Its paired forgetting difference was **+12.50 pp** (positive
means MoE forgot more).

| Seed | MoE − dense final accuracy |
|---:|---:|
| 42 | −2.33 pp |
| 43 | −8.59 pp |
| 44 | −8.88 pp |
| 45 | −17.39 pp |
| 46 | −12.41 pp |

Each method received 180,000 incoming examples, 1,413 incoming optimizer
steps, 43 replay bursts, 1,376 replay examples, and 1,456 total optimizer
steps per seed. The dense and MoE parameter totals differ by about 0.7%.
The forward MAC estimates are analytical counts from the layer dimensions,
not measured FLOPs. MoE had about 17.9% fewer estimated active forward MACs
per example but took about 11.2% longer wall time in this run; theoretical
sparsity did not translate to a speedup here.

## Router behavior

The MoE has four top-1 experts and a Switch-style load-balancing auxiliary
loss with coefficient 0.01. Despite it, router assignments collapsed:
normalized route entropy averaged **0.128** on the first task and **0.000**
on each subsequent task. Maximum expert load averaged **96.3%** on the first
task and **100%** thereafter. By seed, all incoming examples in each later
task were routed to a single expert; which expert was used varied across
seeds. This is not stable task-specific expert specialization.

The result does not support the claim that adding this MoE automatically
protects old skills or improves replay efficiency. It is limited to one
small MLP, one Split-MNIST order, one router/balance-loss setting, and five
seeds. A development-only router/load-balance study would be needed before
integrating MoE with an adaptive replay trigger or making broader claims.

## Artifacts

- [`phase_03_moe_split_mnist_t4.json`](phase_03_moe_split_mnist_t4.json):
  per-seed accuracy matrices, replay schedules, router loads, model profiles,
  environment, and paired differences.
- [`phase_03_moe_split_mnist_t4.png`](phase_03_moe_split_mnist_t4.png):
  accuracy, forgetting, routing, and estimated-MAC summary plot.
- Source revision used: `1ad081f`.
- Literature context: [Switch Transformers](https://arxiv.org/abs/2101.03961)
  and [the sparsely-gated MoE paper](https://arxiv.org/abs/1701.06538).
