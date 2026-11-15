# Phase 3: Sparse MoE architecture pilot

**Status: original five-seed pilot complete; compute-matched diagnostic
follow-up pending.** Original results are in
[`../artifacts/PHASE_03_RESULTS.md`](../artifacts/PHASE_03_RESULTS.md); the
follow-up is specified in
[`phase-03-moe-diagnostic.md`](phase-03-moe-diagnostic.md). This sidecar does
not replace the thesis's primary replay-scheduling question.

## Question

Under the Phase 2 periodic replay schedule, does a small top-1 sparse
Mixture-of-Experts classifier retain Split-MNIST tasks better than the existing
dense classifier, and does its router actually specialize without collapsing?

This tests the proposed architectural analogy at classifier scale. It does not
reproduce an LLM sleep system, grow experts during consolidation, or establish
that MoE prevents catastrophic forgetting.

## Comparison

- Dense baseline: existing `DigitMLP`.
- Sparse MoE: shared input/output layers plus a learned top-1 router and four
  256→32→256 ReLU experts in the original pilot. A Switch-style load-balancing auxiliary loss uses
  coefficient `0.01`.
- Both use the same Split-MNIST order, paired seeds, incoming stream,
  reservoir, periodic `k=25` replay schedule, replay batch size, SGD settings,
  stage-end evaluation, and replay samples.
- Report total parameters, active parameters per example, estimated forward
  multiply-accumulates, wall time, peak GPU allocation, route fractions,
  normalized route entropy, and maximum expert load.

The dense model has 269,322 parameters and an estimated 268,800 forward
multiply-accumulates per example. The MoE has 271,246 total parameters and
221,230 active parameters per example; its estimated 220,672 forward
multiply-accumulates are analytical, not measured FLOPs. These are similar
total parameter budgets but different active compute. Wall-clock time remains
necessary because routing and dispatch overhead may erase theoretical savings.

## Run

```bash
python -m unittest discover -s experiments -v
python experiments/moe_split_mnist.py \
  --epochs 3 --seeds 42 43 44 45 46 \
  --out artifacts/phase_03_moe_split_mnist_t4.json \
  --plot artifacts/phase_03_moe_split_mnist_t4.png
```

The router's load-balancing penalty follows the auxiliary-loss idea in
[Switch Transformers](https://arxiv.org/abs/2101.03961). Sparse MoE's
conditional-compute motivation comes from
[Shazeer et al.](https://arxiv.org/abs/1701.06538); these works also discuss
training and systems challenges. This small pilot is not evidence that MoE is
automatically more accurate or faster.
