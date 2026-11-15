# Phase 3 diagnostic follow-up: match active compute and log router collapse

**Status: five-seed diagnostic run complete.** See
[`../artifacts/PHASE_03_DIAGNOSTIC_RESULTS.md`](../artifacts/PHASE_03_DIAGNOSTIC_RESULTS.md).

**Purpose:** Correct two limits of the original Phase 3 pilot. The original
MoE used a 32-wide bottleneck per expert, so its active path was substantially
smaller than the dense model. It also logged only task-aggregate hard routing,
not the training signals needed to locate router collapse.

## Frozen comparison

- Dense baseline: existing `DigitMLP`.
- Sparse MoE: four top-1 experts with 256→128→256 layers.
- The 128-wide expert gives approximately matched active forward compute:
  dense 268,800 MACs/example; MoE 269,824 MACs/example (0.38% difference).
- Total MoE capacity is larger by design: 468,238 parameters versus 269,322
  dense parameters. This tests sparse routing with matched active compute, not
  a matched total-parameter budget.
- Same paired seeds 42–46, Split-MNIST order, three epochs/task, periodic
  replay schedule, replay budget, optimizer, and stage-end evaluation.
- Add four chronological windows per task with hard route fraction, mean
  router probability, task loss, auxiliary balance loss, and the weighted
  auxiliary/classification loss ratio. Also save replay-route counts by the
  task during which the replay occurred.

The auxiliary loss is the Switch-style top-1 load-balancing objective with
coefficient `0.01`. These diagnostics are descriptive and are not used to tune
the router or inspect the final test set.

## Run

```bash
python -m unittest discover -s experiments -v
python experiments/moe_split_mnist.py \
  --moe-bottleneck 128 --epochs 3 --seeds 42 43 44 45 46 \
  --out artifacts/phase_03_moe_compute_matched_t4.json \
  --plot artifacts/phase_03_moe_compute_matched_t4.png
```

This follow-up can tell whether collapse happens early and whether the
auxiliary gradient is small relative to task loss. It still cannot isolate
the effect of load-balancing coefficient without a separate development-only
ablation. It is not an LLM-sleep reproduction.
