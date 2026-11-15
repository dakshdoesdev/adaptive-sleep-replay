# Phase 4: adaptive replay-trigger pilot

Status: pilot implementation; five-seed results are recorded in
[`../artifacts/PHASE_04_RESULTS.md`](../artifacts/PHASE_04_RESULTS.md).
This is not a final hypothesis test or a matched-realized-replay comparison.

## Question

Do a fixed-probe loss-spike trigger or a gradient-conflict trigger place
memory-only replay bursts more effectively than periodic replay under the same
maximum burst allowance?

## Methods

- **Periodic control:** one 32-example replay update every 25 incoming steps,
  yielding up to 43 bursts over the four post-warmup tasks.
- **Probe-loss trigger:** every 10 incoming steps, trigger when fixed-probe
  cross-entropy exceeds its lagged EMA by more than 10%; update the EMA after
  making the trigger decision, with beta 0.90.
- **Gradient-conflict trigger:** every 10 incoming steps, trigger when cosine
  similarity between the current incoming-batch gradient and the fixed-probe
  gradient is below zero. The incoming gradient is reused for the normal
  update; the probe gradient is an additional backward pass.
- **Adaptive budget:** at most 43 single-update replay bursts, with a 25-step
  cooldown. Adaptive methods may use fewer than 43 bursts; unused bursts are
  not filled with scheduled replay.

All methods use the same five Split-MNIST tasks, three epochs per task,
class-incremental 10-output MLP, paired seeds, incoming batch size 128, and
32-example replay batches. The initial 500-example reservoir is divided into
468 mutable replay examples and a fixed 32-example probe. The probe never
changes membership and is included in the 500-example capacity. The first
task is warmup without replay or trigger checks.

Trigger signals are computed before the current incoming update. The incoming
update runs before any triggered replay update. Trigger checks use training
memory only; test accuracy is used only for stage-end evaluation.

## Accounting and interpretation

Raw per-seed JSON logs trigger values and positions, replay examples/bursts,
optimizer updates, probe forward/backward example counts, trigger monitoring
time, incoming-gradient time, replay time, evaluation time, total time,
device, versions, and code commit. Wall-clock measurements include the entire
run; component times are diagnostic breakdowns.

The periodic control and adaptive methods share the 500-example memory cap and
incoming exposure, but realized replay counts can differ. This is an
**at-most-budget pilot**, not the equal-replay-example test for H1 and not the
matched-total-compute test for H2. Thresholds are prespecified pilot settings,
not tuned on held-out development task orders. The test set is never used for
triggering or tuning.

## Reproduction

```bash
python experiments/phase4_adaptive_replay.py
```

Use the CLI flags for a smaller runtime pilot or to change seeds and trigger
settings. Any threshold change must be reported as a new configuration rather
than silently folded into these results.
