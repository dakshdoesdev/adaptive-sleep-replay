# Phase 2: Replay-burst timing under a matched replay budget

**Question:** With the same incoming stream, reservoir, replay batch size, and
number of replay bursts, does periodic placement outperform random placement?
This isolates replay timing from replay volume before adding a forgetting
trigger.

## Frozen pilot configuration

- Split-MNIST class-incremental task order: `(0,1), (2,3), (4,5), (6,7), (8,9)`.
- Paired seeds `42–46`, three epochs per task, batch size 128.
- Reservoir capacity 500; replay batch size 32; SGD learning rate 0.05.
- A one-step replay burst after every 25 incoming minibatches for the periodic
  method. The random method samples the same number of positions uniformly
  without replacement within each task.
- No replay in the first task, before historical examples are available.
- The final test set is used only for the existing stage-end metrics, never to
  schedule replay.

Both methods receive the same incoming training examples and incoming optimizer
steps. Their replay-example and replay-step budgets are exactly equal within
each seed. Replay adds extra optimizer work, so this is a matched-replay-budget
placement comparison, **not** a matched-total-computation result. Runtime also
includes the existing stage-end evaluation.

## Run

```bash
python -m unittest discover -s experiments -v
python experiments/replay_burst_scheduling.py \
  --epochs 3 --seeds 42 43 44 45 46 \
  --out artifacts/phase_02_replay_bursts.json \
  --plot artifacts/phase_02_replay_bursts.png
```

The JSON records each seed's accuracy matrix, schedule positions, replay counts,
runtime, device, and paired differences. This is a pilot; do not treat it as
evidence that periodic replay is generally optimal or that adaptive replay saves
computation. Later work should add development-only interval tuning and
forgetting-triggered bursts under the protocol's monitoring and total-compute
accounting.
