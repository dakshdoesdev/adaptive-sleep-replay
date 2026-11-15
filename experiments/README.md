# First executable baseline

`split_mnist_baseline.py` compares sequential training with uniform experience
replay on class-incremental Split-MNIST. It is the first baseline in the
project's [experimental protocol](protocol.md), not an adaptive-scheduling
result and not a reproduction of an LLM sleep paper.

The pilot compares sequential and uniform replay over paired seeds. Uniform
replay adds 32 remembered examples to each incoming minibatch, so it performs
more work. These pilots do not claim a compute-matched comparison or a general
result. The five-seed one- and three-epoch pilots are in
[`../artifacts/RESULTS.md`](../artifacts/RESULTS.md).

Run in Colab or another environment with PyTorch, torchvision, and matplotlib:

```bash
python experiments/split_mnist_baseline.py
```

For a paired multi-seed pilot:

```bash
python experiments/split_mnist_baseline.py --epochs 1 --seeds 42 43 44 45 46 \
  --out artifacts/split_mnist_five_seed.json \
  --plot artifacts/split_mnist_five_seed.png
```

For the three-epoch follow-up:

```bash
python experiments/split_mnist_baseline.py --epochs 3 --seeds 42 43 44 45 46 \
  --out artifacts/split_mnist_five_seed_3epoch.json \
  --plot artifacts/split_mnist_five_seed_3epoch.png
```

MNIST is downloaded to `data/mnist/`. The script writes JSON metrics and a PNG
plot under `artifacts/`. Do not commit the downloaded dataset. The default
run trains on all MNIST training examples for three epochs per task.

Before thesis claims, extend this pilot with multiple paired seeds,
development-only tuning, matched-compute comparisons, trigger monitoring
costs, and held-out final evaluation as specified in `protocol.md`.

## Phase 2: replay-burst timing

[`replay_burst_scheduling.py`](replay_burst_scheduling.py) compares periodic
bursts with randomly placed bursts at the same per-task burst count and length.
It is a matched-replay-budget placement test, not a matched-total-compute or
adaptive-trigger result. The phase protocol and pilot settings are in
[`phase-02-replay-bursts.md`](phase-02-replay-bursts.md).
The five-seed Tesla T4 pilot is reported in
[`../artifacts/PHASE_02_RESULTS.md`](../artifacts/PHASE_02_RESULTS.md), with
raw metrics and plot alongside it.

```bash
python experiments/replay_burst_scheduling.py \
  --epochs 3 --seeds 42 43 44 45 46 \
  --out artifacts/phase_02_replay_bursts.json \
  --plot artifacts/phase_02_replay_bursts.png
```

## Phase 3: sparse MoE architecture sidecar

[`moe_split_mnist.py`](moe_split_mnist.py) compares a dense classifier with a
four-expert top-1 sparse MoE under the same periodic replay schedule. It logs
router load and entropy alongside parameter and estimated active-compute
counts. This is an exploratory architecture pilot, not an LLM sleep
reproduction or a replacement for the replay-scheduling thesis question. See
[`phase-03-moe-pilot.md`](phase-03-moe-pilot.md) for the comparison and
[`../artifacts/PHASE_03_RESULTS.md`](../artifacts/PHASE_03_RESULTS.md) for the
five-seed Tesla T4 results.

The original pilot used narrow experts and therefore did **not** match active
compute. Its diagnostic follow-up uses 128-wide experts, approximately matches
dense active MACs, and records router/loss signals over training. See
[`phase-03-moe-diagnostic.md`](phase-03-moe-diagnostic.md) before interpreting
that run, and [`../artifacts/PHASE_03_DIAGNOSTIC_RESULTS.md`](../artifacts/PHASE_03_DIAGNOSTIC_RESULTS.md)
for the five-seed RTX 3050 result.
