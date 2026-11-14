# Split-MNIST baseline pilots

**Status: exploratory pilots, not final thesis results.** Both five-seed runs
use seeds 42–46, the full 60,000-example MNIST training set, a ten-output
unmasked classifier, and a paired sequential-versus-uniform-replay comparison.
Uniform replay uses a 500-example reservoir and adds 32 sampled memories to
each incoming minibatch once the buffer is populated.

## Five seeds, one epoch per task

| Method | Final average accuracy | Average forgetting | Mean new-task accuracy | Replay examples/seed |
|---|---:|---:|---:|---:|
| Sequential | 19.21 ± 0.10% | 98.39 ± 0.09 pp | 97.93% | 0 |
| Uniform replay | 67.87 ± 2.63% | 36.19 ± 3.71 pp | 96.82% | 15,040 |

Each method saw 60,000 incoming examples and made 471 optimizer updates per
seed. The mean paired final-accuracy difference was **+48.65 pp**; replay was
better on all five seeds.

## Five seeds, three epochs per task

| Method | Final average accuracy | Average forgetting | Mean new-task accuracy | Replay examples/seed | Mean train + eval time/seed |
|---|---:|---:|---:|---:|---:|
| Sequential | 19.45 ± 0.01% | 98.89 ± 0.10 pp | 98.56% | 0 | 21.11 s |
| Uniform replay | 79.39 ± 1.98% | 22.84 ± 2.75 pp | 97.66% | 45,184 | 23.34 s |

Each method saw 180,000 incoming examples and made 1,413 optimizer updates
per seed. Replay's mean paired final-accuracy advantage was **+59.93 pp**,
positive on all five seeds (57.43, 62.54, 60.30, 60.77, and 58.63 pp).
Sequential training learned each new digit pair well but ended near 0% on
earlier pairs after the final task. Replay retained substantially more earlier
performance, using additional examples and runtime.

Values are mean ± sample standard deviation across five seeds. The runtime was
a Colab Tesla T4 (`torch 2.14.1+cu130`, CUDA 13.0); reported time includes
evaluation and is specific to this run. Both comparisons match incoming
exposure and optimizer-step count, **not total computation**. Five seeds and
one task order do not establish a general effect or thesis conclusion.
Neither run tests adaptive scheduling or an LLM sleep/dreaming method.

## Artifacts

- [`split_mnist_five_seed_3epoch_t4.json`](split_mnist_five_seed_3epoch_t4.json)
  and [plot](split_mnist_five_seed_3epoch_t4.png): primary three-epoch
  multi-seed pilot with per-seed matrices, paired differences, and environment.
- [`split_mnist_five_seed_t4.json`](split_mnist_five_seed_t4.json) and
  [plot](split_mnist_five_seed_t4.png): one-epoch five-seed follow-up.
- [`split_mnist_pilot_cpu.json`](split_mnist_pilot_cpu.json) and
  [plot](split_mnist_pilot_cpu.png): initial one-seed CPU smoke run; seed 42
  is included in the later paired runs.

Next: tune baselines on development streams and compare matched replay counts
and matched total computation as specified in
[`../experiments/protocol.md`](../experiments/protocol.md).
