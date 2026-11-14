# Split-MNIST baseline pilot

**Status: exploratory pipeline check, not a final thesis result.** The run used
seed 42, one epoch per task, all 60,000 MNIST training examples, an MLP with
10 unmasked output classes, and a 500-example reservoir. Uniform replay added
32 sampled memory examples to each incoming minibatch after the first buffer
updates. Both methods received the same incoming examples and optimizer-step
count; replay therefore used additional training examples and compute.

## Result

| Method | Final average accuracy | Average forgetting | Mean new-task accuracy | Replay examples | Train + evaluation time |
|---|---:|---:|---:|---:|---:|
| Sequential | 19.22% | 98.36 pp | 97.91% | 0 | 11.90 s |
| Uniform replay | 63.91% | 41.48 pp | 97.09% | 15,040 | 13.04 s |

Both methods made 471 optimizer updates and saw 60,000 incoming training
examples. This single run shows the expected failure mode: sequential training learned
each new digit pair well but ended at 0% on every earlier pair. Uniform replay
retained part of that earlier performance while achieving similar accuracy on
the current pair. The result is consistent with replay helping retention in
this pilot, but one seed cannot establish a reliable effect size or support a
general claim.

The measured setup was CPU-only (`torch 2.14.1+cpu`, `torchvision 0.29.1+cpu`);
the timings are for this runtime and include evaluation. The run did not test
adaptive scheduling, matched total compute, or any LLM sleep/dreaming method.
It is a baseline for the separate replay-scheduling thesis project.

## Artifacts

- [`split_mnist_pilot_cpu.json`](split_mnist_pilot_cpu.json): configuration,
  per-task accuracy matrix, exposure, update count, runtime and environment.
- [`split_mnist_pilot_cpu.png`](split_mnist_pilot_cpu.png): accuracy and
  forgetting visualization.

Next: run multiple paired seeds, tune all methods on development streams, and
compare under both matched replay counts and matched total computation as
specified in [`../experiments/protocol.md`](../experiments/protocol.md).
