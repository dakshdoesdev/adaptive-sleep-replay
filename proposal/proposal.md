# Research proposal draft

## When Is Replay Worth Its Cost? Evaluating Forgetting-Triggered Consolidation in Continual Learning

Author: Daksh Verma. Institution: Chitkara University. Supervisor, programme requirements, deadline and final submission format: to be confirmed.

## Abstract
Continual learning requires a neural network to acquire new information while retaining previously learned capabilities. Replay can reduce forgetting by revisiting stored examples, but both replay and the signals used to schedule it consume computation. This study will evaluate whether simple memory-loss and gradient-interference signals can guide short consolidation phases more effectively than tuned periodic scheduling. Using small classification networks, fixed memory budgets and controlled data exposure, it will compare retention, new-task learning, replay usage and total computational cost. The intended contribution is an empirical characterization of when adaptive monitoring is useful and when its overhead or noise outweighs its benefits. Improved performance is a hypothesis, not an assumed result.

## Background and gap to investigate
Sleep Replay Consolidation demonstrates a biologically inspired consolidation mechanism (Tadros et al., 2022). However, adaptive scheduling is also established: Replay Scheduling studies historical task allocation, S-TRIGGER uses environment change detection, and TRAF adapts the frequency of memory updates. Smart Replay further examines resource-aware scheduling. Consequently, this project does not claim to introduce adaptive replay itself. It investigates the practical value of inexpensive signals within a tightly controlled, resource-limited classification setting. Whether this exact comparison is sufficiently distinct remains subject to full-text literature review and supervisor assessment. Source links and reading priorities are in ../notes/literature-map.md.

## Research question and hypotheses
Can forgetting or gradient-conflict signals improve the retention-versus-total-computation trade-off relative to tuned periodic replay under a fixed memory budget?

H1: At comparable retention and new-task performance, an adaptive schedule can require fewer replay examples. H2: Those savings remain after monitoring costs are included. These are separate claims; supporting H1 does not establish H2. The null is that the adaptive schedules provide no improvement under the chosen comparisons.

## Method
Train a classifier on a sequence of disjoint class groups. Maintain a fixed-capacity reservoir of past training examples. At specified check intervals, compute either an increase in probe loss or negative normalized gradient alignment. Trigger a fixed-length memory-only replay burst when a tuned threshold is crossed, subject to cooldown and budget limits. Keep replay sampling unchanged across burst methods. In this project sleep denotes an offline rehearsal interval, not a validated model of biological sleep.

## Evaluation
Debug on Split MNIST, then evaluate Split CIFAR-10, with CIFAR-100 conditional on resources. Compare sequential learning, mixed replay, periodic bursts, random bursts and the two adaptive signals. Conduct matched replay-count and matched total-compute analyses separately, with multiple paired random seeds and equal hyperparameter-search budgets. Evaluate average accuracy, average forgetting, backward transfer, new-task accuracy and runtime including trigger overhead. Reserve final test data for reporting only. The detailed protocol is in ../experiments/protocol.md.

## Expected contribution and limitations
The output will be a reproducible implementation, an empirical comparison, and an analysis of failure conditions. A negative result can establish when simple periodic replay is the more effective choice. Small buffers may provide biased probes, threshold choices may not transfer across streams, and results on small image classifiers will not establish effects for language models or biological sleep. No performance gains or publication acceptance are promised.

## Plan
First reproduce baselines and understand the literature. Next pilot signals and instrumentation, freeze the final protocol, run repeated experiments, and write the results from saved logs. Obtain department-specific submission and supervision requirements before formatting a final synopsis.
