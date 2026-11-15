# Draft experimental protocol

Status: draft protocol; final evaluation is not complete. An exploratory Phase 4
pilot has been run, but it does not complete the matched-replay or
matched-total-compute evaluations described here. Freeze the final protocol
after development-only tuning and before held-out evaluation.

## Primary question
At equal memory and incoming-data exposure, do loss-triggered or gradient-triggered replay bursts improve the accuracy/total-compute frontier over tuned periodic bursts?

Use Split MNIST for debugging and Split CIFAR-10 for substantive evaluation. Add CIFAR-100 only after a runtime pilot. Evaluate class-incrementally with a single classifier and no task identity at test time. Report whether unseen classes are masked. Keep the rule identical across methods. Task boundaries may define evaluation checkpoints without being passed to the controller.

## Minimal methods
1. Sequential fine-tuning, no replay.
2. Uniform replay mixed with incoming minibatches.
3. Periodic replay bursts, sweeping check interval k on development streams.
4. Random replay bursts with the same count and burst length, scheduled independently of test performance.
5. Memory-loss-triggered fixed bursts.
6. Gradient-conflict-triggered fixed bursts.

Add a faithful TRAF or Smart Replay comparator if a reproducible implementation and comparable assumptions are available. Until then, do not claim superiority over adaptive replay generally.

Fix architecture, initialization per paired seed, optimizer, data order, incoming exposure, buffer capacity, reservoir update policy, replay sampling, augmentation and batch size. Start with fixed replay burst length to isolate scheduling. Tune all methods with equal development budgets; reserve final seeds/task orders for evaluation. Never use final test accuracy to trigger, stop, or tune.

## Signals
Loss signal: evaluate the same probe examples at adjacent checks; compare current loss with a lagged moving reference, updating the reference only after computing the signal. A changing probe can create false spikes; use a stable probe window, and reset its reference when membership changes. Count probe storage and reference scalars within the memory accounting. Tiny gradual forgetting may escape a moving average: include this as a failure case.

Gradient signal: c = dot(g_current,g_memory)/(norm(g_current)*norm(g_memory)+epsilon), trigger if c < -tau. Handle near-zero gradients explicitly. This is a proxy for the local effect of a gradient update, not proof of forgetting. Initially use plain SGD to make interpretation clearer; optimizer extensions require separate checks.

Use a cooldown and hard replay budget. Warm up when no historical memory exists. Cap the number of replay steps so no sleep phase can loop indefinitely. Thresholds and check frequencies are development hyperparameters.

## Two different resource comparisons
A. Match replay examples/updates and compare placement. Include extra monitoring cost in the report; this comparison does not establish compute savings.
B. Match total computation, including probe forwards, memory backwards, replay and incoming training. Plot accuracy against total cost. Separately report a fixed incoming-exposure comparison so compute constraints cannot silently starve new learning.

Measure wall time with identical evaluation cadence and device; account for GPU synchronization and warmup. Report training, trigger, replay, and evaluation time separately. Claim FLOPs only if measured or explicitly estimated with a supported accounting method. Never substitute replay count for total compute.

## Metrics
Let a[t,i] be accuracy on test task i after training stage t; T is the last stage.
Final average accuracy = mean over i=1..T of a[T,i].
Average forgetting = mean over i=1..T-1 of (max over t=i..T of a[t,i] - a[T,i]).
Backward transfer = mean over i=1..T-1 of (a[T,i] - a[i,i]).
Report new-task accuracy too: retaining old tasks by failing to learn new tasks is not success.

Log replay batches, replay examples, total optimizer updates, monitoring passes, total runtime, peak GPU allocation, buffer bytes, sleep timestamps, trigger values, seed, class order, exact configuration, code commit, package versions and hardware.

Aim for five paired seeds for final comparisons; pilot runtime before committing. Show individual runs, paired differences, mean and uncertainty, not just best runs. If claiming comparable accuracy with lower cost, choose an acceptable accuracy margin with the supervisor before final testing. No numeric improvement is assumed.

## Essential validity checks before experiments
Assert buffer capacity; no test examples enter training/probes; random timing does not depend on test outcomes; incoming sample counts match intended protocol; budgets are enforced; accuracy matrix and metrics have hand-checked examples; trigger-disabled runs reduce to the expected baseline.

## Results artifacts and remaining deliverables
Accuracy-versus-compute curves; old/new accuracy over time; trigger timeline; signal-versus-subsequent held-out loss association (analysis only); threshold/check-frequency sensitivity; runtime breakdown; seed-level results. Empty results should remain empty until measured.

The exploratory adaptive-trigger pilot and its raw seed-level artifact are
documented in [`../artifacts/PHASE_04_RESULTS.md`](../artifacts/PHASE_04_RESULTS.md).
The matched-replay, matched-total-compute, held-out development/final split,
and scale-up deliverables remain outstanding.
