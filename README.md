# Your first continual-learning research project

Prepared 14 November 2026. Status: proposal plus one exploratory Split-MNIST pilot; not final evaluation.

Start with [the research decision](notes/research-decision.md), then [the beginner guide](notes/learning-guide.md), [the literature map](notes/literature-map.md), and [the proposal](proposal/proposal.md). The experimental protocol and first executable baseline are in [experiments/](experiments/README.md).

## Working title
**When Is Replay Worth Its Cost? Evaluating Forgetting-Triggered Consolidation in Continual Learning**

This is a proposed undergraduate research direction, not a claim of a new algorithm or a completed thesis. Supervisor requirements determine the final format and scope. A negative result can still support a useful thesis if the question, controls, and evidence are sound.

## Next milestone
The first Split-MNIST sequential-versus-uniform-replay pilot is recorded in [the results report](artifacts/RESULTS.md). It uses one seed and one epoch, so it only validates the pipeline and illustrates forgetting. Next, run paired seeds and tune the baseline before adding an adaptive trigger.

The downloaded paper corpus is local-only and is not included in this repository. See the [literature map](notes/literature-map.md) for public source links. Do not treat the pilot as a final thesis result.
