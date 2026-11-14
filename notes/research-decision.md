# Research analyst assessment

## Recommendation
Study whether inexpensive forgetting signals improve the retention-versus-total-compute trade-off relative to well-tuned periodic replay. Keep the model, memory sampling, replay batch size, and sleep burst length fixed initially; vary only the trigger timing.

This is the most defensible starting direction for your current stated experience and locally verified NVIDIA RTX 3050 Laptop GPU with 4096 MiB VRAM. Feasibility is an assessment, not a benchmark. Begin with small classifiers; large-language-model sleep and reinforcement-learning dreaming are substantially broader projects.

| Candidate | Feasibility for first project | Main concern | Decision |
|---|---|---|---|
| Cheap loss signal versus gradient signal for replay timing | Good with small classifiers | Timing and adaptive replay already studied | Recommended empirical thesis |
| Adaptive start and adaptive end of sleep | Moderate | Two changing mechanisms obscure attribution | Extension after fixed-burst experiment |
| Generative sleep without stored examples | Lower | Generator training and quality confound scheduling | Defer |
| Reproduce LLM sleep and dreaming | Lower on current hardware | Distillation/RL/model scale | Background reading |

## Correction to earlier advice
The earlier claim that deciding when to replay could itself establish novelty was too strong. Learn the Time to Learn, TRAF, S-TRIGGER, and Smart Replay directly overlap scheduling or triggering. A full related-work comparison must precede any originality claim. The recommended contribution is a controlled, reproducible comparison and identification of conditions where monitoring pays for itself. A new trigger is a possible outcome, not an assumed contribution.

## Problems in the supplied draft
- Fixed replay wasting compute or responding too late is a hypothesis to test, not an established universal fact.
- Biological sleep is not adequately described as simply being triggered by synaptic fatigue and novelty. Remove this causal assertion unless supported by neuroscience sources.
- Negative gradient dot product predicts a local first-order conflict for an SGD-like update. It does not prove actual erasure, especially with momentum, adaptive optimizers, finite steps, or a noisy memory sample.
- A negative dot product is not a threshold on how strongly negative cosine similarity is. Normalize and specify the threshold.
- Loss stabilizing can mean staying at a poor level. Start with fixed replay bursts and a hard budget rather than a vague stop rule.
- Backward transfer and forgetting are distinct metrics, and neither should be defined using Task 1 alone.
- A proposal is a sensible first submission, but page limits, approval rules, and topic changes depend on your department. Approval does not universally lock a student into a topic.

## Decision checkpoints
1. Explain backpropagation, replay, and the comparison in your own words.
2. Demonstrate forgetting and replay benefits using reproducible baselines.
3. Test whether signals predict held-out deterioration beyond random timing.
4. Compare matched replay counts and matched total compute separately.
5. If adaptive replay loses, investigate monitoring overhead, buffer bias, or threshold sensitivity and report it honestly.
