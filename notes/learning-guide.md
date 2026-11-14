# Understand your thesis

Imagine learning digits 0 and 1, then 2 and 3, and continuing until 9. The same weights must support every digit. Updates that help the new digits can damage decisions for earlier digits. That observed performance decline is forgetting.

A replay buffer is a small collection of previous training examples. Replaying them reminds the model what earlier decisions should look like. A sleep phase here means temporarily training on that buffer. This is an engineering analogy; ordinary supervised rehearsal is not a biological simulation.

A gradient points toward increasing loss. Gradient descent moves against it. If gradients on old and new data point in conflicting directions, an update helping one can locally hurt the other. This prediction can be noisy. Measuring a memory gradient requires additional work, which can cancel the savings from skipping replay.

A loss probe asks: how badly is the model currently predicting a small set of remembered examples? The probe is available training memory, never the final test set. Its loss can be unrepresentative if the buffer is tiny or repeatedly memorized.

Your central experiment asks: does checking for trouble and replaying only then beat replaying every few steps, after paying for the checks?

## Learning sequence, paced by understanding
1. Train a small classifier: explain input, output, loss, optimizer, train/validation/test separation.
2. Explain a forward pass, backward pass, batch, epoch, and learning rate. Check gradients on a tiny example.
3. Train sequentially on two digit groups; plot old and new accuracy.
4. Add a small replay buffer; show the same plot with replay.
5. Read SRC, Replay Scheduling, and TRAF using the notes template.
6. Add a loss trigger, then a gradient trigger; compare under the protocol.
7. Write methods from the actual code and results from saved logs.

For each step, write a paragraph yourself before using an assistant to improve the wording. Keep original notes, citations, code versions, and failed runs. Your ownership comes from understanding the decisions and being able to reproduce and defend them.

## First supervisor discussion
Bring a short proposal and a literature comparison. Ask about degree-specific thesis requirements, expected originality, available time, supervision, and acceptable compute. These requirements have not been confirmed from your inbox. Do not promise a positive result or publication.

## Suggested working sequence
Week 1: fundamentals and paper notes. Week 2: baseline reproduction. Week 3: trigger pilot and cost instrumentation. Week 4: freeze primary protocol. Weeks 5–6: repeated experiments. Week 7: analysis and failure cases. Week 8: draft and supervisor revision. This is a provisional schedule, not an estimate of your personal learning speed.
