# Reading map and source register

Evidence level: initial screening from primary abstracts and accessible article text, with selected PDFs downloaded for deeper study. This is not an exhaustive systematic review. Do not cite an abstract as proof of implementation details. Read methods, appendix, and evaluation protocol before reproducing.

| Priority | Paper/source | Why it matters |
|---|---|---|
| 1 | [Sleep-like unsupervised replay reduces catastrophic forgetting, Tadros et al., Nature Communications 2022](https://www.nature.com/articles/s41467-022-34938-7) | Sleep consolidation uses noisy inputs and Hebbian plasticity; distinct from supervised replay bursts. |
| 2 | [Learn the Time to Learn: Replay Scheduling, Klasson et al., TMLR 2023](https://openreview.net/forum?id=Q4aAITDgdP) | Schedules which historical tasks to replay; directly relevant to scheduling novelty. Preprint: https://arxiv.org/abs/2209.08660 |
| 3 | [Trust-Region Adaptive Frequency, Kong et al., IJCV 2023](https://doi.org/10.1007/s11263-023-01775-0) | Confidence controls frequency of memory-only updates; a particularly close comparator. |
| 4 | [S-TRIGGER](https://arxiv.org/abs/1902.09434) | Environment change detection triggers generative replay. Different learning setting, but invalidates broad first self-triggered replay claims. |
| 5 | [Maximal Interfered Retrieval, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html) | Selects memories harmed by a predicted update. Distinguish what to replay from when. |
| 6 | [Smart Replay, Chen et al., CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Smart_Replay_Adaptive_Scheduling_of_Memory_Rehearsal_for_Computational_Resource-Aware_CVPR_2026_paper.pdf) | Directly addresses adaptive scheduling under computational constraints; mandatory full-text comparison before method claims. |
| 7 | [Adaptive Memory Replay, Smith et al., CVPR Workshops 2024](https://openaccess.thecvf.com/content/CVPR2024W/ELVM/html/Smith_Adaptive_Memory_Replay_for_Continual_Learning_CVPRW_2024_paper.html) | Adaptive selection with access to all historical data; memory assumptions differ from a small fixed buffer. |
| 8 | [GRASP, Harun et al., 2025](https://proceedings.mlr.press/v274/harun25a.html) | Efficient sample selection is another route to fewer updates; don't attribute every gain to timing. |
| 9 | [Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories, Behrouz et al.](https://arxiv.org/abs/2606.03979) | The paper mentioned in your email. Consolidation/distillation and dreaming; background inspiration, not a small-classifier baseline. Treat as preprint unless a venue is independently confirmed. |

Additional screening leads: https://arxiv.org/abs/2606.08447 (sleep after several sequential tasks); https://www.sciencedirect.com/science/article/pii/S0031320326017097 (Learning to Replay, closed-loop timing/content). These need full-text/date/version verification before being used in the novelty argument.

## Existing implementations to inspect before coding
- https://github.com/aimagelab/mammoth — experience replay and other continual-learning methods; inspect models/er.py, licensing, version, and benchmark definitions.
- https://github.com/ContinualAI/avalanche — established continual-learning training and evaluation components.
- MIR implementation is linked by the original NeurIPS paper. Follow the publication link and verify current repository availability.

Select one framework after a minimal baseline run. Pin its commit and dependencies. Do not combine frameworks with differing evaluation assumptions.

## Paper notes template
For each paper write: problem; setting; data and memory access; what changes (sampling, timing, duration, objective); trigger inputs; monitoring cost; baselines; evaluation masks/task labels; strongest result; limitations; exact page/section supporting each statement; relation to our question; something you still do not understand.

Search terms used: adaptive replay scheduling; replay frequency; forgetting-triggered replay; gradient interference; sleep consolidation; computational resource-aware incremental learning. Search primary publishers, arXiv, OpenReview, and author-linked repositories. Record rejected or duplicate candidates too.
