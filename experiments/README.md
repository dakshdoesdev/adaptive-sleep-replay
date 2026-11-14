# First executable baseline

`split_mnist_baseline.py` compares sequential training with uniform experience
replay on class-incremental Split-MNIST. It is the first baseline in the
project's [experimental protocol](protocol.md), not an adaptive-scheduling
result and not a reproduction of an LLM sleep paper.

The pilot uses one paired seed to verify the pipeline and show whether
sequential learning forgets earlier digit pairs. Uniform replay adds 32
remembered examples to each incoming minibatch, so it performs more work.
This pilot does not claim a compute-matched comparison or a general result.
The five-seed one- and three-epoch pilots and their interpretation are in
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

MNIST is downloaded to `data/mnist/`. The script writes JSON metrics and a PNG
plot under `artifacts/`. Do not commit the downloaded dataset. The default
run trains on all MNIST training examples for three epochs per task.

Before thesis claims, extend this pilot with multiple paired seeds,
development-only tuning, matched-compute comparisons, trigger monitoring
costs, and held-out final evaluation as specified in `protocol.md`.
