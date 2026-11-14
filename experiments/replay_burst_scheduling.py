#!/usr/bin/env python3
"""Compare periodic and random replay bursts at a matched replay budget."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from split_mnist_baseline import (
    TASKS,
    DigitMLP,
    _reservoir_add,
    _sync,
    _task_accuracy,
    aggregate_runs,
    summarize_accuracy,
)

METHODS = ("periodic_replay", "random_replay")


def make_burst_schedule(
    method: str,
    incoming_steps: int,
    interval: int,
    rng: random.Random,
) -> list[int]:
    if method not in METHODS:
        raise ValueError(f"unknown replay schedule: {method}")
    if incoming_steps < 0 or interval < 1:
        raise ValueError("incoming_steps must be non-negative and interval positive")

    burst_count = incoming_steps // interval
    if method == "periodic_replay":
        return list(range(interval, incoming_steps + 1, interval))
    return sorted(rng.sample(range(1, incoming_steps + 1), burst_count))


def run_method(
    method: str,
    train_data: MNIST,
    test_data: MNIST,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    replay_batch_size: int,
    memory_capacity: int,
    learning_rate: float,
    burst_interval: int,
    burst_steps: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = DigitMLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    memory: list[tuple[torch.Tensor, int]] = []
    reservoir_rng = random.Random(seed + 1000)
    replay_rng = random.Random(seed + 2000)
    stream_seen = replay_examples = incoming_examples = 0
    incoming_optimizer_steps = replay_optimizer_steps = replay_bursts = 0
    accuracy_matrix: list[list[float | None]] = []
    new_task_accuracy: list[float] = []
    burst_positions_by_task: dict[str, list[int]] = {}

    _sync(device)
    started = time.perf_counter()
    for task_index, classes in enumerate(TASKS):
        indices = torch.where(
            (train_data.targets == classes[0]) | (train_data.targets == classes[1])
        )[0].tolist()
        generator = torch.Generator().manual_seed(seed + task_index)
        loader = DataLoader(
            Subset(train_data, indices),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        task_steps = epochs * len(loader)
        schedule_rng = random.Random(seed + 50_000 + task_index)
        positions = (
            make_burst_schedule(method, task_steps, burst_interval, schedule_rng)
            if task_index > 0
            else []
        )
        burst_positions_by_task[str(task_index)] = positions
        scheduled_positions = set(positions)
        task_step = 0

        model.train()
        for epoch in range(epochs):
            for images, labels in loader:
                task_step += 1
                images = images.to(device)
                labels = labels.to(device)
                incoming_examples += len(labels)

                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                incoming_optimizer_steps += 1

                if epoch == 0:
                    stream_seen = _reservoir_add(
                        memory,
                        images,
                        labels,
                        stream_seen,
                        memory_capacity,
                        reservoir_rng,
                    )

                if task_step in scheduled_positions:
                    if not memory:
                        raise RuntimeError("replay burst scheduled before memory exists")
                    replay_bursts += 1
                    for _ in range(burst_steps):
                        sampled = [
                            memory[replay_rng.randrange(len(memory))]
                            for _ in range(replay_batch_size)
                        ]
                        replay_images = torch.stack(
                            [item[0] for item in sampled]
                        ).to(device)
                        replay_labels = torch.tensor(
                            [item[1] for item in sampled],
                            dtype=torch.long,
                            device=device,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        replay_loss = criterion(model(replay_images), replay_labels)
                        replay_loss.backward()
                        optimizer.step()
                        replay_optimizer_steps += 1
                        replay_examples += replay_batch_size

        row: list[float | None] = [None] * len(TASKS)
        for evaluated_task, evaluated_classes in enumerate(TASKS[: task_index + 1]):
            row[evaluated_task] = _task_accuracy(
                model, test_data, evaluated_classes, device, batch_size * 4
            )
        accuracy_matrix.append(row)
        new_score = row[task_index]
        if new_score is None:
            raise RuntimeError(f"task {task_index} was not evaluated after training")
        new_task_accuracy.append(new_score)

    _sync(device)
    elapsed = time.perf_counter() - started
    summary = summarize_accuracy(accuracy_matrix)
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else None
    )
    return {
        "accuracy_matrix": accuracy_matrix,
        "new_task_accuracy": new_task_accuracy,
        **summary,
        "incoming_training_examples": incoming_examples,
        "incoming_optimizer_steps": incoming_optimizer_steps,
        "replay_examples": replay_examples,
        "replay_bursts": replay_bursts,
        "replay_optimizer_steps": replay_optimizer_steps,
        "optimizer_steps": incoming_optimizer_steps + replay_optimizer_steps,
        "replay_buffer_capacity": memory_capacity,
        "monitoring_passes": 0,
        "replay_burst_positions_by_task": burst_positions_by_task,
        "training_and_evaluation_seconds": elapsed,
        "peak_gpu_memory_mib": peak_memory,
    }


def make_plot(results: dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"periodic_replay": "#1b9e77", "random_replay": "#7570b3"}
    labels = {"periodic_replay": "Periodic bursts", "random_replay": "Random bursts"}

    for method in METHODS:
        matrix = results["methods"][method]["accuracy_matrix"]
        mean_accuracy = [
            statistics.mean(score for score in row if score is not None) * 100
            for row in matrix
        ]
        axes[0].plot(
            range(1, len(TASKS) + 1),
            mean_accuracy,
            marker="o",
            color=colors[method],
            label=labels[method],
        )
    axes[0].set(
        title="Mean accuracy across learned tasks",
        xlabel="Training stage",
        ylabel="Test accuracy (%)",
        xticks=range(1, len(TASKS) + 1),
        ylim=(0, 100),
    )
    axes[0].legend()

    metrics = ("final_average_accuracy", "average_forgetting")
    metric_labels = ("Final average accuracy", "Average forgetting")
    width = 0.36
    positions = range(len(METHODS))
    for metric, offset, label in zip(metrics, (-width / 2, width / 2), metric_labels):
        values = [results["methods"][method][metric] * 100 for method in METHODS]
        errors = [
            (results["methods"][method][f"{metric}_std"] or 0) * 100
            for method in METHODS
        ]
        axes[1].bar(
            [position + offset for position in positions],
            values,
            width,
            yerr=errors,
            capsize=4,
            label=label,
            alpha=1 if metric == metrics[0] else 0.55,
        )
    axes[1].set(
        title="Final accuracy and average forgetting",
        ylabel="Percentage points",
        xticks=list(positions),
        xticklabels=[labels[method] for method in METHODS],
        ylim=(0, 100),
    )
    axes[1].legend()
    fig.suptitle(
        f"Matched replay budget ({len(results['configuration']['seeds'])} paired seeds)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--out", default="artifacts/phase_02_replay_bursts.json")
    parser.add_argument("--plot", default="artifacts/phase_02_replay_bursts.png")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-batch-size", type=int, default=32)
    parser.add_argument("--memory-capacity", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--burst-interval", type=int, default=25)
    parser.add_argument("--burst-steps", type=int, default=1)
    args = parser.parse_args()

    if min(
        args.epochs,
        args.batch_size,
        args.replay_batch_size,
        args.memory_capacity,
        args.burst_interval,
        args.burst_steps,
    ) < 1:
        parser.error("epochs, batch sizes, capacities, and burst settings must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train_data = MNIST(args.data_dir, train=True, download=True, transform=ToTensor())
    test_data = MNIST(args.data_dir, train=False, download=True, transform=ToTensor())

    results: dict[str, Any] = {
        "experiment": "Phase 2: periodic versus random replay bursts",
        "scope": (
            "Exploratory paired-seed scheduling pilot. Burst timing is the only "
            "intended difference; this is not a matched-total-compute or final "
            "evaluation and does not test adaptive triggers or LLM sleep."
        ),
        "configuration": {
            "seeds": args.seeds,
            "task_order": TASKS,
            "epochs_per_task": args.epochs,
            "batch_size": args.batch_size,
            "memory_capacity": args.memory_capacity,
            "replay_batch_size": args.replay_batch_size,
            "burst_interval": args.burst_interval,
            "burst_steps": args.burst_steps,
            "learning_rate": args.learning_rate,
            "test_classes_masked": False,
            "buffer_policy": "reservoir; updated once per incoming training example",
            "warmup": "no replay during the first task",
            "random_schedule": "seeded uniform sample without replacement per task",
        },
        "environment": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "runs": {},
    }

    for seed in args.seeds:
        run = {"methods": {}}
        for method in METHODS:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            run["methods"][method] = run_method(
                method,
                train_data,
                test_data,
                device,
                seed,
                args.epochs,
                args.batch_size,
                args.replay_batch_size,
                args.memory_capacity,
                args.learning_rate,
                args.burst_interval,
                args.burst_steps,
            )
        results["runs"][str(seed)] = run

    results.update(
        aggregate_runs(
            results["runs"],
            methods_to_aggregate=METHODS,
            comparison=("periodic_replay", "random_replay"),
        )
    )
    out_path = Path(args.out)
    plot_path = Path(args.plot)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    make_plot(results, plot_path)
    print(json.dumps(results, indent=2))
    print(f"\nWrote {out_path} and {plot_path}")


if __name__ == "__main__":
    main()
