#!/usr/bin/env python3
"""Paired Split-MNIST pilot: sequential learning versus uniform experience replay."""

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
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
METHODS = ("sequential", "uniform_replay")


class DigitMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


def summarize_accuracy(matrix: list[list[float | None]]) -> dict[str, float]:
    final = [score for score in matrix[-1] if score is not None]
    forgetting = []
    for task_index in range(len(matrix) - 1):
        final_score = matrix[-1][task_index]
        observed = [
            row[task_index]
            for row in matrix[task_index:]
            if row[task_index] is not None
        ]
        if final_score is not None and observed:
            forgetting.append(max(observed) - final_score)
    return {
        "final_average_accuracy": sum(final) / len(final),
        "average_forgetting": sum(forgetting) / len(forgetting) if forgetting else 0.0,
    }


def aggregate_runs(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    methods = {}
    for method in METHODS:
        entries = [run["methods"][method] for run in runs.values()]
        matrices = [entry["accuracy_matrix"] for entry in entries]
        mean_matrix: list[list[float | None]] = []
        std_matrix: list[list[float | None]] = []
        for stage in range(len(TASKS)):
            mean_row: list[float | None] = []
            std_row: list[float | None] = []
            for task_index in range(len(TASKS)):
                scores = [
                    matrix[stage][task_index]
                    for matrix in matrices
                    if matrix[stage][task_index] is not None
                ]
                mean_row.append(statistics.mean(scores) if scores else None)
                std_row.append(statistics.stdev(scores) if len(scores) > 1 else None)
            mean_matrix.append(mean_row)
            std_matrix.append(std_row)

        result: dict[str, Any] = {
            "accuracy_matrix": mean_matrix,
            "accuracy_std_matrix": std_matrix,
        }
        for metric in ("final_average_accuracy", "average_forgetting"):
            values = [entry[metric] for entry in entries]
            result[metric] = statistics.mean(values)
            result[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        new_task_values = [
            score for entry in entries for score in entry["new_task_accuracy"]
        ]
        result["mean_new_task_accuracy"] = statistics.mean(new_task_values)
        result["new_task_accuracy_by_stage"] = [
            statistics.mean(entry["new_task_accuracy"][stage] for entry in entries)
            for stage in range(len(TASKS))
        ]
        for metric in (
            "incoming_training_examples",
            "replay_examples",
            "optimizer_steps",
            "replay_buffer_capacity",
        ):
            values = {entry[metric] for entry in entries}
            if len(values) != 1:
                raise ValueError(f"{metric} differs across paired seeds: {values}")
            result[metric] = values.pop()
        result["mean_training_and_evaluation_seconds"] = statistics.mean(
            entry["training_and_evaluation_seconds"] for entry in entries
        )
        methods[method] = result

    paired_differences = {}
    for metric in ("final_average_accuracy", "average_forgetting"):
        deltas = {
            seed: runs[seed]["methods"]["uniform_replay"][metric]
            - runs[seed]["methods"]["sequential"][metric]
            for seed in runs
        }
        paired_differences[metric] = {
            "uniform_replay_minus_sequential_by_seed": deltas,
            "mean": statistics.mean(deltas.values()),
            "std": statistics.stdev(deltas.values()) if len(deltas) > 1 else None,
        }
    return {"methods": methods, "paired_differences": paired_differences}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reservoir_add(
    memory: list[tuple[torch.Tensor, int]],
    images: torch.Tensor,
    labels: torch.Tensor,
    stream_seen: int,
    capacity: int,
    rng: random.Random,
) -> int:
    for image, label in zip(images, labels):
        stream_seen += 1
        item = (image.detach().cpu(), int(label))
        if len(memory) < capacity:
            memory.append(item)
        else:
            slot = rng.randrange(stream_seen)
            if slot < capacity:
                memory[slot] = item
    return stream_seen


@torch.no_grad()
def _task_accuracy(
    model: nn.Module,
    dataset: MNIST,
    classes: tuple[int, int],
    device: torch.device,
    batch_size: int,
) -> float:
    targets = dataset.targets
    indices = torch.where((targets == classes[0]) | (targets == classes[1]))[0].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    correct = total = 0
    model.eval()
    for images, labels in loader:
        logits = model(images.to(device))
        correct += int((logits.argmax(dim=1).cpu() == labels).sum())
        total += len(labels)
    return correct / total


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
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = DigitMLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    memory: list[tuple[torch.Tensor, int]] = []
    reservoir_rng = random.Random(seed + 1000)
    replay_rng = random.Random(seed + 2000)
    stream_seen = replay_examples = optimizer_steps = incoming_examples = 0
    accuracy_matrix: list[list[float | None]] = []
    new_task_accuracy: list[float] = []

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
        model.train()
        for epoch in range(epochs):
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                train_images, train_labels = images, labels
                incoming_examples += len(labels)

                if method == "uniform_replay" and memory:
                    count = replay_batch_size
                    sampled = [memory[replay_rng.randrange(len(memory))] for _ in range(count)]
                    replay_images = torch.stack([item[0] for item in sampled]).to(device)
                    replay_labels = torch.tensor(
                        [item[1] for item in sampled], dtype=torch.long, device=device
                    )
                    train_images = torch.cat((images, replay_images), dim=0)
                    train_labels = torch.cat((labels, replay_labels), dim=0)
                    replay_examples += count

                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(train_images), train_labels)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1

                if method == "uniform_replay" and epoch == 0:
                    stream_seen = _reservoir_add(
                        memory,
                        images,
                        labels,
                        stream_seen,
                        memory_capacity,
                        reservoir_rng,
                    )

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
        "replay_examples": replay_examples,
        "optimizer_steps": optimizer_steps,
        "replay_buffer_capacity": memory_capacity if method == "uniform_replay" else 0,
        "training_and_evaluation_seconds": elapsed,
        "peak_gpu_memory_mib": peak_memory,
    }


def make_plot(results: dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"sequential": "#d95f02", "uniform_replay": "#1b9e77"}
    labels = {"sequential": "Sequential", "uniform_replay": "Uniform replay"}

    for method in METHODS:
        matrix = results["methods"][method]["accuracy_matrix"]
        for task_index in range(len(TASKS)):
            stages = [
                stage
                for stage, row in enumerate(matrix, start=1)
                if row[task_index] is not None
            ]
            scores = [
                row[task_index] * 100
                for row in matrix
                if row[task_index] is not None
            ]
            axes[0].plot(
                stages,
                scores,
                marker="o",
                color=plt.cm.tab10(task_index),
                linestyle="-" if method == "sequential" else "--",
                label=labels[method] if task_index == 0 else None,
            )
    axes[0].set(
        title="Accuracy on each learned task",
        xlabel="Training stage",
        ylabel="Test accuracy (%)",
        xticks=range(1, len(TASKS) + 1),
        ylim=(0, 100),
    )
    method_handles = [
        Line2D([0], [0], color="black", linestyle="-", label="Sequential"),
        Line2D([0], [0], color="black", linestyle="--", label="Uniform replay"),
    ]
    task_handles = [
        Line2D(
            [0],
            [0],
            color=plt.cm.tab10(index),
            label=f"Digits {classes[0]}–{classes[1]}",
        )
        for index, classes in enumerate(TASKS)
    ]
    first_legend = axes[0].legend(handles=method_handles, loc="lower left")
    axes[0].add_artist(first_legend)
    axes[0].legend(handles=task_handles, loc="lower right", fontsize=8)

    names = [labels[method] for method in METHODS]
    averages = [
        results["methods"][method]["final_average_accuracy"] * 100
        for method in METHODS
    ]
    forgetting = [
        results["methods"][method]["average_forgetting"] * 100 for method in METHODS
    ]
    positions = range(len(METHODS))
    width = 0.36
    accuracy_errors = [
        results["methods"][method]["final_average_accuracy_std"] or 0
        for method in METHODS
    ]
    forgetting_errors = [
        results["methods"][method]["average_forgetting_std"] or 0
        for method in METHODS
    ]
    axes[1].bar(
        [position - width / 2 for position in positions],
        averages,
        width,
        color=[colors[method] for method in METHODS],
        yerr=[value * 100 for value in accuracy_errors],
        capsize=4,
        label="Final average accuracy",
    )
    axes[1].bar(
        [position + width / 2 for position in positions],
        forgetting,
        width,
        color=[colors[method] for method in METHODS],
        alpha=0.4,
        yerr=[value * 100 for value in forgetting_errors],
        capsize=4,
        label="Average forgetting",
    )
    axes[1].set(
        title="Final accuracy and average forgetting",
        ylabel="Percentage points",
        xticks=list(positions),
        xticklabels=names,
        ylim=(0, 100),
    )
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].legend()
    fig.suptitle(
        f"Split-MNIST pilot ({len(results['configuration']['seeds'])} paired seed(s); "
        "replay adds training work)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--out", default="artifacts/split_mnist_pilot.json")
    parser.add_argument("--plot", default="artifacts/split_mnist_pilot.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-batch-size", type=int, default=32)
    parser.add_argument("--memory-capacity", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.memory_capacity < 1:
        parser.error("epochs, batch size, and memory capacity must be positive")
    if args.replay_batch_size < 1:
        parser.error("replay batch size must be positive")
    seeds = args.seeds or [args.seed]
    if len(set(seeds)) != len(seeds):
        parser.error("seeds must be unique")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train_data = MNIST(args.data_dir, train=True, download=True, transform=ToTensor())
    test_data = MNIST(args.data_dir, train=False, download=True, transform=ToTensor())

    results: dict[str, Any] = {
        "experiment": "Split-MNIST sequential versus uniform replay pilot",
        "scope": (
            "One paired seed for baseline debugging; not a final evaluation and "
            "not a test of adaptive scheduling or LLM sleep."
        ),
        "configuration": {
            "seeds": seeds,
            "task_order": TASKS,
            "epochs_per_task": args.epochs,
            "batch_size": args.batch_size,
            "memory_capacity": args.memory_capacity,
            "replay_batch_size": args.replay_batch_size,
            "learning_rate": args.learning_rate,
            "test_classes_masked": False,
        },
        "environment": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "runs": {},
    }

    for seed in seeds:
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
            )
        results["runs"][str(seed)] = run
    results.update(aggregate_runs(results["runs"]))

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
