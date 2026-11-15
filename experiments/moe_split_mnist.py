#!/usr/bin/env python3
"""Compare dense and sparse-MoE models under the Phase 2 replay schedule."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from replay_burst_scheduling import make_burst_schedule
from split_mnist_baseline import (
    TASKS,
    DigitMLP,
    _reservoir_add,
    _sync,
    _task_accuracy,
    aggregate_runs,
    summarize_accuracy,
)

ARCHITECTURES = ("dense", "sparse_moe")
NUM_EXPERTS = 4
MOE_BOTTLENECK = 128
LOAD_BALANCE_COEFFICIENT = 0.01
REPLAY_METHOD = "periodic_replay"
DIAGNOSTIC_WINDOWS = 4


class SparseMoEClassifier(nn.Module):
    def __init__(
        self,
        num_experts: int = NUM_EXPERTS,
        bottleneck: int = MOE_BOTTLENECK,
    ) -> None:
        super().__init__()
        self.input_layer = nn.Linear(28 * 28, 256)
        self.router = nn.Linear(256, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(256, bottleneck),
                    nn.ReLU(),
                    nn.Linear(bottleneck, 256),
                )
                for _ in range(num_experts)
            ]
        )
        self.output_layer = nn.Linear(256, 10)
        self.num_experts = num_experts
        self.bottleneck = bottleneck

    def _forward_components(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.input_layer(images.flatten(start_dim=1)))
        probabilities = self.router(hidden).softmax(dim=-1)
        routes = probabilities.argmax(dim=-1)
        selected_gates = probabilities.gather(1, routes.unsqueeze(1))
        expert_outputs = torch.zeros_like(hidden)

        for expert_index, expert in enumerate(self.experts):
            token_indices = torch.where(routes == expert_index)[0]
            if token_indices.numel() == 0:
                continue
            selected = expert(hidden.index_select(0, token_indices))
            selected = selected * selected_gates.index_select(0, token_indices)
            expert_outputs = expert_outputs.index_copy(0, token_indices, selected)

        logits = self.output_layer(F.relu(expert_outputs))
        return logits, probabilities, routes

    def forward_with_auxiliary(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, probabilities, routes = self._forward_components(images)
        importance = probabilities.mean(dim=0)
        load = F.one_hot(routes, num_classes=self.num_experts).float().mean(dim=0)
        auxiliary_loss = self.num_experts * torch.sum(load.detach() * importance)
        return logits, auxiliary_loss, routes, probabilities

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self._forward_components(images)[0]

    def parameter_profile(self) -> dict[str, int | float]:
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        shared_parameters = (
            sum(parameter.numel() for parameter in self.input_layer.parameters())
            + sum(parameter.numel() for parameter in self.router.parameters())
            + sum(parameter.numel() for parameter in self.output_layer.parameters())
        )
        one_expert_parameters = sum(
            parameter.numel() for parameter in self.experts[0].parameters()
        )
        active_parameters = shared_parameters + one_expert_parameters
        macs_per_example = (
            28 * 28 * 256
            + 256 * self.num_experts
            + 256 * self.bottleneck
            + self.bottleneck * 256
            + 256 * 10
        )
        return {
            "total_parameters": total_parameters,
            "active_parameters_per_example": active_parameters,
            "estimated_forward_macs_per_example": macs_per_example,
            "num_experts": self.num_experts,
            "top_k": 1,
            "expert_bottleneck": self.bottleneck,
        }


def _model_and_profile(
    architecture: str, moe_bottleneck: int = MOE_BOTTLENECK
) -> tuple[nn.Module, dict[str, Any]]:
    if architecture == "dense":
        model = DigitMLP()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        macs_per_example = 28 * 28 * 256 + 256 * 256 + 256 * 10
        profile: dict[str, Any] = {
            "total_parameters": parameters,
            "active_parameters_per_example": parameters,
            "estimated_forward_macs_per_example": macs_per_example,
            "num_experts": 0,
            "top_k": 0,
            "expert_bottleneck": None,
        }
        return model, profile
    if architecture == "sparse_moe":
        model = SparseMoEClassifier(bottleneck=moe_bottleneck)
        return model, model.parameter_profile()
    raise ValueError(f"unknown architecture: {architecture}")


def _forward_with_auxiliary(
    model: nn.Module, images: torch.Tensor
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if isinstance(model, SparseMoEClassifier):
        return model.forward_with_auxiliary(images)
    logits = model(images)
    return logits, logits.new_zeros(()), None, None


def _route_summary(counts_by_task: list[list[int]]) -> dict[str, Any]:
    fractions = []
    normalized_entropies = []
    maximum_loads = []
    for counts in counts_by_task:
        total = sum(counts)
        shares = [count / total for count in counts] if total else [0.0] * len(counts)
        fractions.append(shares)
        positive = [share for share in shares if share > 0]
        entropy = -sum(share * math.log(share) for share in positive)
        normalized_entropies.append(entropy / math.log(len(counts)))
        maximum_loads.append(max(shares))
    return {
        "incoming_route_fraction_by_task": fractions,
        "normalized_route_entropy_by_task": normalized_entropies,
        "maximum_expert_load_by_task": maximum_loads,
    }


def _diagnostic_windows(
    route_counts: list[list[list[int]]],
    probability_sums: list[list[list[float]]],
    classification_loss_sums: list[list[float]],
    auxiliary_loss_sums: list[list[float]],
    example_counts: list[list[int]],
) -> list[dict[str, Any]]:
    task_diagnostics = []
    for task_index, task in enumerate(TASKS):
        windows = []
        for window_index in range(DIAGNOSTIC_WINDOWS):
            count = example_counts[task_index][window_index]
            routes = route_counts[task_index][window_index]
            probabilities = probability_sums[task_index][window_index]
            route_fraction = (
                [value / count for value in routes]
                if count
                else [0.0] * NUM_EXPERTS
            )
            mean_probability = (
                [value / count for value in probabilities]
                if count
                else [0.0] * NUM_EXPERTS
            )
            positive = [value for value in route_fraction if value > 0]
            entropy = -sum(value * math.log(value) for value in positive)
            normalized_entropy = entropy / math.log(NUM_EXPERTS)
            mean_classification_loss = (
                classification_loss_sums[task_index][window_index] / count
                if count
                else None
            )
            mean_auxiliary_loss = (
                auxiliary_loss_sums[task_index][window_index] / count
                if count
                else None
            )
            weighted_auxiliary_ratio = (
                LOAD_BALANCE_COEFFICIENT
                * mean_auxiliary_loss
                / mean_classification_loss
                if mean_classification_loss
                and mean_auxiliary_loss is not None
                else None
            )
            windows.append(
                {
                    "window": window_index,
                    "incoming_examples": count,
                    "route_fraction": route_fraction,
                    "mean_router_probability": mean_probability,
                    "normalized_route_entropy": normalized_entropy,
                    "maximum_expert_load": max(route_fraction),
                    "mean_classification_loss": mean_classification_loss,
                    "mean_unweighted_auxiliary_loss": mean_auxiliary_loss,
                    "weighted_auxiliary_to_classification_loss_ratio": weighted_auxiliary_ratio,
                }
            )
        task_diagnostics.append(
            {
                "task_index": task_index,
                "classes": task,
                "windows": windows,
            }
        )
    return task_diagnostics


def run_architecture(
    architecture: str,
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
    moe_bottleneck: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model, model_profile = _model_and_profile(architecture, moe_bottleneck)
    model = model.to(device)
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
    route_counts_by_task = torch.zeros(
        (len(TASKS), NUM_EXPERTS), dtype=torch.long, device=device
    )
    route_counts_by_window = torch.zeros(
        (len(TASKS), DIAGNOSTIC_WINDOWS, NUM_EXPERTS),
        dtype=torch.long,
        device=device,
    )
    probability_sums_by_window = torch.zeros(
        (len(TASKS), DIAGNOSTIC_WINDOWS, NUM_EXPERTS),
        dtype=torch.float64,
        device=device,
    )
    classification_loss_sums_by_window = torch.zeros(
        (len(TASKS), DIAGNOSTIC_WINDOWS), dtype=torch.float64, device=device
    )
    auxiliary_loss_sums_by_window = torch.zeros_like(
        classification_loss_sums_by_window
    )
    examples_by_window = torch.zeros(
        (len(TASKS), DIAGNOSTIC_WINDOWS), dtype=torch.long, device=device
    )
    replay_route_counts_by_task = torch.zeros(
        (len(TASKS), NUM_EXPERTS), dtype=torch.long, device=device
    )

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
            make_burst_schedule(
                REPLAY_METHOD, task_steps, burst_interval, schedule_rng
            )
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
                window_index = min(
                    (task_step - 1) * DIAGNOSTIC_WINDOWS // task_steps,
                    DIAGNOSTIC_WINDOWS - 1,
                )
                images = images.to(device)
                labels = labels.to(device)
                incoming_examples += len(labels)

                optimizer.zero_grad(set_to_none=True)
                logits, auxiliary_loss, routes, probabilities = _forward_with_auxiliary(
                    model, images
                )
                classification_loss = criterion(logits, labels)
                loss = classification_loss
                if architecture == "sparse_moe":
                    loss = loss + LOAD_BALANCE_COEFFICIENT * auxiliary_loss
                    if routes is None or probabilities is None:
                        raise RuntimeError("sparse MoE did not return routing diagnostics")
                    route_counts = torch.bincount(
                        routes.detach(), minlength=NUM_EXPERTS
                    )
                    route_counts_by_task[task_index] += route_counts
                    route_counts_by_window[task_index, window_index] += route_counts
                    probability_sums_by_window[task_index, window_index] += (
                        probabilities.detach().to(torch.float64).sum(dim=0)
                    )
                    classification_loss_sums_by_window[
                        task_index, window_index
                    ] += classification_loss.detach().to(torch.float64) * len(labels)
                    auxiliary_loss_sums_by_window[
                        task_index, window_index
                    ] += auxiliary_loss.detach().to(torch.float64) * len(labels)
                    examples_by_window[task_index, window_index] += len(labels)
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
                        (
                            replay_logits,
                            replay_auxiliary,
                            replay_routes,
                            replay_probabilities,
                        ) = _forward_with_auxiliary(model, replay_images)
                        replay_loss = criterion(replay_logits, replay_labels)
                        if architecture == "sparse_moe":
                            replay_loss = (
                                replay_loss
                                + LOAD_BALANCE_COEFFICIENT * replay_auxiliary
                            )
                            if replay_routes is None or replay_probabilities is None:
                                raise RuntimeError(
                                    "sparse MoE replay did not return routing diagnostics"
                                )
                            replay_route_counts_by_task[task_index] += torch.bincount(
                                replay_routes.detach(), minlength=NUM_EXPERTS
                            )
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
    route_summary = (
        _route_summary(route_counts_by_task.cpu().tolist())
        if architecture == "sparse_moe"
        else {
            "incoming_route_fraction_by_task": None,
            "normalized_route_entropy_by_task": None,
            "maximum_expert_load_by_task": None,
        }
    )
    diagnostics = (
        _diagnostic_windows(
            route_counts_by_window.cpu().tolist(),
            probability_sums_by_window.cpu().tolist(),
            classification_loss_sums_by_window.cpu().tolist(),
            auxiliary_loss_sums_by_window.cpu().tolist(),
            examples_by_window.cpu().tolist(),
        )
        if architecture == "sparse_moe"
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
        "model_profile": model_profile,
        "incoming_router_diagnostics_by_task": diagnostics,
        "replay_route_counts_by_incoming_task": (
            replay_route_counts_by_task.cpu().tolist()
            if architecture == "sparse_moe"
            else None
        ),
        **route_summary,
    }


def _aggregate_routing(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    methods = {}
    for architecture in ARCHITECTURES:
        entries = [
            run["methods"][architecture] for run in runs.values()
        ]
        route_matrices = [
            entry["incoming_route_fraction_by_task"]
            for entry in entries
        ]
        if route_matrices[0] is None:
            methods[architecture] = None
            continue
        mean_routes = []
        std_routes = []
        for task_index in range(len(TASKS)):
            mean_row = []
            std_row = []
            for expert_index in range(NUM_EXPERTS):
                values = [
                    matrix[task_index][expert_index] for matrix in route_matrices
                ]
                mean_row.append(statistics.mean(values))
                std_row.append(statistics.stdev(values) if len(values) > 1 else None)
            mean_routes.append(mean_row)
            std_routes.append(std_row)
        methods[architecture] = {
            "mean_route_fraction_by_task": mean_routes,
            "route_fraction_std_by_task": std_routes,
            "mean_normalized_route_entropy_by_task": [
                statistics.mean(
                    entry["normalized_route_entropy_by_task"][task_index]
                    for entry in entries
                )
                for task_index in range(len(TASKS))
            ],
            "mean_maximum_expert_load_by_task": [
                statistics.mean(
                    entry["maximum_expert_load_by_task"][task_index]
                    for entry in entries
                )
                for task_index in range(len(TASKS))
            ],
        }
    return methods


def make_plot(results: dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"dense": "#d95f02", "sparse_moe": "#1b9e77"}
    labels = {"dense": "Dense", "sparse_moe": "Sparse MoE"}

    for architecture in ARCHITECTURES:
        matrix = results["methods"][architecture]["accuracy_matrix"]
        mean_accuracy = [
            statistics.mean(score for score in row if score is not None) * 100
            for row in matrix
        ]
        axes[0, 0].plot(
            range(1, len(TASKS) + 1),
            mean_accuracy,
            marker="o",
            color=colors[architecture],
            label=labels[architecture],
        )
    axes[0, 0].set(
        title="Mean accuracy across learned tasks",
        xlabel="Training stage",
        ylabel="Test accuracy (%)",
        xticks=range(1, len(TASKS) + 1),
        ylim=(0, 100),
    )
    axes[0, 0].legend()

    positions = range(len(ARCHITECTURES))
    width = 0.36
    for metric, offset, label in (
        ("final_average_accuracy", -width / 2, "Final average accuracy"),
        ("average_forgetting", width / 2, "Average forgetting"),
    ):
        values = [
            results["methods"][architecture][metric] * 100
            for architecture in ARCHITECTURES
        ]
        errors = [
            (results["methods"][architecture][f"{metric}_std"] or 0) * 100
            for architecture in ARCHITECTURES
        ]
        axes[0, 1].bar(
            [position + offset for position in positions],
            values,
            width,
            yerr=errors,
            capsize=4,
            label=label,
            alpha=1 if metric == "final_average_accuracy" else 0.55,
        )
    axes[0, 1].set(
        title="Final accuracy and average forgetting",
        ylabel="Percentage points",
        xticks=list(positions),
        xticklabels=[labels[name] for name in ARCHITECTURES],
        ylim=(0, 100),
    )
    axes[0, 1].legend()

    route_summary = results["routing_summary"]["sparse_moe"]
    route_matrix = route_summary["mean_route_fraction_by_task"]
    image = axes[1, 0].imshow(route_matrix, vmin=0, vmax=1, cmap="viridis")
    axes[1, 0].set(
        title="Sparse MoE incoming-task routing",
        xlabel="Expert",
        ylabel="Training task",
        xticks=range(NUM_EXPERTS),
        yticks=range(len(TASKS)),
        yticklabels=[f"{a}–{b}" for a, b in TASKS],
    )
    fig.colorbar(image, ax=axes[1, 0], label="Mean fraction routed")

    profiles = [
        results["methods"][architecture]["model_profile"]
        for architecture in ARCHITECTURES
    ]
    macs = [
        profile["estimated_forward_macs_per_example"] / 1000
        for profile in profiles
    ]
    axes[1, 1].bar(
        [labels[name] for name in ARCHITECTURES],
        macs,
        color=[colors[name] for name in ARCHITECTURES],
    )
    axes[1, 1].set(
        title="Estimated active forward compute",
        ylabel="Multiply-accumulates / example (thousands)",
    )

    fig.suptitle(
        f"Dense vs. sparse MoE under matched periodic replay "
        f"({len(results['configuration']['seeds'])} paired seeds)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _git_commit() -> str:
    project_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--out", default="artifacts/phase_03_moe_split_mnist.json")
    parser.add_argument("--plot", default="artifacts/phase_03_moe_split_mnist.png")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-batch-size", type=int, default=32)
    parser.add_argument("--memory-capacity", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--burst-interval", type=int, default=25)
    parser.add_argument("--burst-steps", type=int, default=1)
    parser.add_argument(
        "--moe-bottleneck",
        type=int,
        default=MOE_BOTTLENECK,
        help="per-expert bottleneck width; 128 approximately matches dense active MACs",
    )
    args = parser.parse_args()

    if min(
        args.epochs,
        args.batch_size,
        args.replay_batch_size,
        args.memory_capacity,
        args.burst_interval,
        args.burst_steps,
        args.moe_bottleneck,
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
        "experiment": "Phase 3 diagnostic: active-compute-matched dense versus sparse MoE",
        "scope": (
            "Exploratory diagnostic follow-up to Phase 3. Dense and MoE active "
            "forward MACs are approximately matched; total MoE capacity is "
            "larger. This does not test adaptive replay, expert growth, permanent "
            "consolidation, or LLM sleep."
        ),
        "code_commit": _git_commit(),
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
            "replay_schedule": "periodic, no replay during first task",
            "moe_num_experts": NUM_EXPERTS,
            "moe_top_k": 1,
            "moe_expert_bottleneck": args.moe_bottleneck,
            "moe_load_balance_coefficient": LOAD_BALANCE_COEFFICIENT,
            "moe_load_balance_loss": "num_experts * sum(stop_gradient(load) * mean_router_probability)",
            "router_diagnostic_windows_per_task": DIAGNOSTIC_WINDOWS,
            "router_diagnostic_fields": [
                "hard route fraction",
                "mean router probability",
                "classification loss",
                "unweighted auxiliary loss",
                "weighted auxiliary-to-classification loss ratio",
            ],
            "compute_match": "active forward MACs approximately matched; total MoE parameters larger",
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
        for architecture in ARCHITECTURES:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            run["methods"][architecture] = run_architecture(
                architecture,
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
                args.moe_bottleneck,
            )
        results["runs"][str(seed)] = run

    results.update(
        aggregate_runs(
            results["runs"],
            methods_to_aggregate=ARCHITECTURES,
            comparison=("dense", "sparse_moe"),
        )
    )
    for architecture in ARCHITECTURES:
        entries = [
            run["methods"][architecture] for run in results["runs"].values()
        ]
        profiles = {json.dumps(entry["model_profile"], sort_keys=True) for entry in entries}
        if len(profiles) != 1:
            raise ValueError(f"model profile differs across seeds: {profiles}")
        results["methods"][architecture]["model_profile"] = entries[0][
            "model_profile"
        ]
    results["routing_summary"] = _aggregate_routing(results["runs"])

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
