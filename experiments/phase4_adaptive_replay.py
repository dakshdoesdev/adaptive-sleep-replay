#!/usr/bin/env python3
"""Compare periodic replay with fixed-budget loss and gradient triggers."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
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

METHODS = ("periodic_replay", "loss_triggered", "gradient_conflict")
PERIODIC_BUDGET = 43
PROBE_SIZE = 32


def loss_spike(
    current_loss: float, reference_loss: float, relative_threshold: float
) -> bool:
    if relative_threshold < 0:
        raise ValueError("relative_threshold must be non-negative")
    return current_loss > reference_loss * (1.0 + relative_threshold)


def accept_adaptive_trigger(
    triggered: bool,
    global_step: int,
    last_trigger_step: int | None,
    bursts_used: int,
    max_bursts: int,
    cooldown: int,
) -> bool:
    if max_bursts < 0 or cooldown < 0:
        raise ValueError("burst budget and cooldown must be non-negative")
    return (
        triggered
        and bursts_used < max_bursts
        and (
            last_trigger_step is None
            or global_step - last_trigger_step >= cooldown
        )
    )


def reserve_probe(
    memory: list[tuple[torch.Tensor, int]],
    probe_size: int,
    rng: random.Random,
) -> tuple[list[tuple[torch.Tensor, int]], list[tuple[torch.Tensor, int]]]:
    if not 0 < probe_size < len(memory):
        raise ValueError("probe_size must be positive and smaller than memory")
    selected = set(rng.sample(range(len(memory)), probe_size))
    probe = [item for index, item in enumerate(memory) if index in selected]
    replay_memory = [item for index, item in enumerate(memory) if index not in selected]
    return replay_memory, probe


def gradient_cosine(
    current_gradients: tuple[torch.Tensor | None, ...],
    memory_gradients: tuple[torch.Tensor | None, ...],
    epsilon: float = 1e-12,
) -> float | None:
    dot = None
    current_norm_squared = None
    memory_norm_squared = None
    for current, memory in zip(current_gradients, memory_gradients):
        if current is None or memory is None:
            continue
        current_flat = current.detach().reshape(-1)
        memory_flat = memory.detach().reshape(-1)
        pair_dot = torch.dot(current_flat, memory_flat)
        pair_current_norm = torch.dot(current_flat, current_flat)
        pair_memory_norm = torch.dot(memory_flat, memory_flat)
        dot = pair_dot if dot is None else dot + pair_dot
        current_norm_squared = (
            pair_current_norm
            if current_norm_squared is None
            else current_norm_squared + pair_current_norm
        )
        memory_norm_squared = (
            pair_memory_norm
            if memory_norm_squared is None
            else memory_norm_squared + pair_memory_norm
        )
    if (
        dot is None
        or current_norm_squared is None
        or memory_norm_squared is None
    ):
        return None
    current_norm = current_norm_squared.sqrt()
    memory_norm = memory_norm_squared.sqrt()
    if current_norm.item() <= epsilon or memory_norm.item() <= epsilon:
        return None
    return float((dot / (current_norm * memory_norm + epsilon)).item())


def _probe_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    return criterion(model(images), labels)


def _sample_batch(
    memory: list[tuple[torch.Tensor, int]],
    probe: list[tuple[torch.Tensor, int]],
    replay_batch_size: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pool = memory + probe
    if not pool:
        raise RuntimeError("replay requested before the buffer was populated")
    sampled = [pool[rng.randrange(len(pool))] for _ in range(replay_batch_size)]
    images = torch.stack([item[0] for item in sampled]).to(device)
    labels = torch.tensor(
        [item[1] for item in sampled], dtype=torch.long, device=device
    )
    return images, labels


def _flat_gradient(
    gradients: tuple[torch.Tensor | None, ...],
) -> tuple[torch.Tensor | None, ...]:
    return tuple(None if gradient is None else gradient.detach() for gradient in gradients)


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
    burst_interval: int,
    check_interval: int,
    cooldown: int,
    max_bursts: int,
    loss_relative_threshold: float,
    loss_ema_beta: float,
    gradient_threshold: float,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unknown replay method: {method}")
    if memory_capacity <= PROBE_SIZE:
        raise ValueError("memory capacity must exceed the fixed probe size")

    torch.manual_seed(seed)
    model = DigitMLP().to(device)
    parameters = tuple(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()
    memory: list[tuple[torch.Tensor, int]] = []
    probe: list[tuple[torch.Tensor, int]] = []
    reservoir_rng = random.Random(seed + 1000)
    replay_rng = random.Random(seed + 2000)
    probe_rng = random.Random(seed + 3000)
    stream_seen = incoming_examples = incoming_steps = 0
    replay_examples = replay_steps = replay_bursts = 0
    probe_forwards = probe_backward_examples = trigger_checks = 0
    skipped_near_zero_gradients = 0
    incoming_training_seconds = replay_training_seconds = 0.0
    trigger_seconds = evaluation_seconds = 0.0
    signal_log: list[dict[str, Any]] = []
    accuracy_matrix: list[list[float | None]] = []
    new_task_accuracy: list[float] = []
    burst_positions_by_task: dict[str, list[int]] = {}
    loss_ema: float | None = None
    last_trigger_step: int | None = None

    _sync(device)
    run_started = time.perf_counter()
    for task_index, classes in enumerate(TASKS):
        model.train()
        task_indices = torch.where(
            (train_data.targets == classes[0]) | (train_data.targets == classes[1])
        )[0].tolist()
        loader = DataLoader(
            Subset(train_data, task_indices),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + task_index),
            num_workers=0,
        )
        task_steps = epochs * len(loader)
        periodic_positions = (
            make_burst_schedule(
                "periodic_replay",
                task_steps,
                burst_interval,
                random.Random(seed + 50_000 + task_index),
            )
            if task_index > 0
            else []
        )
        periodic_positions_set = set(periodic_positions)
        burst_positions_by_task[str(task_index)] = []
        task_step = 0
        for epoch in range(epochs):
            for images, labels in loader:
                task_step += 1
                global_step = incoming_steps + 1
                images = images.to(device)
                labels = labels.to(device)
                incoming_examples += len(labels)

                incoming_step_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                incoming_loss = criterion(logits, labels)
                should_check = (
                    method != "periodic_replay"
                    and task_index > 0
                    and task_step % check_interval == 0
                    and replay_bursts < max_bursts
                )
                trigger = False
                signal_value: float | None = None
                reference_value: float | None = None
                incoming_gradient_elapsed = 0.0
                incoming_gradients: tuple[torch.Tensor | None, ...] | None = None

                if should_check:
                    trigger_checks += 1
                    if method == "loss_triggered":
                        _sync(device)
                        incoming_training_seconds += (
                            time.perf_counter() - incoming_step_started
                        )
                        monitor_started = time.perf_counter()
                        with torch.no_grad():
                            current_probe_loss = float(
                                _probe_loss(
                                    model,
                                    probe_images,
                                    probe_labels,
                                    criterion,
                                ).item()
                            )
                        probe_forwards += len(probe_labels)
                        reference_value = loss_ema
                        signal_value = current_probe_loss
                        if loss_ema is not None:
                            trigger = loss_spike(
                                current_probe_loss,
                                loss_ema,
                                loss_relative_threshold,
                            )
                            loss_ema = (
                                loss_ema_beta * loss_ema
                                + (1.0 - loss_ema_beta) * current_probe_loss
                            )
                        else:
                            loss_ema = current_probe_loss
                        _sync(device)
                        trigger_seconds += time.perf_counter() - monitor_started
                    else:
                        _sync(device)
                        incoming_training_seconds += (
                            time.perf_counter() - incoming_step_started
                        )
                        incoming_gradient_started = time.perf_counter()
                        incoming_gradients = _flat_gradient(
                            torch.autograd.grad(
                                incoming_loss,
                                parameters,
                                retain_graph=False,
                                allow_unused=True,
                            )
                        )
                        _sync(device)
                        incoming_training_seconds += (
                            time.perf_counter() - incoming_gradient_started
                        )
                        monitor_started = time.perf_counter()
                        with torch.enable_grad():
                            memory_loss = _probe_loss(
                                model,
                                probe_images,
                                probe_labels,
                                criterion,
                            )
                            memory_gradients = _flat_gradient(
                                torch.autograd.grad(
                                    memory_loss,
                                    parameters,
                                    retain_graph=False,
                                    allow_unused=True,
                                )
                            )
                        probe_forwards += len(probe_labels)
                        probe_backward_examples += len(probe_labels)
                        signal_value = gradient_cosine(
                            incoming_gradients, memory_gradients
                        )
                        if signal_value is None:
                            skipped_near_zero_gradients += 1
                        else:
                            trigger = signal_value < -gradient_threshold
                        _sync(device)
                        trigger_seconds += time.perf_counter() - monitor_started
                    incoming_step_started = time.perf_counter()

                if incoming_gradients is None:
                    incoming_loss.backward()
                    incoming_steps_gradients: tuple[
                        torch.Tensor | None, ...
                    ] | None = None
                else:
                    incoming_steps_gradients = incoming_gradients
                if incoming_steps_gradients is not None:
                    for parameter, gradient in zip(parameters, incoming_steps_gradients):
                        parameter.grad = None if gradient is None else gradient.clone()
                optimizer.step()
                _sync(device)
                incoming_training_seconds += time.perf_counter() - incoming_step_started
                incoming_steps += 1

                if epoch == 0:
                    stream_seen = _reservoir_add(
                        memory,
                        images,
                        labels,
                        stream_seen,
                        memory_capacity,
                        reservoir_rng,
                    )

                if method == "periodic_replay":
                    trigger = task_step in periodic_positions_set
                elif should_check:
                    trigger = accept_adaptive_trigger(
                        trigger,
                        global_step,
                        last_trigger_step,
                        replay_bursts,
                        max_bursts,
                        cooldown,
                    )
                if trigger and replay_bursts < max_bursts:
                    burst_positions_by_task[str(task_index)].append(task_step)
                    replay_bursts += 1
                    last_trigger_step = global_step
                    replay_started = time.perf_counter()
                    replay_images, replay_labels = _sample_batch(
                        memory,
                        probe,
                        replay_batch_size,
                        replay_rng,
                        device,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    replay_loss = criterion(model(replay_images), replay_labels)
                    replay_loss.backward()
                    optimizer.step()
                    _sync(device)
                    replay_training_seconds += time.perf_counter() - replay_started
                    replay_steps += 1
                    replay_examples += replay_batch_size

                if should_check:
                    signal_log.append(
                        {
                            "task_index": task_index,
                            "task_step": task_step,
                            "global_incoming_step": global_step,
                            "signal": signal_value,
                            "ema_reference": reference_value,
                            "triggered": bool(
                                task_step in burst_positions_by_task[str(task_index)]
                            ),
                            "replay_budget_used": replay_bursts,
                        }
                    )

        if task_index == 0:
            if len(memory) <= PROBE_SIZE:
                raise RuntimeError("not enough reservoir examples to create probe")
            memory, probe = reserve_probe(memory, PROBE_SIZE, probe_rng)
            if len(memory) + len(probe) != memory_capacity:
                raise RuntimeError("replay buffer plus fixed probe does not match capacity")
            probe_images = torch.stack([item[0] for item in probe]).to(device)
            probe_labels = torch.tensor(
                [item[1] for item in probe], dtype=torch.long, device=device
            )
            if method == "loss_triggered":
                _sync(device)
                baseline_started = time.perf_counter()
                with torch.no_grad():
                    loss_ema = float(
                        _probe_loss(
                            model, probe_images, probe_labels, criterion
                        ).item()
                    )
                _sync(device)
                trigger_seconds += time.perf_counter() - baseline_started
                probe_forwards += len(probe_labels)

        evaluation_started = time.perf_counter()
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
        evaluation_seconds += time.perf_counter() - evaluation_started

    _sync(device)
    total_seconds = time.perf_counter() - run_started
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
        "incoming_optimizer_steps": incoming_steps,
        "replay_examples": replay_examples,
        "replay_bursts": replay_bursts,
        "replay_optimizer_steps": replay_steps,
        "optimizer_steps": incoming_steps + replay_steps,
        "replay_buffer_capacity": memory_capacity,
        "replay_memory_capacity_after_probe": memory_capacity - PROBE_SIZE,
        "probe_size": len(probe),
        "probe_stored_within_buffer": True,
        "monitoring_passes": trigger_checks,
        "probe_forward_examples": probe_forwards,
        "probe_backward_examples": probe_backward_examples,
        "skipped_near_zero_gradients": skipped_near_zero_gradients,
        "trigger_checks": trigger_checks,
        "trigger_log": signal_log,
        "replay_burst_positions_by_task": burst_positions_by_task,
        "incoming_training_seconds": incoming_training_seconds,
        "trigger_monitoring_seconds": trigger_seconds,
        "replay_training_seconds": replay_training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": total_seconds,
        "training_and_evaluation_seconds": total_seconds,
        "peak_gpu_memory_mib": peak_memory,
    }


def _add_paired_comparisons(
    results: dict[str, Any],
    runs: dict[str, dict[str, Any]],
) -> None:
    for candidate in METHODS[1:]:
        for metric in ("final_average_accuracy", "average_forgetting"):
            deltas = {
                seed: runs[seed]["methods"][candidate][metric]
                - runs[seed]["methods"]["periodic_replay"][metric]
                for seed in runs
            }
            results["paired_differences"][metric][
                f"{candidate}_minus_periodic_replay_by_seed"
            ] = deltas
            results["paired_differences"][metric][candidate] = {
                "mean": statistics.mean(deltas.values()),
                "std": (
                    statistics.stdev(deltas.values())
                    if len(deltas) > 1
                    else None
                ),
            }


def make_plot(results: dict[str, Any], out_path: Path) -> None:
    labels = {
        "periodic_replay": "Periodic",
        "loss_triggered": "Probe-loss",
        "gradient_conflict": "Gradient-conflict",
    }
    colors = {
        "periodic_replay": "#1b9e77",
        "loss_triggered": "#d95f02",
        "gradient_conflict": "#7570b3",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for method in METHODS:
        values = results["methods"][method]
        axes[0].bar(
            labels[method],
            values["final_average_accuracy"] * 100,
            yerr=(values["final_average_accuracy_std"] or 0) * 100,
            color=colors[method],
            capsize=4,
        )
        axes[1].bar(
            labels[method],
            values["replay_bursts"],
            yerr=(values.get("replay_bursts_std") or 0),
            color=colors[method],
            capsize=4,
        )
    axes[0].set(title="Final average accuracy", ylabel="Accuracy (%)", ylim=(0, 100))
    axes[1].set(
        title="Replay bursts used",
        ylabel="Bursts (maximum 43)",
        ylim=(0, PERIODIC_BUDGET + 5),
    )
    fig.suptitle("Adaptive replay trigger pilot (five paired seeds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--out", default="artifacts/phase_04_adaptive_replay_rtx3050.json")
    parser.add_argument("--plot", default="artifacts/phase_04_adaptive_replay_rtx3050.png")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-batch-size", type=int, default=32)
    parser.add_argument("--memory-capacity", type=int, default=500)
    parser.add_argument("--burst-interval", type=int, default=25)
    parser.add_argument("--check-interval", type=int, default=10)
    parser.add_argument("--cooldown", type=int, default=25)
    parser.add_argument("--max-bursts", type=int, default=PERIODIC_BUDGET)
    parser.add_argument("--loss-relative-threshold", type=float, default=0.10)
    parser.add_argument("--loss-ema-beta", type=float, default=0.90)
    parser.add_argument("--gradient-threshold", type=float, default=0.0)
    args = parser.parse_args()

    if min(
        args.epochs,
        args.batch_size,
        args.replay_batch_size,
        args.memory_capacity,
        args.burst_interval,
        args.check_interval,
        args.cooldown,
        args.max_bursts,
    ) < 1:
        parser.error("epoch, batch, memory, interval, cooldown, and budget values must be positive")
    if not 0 <= args.loss_ema_beta < 1:
        parser.error("loss EMA beta must be in [0, 1)")
    if args.loss_relative_threshold < 0 or args.gradient_threshold < 0:
        parser.error("trigger thresholds must be non-negative")
    if args.memory_capacity <= PROBE_SIZE:
        parser.error("memory capacity must exceed the 32-example probe")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train_data = MNIST(args.data_dir, train=True, download=True, transform=ToTensor())
    test_data = MNIST(args.data_dir, train=False, download=True, transform=ToTensor())
    project_root = Path(__file__).resolve().parents[1]
    code_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    results: dict[str, Any] = {
        "experiment": "Phase 4: periodic versus fixed-budget adaptive replay",
        "scope": (
            "Exploratory Split-MNIST pilot. Adaptive methods have a maximum of "
            "43 one-step replay bursts; trigger thresholds are prespecified "
            "pilot settings, not development-tuned or final results."
        ),
        "code_commit": code_commit,
        "configuration": {
            "seeds": args.seeds,
            "task_order": TASKS,
            "epochs_per_task": args.epochs,
            "batch_size": args.batch_size,
            "memory_capacity": args.memory_capacity,
            "probe_size": PROBE_SIZE,
            "replay_batch_size": args.replay_batch_size,
            "burst_steps": 1,
            "periodic_interval": args.burst_interval,
            "check_interval": args.check_interval,
            "cooldown_in_incoming_steps": args.cooldown,
            "adaptive_burst_budget": args.max_bursts,
            "loss_relative_threshold": args.loss_relative_threshold,
            "loss_ema_beta": args.loss_ema_beta,
            "gradient_cosine_threshold": -args.gradient_threshold,
            "trigger_order": "signal checked pre-update; EMA updated after loss signal; incoming update then replay burst",
            "probe_policy": "fixed 32-example sample from initial reservoir, reserved inside total 500-example memory",
            "test_classes_masked": False,
            "test_metrics_used_for_trigger": False,
            "evaluation": "stage-end task test accuracy only",
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
                args.burst_interval,
                args.check_interval,
                args.cooldown,
                args.max_bursts,
                args.loss_relative_threshold,
                args.loss_ema_beta,
                args.gradient_threshold,
            )
        results["runs"][str(seed)] = run

    results.update(
        aggregate_runs(
            results["runs"],
            methods_to_aggregate=METHODS,
            comparison=("periodic_replay", "loss_triggered"),
            variable_metrics=(
                "replay_examples",
                "optimizer_steps",
                "replay_bursts",
                "replay_optimizer_steps",
                "monitoring_passes",
            ),
        )
    )
    _add_paired_comparisons(results, results["runs"])
    for method in METHODS:
        entries = [run["methods"][method] for run in results["runs"].values()]
        for metric in (
            "trigger_checks",
            "probe_forward_examples",
            "probe_backward_examples",
            "incoming_training_seconds",
            "trigger_monitoring_seconds",
            "replay_training_seconds",
            "evaluation_seconds",
            "total_seconds",
        ):
            values = [entry[metric] for entry in entries]
            results["methods"][method][metric] = statistics.mean(values)
            results["methods"][method][f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else None
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
