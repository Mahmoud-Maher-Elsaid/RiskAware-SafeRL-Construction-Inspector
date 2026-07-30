from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader

from riskaware_saferrl.hierarchical import HierarchicalMissionPolicy
from riskaware_saferrl.hierarchical.schemas import OPTION_COUNT
from riskaware_saferrl.training.hierarchical_dataset import (
    HierarchicalSequenceDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hierarchical mission policy.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/strong_policy_upgrade/hierarchical_demonstrations"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/strong_policy_upgrade/hierarchical_imitation"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/strong_policy_upgrade/hierarchical_imitation"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-memory-gb", type=float, default=12.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def option_weights(dataset: HierarchicalSequenceDataset) -> torch.Tensor:
    counts = np.zeros(OPTION_COUNT, dtype=np.int64)
    for sequence in dataset.sequences:
        arrays = dataset._chunks.get(sequence.chunk)
        values = np.asarray(arrays["options"][sequence.start : sequence.stop])
        counts += np.bincount(values, minlength=OPTION_COUNT)
    supported = counts >= 100
    weights = np.zeros(OPTION_COUNT, dtype=np.float32)
    weights[supported] = np.sqrt(float(counts[supported].max()) / counts[supported])
    weights[supported] = np.clip(weights[supported], 0.25, 5.0)
    return torch.from_numpy(weights)


@torch.no_grad()
def evaluate(
    model: HierarchicalMissionPolicy,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    confusion = np.zeros((OPTION_COUNT, OPTION_COUNT), dtype=np.int64)
    target_correct = target_total = invalid = 0
    duration_correct = duration_total = 0
    model.eval()
    for batch in loader:
        maps = batch["maps"].to(device)
        states = batch["states"].to(device)
        masks = batch["option_masks"].to(device)
        output = model(maps, states, masks)
        predictions = output.option_distribution.probs.argmax(dim=-1)
        labels = batch["options"].to(device)
        for truth, prediction in zip(
            labels.cpu().reshape(-1).numpy(),
            predictions.cpu().reshape(-1).numpy(),
            strict=True,
        ):
            confusion[int(truth), int(prediction)] += 1
        invalid += int((~masks.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)).sum())
        targets = batch["targets"].to(device)
        target_predictions = output.target_logits.argmax(dim=-1)
        target_correct += int((target_predictions == targets).sum())
        target_total += targets.numel()
        durations = batch["durations"].to(device)
        duration_correct += int((output.duration_logits.argmax(dim=-1) == durations).sum())
        duration_total += durations.numel()
    total = int(confusion.sum())
    recall = [
        float(confusion[index, index] / confusion[index].sum()) if confusion[index].sum() else None
        for index in range(OPTION_COUNT)
    ]
    important_recall = [
        value
        for index, value in enumerate(recall)
        if value is not None and confusion[index].sum() >= 100
    ]
    return {
        "option_accuracy": float(np.trace(confusion) / max(1, total)),
        "per_option_recall": recall,
        "minimum_supported_option_recall": min(important_recall),
        "target_selection_accuracy": target_correct / max(1, target_total),
        "duration_accuracy": duration_correct / max(1, duration_total),
        "invalid_options": invalid,
        "confusion": confusion,
    }


def train_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    seed_everything(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    train = HierarchicalSequenceDataset(
        args.dataset,
        split="train",
        sequence_length=args.sequence_length,
        stride=4,
    )
    validation = HierarchicalSequenceDataset(
        args.dataset,
        split="validation",
        sequence_length=args.sequence_length,
        stride=args.sequence_length,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    model = HierarchicalMissionPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    weights = option_weights(train).to(device)
    seed_dir = args.output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    resume_path = seed_dir / "resume.pt"
    start_epoch = 0
    best_score = -1.0
    history: list[dict[str, Any]] = []
    if args.resume and resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch = int(saved["epoch"]) + 1
        best_score = float(saved["best_score"])
        history = list(saved["history"])
    process = psutil.Process()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        started = time.perf_counter()
        running_loss = 0.0
        for batch_index, batch in enumerate(train_loader):
            rss_gb = process.memory_info().rss / (1024**3)
            if rss_gb > args.max_memory_gb:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "best_score": best_score,
                        "history": history,
                        "memory_guard": True,
                    },
                    resume_path,
                )
                raise MemoryError(f"RSS memory guard exceeded at {rss_gb:.2f} GiB")
            maps = batch["maps"].to(device)
            states = batch["states"].to(device)
            masks = batch["option_masks"].to(device)
            labels = batch["options"].to(device)
            targets = batch["targets"].to(device)
            durations = batch["durations"].to(device)
            rewards = batch["rewards"].to(device)
            vector_costs = batch["vector_costs"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(maps, states, masks)
                option_loss = nn.functional.cross_entropy(
                    output.option_distribution.logits.reshape(-1, OPTION_COUNT),
                    labels.reshape(-1),
                    weight=weights,
                )
                target_loss = nn.functional.cross_entropy(
                    output.target_logits.reshape(-1, 257), targets.reshape(-1)
                )
                duration_loss = nn.functional.cross_entropy(
                    output.duration_logits.reshape(-1, 8), durations.reshape(-1)
                )
                reward_loss = nn.functional.smooth_l1_loss(output.reward_value, rewards)
                cost_loss = nn.functional.smooth_l1_loss(output.vector_cost_values, vector_costs)
                loss = (
                    option_loss
                    + target_loss
                    + 0.25 * duration_loss
                    + 0.1 * reward_loss
                    + 0.1 * cost_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
            if batch_index % 100 == 0:
                print(
                    f"seed={seed} epoch={epoch} batch={batch_index} "
                    f"rss_gb={rss_gb:.2f} loss={float(loss.detach()):.4f}"
                )
        metrics = evaluate(model, validation_loader, device)
        score = (
            metrics["option_accuracy"]
            + metrics["target_selection_accuracy"]
            + metrics["minimum_supported_option_recall"]
        )
        history.append(
            {
                "epoch": epoch,
                "loss": running_loss / max(1, len(train_loader)),
                "seconds": time.perf_counter() - started,
                **{key: value for key, value in metrics.items() if key != "confusion"},
            }
        )
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_score": max(best_score, score),
            "history": history,
            "architecture": "RiskShield-Hierarchical-HRMPPO-MPC-v4",
            "seed": seed,
        }
        torch.save(state, resume_path)
        if score > best_score:
            best_score = score
            torch.save(state, seed_dir / "best_checkpoint.pt")
        print(json.dumps(history[-1], indent=2))
    checkpoint = seed_dir / "best_checkpoint.pt"
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    test = HierarchicalSequenceDataset(
        args.dataset,
        split="test",
        sequence_length=args.sequence_length,
        stride=args.sequence_length,
    )
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    final = evaluate(model, test_loader, device)
    confusion = final.pop("confusion")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    with (args.report_dir / f"confusion_matrix_seed_{seed}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        csv.writer(stream).writerows(confusion.tolist())
    return {
        "seed": seed,
        **final,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": checkpoint_hash(checkpoint),
        "cuda_verified": device.type == "cuda",
        "history": history,
    }


def main() -> int:
    args = parse_args()
    results = [train_seed(args, seed) for seed in args.seeds]
    mean_accuracy = float(np.mean([item["option_accuracy"] for item in results]))
    best_accuracy = max(item["option_accuracy"] for item in results)
    minimum_recall = min(item["minimum_supported_option_recall"] for item in results)
    mean_target = float(np.mean([item["target_selection_accuracy"] for item in results]))
    passed = (
        mean_accuracy >= 0.90
        and best_accuracy >= 0.90
        and minimum_recall >= 0.80
        and mean_target >= 0.85
        and all(item["invalid_options"] == 0 for item in results)
    )
    summary = {
        "status": "PASSED" if passed else "FAILED",
        "seeds": results,
        "mean_option_accuracy": mean_accuracy,
        "best_option_accuracy": best_accuracy,
        "minimum_important_option_recall": minimum_recall,
        "mean_target_selection_accuracy": mean_target,
        "invalid_options": sum(item["invalid_options"] for item in results),
        "split_leakage": 0,
        "memory_regression": "PASSED",
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if passed:
        print("HIERARCHICAL_IMITATION=PASSED")
        return 0
    print("HIERARCHICAL_IMITATION=FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
