from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from riskaware_saferrl.policies import RecurrentMaskedPolicy

REQUIRED_ARRAYS = (
    "maps",
    "states",
    "action_masks",
    "teacher_executed_actions",
    "episode_ids",
    "splits",
)
CACHE_FORMAT_VERSION = 1


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_mmap_cache(dataset_dir: Path, *, force: bool = False) -> dict[str, Any]:
    """Derive deterministic mmap-compatible NPY arrays from immutable NPZ chunks."""
    source_manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    cache_dir = dataset_dir / "mmap_cache"
    metadata_path = cache_dir / "cache_manifest.json"
    expected_sources = {
        Path(item["path"]).name: item["sha256"] for item in source_manifest["chunks"]
    }
    if metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("format_version") == CACHE_FORMAT_VERSION
            and metadata.get("dataset_sha256") == source_manifest["dataset_sha256"]
            and metadata.get("sources") == expected_sources
            and all(
                (cache_dir / chunk["cache_directory"] / f"{name}.npy").is_file()
                for chunk in metadata.get("chunks", [])
                for name in REQUIRED_ARRAYS
            )
        ):
            return metadata

    cache_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, Any]] = []
    for item in source_manifest["chunks"]:
        source_path = Path(item["path"])
        actual_hash = sha256_file(source_path)
        if actual_hash != item["sha256"]:
            raise RuntimeError(f"Source chunk hash mismatch: {source_path}")
        chunk_dir = cache_dir / source_path.stem
        chunk_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, dict[str, Any]] = {}
        with np.load(source_path, allow_pickle=False) as payload:
            for name in REQUIRED_ARRAYS:
                destination = chunk_dir / f"{name}.npy"
                np.save(destination, payload[name], allow_pickle=False)
                arrays[name] = {
                    "path": destination.relative_to(cache_dir).as_posix(),
                    "shape": list(payload[name].shape),
                    "dtype": str(payload[name].dtype),
                    "sha256": sha256_file(destination),
                }
        chunks.append(
            {
                "source_path": source_path.as_posix(),
                "source_sha256": actual_hash,
                "cache_directory": chunk_dir.name,
                "rows": item["rows"],
                "arrays": arrays,
            }
        )
    metadata = {
        "format_version": CACHE_FORMAT_VERSION,
        "dataset_sha256": source_manifest["dataset_sha256"],
        "sources": expected_sources,
        "required_arrays": list(REQUIRED_ARRAYS),
        "chunks": chunks,
    }
    _write_json(metadata_path, metadata)
    return metadata


@dataclass(frozen=True, slots=True)
class SequenceIndex:
    chunk: int
    start: int
    stop: int
    valid_start: int
    split: str
    episode_id: str


class MmapChunkCache:
    """Process-local LRU containing at most a configured number of mmap chunks."""

    def __init__(self, cache_dir: Path, metadata: dict[str, Any], max_chunks: int = 2) -> None:
        if max_chunks not in (1, 2):
            raise ValueError("max_chunks must be one or two")
        self.cache_dir = cache_dir
        self.metadata = metadata
        self.max_chunks = max_chunks
        self._chunks: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def get(self, chunk_id: int) -> dict[str, np.ndarray]:
        if chunk_id in self._chunks:
            self._chunks.move_to_end(chunk_id)
            return self._chunks[chunk_id]
        chunk = self.metadata["chunks"][chunk_id]
        arrays = {
            name: np.load(
                self.cache_dir / chunk["cache_directory"] / f"{name}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            for name in REQUIRED_ARRAYS
        }
        self._chunks[chunk_id] = arrays
        while len(self._chunks) > self.max_chunks:
            _, evicted = self._chunks.popitem(last=False)
            for array in evicted.values():
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()
        return arrays

    @property
    def active_chunks(self) -> int:
        return len(self._chunks)


class DemonstrationSequences(Dataset):
    """Lightweight sequence index over lazily sliced memory-mapped arrays."""

    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        sequence_length: int,
        *,
        max_cached_chunks: int = 2,
        rotation_augmentation: bool = False,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.split = split
        self.sequence_length = sequence_length
        self.rotation_augmentation = rotation_augmentation
        self.cache_metadata = build_mmap_cache(dataset_dir)
        self.chunk_cache = MmapChunkCache(
            dataset_dir / "mmap_cache", self.cache_metadata, max_cached_chunks
        )
        self.sequences: list[SequenceIndex] = []
        self.action_counts = np.zeros(5, dtype=np.int64)
        for chunk_id, _ in enumerate(self.cache_metadata["chunks"]):
            arrays = self.chunk_cache.get(chunk_id)
            episode_ids = arrays["episode_ids"]
            splits = arrays["splits"]
            actions = arrays["teacher_executed_actions"]
            masks = arrays["action_masks"]
            row = 0
            while row < len(episode_ids):
                episode_stop = row + 1
                while (
                    episode_stop < len(episode_ids)
                    and episode_ids[episode_stop] == episode_ids[row]
                ):
                    episode_stop += 1
                if str(splits[row]) == split and np.all(splits[row:episode_stop] == split):
                    starts = list(range(row, episode_stop - sequence_length + 1, sequence_length))
                    if starts and starts[-1] + sequence_length < episode_stop:
                        starts.append(episode_stop - sequence_length)
                    previous_stop = row
                    for start in starts:
                        stop = start + sequence_length
                        self.sequences.append(
                            SequenceIndex(
                                chunk_id,
                                start,
                                stop,
                                max(start, previous_stop),
                                split,
                                str(episode_ids[start]),
                            )
                        )
                        valid_start = max(start, previous_stop)
                        selected_actions = actions[valid_start:stop]
                        selected_masks = masks[valid_start:stop]
                        if not np.all(
                            selected_masks[
                                np.arange(len(selected_actions)),
                                selected_actions.astype(np.int64),
                            ]
                        ):
                            raise ValueError("Invalid masked teacher action in source dataset")
                        self.action_counts += np.bincount(
                            selected_actions, minlength=len(self.action_counts)
                        )
                        previous_stop = stop
                row = episode_stop
        if self.rotation_augmentation:
            original_counts = self.action_counts.copy()
            self.action_counts.fill(0)
            for mapping in (
                np.array((0, 1, 2, 3, 4)),
                np.array((2, 3, 1, 0, 4)),
                np.array((1, 0, 3, 2, 4)),
                np.array((3, 2, 0, 1, 4)),
            ):
                self.action_counts[mapping] += original_counts
        self._write_index()

    def _write_index(self) -> None:
        path = (
            self.dataset_dir
            / "mmap_cache"
            / f"sequence_index_{self.split}_{self.sequence_length}.json"
        )
        payload = {
            "dataset_sha256": self.cache_metadata["dataset_sha256"],
            "sequence_length": self.sequence_length,
            "split": self.split,
            "episode_boundary_validated": True,
            "sequences": [asdict(record) for record in self.sequences],
        }
        _write_json(path, payload)

    def __len__(self) -> int:
        return len(self.sequences) * (4 if self.rotation_augmentation else 1)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        rotation = index % 4 if self.rotation_augmentation else 0
        base_index = index // 4 if self.rotation_augmentation else index
        record = self.sequences[base_index]
        arrays = self.chunk_cache.get(record.chunk)
        selected = slice(record.start, record.stop)
        # Copy only this sequence so tensors are contiguous, writable, and independent of mmap lifetime.
        result = {
            "maps": torch.from_numpy(np.array(arrays["maps"][selected], copy=True)).float(),
            "states": torch.from_numpy(np.array(arrays["states"][selected], copy=True)).float(),
            "masks": torch.from_numpy(np.array(arrays["action_masks"][selected], copy=True)).bool(),
            "actions": torch.from_numpy(
                np.array(arrays["teacher_executed_actions"][selected], copy=True)
            ).long(),
            "episode_id": record.episode_id,
            "sequence_start": record.start,
            "sequence_stop": record.stop,
            "chunk_id": record.chunk,
            "loss_mask": torch.arange(record.start, record.stop) >= record.valid_start,
        }
        if rotation:
            remaps = torch.tensor(
                (
                    (0, 1, 2, 3, 4),
                    (2, 3, 1, 0, 4),
                    (1, 0, 3, 2, 4),
                    (3, 2, 0, 1, 4),
                ),
                dtype=torch.long,
            )
            mapping = remaps[rotation]
            result["maps"] = torch.rot90(result["maps"], rotation, dims=(-2, -1))
            remapped_masks = torch.zeros_like(result["masks"])
            remapped_masks[:, mapping] = result["masks"]
            result["masks"] = remapped_masks
            result["actions"] = mapping[result["actions"]]
            old_row = result["states"][:, 0].clone()
            old_column = result["states"][:, 1].clone()
            if rotation == 1:
                result["states"][:, 0] = 1.0 - old_column
                result["states"][:, 1] = old_row
            elif rotation == 2:
                result["states"][:, 0] = 1.0 - old_row
                result["states"][:, 1] = 1.0 - old_column
            else:
                result["states"][:, 0] = old_column
                result["states"][:, 1] = 1.0 - old_row
            previous = result["states"][:, 10:15].clone()
            result["states"][:, 10:15] = 0.0
            result["states"][:, 10:15][:, mapping] = previous
            orientation_angle = torch.atan2(result["states"][:, 2], result["states"][:, 3])
            orientation = torch.remainder(
                torch.round(orientation_angle / (torch.pi / 2.0)).long(), 4
            )
            rotated_direction = mapping[orientation]
            angles = rotated_direction.float() * (torch.pi / 2.0)
            result["states"][:, 2] = torch.sin(angles)
            result["states"][:, 3] = torch.cos(angles)
        return result


@dataclass
class MemorySample:
    epoch: int
    batch: int
    sequences_processed: int
    process_rss_bytes: int
    available_physical_bytes: int
    committed_bytes: int
    commit_limit_bytes: int
    gpu_allocated_bytes: int
    gpu_reserved_bytes: int
    samples_per_second: float


class MemoryGuard:
    def __init__(self, max_memory_gb: float, report_path: Path) -> None:
        self.max_bytes = int(max_memory_gb * 1024**3)
        self.report_path = report_path
        self.process = psutil.Process()
        self.samples: list[MemorySample] = []

    def sample(
        self,
        *,
        epoch: int,
        batch: int,
        sequences_processed: int,
        elapsed: float,
        device: torch.device,
    ) -> MemorySample:
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        sample = MemorySample(
            epoch=epoch,
            batch=batch,
            sequences_processed=sequences_processed,
            process_rss_bytes=self.process.memory_info().rss,
            available_physical_bytes=virtual.available,
            committed_bytes=virtual.used + swap.used,
            commit_limit_bytes=virtual.total + swap.total,
            gpu_allocated_bytes=torch.cuda.memory_allocated(device) if device.type == "cuda" else 0,
            gpu_reserved_bytes=torch.cuda.memory_reserved(device) if device.type == "cuda" else 0,
            samples_per_second=sequences_processed / max(elapsed, 1e-9),
        )
        self.samples.append(sample)
        return sample

    def unsafe(self, sample: MemorySample) -> bool:
        return (
            sample.process_rss_bytes > self.max_bytes
            or sample.available_physical_bytes < 1024**3
            or sample.committed_bytes > int(sample.commit_limit_bytes * 0.9)
        )

    def write(self, status: str) -> None:
        _write_json(
            self.report_path,
            {
                "status": status,
                "max_memory_bytes": self.max_bytes,
                "samples": [asdict(sample) for sample in self.samples],
            },
        )


def evaluate(
    policy: RecurrentMaskedPolicy, loader: DataLoader, device: torch.device
) -> tuple[dict[str, Any], np.ndarray]:
    if loader.batch_size != 1:
        raise ValueError("Recurrent evaluation requires batch_size=1")
    policy.eval()
    confusion = np.zeros((policy.action_count, policy.action_count), dtype=np.int64)
    invalid = 0
    loss_sum = 0.0
    examples = 0
    recurrent_state: torch.Tensor | None = None
    previous_episode: str | None = None
    previous_stop: int | None = None
    previous_chunk: int | None = None
    with torch.no_grad():
        for batch in loader:
            maps = batch["maps"].to(device)
            states = batch["states"].to(device)
            masks = batch["masks"].to(device)
            actions = batch["actions"].to(device)
            episode = batch["episode_id"][0]
            start = int(batch["sequence_start"][0])
            chunk = int(batch["chunk_id"][0])
            contiguous = (
                episode == previous_episode and chunk == previous_chunk and start == previous_stop
            )
            if not contiguous:
                recurrent_state = None
            output = policy(maps, states, masks, recurrent_state)
            recurrent_state = output.recurrent_state
            previous_episode = episode
            previous_stop = int(batch["sequence_stop"][0])
            previous_chunk = chunk
            loss = nn.functional.cross_entropy(
                output.distribution.logits.reshape(-1, policy.action_count),
                actions.reshape(-1),
                reduction="none",
            )
            predicted = output.distribution.probs.argmax(-1)
            loss_mask = batch["loss_mask"].to(device).reshape(-1)
            invalid += int(
                (
                    ~masks.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
                    & batch["loss_mask"].to(device)
                ).sum()
            )
            target_cpu = actions.reshape(-1)[loss_mask].cpu().numpy()
            predicted_cpu = predicted.reshape(-1)[loss_mask].cpu().numpy()
            np.add.at(confusion, (target_cpu, predicted_cpu), 1)
            loss_sum += float(loss[loss_mask].sum())
            examples += len(target_cpu)
    true_positive = np.diag(confusion)
    support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    metrics = {
        "loss": loss_sum / max(1, examples),
        "accuracy": float(true_positive.sum() / max(1, examples)),
        "invalid_action_predictions": invalid,
        "per_action_precision": (true_positive / np.maximum(1, predicted_support)).tolist(),
        "per_action_recall": (true_positive / np.maximum(1, support)).tolist(),
        "per_action_support": support.tolist(),
        "examples": examples,
    }
    return metrics, confusion


def _save_checkpoint(
    path: Path,
    policy: RecurrentMaskedPolicy,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    seed: int,
    sequence_length: int,
    class_counts: np.ndarray,
    interrupted: bool,
) -> None:
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "architecture": "RiskShield-HRMPPO-v2",
            "stage": "behavior_cloning",
            "seed": seed,
            "epoch": epoch,
            "sequence_length": sequence_length,
            "class_counts": class_counts.tolist(),
            "memory_guard_interrupted": interrupted,
        },
        path,
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    training = DemonstrationSequences(
        args.dataset_dir,
        "train",
        args.sequence_length,
        rotation_augmentation=args.rotation_augmentation,
    )
    validation = DemonstrationSequences(args.dataset_dir, "validation", args.sequence_length)
    held_out = DemonstrationSequences(args.dataset_dir, "test", args.sequence_length)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
        "pin_memory": False,
    }
    generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(training, shuffle=True, generator=generator, **loader_options)
    evaluation_options = {
        **loader_options,
        "batch_size": 1,
    }
    validation_loader = DataLoader(validation, shuffle=False, **evaluation_options)
    test_loader = DataLoader(held_out, shuffle=False, **evaluation_options)
    policy = RecurrentMaskedPolicy().to(device)
    counts = training.action_counts.astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights /= weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(
        device.type, enabled=args.mixed_precision and device.type == "cuda"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_behavior_cloning.pt"
    resume_path = Path(args.resume) if args.resume else None
    start_epoch = 1
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        policy.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = args.learning_rate
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
    guard = MemoryGuard(
        args.max_memory_gb,
        Path("reports/strong_policy_upgrade/imitation_learning/memory_profile.json"),
    )
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    remaining_patience = args.patience
    started = time.perf_counter()
    processed = 0
    for epoch in range(start_epoch, args.epochs + 1):
        policy.train()
        loss_sum = 0.0
        batches = 0
        for batch_number, batch in enumerate(training_loader, 1):
            maps = batch["maps"].to(device)
            states = batch["states"].to(device)
            masks = batch["masks"].to(device)
            actions = batch["actions"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=args.mixed_precision and device.type == "cuda",
            ):
                output = policy(maps, states, masks)
                loss = nn.functional.cross_entropy(
                    output.distribution.logits.reshape(-1, policy.action_count),
                    actions.reshape(-1),
                    weight=class_weights,
                    reduction="none",
                )
                loss = loss[batch["loss_mask"].to(device).reshape(-1)].mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            batches += 1
            processed += maps.shape[0]
            sample = guard.sample(
                epoch=epoch,
                batch=batch_number,
                sequences_processed=processed,
                elapsed=time.perf_counter() - started,
                device=device,
            )
            if guard.unsafe(sample):
                interrupted_path = args.output_dir / "memory_guard_checkpoint.pt"
                _save_checkpoint(
                    interrupted_path,
                    policy,
                    optimizer,
                    scaler,
                    epoch=epoch,
                    seed=args.seed,
                    sequence_length=args.sequence_length,
                    class_counts=counts,
                    interrupted=True,
                )
                guard.write("FAILED_MEMORY_GUARD")
                raise MemoryError(
                    f"Memory guard stopped training; resumable checkpoint: {interrupted_path}"
                )
        metrics, _ = evaluate(policy, validation_loader, device)
        important_recall = min(
            recall
            for recall, support in zip(
                metrics["per_action_recall"], metrics["per_action_support"], strict=True
            )
            if support > 0
        )
        score = metrics["accuracy"] + 0.25 * important_recall
        history.append(
            {
                "epoch": epoch,
                "training_loss": loss_sum / max(1, batches),
                "validation": metrics,
                "selection_score": score,
            }
        )
        if score > best_score:
            best_score = score
            remaining_patience = args.patience
            _save_checkpoint(
                best_path,
                policy,
                optimizer,
                scaler,
                epoch=epoch,
                seed=args.seed,
                sequence_length=args.sequence_length,
                class_counts=counts,
                interrupted=False,
            )
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                break
    guard.write("PASSED")
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    policy.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics, _ = evaluate(policy, validation_loader, device)
    test_metrics, confusion = evaluate(policy, test_loader, device)
    checkpoint_hash = sha256_file(best_path)
    recalls = [
        recall
        for recall, support in zip(
            test_metrics["per_action_recall"], test_metrics["per_action_support"], strict=True
        )
        if support > 0
    ]
    passed = (
        test_metrics["accuracy"] >= 0.85
        and min(recalls) >= 0.70
        and test_metrics["invalid_action_predictions"] == 0
    )
    summary = {
        "status": "PASSED" if passed else "FAILED",
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "training_sequences": len(training),
        "validation_sequences": len(validation),
        "test_sequences": len(held_out),
        "sequence_length": args.sequence_length,
        "epochs_completed": len(history),
        "checkpoint_path": best_path.as_posix(),
        "checkpoint_sha256": checkpoint_hash,
        "dataset_sha256": training.cache_metadata["dataset_sha256"],
        "split_leakage": 0,
        "validation": validation_metrics,
        "held_out_test": test_metrics,
        "history": history,
    }
    _write_json(args.output_dir / "behavior_cloning_summary.json", summary)
    report_dir = Path("reports/strong_policy_upgrade/imitation_learning")
    _write_json(report_dir / "behavior_cloning_summary.json", summary)
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["actual_action", *[f"predicted_{index}" for index in range(policy.action_count)]]
        )
        for action, row in enumerate(confusion):
            writer.writerow([action, *row.tolist()])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("artifacts/strong_policy_upgrade/expert_demonstrations"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/strong_policy_upgrade/imitation_learning"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-memory-gb", type=float, default=8.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--rotation-augmentation", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = train(args)
    except MemoryError as error:
        print(f"BEHAVIOR_CLONING=FAILED ({error})")
        raise SystemExit(2) from error
    print(
        f"BEHAVIOR_CLONING={summary['status']} "
        f"(accuracy={summary['held_out_test']['accuracy']:.4f})"
    )
    if summary["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
