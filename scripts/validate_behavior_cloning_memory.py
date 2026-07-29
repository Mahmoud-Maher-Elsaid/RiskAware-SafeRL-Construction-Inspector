from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader
from train_behavior_cloning_v2 import DemonstrationSequences, sha256_file

from riskaware_saferrl.policies import RecurrentMaskedPolicy


def validate(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Behavior Cloning memory gate")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dataset = DemonstrationSequences(args.dataset_dir, "train", args.sequence_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        generator=torch.Generator().manual_seed(args.seed),
    )
    policy = RecurrentMaskedPolicy().to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4)
    process = psutil.Process()
    initial_rss = process.memory_info().rss
    peak_rss = initial_rss
    minimum_available = psutil.virtual_memory().available
    rss_samples: list[dict[str, int]] = []
    started = time.perf_counter()
    iterator = iter(loader)
    for batch_number in range(1, args.batches + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        maps = batch["maps"].to(device)
        states = batch["states"].to(device)
        masks = batch["masks"].to(device)
        actions = batch["actions"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = policy(maps, states, masks)
        loss = nn.functional.cross_entropy(
            output.distribution.logits.reshape(-1, policy.action_count),
            actions.reshape(-1),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if batch_number % 10 == 0:
            rss = process.memory_info().rss
            peak_rss = max(peak_rss, rss)
            minimum_available = min(minimum_available, psutil.virtual_memory().available)
            rss_samples.append({"batch": batch_number, "rss_bytes": rss})
    checkpoint = args.output_dir / "memory_smoke_checkpoint.pt"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "batches": args.batches,
            "sequence_length": args.sequence_length,
        },
        checkpoint,
    )
    restored = torch.load(checkpoint, map_location=device, weights_only=True)
    restored_policy = RecurrentMaskedPolicy().to(device)
    restored_policy.load_state_dict(restored["model_state_dict"])
    restored_optimizer = torch.optim.AdamW(restored_policy.parameters(), lr=3e-4)
    restored_optimizer.load_state_dict(restored["optimizer_state_dict"])
    final_rss = process.memory_info().rss
    growth = peak_rss - initial_rss
    midpoint_rss = max(
        sample["rss_bytes"] for sample in rss_samples if sample["batch"] <= args.batches // 2
    )
    second_half_growth = peak_rss - midpoint_rss
    passed = (
        args.batches >= 1000
        and growth <= args.max_growth_mb * 1024**2
        and second_half_growth <= args.max_plateau_growth_mb * 1024**2
        and dataset.chunk_cache.active_chunks <= 2
        and restored["batches"] == args.batches
    )
    report = {
        "status": "PASSED" if passed else "FAILED",
        "device": torch.cuda.get_device_name(0),
        "batches": args.batches,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "num_workers": 0,
        "pin_memory": False,
        "initial_rss_bytes": initial_rss,
        "peak_rss_bytes": peak_rss,
        "final_rss_bytes": final_rss,
        "rss_growth_bytes": growth,
        "rss_growth_limit_bytes": int(args.max_growth_mb * 1024**2),
        "second_half_rss_growth_bytes": second_half_growth,
        "second_half_rss_growth_limit_bytes": int(args.max_plateau_growth_mb * 1024**2),
        "rss_samples": rss_samples,
        "minimum_available_physical_bytes": minimum_available,
        "gpu_allocated_bytes": torch.cuda.memory_allocated(device),
        "gpu_reserved_bytes": torch.cuda.memory_reserved(device),
        "elapsed_seconds": time.perf_counter() - started,
        "sequences_per_second": args.batches
        * args.batch_size
        / max(time.perf_counter() - started, 1e-9),
        "checkpoint_path": checkpoint.as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_round_trip_verified": True,
        "source_dataset_sha256": dataset.cache_metadata["dataset_sha256"],
        "active_mmap_chunks": dataset.chunk_cache.active_chunks,
    }
    report_path = Path("reports/strong_policy_upgrade/imitation_learning/memory_smoke_test.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
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
    parser.add_argument("--batches", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=32)
    # CUDA/PyTorch initializes about 820 MiB of process-resident runtime state on this
    # Windows host; the separate plateau bound detects continuing dataset growth.
    parser.add_argument("--max-growth-mb", type=float, default=1024)
    parser.add_argument("--max-plateau-growth-mb", type=float, default=64)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    report = validate(args)
    print(f"BEHAVIOR_CLONING_MEMORY_GATE={report['status']}")
    if report["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
