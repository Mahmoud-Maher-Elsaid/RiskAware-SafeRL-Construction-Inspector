from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def package_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def run_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        return {
            "command": command,
            "exit_code": completed.returncode,
            "output": completed.stdout,
        }
    except OSError as error:
        return {
            "command": command,
            "exit_code": None,
            "output": repr(error),
        }


def main() -> int:
    integration_report_path = (
        PROJECT_ROOT / "data" / "manifests" / "ppe_dataset_integration_validation.json"
    )

    integration_report = json.loads(integration_report_path.read_text(encoding="utf-8"))

    torch_status: dict[str, Any]

    try:
        import torch

        cuda_available = torch.cuda.is_available()

        torch_status = {
            "imported": True,
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": cuda_available,
            "gpu_count": torch.cuda.device_count(),
            "cudnn_available": torch.backends.cudnn.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
        }

        if cuda_available:
            properties = torch.cuda.get_device_properties(0)

            torch_status.update(
                {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_total_memory_bytes": properties.total_memory,
                    "gpu_total_memory_gib": round(
                        properties.total_memory / (1024**3),
                        3,
                    ),
                    "compute_capability": [
                        properties.major,
                        properties.minor,
                    ],
                }
            )
    except Exception as error:
        torch_status = {
            "imported": False,
            "error": repr(error),
            "cuda_available": False,
        }

    packages = {
        "torch": package_version("torch"),
        "torchvision": package_version("torchvision"),
        "opencv_python": package_version("opencv-python"),
        "pillow": package_version("Pillow"),
        "numpy": package_version("numpy"),
        "pyyaml": package_version("PyYAML"),
        "ultralytics": package_version("ultralytics"),
        "onnx": package_version("onnx"),
        "onnxruntime": package_version("onnxruntime"),
        "onnxruntime_gpu": package_version("onnxruntime-gpu"),
        "tensorrt": package_version("tensorrt"),
    }

    modules = {
        "torch": module_available("torch"),
        "torchvision": module_available("torchvision"),
        "cv2": module_available("cv2"),
        "PIL": module_available("PIL"),
        "yaml": module_available("yaml"),
        "ultralytics": module_available("ultralytics"),
        "onnx": module_available("onnx"),
        "onnxruntime": module_available("onnxruntime"),
        "tensorrt": module_available("tensorrt"),
    }

    training_ready = (
        integration_report.get("status") == "PASS"
        and torch_status.get("imported") is True
        and torch_status.get("cuda_available") is True
    )

    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "dataset_integration_status": integration_report.get("status"),
        "torch": torch_status,
        "packages": packages,
        "modules": modules,
        "nvidia_smi": run_command(["nvidia-smi"]),
        "pip_check": run_command([sys.executable, "-m", "pip", "check"]),
        "training_ready": training_ready,
        "notes": [
            ("A detector framework has not been selected solely from this preflight report."),
            ("Historical custom PPE checkpoints must not be used to resume production training."),
            (
                "A generic official pretrained backbone may be used "
                "for a fresh project-specific experiment."
            ),
        ],
    }

    report_path = PROJECT_ROOT / "reports" / "perception" / "training_environment.json"

    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("PERCEPTION TRAINING PREFLIGHT")
    print("=" * 72)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch_status.get('version')}")
    print(f"CUDA build: {torch_status.get('cuda_build')}")
    print(f"CUDA available: {torch_status.get('cuda_available')}")
    print(f"GPU: {torch_status.get('gpu_name')}")
    print(f"GPU memory GiB: {torch_status.get('gpu_total_memory_gib')}")
    print(f"Ultralytics: {packages['ultralytics']}")
    print(f"ONNX: {packages['onnx']}")
    print(f"ONNX Runtime: {packages['onnxruntime']}")
    print(f"TensorRT: {packages['tensorrt']}")
    print(f"Training ready: {training_ready}")
    print(f"Report: {report_path}")
    print("=" * 72)

    return 0 if training_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
