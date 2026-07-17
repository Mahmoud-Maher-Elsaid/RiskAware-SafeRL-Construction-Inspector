from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import (
    BaseCallback,
)


class DualCriticDiagnosticsCallback(BaseCallback):
    """Persist one dual-controller record per rollout."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.output_path = Path(output_path)

    def _init_callback(self) -> None:
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        provider = getattr(
            self.model,
            "get_dual_diagnostics",
            None,
        )
        if provider is None or not callable(provider):
            raise TypeError("The model does not provide dual diagnostics.")

        record: dict[str, Any] = dict(provider())
        with self.output_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(json.dumps(record) + "\n")

        if self.verbose:
            print(
                "Dual-critic update: "
                f"episode_cost={record.get('mean_episode_cost')}, "
                f"ema_cost={record.get('ema_episode_cost')}, "
                f"limit={record.get('cost_limit')}, "
                f"lambda={record.get('multiplier_after')}"
            )
