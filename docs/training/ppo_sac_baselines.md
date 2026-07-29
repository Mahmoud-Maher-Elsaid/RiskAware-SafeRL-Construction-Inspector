# PPO and SAC Baselines

The historical PPO training pipeline and its 102,400-step production MaskablePPO
checkpoint remain unchanged. The final research benchmark adds a fixed 16×16
observation adapter so one policy can be evaluated across small, medium, and
dynamic sites.

PPO uses Stable-Baselines3 `PPO` with a discrete five-action policy. SAC requires
a continuous action space, so `ContinuousActionAdapter` maps one learned scalar
in `[-1, 1]` into five equal bins corresponding to the documented environment
actions. SAC still learns its actor and critics from experience; the adapter is
only an action-space interface and contains no navigation heuristic.

Both trainers provide deterministic seeds, CUDA device selection, resumable
checkpoints, periodic checkpoints, TensorBoard logs, JSONL episode telemetry,
configuration metadata, safety-adjusted early stopping, and SHA-256 checkpoint
hashes.

The final short-budget cross-site evaluation is intentionally reported as a
negative result: neither unconstrained baseline completed the full task. PPO
obtained mean hazard recall 0.061; SAC obtained 0.092 but incurred substantially
higher safety cost. These results are not promoted as successful navigation.

```powershell
.venv\Scripts\python.exe scripts\train_ppo_research.py
.venv\Scripts\python.exe scripts\train_sac.py
```

Training artifacts remain outside Git under `artifacts/final_submission`. Their
metadata and hashes are curated in final-submission reports.
