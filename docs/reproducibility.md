# Reproducibility

Use Python 3.11 and the pinned dependency ranges in `pyproject.toml`.
Checkpoint metadata and hashes must validate before evaluation. All grid resets
and perturbations accept deterministic seeds.

Run the production, benchmark, and acceptance commands from the README. The
benchmark uses atomic per-run JSON caches and resumes only missing runs. Raw CSV
and Parquet records, aggregates, statistics, figure source data, and manifests
are committed. Local checkpoint files remain excluded from Git.
