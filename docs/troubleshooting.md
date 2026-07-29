# Troubleshooting

- **CUDA false:** verify the NVIDIA driver and the PyTorch CUDA build using the
  project Python executable.
- **Checkpoint missing or hash mismatch:** restore the artifact to the path in
  the model metadata; do not substitute an unverified checkpoint.
- **Webots exits early:** inspect the production stdout/stderr and controller
  logs; terminate stale Webots processes before retrying.
- **Benchmark interrupted:** rerun the same PowerShell command. Valid cache
  records are retained and only missing runs execute.
- **`latexmk` requests Perl:** use the documented `pdflatex`, `bibtex`,
  `pdflatex`, `pdflatex` sequence.
