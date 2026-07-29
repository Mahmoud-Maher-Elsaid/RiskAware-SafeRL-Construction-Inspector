# Paper

The paper reports only committed benchmark and runtime evidence. Tables and
figures are generated from Stage 9 raw data before being copied here.

Compile from this directory:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The expected output is `paper/main.pdf`. Rebuild the benchmark figures first
with `scripts/generate_final_figures.py` whenever raw results change.
