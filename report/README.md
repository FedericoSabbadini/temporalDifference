# Report - TemporalDifference

Final report (in Italian) for the Reinforcement Learning project.

## Structure

```
report/
├── main.tex              # entry point (loads all sections)
├── references.bib        # bibliography
├── sections/             # one .tex per section
│   ├── introduction.tex
│   ├── theory.tex
│   ├── methodology.tex
│   ├── experiments.tex
│   ├── results.tex
│   ├── discussion.tex    # critical analysis (incl. limitations)
│   ├── compliance.tex    # requirement - deliverable matrix
│   ├── code_origin.tex
│   └── conclusions.tex
├── figures/              # PNGs copied from ../figures (env-prefixed)
└── tables/               # .tex tables generated from real data
```

## Compiling

```bash
# From the report/ directory:
latexmk -pdf main.tex
# or manually:
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Produces `main.pdf` (~16 pages).

## Regenerating figures and tables

Figures and tables are derived from the saved experiment outputs, not
hand-written:

```bash
# 1. (Re)run experiments to produce results/ and figures/  (from repo root)
uv run python scripts/run_cartpole.py
uv run python scripts/run_acrobot.py
uv run python scripts/run_pong.py

# 2. Regenerate the LaTeX tables from the saved .npz files
uv run python scripts/make_report_tables.py

# 3. Refresh the report's figure copies
cd report/figures
for f in ../../figures/*/*.png; do
  d=$(basename $(dirname "$f")); cp "$f" "${d}_$(basename $f)"
done
```

All reported numbers come from `results/*/*.npz`; `make_report_tables.py`
computes them so nothing is manually transcribed.
