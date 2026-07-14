# Report — TemporalDifference

Final report (in Italian) for the Reinforcement Learning project. 15 pages, which is the
assignment's limit.

## Structure

```
report/
├── main.tex              # entry point (preamble + loads all sections)
├── references.bib
├── make_report_tables.py # generates tables/*.tex from results/*.npz
├── sections/
│   ├── introduction.tex
│   ├── theory.tex        # the four algorithms + DQN
│   ├── methodology.tex   # implementation, discretization choices, code origin
│   ├── experiments.tex   # setup, hyperparameters, how to read the figures
│   ├── results.tex       # per-environment results (all figures live here)
│   ├── failure.tex       # failure analysis (mandatory section)
│   ├── discussion.tex    # cross-cutting critical analysis + known limits
│   └── conclusions.tex   # what failed / what surprised me / algorithm limits
├── figures/              # PNGs copied from ../figures, env-prefixed. Every file
│                         #   here is cited in the text.
└── tables/               # generated — do not edit by hand
```

## Compiling

```bash
cd report && latexmk -pdf main.tex     # -> main.pdf
```

## Regenerating figures and tables

No number in the report is typed by hand: the tables are computed from the saved `.npz`.

```bash
# 1. (re)run experiments — from the repo root
uv run python scripts/run_cartpole.py
uv run python scripts/run_acrobot.py
uv run python scripts/run_pong.py

# 2. regenerate the LaTeX tables from the saved .npz
uv run python report/make_report_tables.py

# 3. refresh the report's figure copies (env-prefixed names)
for f in figures/*/*.png; do
  d=$(basename $(dirname "$f")); cp "$f" "report/figures/${d}_$(basename $f)"
done
```

Step 3 copies *all* figures; two of them (`*_prediction_return.png`) are intentionally
not cited — the report explains why — so they can be deleted from `report/figures/` to
keep that folder exactly equal to what the text references.
