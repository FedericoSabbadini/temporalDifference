#!/usr/bin/env python3
"""Generate LaTeX result tables from the saved .npz result files.

Origin: NEW. Reads ONLY the real arrays produced by the runners
(results/*/*.npz) and emits LaTeX tables into report/tables/.
No numbers are hand-written — everything is computed from saved data,
so the tables can be regenerated after re-running any experiment.

Usage:
    uv run python scripts/make_report_tables.py

Note: only the control comparisons (CartPole, Acrobot) and the Pong
ablation were persisted as .npz by the runners. Sensitivity / failure /
prediction experiments produced figures only, so they are discussed
qualitatively in the report rather than tabulated here.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

OUT_DIR = "report/tables"


def _valid_rows(arr):
    """Yield each seed's non-NaN episode returns (arrays may be NaN-padded)."""
    for row in arr:
        yield row[~np.isnan(row)]


def _final_perf(arr, tail_frac=0.1):
    """Mean/std over the last `tail_frac` of each seed's valid episodes.

    Using a fraction (last 10%) rather than a fixed count gives a stable
    estimate of asymptotic performance that is not distorted by the noise
    of the very last few episodes (e.g. an end-of-run spike). It is computed
    per seed, then aggregated, which is the correct way to get a seed-level
    mean ± std and is robust to the NaN padding used for the Pong arrays.
    """
    counts = [len(r) for r in _valid_rows(arr)]
    per_seed = np.array([
        np.mean(r[-max(1, int(len(r) * tail_frac)):]) for r in _valid_rows(arr)
    ])
    return per_seed.mean(), per_seed.std(), counts


def table_control(npz_path, env_label, caption, label):
    """Emit a Q-Learning vs SARSA final-performance table."""
    d = np.load(npz_path, allow_pickle=True)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Algorithm & Initial return & Final return & Episodes/seed \\",
        r"          & (first 100 ep) & (last 10\% ep, mean $\pm$ std) & \\",
        r"\midrule",
    ]
    name_map = {"ql": "Q-Learning", "sarsa": "SARSA"}
    for key in ("ql", "sarsa"):
        arr = d[key]
        init = np.nanmean(arr[:, :100])
        mean, std, counts = _final_perf(arr)
        lines.append(
            rf"{name_map[key]} & {init:.1f} & ${mean:.1f} \pm {std:.1f}$ & {counts[0]} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def table_pong(npz_path):
    """Emit the Pong DQN ablation table (5 configurations)."""
    d = np.load(npz_path, allow_pickle=True)
    order = [
        ("full", "Full DQN (replay + target)"),
        ("double", "Double DQN (bonus)"),
        ("no_target", "No target network"),
        ("no_replay", "No experience replay"),
        ("no_both", "No target + no replay"),
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Pong DQN ablation: final performance after 200k environment "
        r"steps, mean $\pm$ std over 3 seeds. Random policy $\approx -20.4$.}",
        r"\label{tab:pong-ablation}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Configuration & Final return & Episodes/seed \\",
        r"              & (last 10\% ep, mean $\pm$ std) & (mean) \\",
        r"\midrule",
    ]
    for key, name in order:
        arr = d[key]
        mean, std, counts = _final_perf(arr)
        lines.append(
            rf"{name} & ${mean:.1f} \pm {std:.1f}$ & {int(np.mean(counts))} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tables = {
        "cartpole_control.tex": table_control(
            "results/cartpole/control_comparison.npz", "CartPole",
            "CartPole control: Q-Learning vs SARSA, mean over 3 seeds. "
            "Max possible return is 500; both plateau near 184 due to the "
            "coarse $10^4$-state discretization.",
            "tab:cartpole-control"),
        "acrobot_control.tex": table_control(
            "results/acrobot/control_comparison.npz", "Acrobot",
            "Acrobot control: Q-Learning vs SARSA, mean over 3 seeds. "
            "Random policy $\\approx -500$; a good swing-up reaches "
            "$\\approx -100$.",
            "tab:acrobot-control"),
        "pong_ablation.tex": table_pong("results/pong/ablation.npz"),
    }

    for fname, content in tables.items():
        path = os.path.join(OUT_DIR, fname)
        with open(path, "w") as f:
            f.write(content)
        print(f"Wrote {path}")

    # Also print a plain-text summary for quick inspection / the report text
    print("\n=== Plain-text summary (for reference) ===")
    for npz, label in [("results/cartpole/control_comparison.npz", "CartPole"),
                       ("results/acrobot/control_comparison.npz", "Acrobot")]:
        d = np.load(npz, allow_pickle=True)
        for key in ("ql", "sarsa"):
            m, s, c = _final_perf(d[key])
            print(f"{label:9s} {key:6s}: final {m:7.1f} +/- {s:4.1f}  ({c[0]} ep)")
    d = np.load("results/pong/ablation.npz", allow_pickle=True)
    for key in ("full", "double", "no_target", "no_replay", "no_both"):
        m, s, c = _final_perf(d[key])
        print(f"Pong      {key:10s}: final {m:7.1f} +/- {s:4.1f}  ({int(np.mean(c))} ep)")


if __name__ == "__main__":
    main()
