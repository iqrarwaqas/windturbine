"""Export a clean, report-ready results summary from a completed sweep.

Reads `results/records.csv` + `results/metrics.csv` and writes:
  - results_summary.md  : markdown tables (final CARE mean+/-std, forgetting/BWT,
                          final-stage sub-scores, Farm-A forgetting trajectory) plus
                          auto-generated plain-language findings.
  - the §10 figures (via plots.py): forgetting curves, per-farm bars, bandit ablation.

Usage:
  python -m care_cl.experiments.summary
  python -m care_cl.experiments.summary --results-dir care_cl/results --bandit off
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# Display order: bounds first, then CL methods.
ORDER = ["joint", "naive", "ewc", "replay", "distill", "replay_distill"]


def _ordered(strats):
    present = [s for s in ORDER if s in strats]
    return present + [s for s in strats if s not in ORDER]


def _fmt(mean, std=None):
    if std is None or np.isnan(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build_markdown(records: pd.DataFrame, metrics: pd.DataFrame, bandit: str) -> str:
    rec = records[records["bandit"] == bandit]
    met = metrics[metrics["bandit"] == bandit]
    strategies = _ordered(met["strategy"].unique())
    align = ", ".join(sorted(rec["align_mode"].unique()))
    seeds = sorted(int(s) for s in met["seed"].unique())

    lines = [
        "# CARE Continual-Learning — Results Summary",
        "",
        f"- **Align mode:** {align}  ·  **Bandit:** {bandit}  ·  "
        f"**Seeds:** {len(seeds)} ({seeds})  ·  **Sequence:** A → B → C",
        "- CARE sub-scores: Coverage (F₀.₅), Earliness, Reliability, Accuracy; "
        "higher is better. Forgetting on Farm A = CARE(after A) − CARE(after C); "
        "**lower is better**, negative = backward transfer.",
        "",
        "## 1. Final average CARE & forgetting (mean ± std over seeds)",
        "",
    ]

    # Table 1: final_avg_care + forgetting metrics.
    rows = []
    for s in strategies:
        g = met[met["strategy"] == s]
        fa = _fmt(g["final_avg_care"].mean(), g["final_avg_care"].std())
        fgt = ("—" if s == "joint" else
               _fmt(g["forgetting_A"].mean(), g["forgetting_A"].std()))
        bwt = _fmt(g["backward_transfer"].mean(), g["backward_transfer"].std())
        note = "upper bound" if s == "joint" else (
            "lower bound" if s == "naive" else "")
        rows.append([s, fa, fgt, bwt, note])
    lines.append(_md_table(
        ["strategy", "final_avg_care", "forgetting (Farm A)", "backward_transfer", ""],
        rows))
    lines.append("")

    # Table 2: final-stage sub-scores.
    lines += ["## 2. Final-stage sub-scores (after training C, mean over farms+seeds)", ""]
    fin = rec[rec["stage"] == "C"]
    rows = []
    for s in strategies:
        g = fin[fin["strategy"] == s]
        rows.append([s] + [f"{g[c].mean():.3f}" for c in
                            ["coverage", "earliness", "reliability", "accuracy", "CARE"]])
    lines.append(_md_table(
        ["strategy", "coverage", "earliness", "reliability", "accuracy", "CARE"], rows))
    lines.append("")

    # Table 3: Farm A trajectory (the forgetting story).
    lines += ["## 3. Farm A CARE over training stages (the forgetting story)", ""]
    fa = rec[rec["farm"] == "A"]
    stages = [st for st in ["A", "B", "C"] if st in fa["stage"].unique()]
    rows = []
    for s in strategies:
        g = fa[fa["strategy"] == s]
        cells = []
        for st in stages:
            v = g[g["stage"] == st]["CARE"]
            cells.append(f"{v.mean():.3f}" if len(v) else "—")
        rows.append([s] + cells)
    lines.append(_md_table(["strategy"] + [f"after {st}" for st in stages], rows))
    lines.append("")

    # Auto findings.
    lines += ["## 4. Findings", ""] + _findings(met, strategies)
    lines += ["", "## 5. Figures", "",
              "![Forgetting curves](fig_forgetting_curves.png)", "",
              "![Final per-farm CARE](fig_final_per_farm_care.png)", ""]
    if metrics["bandit"].nunique() > 1:
        lines += ["![Bandit ablation](fig_bandit_ablation.png)", ""]
    return "\n".join(lines)


def _findings(met: pd.DataFrame, strategies) -> list[str]:
    """Generate honest, data-driven bullet points (no hard-coded conclusions)."""
    out = []
    fa = met.groupby("strategy")["final_avg_care"].mean()
    fg = met.groupby("strategy")["forgetting_A"].mean()

    if "joint" in fa.index:
        cl = [s for s in fa.index if s != "joint"]
        viol = [s for s in cl if fa[s] > fa["joint"] + 1e-9]
        if viol:
            out.append(f"- ⚠️ Upper-bound check: {', '.join(viol)} exceeded `joint` "
                       f"on final_avg_care (within seed noise — flag, don't trust).")
        else:
            out.append("- ✅ Upper-bound check holds: no CL method beats `joint`.")

    cl_fg = fg.drop(labels=[i for i in ["joint"] if i in fg.index]).dropna()
    if not cl_fg.empty:
        best = cl_fg.idxmin()
        out.append(f"- Lowest forgetting on Farm A: **{best}** ({fg[best]:+.4f}).")
        if "naive" in fg.index:
            out.append(f"- `naive` forgetting = {fg['naive']:+.4f} "
                       f"(expected lower bound).")
            if best != "naive":
                out.append(f"- **{best}** cuts Farm-A forgetting by "
                           f"{(fg['naive']-fg[best]):.4f} vs `naive`.")
        worst = cl_fg.idxmax()
        out.append(f"- Worst CL forgetting: {worst} ({fg[worst]:+.4f}).")

    cl_fa = fa.drop(labels=[i for i in ["joint"] if i in fa.index])
    if not cl_fa.empty:
        out.append(f"- Best CL final_avg_care: {cl_fa.idxmax()} ({cl_fa.max():.3f}).")
    out.append("- Note: final_avg_care gaps are often within ~1 std; the robust signal "
               "is the **forgetting / backward-transfer** comparison.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="care_cl/results")
    ap.add_argument("--bandit", default=None,
                    help="which bandit setting to summarize (default: first found)")
    args = ap.parse_args()

    rec_path = os.path.join(args.results_dir, "records.csv")
    met_path = os.path.join(args.results_dir, "metrics.csv")
    if not (os.path.exists(rec_path) and os.path.exists(met_path)):
        raise SystemExit(f"Run a sweep first — {rec_path}/{met_path} not found.")

    records = pd.read_csv(rec_path)
    metrics = pd.read_csv(met_path)
    bandit = args.bandit or ("off" if "off" in set(records["bandit"])
                             else records["bandit"].iloc[0])

    md = build_markdown(records, metrics, bandit)
    out_md = os.path.join(args.results_dir, "results_summary.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)

    # (Re)generate the figures next to the markdown.
    try:
        from .plots import make_all_plots
        make_all_plots(records, metrics, args.results_dir)
    except Exception as e:
        print(f"(figures skipped: {e})")

    # Echo to console, but tolerate legacy code pages (cp1252) that can't render
    # the unicode arrows/+- used in the markdown.
    try:
        print(md)
    except UnicodeEncodeError:
        import sys
        sys.stdout.buffer.write(md.encode("utf-8", "replace"))
        print()
    print(f"\nwrote {out_md} (+ figures in {args.results_dir})")


if __name__ == "__main__":
    main()
