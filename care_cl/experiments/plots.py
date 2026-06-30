"""Figures (§10): forgetting curves, per-farm CARE bars, ablation panels."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def forgetting_curves(records: pd.DataFrame, out_dir: str, bandit="off"):
    """CARE on Farm A (and B) vs training stage, one line per strategy."""
    df = records[records["bandit"] == bandit]
    stages = ["A", "B", "C"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, farm in zip(axes, ["A", "B"]):
        for strat in sorted(df["strategy"].unique()):
            sub = df[(df["strategy"] == strat) & (df["farm"] == farm)]
            xs, ys = [], []
            for i, st in enumerate(stages):
                v = sub[sub["stage"] == st]["CARE"]
                if len(v):
                    xs.append(i); ys.append(v.mean())
            if xs:
                ax.plot(xs, ys, marker="o", label=strat)
        ax.set_title(f"CARE on Farm {farm} over stages")
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([f"after {s}" for s in stages])
        ax.set_xlabel("training stage"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("CARE")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_forgetting_curves.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def final_per_farm_bars(records: pd.DataFrame, out_dir: str, bandit="off"):
    df = records[(records["bandit"] == bandit) & (records["stage"] == "C")]
    strategies = sorted(df["strategy"].unique())
    farms = ["A", "B", "C"]
    x = np.arange(len(farms)); w = 0.8 / max(1, len(strategies))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, strat in enumerate(strategies):
        means = [df[(df["strategy"] == strat) & (df["farm"] == f)]["CARE"].mean() for f in farms]
        ax.bar(x + i * w, means, w, label=strat)
    ax.set_xticks(x + w * (len(strategies) - 1) / 2)
    ax.set_xticklabels(farms); ax.set_ylabel("final CARE"); ax.set_xlabel("farm")
    ax.set_title("Final per-farm CARE (after training C)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_final_per_farm_care.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def bandit_ablation(metrics: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    piv = metrics.groupby(["strategy", "bandit"])["final_avg_care"].mean().unstack()
    piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("final_avg_care"); ax.set_title("Bandit on/off ablation")
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    p = os.path.join(out_dir, "fig_bandit_ablation.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def make_all_plots(records: pd.DataFrame, metrics: pd.DataFrame, out_dir: str):
    paths = []
    bandit = "off" if "off" in set(records["bandit"]) else records["bandit"].iloc[0]
    paths.append(forgetting_curves(records, out_dir, bandit))
    paths.append(final_per_farm_bars(records, out_dir, bandit))
    if metrics["bandit"].nunique() > 1:
        paths.append(bandit_ablation(metrics, out_dir))
    return paths


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "care_cl/results"
    make_all_plots(pd.read_csv(os.path.join(d, "records.csv")),
                   pd.read_csv(os.path.join(d, "metrics.csv")), d)
