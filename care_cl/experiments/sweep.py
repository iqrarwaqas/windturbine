"""Run the full experiment grid and aggregate to results/ (§9).

One command reproduces the results table + figures:
    python -m care_cl.experiments.sweep
    python -m care_cl.experiments.sweep --quick   # smaller grid for a fast pass
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from .run import load_cfg, DEFAULT_CFG
from ..cl.protocol import run_sequence
from ..cl.strategies import build_strategy
from ..cl.protocol import resolve_device
from ..eval.cl_metrics import summarize


def run_grid(cfg_path: str, strategies, bandits, seeds, align_modes, quick=False):
    rows = []          # per (strategy,bandit,seed,stage,farm)
    metric_rows = []   # per (strategy,bandit,seed) derived metrics
    for align_mode in align_modes:
        for strat_name in strategies:
            for bandit in bandits:
                for seed in seeds:
                    cfg = load_cfg(cfg_path, {
                        "threshold.bandit": bandit, "align.mode": align_mode, "seed": seed})
                    if quick:
                        cfg["train"]["epochs"] = 4
                        cfg["train"]["max_train_rows_per_farm"] = 40000
                    device = resolve_device(cfg["train"]["device"])
                    strat = build_strategy(strat_name, cfg, device)
                    run = run_sequence(cfg, strat, seed)
                    rows.extend(run["records"])
                    m = summarize(run["matrix"], cfg["cl"]["sequence"])
                    metric_rows.append({
                        "strategy": strat_name, "bandit": bandit, "align_mode": align_mode,
                        "seed": seed, "final_avg_care": m["final_avg_care"],
                        "mean_forgetting": m["mean_forgetting"],
                        "backward_transfer": m["backward_transfer"],
                        "forgetting_A": m["forgetting_per_farm"].get("A", float("nan")),
                    })
                    print(f"  done: {strat_name} bandit={bandit} mode={align_mode} "
                          f"seed={seed} final_avg_care={m['final_avg_care']:.3f}")
    return pd.DataFrame(rows), pd.DataFrame(metric_rows)


def aggregate(metrics_df: pd.DataFrame) -> pd.DataFrame:
    g = metrics_df.groupby(["strategy", "bandit", "align_mode"])
    agg = g.agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--strategies", nargs="+",
                    default=["naive", "joint", "ewc", "replay", "distill", "replay_distill"])
    ap.add_argument("--bandits", nargs="+", default=["off", "on"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--align-modes", nargs="+", default=["shared_only"])
    args = ap.parse_args()

    with open(args.config) as f:
        import yaml
        results_dir = yaml.safe_load(f)["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    records_df, metrics_df = run_grid(
        args.config, args.strategies, args.bandits, args.seeds, args.align_modes, args.quick)

    records_path = os.path.join(results_dir, "records.csv")
    metrics_path = os.path.join(results_dir, "metrics.csv")
    agg_path = os.path.join(results_dir, "metrics_aggregated.csv")
    records_df.to_csv(records_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    agg = aggregate(metrics_df)
    agg.to_csv(agg_path, index=False)

    print("\n=== Aggregated final_avg_care (mean) ===")
    piv = metrics_df.groupby(["strategy", "bandit"])["final_avg_care"].mean().unstack()
    print(piv.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {records_path}\n      {metrics_path}\n      {agg_path}")

    try:
        from .plots import make_all_plots
        make_all_plots(records_df, metrics_df, results_dir)
        print(f"figures in {results_dir}")
    except Exception as e:  # plotting is non-fatal
        print(f"(plots skipped: {e})")


if __name__ == "__main__":
    main()
