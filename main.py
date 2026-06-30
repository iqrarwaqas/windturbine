#!/usr/bin/env python
"""
main.py — one-file entry point for the CARE continual-learning benchmark.

WHAT THIS DOES
--------------
Runs the whole study end-to-end on the CARE-to-Compare wind-turbine dataset:

  1. (optional) a single-farm CARE "acceptance gate" to confirm the harness is sane,
  2. trains the autoencoder normal-behaviour model continually across the three wind
     farms A -> B -> C for one or more CL strategies and seeds,
  3. evaluates the CARE score on every farm seen so far at each stage,
  4. saves a results table (CSV + JSON) and the figures.

It is a thin, commented wrapper around the `care_cl` package
(`care_cl/experiments/run.py` and `sweep.py`); use those directly for finer control.

GPU / CUDA
----------
Training runs on CUDA by default. The device is read from `care_cl/config/default.yaml`
(`train.device: cuda`). If no GPU is found it automatically falls back to CPU and
prints a notice — so this same file runs on any machine.

REQUIREMENTS
------------
  pip install -r care_cl/requirements.txt
The dataset must be extracted at the path in `data.root` of the config
(default: D:/Datasets/Care), laid out as "Wind Farm A/B/C/datasets/*.csv".

HOW TO RUN
----------
  # Full study: all 6 strategies, bandit off, 3 seeds (this reproduces the paper table)
  python main.py

  # Quick smoke test (fewer epochs/rows) just to see it work end-to-end
  python main.py --quick

  # Only the single-farm sanity gate
  python main.py --gate --farm A

  # A focused comparison (e.g. the headline naive-vs-proposed result)
  python main.py --strategies naive replay_distill --seeds 0 1 2

  # Force CPU even if a GPU is present
  python main.py --device cpu

OUTPUTS  (all under care_cl/results/)
-------------------------------------
  records.csv               one row per (strategy, bandit, seed, stage, farm) + all CARE sub-scores
  metrics.csv               one row per (strategy, bandit, seed): final_avg_care, forgetting, BWT
  metrics_aggregated.csv    mean +/- std across seeds
  fig_forgetting_curves.png CARE on Farm A/B vs training stage, per strategy
  fig_final_per_farm_care.png  grouped bars of final per-farm CARE
  fig_bandit_ablation.png   bandit on/off delta (only if both were run)
  run_<...>.json            per-run JSON (full stage x farm matrix + metrics + records)
  cache/farm_*.npz          standardized farm tensors (speeds up reruns; safe to delete)
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

# Make sure the package is importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from care_cl.experiments.run import DEFAULT_CFG, load_cfg, train_single_farm
from care_cl.experiments.sweep import aggregate, run_grid
from care_cl.experiments.plots import make_all_plots

import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="CARE continual-learning benchmark (one-file runner).")
    ap.add_argument("--config", default=DEFAULT_CFG, help="path to YAML config")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu", "auto"],
                    help="override train.device (default: value in config = cuda)")
    ap.add_argument("--quick", action="store_true",
                    help="fewer epochs / capped rows for a fast end-to-end check")
    ap.add_argument("--gate", action="store_true",
                    help="run ONLY the single-farm CARE acceptance gate, then exit")
    ap.add_argument("--farm", default="A", help="farm for --gate (A|B|C)")
    # Experiment grid (defaults reproduce the full study).
    ap.add_argument("--strategies", nargs="+",
                    default=["naive", "joint", "ewc", "replay", "distill", "replay_distill"])
    ap.add_argument("--bandits", nargs="+", default=["off"], choices=["off", "on"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--align-modes", nargs="+", default=["shared_only"])
    return ap.parse_args()


def main():
    args = parse_args()

    # --- 0. Report the compute device we will actually use -----------------
    cfg0 = load_cfg(args.config, {"train.device": args.device} if args.device else {})
    want = cfg0["train"]["device"]
    on_cuda = want in ("cuda", "auto") and torch.cuda.is_available()
    if on_cuda:
        print(f"[device] Training on CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print(f"[device] Training on CPU (requested '{want}', "
              f"cuda_available={torch.cuda.is_available()}).")

    results_dir = cfg0["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # --- 1. Acceptance gate (optional, fast) -------------------------------
    if args.gate:
        # Train an AE on a single farm and print its CARE score. The harness is
        # trustworthy only if this lands ~0.6-0.7 (see spec section 6).
        res = train_single_farm(cfg0, args.farm, seed=args.seeds[0])
        print(f"\n[GATE] Farm {args.farm}: CARE={res['CARE']:.4f}  "
              f"(coverage={res['coverage']:.3f} earliness={res['earliness']:.3f} "
              f"reliability={res['reliability']:.3f} accuracy={res['accuracy']:.3f})")
        print(f"[GATE] within ~0.6-0.7 band: {0.55 <= res['CARE'] <= 0.75}")
        return

    # --- 2. Run the full experiment grid -----------------------------------
    # Each (strategy, bandit, seed) trains A->B->C and is evaluated on all farms
    # seen so far after every stage. `--device` is threaded through the config.
    if args.device:
        # run_grid reloads the config per run; bake the device override into it.
        import yaml
        with open(args.config) as f:
            tmp = yaml.safe_load(f)
        tmp["train"]["device"] = args.device
        args.config = os.path.join(results_dir, "_config_override.yaml")
        with open(args.config, "w") as f:
            yaml.safe_dump(tmp, f)

    # Each align mode is saved to its OWN subfolder (results/<mode>/) so running
    # both modes never overwrites the other's results. The data cache is shared.
    per_mode_metrics = []
    for mode in args.align_modes:
        out_dir = os.path.join(results_dir, mode)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[run] mode={mode} strategies={args.strategies} "
              f"bandits={args.bandits} seeds={args.seeds} quick={args.quick}")
        records_df, metrics_df = run_grid(
            args.config, args.strategies, args.bandits, args.seeds,
            [mode], quick=args.quick,
        )
        _save_mode_outputs(records_df, metrics_df, out_dir, mode)
        per_mode_metrics.append((mode, metrics_df))

    # --- Cross-mode comparison --------------------------------------------
    # Reads every results/<mode>/metrics.csv on disk, so modes run in separate
    # sessions still get compared (not just the modes from this invocation).
    _write_comparison(results_dir)
    print("Done.")


def _save_mode_outputs(records_df, metrics_df, out_dir, mode):
    """Save tables, figures and the markdown summary for one align mode."""
    os.makedirs(out_dir, exist_ok=True)
    records_path = os.path.join(out_dir, "records.csv")
    metrics_path = os.path.join(out_dir, "metrics.csv")
    agg_path = os.path.join(out_dir, "metrics_aggregated.csv")
    records_df.to_csv(records_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    aggregate(metrics_df).to_csv(agg_path, index=False)

    print(f"\n=== [{mode}] final_avg_care (mean over seeds) ===")
    print(metrics_df.groupby(["strategy", "bandit"])["final_avg_care"]
          .mean().unstack().to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"=== [{mode}] mean forgetting on Farm A (lower = better) ===")
    print(metrics_df.groupby("strategy")["forgetting_A"]
          .mean().to_string(float_format=lambda x: f"{x:+.4f}"))

    try:
        make_all_plots(records_df, metrics_df, out_dir)
    except Exception as e:
        print(f"(plots skipped: {e})")
    try:
        from care_cl.experiments.summary import build_markdown
        bandit = "off" if "off" in set(records_df["bandit"]) else records_df["bandit"].iloc[0]
        with open(os.path.join(out_dir, "results_summary.md"), "w", encoding="utf-8") as f:
            f.write(build_markdown(records_df, metrics_df, bandit))
    except Exception as e:
        print(f"(summary skipped: {e})")
    print(f"[{mode}] wrote tables + figures + results_summary.md to {out_dir}/")


def _write_comparison(results_dir):
    """Write a side-by-side comparison from every results/<mode>/metrics.csv on disk."""
    frames = []
    for mode in ("shared_only", "adapter"):
        mp = os.path.join(results_dir, mode, "metrics.csv")
        if not os.path.exists(mp):
            continue
        m = pd.read_csv(mp)
        g = m.groupby("strategy").agg(
            final_avg_care=("final_avg_care", "mean"),
            forgetting_A=("forgetting_A", "mean")).reset_index()
        g.insert(0, "align_mode", mode)
        frames.append(g)
    if len(frames) < 2:
        return  # need at least two modes to compare
    comp = pd.concat(frames, ignore_index=True)
    comp_csv = os.path.join(results_dir, "comparison_modes.csv")
    comp.to_csv(comp_csv, index=False)

    # Markdown: one block per metric, modes as columns (no tabulate dependency).
    def _table(piv):
        modes = list(piv.columns)
        head = "| strategy | " + " | ".join(modes) + " |"
        sep = "| " + " | ".join("---" for _ in range(len(modes) + 1)) + " |"
        body = []
        for strat, row in piv.iterrows():
            cells = [f"{row[m]:.4f}" if pd.notna(row[m]) else "—" for m in modes]
            body.append(f"| {strat} | " + " | ".join(cells) + " |")
        return "\n".join([head, sep] + body)

    lines = ["# Align-mode comparison (shared_only vs adapter)", "",
             "Mean over seeds. Higher CARE = better; lower forgetting = better.", ""]
    for metric, better in [("final_avg_care", "higher better"),
                           ("forgetting_A", "lower better")]:
        piv = comp.pivot(index="strategy", columns="align_mode", values=metric)
        lines += [f"## {metric} ({better})", "", _table(piv), ""]
    comp_md = os.path.join(results_dir, "comparison_modes.md")
    with open(comp_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nwrote cross-mode comparison:\n  {comp_csv}\n  {comp_md}")


if __name__ == "__main__":
    main()
