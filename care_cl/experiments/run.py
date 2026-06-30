"""CLI entrypoint: run one (strategy, seed) continual-learning config (§8/§11.1).

Examples:
    python -m care_cl.experiments.run --strategy naive --seed 0
    python -m care_cl.experiments.run --strategy replay_distill --bandit on --seed 1
    python -m care_cl.experiments.run --gate --farm A      # §6 single-farm gate
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import yaml

from ..cl.protocol import (build_farm_cache, evaluate_farm, resolve_device,
                           run_sequence, select_threshold)
from ..cl.strategies import build_strategy
from ..eval.cl_metrics import summarize
from ..models.ae_nbm import build_model, reconstruction_errors

DEFAULT_CFG = os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml")


def load_cfg(path: str, overrides: dict) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in overrides.items():
        if v is None:
            continue
        _set(cfg, k, v)
    return cfg


def _set(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _seed_all(seed):
    import random
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train_single_farm(cfg: dict, farm_id: str, seed: int) -> dict:
    """Train an AE on one farm and score it — the §6 acceptance gate."""
    import torch
    _seed_all(seed)
    device = resolve_device(cfg["train"]["device"])
    cache = build_farm_cache(cfg, farm_id)
    model = build_model(cache.input_dim, cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                             weight_decay=cfg["train"]["weight_decay"])
    strat = build_strategy("naive", cfg, device)

    X = cache.train_X
    cap = cfg["train"].get("max_train_rows_per_farm")
    if cap and len(X) > cap:
        X = X[np.random.default_rng(seed).choice(len(X), cap, replace=False)]
    strat.train_on(farm_id, model, optim, X)

    thr = select_threshold(reconstruction_errors(model, cache.train_X, device), cfg)
    return evaluate_farm(model, cache, thr, cfg, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--strategy", default="naive")
    ap.add_argument("--bandit", choices=["on", "off"], default=None)
    ap.add_argument("--align-mode", choices=["shared_only", "adapter"], default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", action="store_true", help="run single-farm CARE gate")
    ap.add_argument("--farm", default="A")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    overrides = {"threshold.bandit": args.bandit, "align.mode": args.align_mode,
                 "seed": args.seed}
    cfg = load_cfg(args.config, overrides)
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    t0 = time.time()

    if args.gate:
        res = train_single_farm(cfg, args.farm, args.seed)
        print(f"[GATE] Farm {args.farm}: CARE={res['CARE']:.4f} "
              f"(coverage={res['coverage']:.3f} earliness={res['earliness']:.3f} "
              f"reliability={res['reliability']:.3f} accuracy={res['accuracy']:.3f})")
        gate_ok = 0.55 <= res["CARE"] <= 0.75
        print(f"[GATE] in ~0.6-0.7 band: {gate_ok}")
        out = {"mode": "gate", "farm": args.farm, **res, "seconds": time.time() - t0}
    else:
        strat = build_strategy(args.strategy, cfg, resolve_device(cfg["train"]["device"]))
        run = run_sequence(cfg, strat, args.seed)
        metrics = summarize(run["matrix"], cfg["cl"]["sequence"])
        print(f"[{args.strategy} seed={args.seed} bandit={cfg['threshold']['bandit']}] "
              f"final_avg_care={metrics['final_avg_care']:.4f} "
              f"mean_forgetting={metrics['mean_forgetting']:.4f}")
        out = {"mode": "cl", "strategy": args.strategy, "seed": args.seed,
               "bandit": cfg["threshold"]["bandit"], "align_mode": cfg["align"]["mode"],
               "matrix": run["matrix"], "metrics": metrics, "records": run["records"],
               "seconds": time.time() - t0}

    out_path = args.out or os.path.join(
        cfg["paths"]["results_dir"],
        f"run_{args.strategy}_bandit-{cfg['threshold']['bandit']}_seed{args.seed}"
        f"{'_gate-'+args.farm if args.gate else ''}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path} ({out['seconds']:.1f}s)")


if __name__ == "__main__":
    main()
