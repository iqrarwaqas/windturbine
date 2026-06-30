"""Continual-learning protocol (§5): A -> B -> C sequence + stage x farm eval.

To stay within consumer RAM (Farm C has 952 features x 58 datasets), we never hold
all raw frames at once. For each farm we build a *compact cache* of standardized
matrices (shared-signal space by default), reading only the columns we need, then
discard the heavy frames. Caches are persisted to results/cache/*.npz for fast reruns.
"""
from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..data.align import (FARM_SIGNAL_MAP, SHARED_SIGNALS, FarmAligner, _impute)
from ..data.loader import FARM_IDS, META_COLS
from ..eval.care_score import DatasetEval, care_score, evaluate_dataset
from ..models.ae_nbm import build_model, reconstruction_errors
from ..rl.bandit_threshold import tune_threshold
from sklearn.preprocessing import StandardScaler


@dataclass
class EvalItem:
    id: str
    is_normal_only: bool
    label: str
    X: np.ndarray                       # standardized prediction-frame matrix
    status: np.ndarray                  # status_type_id per prediction row
    window_rel: tuple[int, int] | None  # event window in prediction-frame index


@dataclass
class FarmCache:
    farm_id: str
    input_dim: int
    train_X: np.ndarray
    eval_items: list[EvalItem]


def _farm_dir(root: str, farm_id: str) -> str:
    return os.path.join(root, f"Wind Farm {farm_id}")


def resolve_device(name: str) -> torch.device:
    if name in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA requested but not available -> falling back to CPU.")
    return torch.device("cpu")


def _shared_cols(farm_id: str) -> list[str]:
    return [FARM_SIGNAL_MAP[farm_id][s] for s in SHARED_SIGNALS]


def _cache_path(cfg: dict, farm_id: str) -> str:
    mode = cfg["align"]["mode"]
    d = os.path.join(cfg["paths"]["results_dir"], "cache")
    os.makedirs(d, exist_ok=True)
    # Tag the cache with the shared-signal set so changing SHARED_SIGNALS (or mode)
    # automatically invalidates stale caches instead of silently reusing them.
    sig = "full" if mode == "adapter" else "_".join(SHARED_SIGNALS)
    key = hashlib.md5(sig.encode()).hexdigest()[:8]
    return os.path.join(d, f"farm_{farm_id}_{mode}_{key}.npz")


def build_farm_cache(cfg: dict, farm_id: str, use_disk: bool = True) -> FarmCache:
    """Build (or load) the compact standardized cache for one farm."""
    path = _cache_path(cfg, farm_id)
    if use_disk and os.path.exists(path):
        return _load_cache(path, farm_id)

    root, sep = cfg["data"]["root"], cfg["data"]["sep"]
    normal_ids = cfg["data"]["normal_status_ids"]
    mode = cfg["align"]["mode"]
    fdir = _farm_dir(root, farm_id)

    event_info = pd.read_csv(os.path.join(fdir, "event_info.csv"), sep=sep)
    csv_paths = sorted(glob.glob(os.path.join(fdir, "datasets", "*.csv")),
                       key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    shared_cols = _shared_cols(farm_id)
    if mode == "shared_only":
        usecols = META_COLS + shared_cols
        feat_cols = shared_cols
    else:  # adapter: need full feature set (heavy)
        usecols = None
        feat_cols = None

    train_raw_parts = []
    raw_items = []  # (id, is_normal, label, raw_pred_matrix, status, window_rel)

    # In adapter mode we keep the full (wide) feature set, so cap the normal-train
    # rows kept per file to bound memory (Farm C is 952 features x 58 files).
    cap_per_file = None
    if mode == "adapter":
        train_cap = cfg["align"].get("adapter_max_train_rows")
        if train_cap:
            cap_per_file = max(1, int(train_cap) // max(1, len(csv_paths)))
    rng = np.random.default_rng(cfg.get("seed", 0))

    for p in csv_paths:
        ev_id = os.path.splitext(os.path.basename(p))[0]
        df = pd.read_csv(p, sep=sep, usecols=usecols)
        if feat_cols is None:
            feat_cols = [c for c in df.columns if c not in META_COLS]

        is_train = df["train_test"] == "train"
        is_pred = df["train_test"] == "prediction"
        normal_mask = df["status_type_id"].isin(normal_ids)

        tr = df.loc[is_train & normal_mask, feat_cols].to_numpy(dtype=np.float64)
        if cap_per_file and len(tr) > cap_per_file:
            tr = tr[rng.choice(len(tr), cap_per_file, replace=False)]
        if len(tr):
            train_raw_parts.append(tr)

        pred_df = df.loc[is_pred]
        pred_raw = pred_df[feat_cols].to_numpy(dtype=np.float64)
        status = pred_df["status_type_id"].to_numpy()

        label, is_normal, window_rel = _event_for(event_info, df, ev_id, is_pred.to_numpy())
        raw_items.append((ev_id, is_normal, label, pred_raw, status, window_rel))

    train_raw = _impute(np.concatenate(train_raw_parts, axis=0))
    scaler = StandardScaler().fit(train_raw)
    train_X = scaler.transform(train_raw).astype(np.float32)

    eval_items = []
    for ev_id, is_normal, label, pred_raw, status, window_rel in raw_items:
        X = scaler.transform(_impute(pred_raw)).astype(np.float32) if len(pred_raw) \
            else np.zeros((0, len(feat_cols)), dtype=np.float32)
        eval_items.append(EvalItem(ev_id, is_normal, label, X, status, window_rel))

    cache = FarmCache(farm_id, train_X.shape[1], train_X, eval_items)
    if use_disk:
        _save_cache(path, cache)
    return cache


def _event_for(event_info, df, ev_id, pred_mask_full):
    """Return (label, is_normal_only, window_rel) where window_rel is in pred-frame index."""
    rows = event_info[event_info["event_id"].astype(str) == str(ev_id)]
    if rows.empty:
        return "normal", True, None
    row = rows.iloc[0]
    label = str(row["event_label"]).strip().lower()
    if label != "anomaly" or pd.isna(row["event_start_id"]) or pd.isna(row["event_end_id"]):
        return label, True, None

    id_to_pos = {v: i for i, v in enumerate(df["id"].to_numpy())}
    s = id_to_pos.get(int(row["event_start_id"]))
    e = id_to_pos.get(int(row["event_end_id"]))
    if s is None or e is None:
        return label, False, None
    if e < s:
        s, e = e, s
    pred_pos = np.where(pred_mask_full)[0]
    in_win = (pred_pos >= s) & (pred_pos <= e)
    if not in_win.any():
        return label, False, None
    rel = np.where(in_win)[0]
    return label, False, (int(rel[0]), int(rel[-1]))


def _save_cache(path: str, cache: FarmCache):
    # Object-typed metadata (ids, labels, windows) is stored separately so the
    # main archive stays a plain numeric npz.
    np.save(path + ".meta.npy",
            np.array([[it.id, it.is_normal_only, it.label, it.window_rel]
                      for it in cache.eval_items], dtype=object),
            allow_pickle=True)
    np.savez_compressed(path, train_X=cache.train_X,
                        input_dim=np.array(cache.input_dim),
                        **{f"X_{i}": it.X for i, it in enumerate(cache.eval_items)},
                        **{f"status_{i}": it.status for i, it in enumerate(cache.eval_items)})


def _load_cache(path: str, farm_id: str) -> FarmCache:
    z = np.load(path)
    meta = np.load(path + ".meta.npy", allow_pickle=True)
    items = []
    for i in range(len(meta)):
        ev_id, is_normal, label, window_rel = meta[i]
        items.append(EvalItem(str(ev_id), bool(is_normal), str(label),
                              z[f"X_{i}"], z[f"status_{i}"],
                              None if window_rel is None else tuple(window_rel)))
    return FarmCache(farm_id, int(z["input_dim"]), z["train_X"], items)


# --------------------------------------------------------------------------
# Threshold selection
# --------------------------------------------------------------------------
def select_threshold(train_errors: np.ndarray, cfg: dict) -> float:
    tcfg = cfg["threshold"]
    if tcfg["bandit"] == "off":
        return float(np.quantile(train_errors, tcfg["fixed_quantile"]))

    # CARE-aligned proxy reward computed on TRAINING errors only (never eval):
    # split into calibration normals + synthetic "fault" errors (elevated tail).
    rng = np.random.default_rng(cfg.get("seed", 0))
    perm = rng.permutation(len(train_errors))
    calib = train_errors[perm[: len(perm) // 2]]
    pseudo_fault = train_errors[perm[len(perm) // 2:]] * 3.0  # simulate elevated recon error

    def reward(thr: float) -> float:
        detect = float((pseudo_fault > thr).mean())          # detection benefit
        false_alarm = float((calib > thr).mean())            # penalty (w4=2 in CARE)
        return detect - 2.0 * false_alarm

    return tune_threshold(train_errors, reward, tcfg)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate_farm(model, cache: FarmCache, threshold: float, cfg: dict,
                  device, farm_id: str | None = None) -> dict:
    evals: list[DatasetEval] = []
    recon_losses = []
    for it in cache.eval_items:
        if len(it.X) == 0:
            continue
        scores = reconstruction_errors(model, it.X, device, farm_id)
        recon_losses.append(float(scores.mean()))
        pred = scores > threshold
        de = evaluate_dataset(
            pred, it.status,
            is_normal_only=it.is_normal_only,
            event_window=it.window_rel,
            normal_status_ids=cfg["data"]["normal_status_ids"],
            beta=cfg["eval"]["beta"],
            criticality_threshold=cfg["eval"]["criticality_threshold"],
        )
        de.id = it.id
        evals.append(de)

    res = care_score(evals, beta=cfg["eval"]["beta"], weights=cfg["eval"]["weights"])
    res["recon_loss"] = float(np.mean(recon_losses)) if recon_losses else 0.0
    res["threshold"] = float(threshold)
    return res


# --------------------------------------------------------------------------
# Main CL run
# --------------------------------------------------------------------------
def run_sequence(cfg: dict, strategy, seed: int) -> dict:
    """Train through the farm sequence; return stage x farm CARE matrix + records."""
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    device = resolve_device(cfg["train"]["device"])
    sequence = cfg["cl"]["sequence"]

    caches = {fid: build_farm_cache(cfg, fid) for fid in sequence}
    input_dim = caches[sequence[0]].input_dim
    for fid in sequence:
        if caches[fid].input_dim != input_dim and cfg["align"]["mode"] == "shared_only":
            raise ValueError("shared_only requires identical input dims across farms")

    mode = cfg["align"]["mode"]
    if mode == "adapter":
        # Shared core sees a fixed-width embedding; each farm gets its own
        # input adapter (full_dim -> embed) and output head (embed -> full_dim).
        shared_dim = cfg["align"].get("adapter_embed_dim", 32)
        model = build_model(shared_dim, cfg)
        for f in sequence:
            model.add_farm_adapter(f, caches[f].input_dim)
        model = model.to(device)
        eval_fid = lambda f: f            # use the farm's adapter at train/eval
    else:
        model = build_model(input_dim, cfg).to(device)
        eval_fid = lambda f: None         # shared-core path; no adapter

    optim = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                             weight_decay=cfg["train"]["weight_decay"])

    matrix: dict[str, dict[str, float]] = {}
    records = []
    cap = cfg["train"].get("max_train_rows_per_farm")

    def _cap(X):
        if cap and len(X) > cap:
            return X[np.random.default_rng(seed).choice(len(X), cap, replace=False)]
        return X

    def _score_farm(stage, f):
        thr = select_threshold(
            reconstruction_errors(model, caches[f].train_X, device, eval_fid(f)), cfg)
        res = evaluate_farm(model, caches[f], thr, cfg, device, eval_fid(f))
        matrix.setdefault(stage, {})[f] = res["CARE"]
        records.append(_record(cfg, strategy, seed, stage, f, res))

    if strategy.name == "joint":
        if mode == "adapter":
            # Per-farm feature dims cannot be pooled into one matrix, so we make
            # one training pass over each farm (a multi-task reference, not a
            # strict pooled upper bound) before evaluating all farms.
            for f in sequence:
                strategy.train_on(f, model, optim, _cap(caches[f].train_X))
        else:
            pooled = np.concatenate([caches[f].train_X for f in sequence], axis=0)
            strategy.train_on(sequence[-1], model, optim, _cap(pooled))
        for f in sequence:
            _score_farm(sequence[-1], f)
        return {"matrix": matrix, "records": records}

    for stage in sequence:
        X = _cap(caches[stage].train_X)
        strategy.train_on(stage, model, optim, X)
        strategy.end_of_farm(stage, model, X)
        for f in sequence[: sequence.index(stage) + 1]:
            _score_farm(stage, f)

    return {"matrix": matrix, "records": records}


def _record(cfg, strategy, seed, stage, farm, res):
    return {
        "strategy": strategy.name,
        "bandit": cfg["threshold"]["bandit"],
        "align_mode": cfg["align"]["mode"],
        "seed": seed,
        "stage": stage,
        "farm": farm,
        "CARE": res["CARE"],
        "coverage": res["coverage"],
        "earliness": res["earliness"],
        "reliability": res["reliability"],
        "accuracy": res["accuracy"],
        "recon_loss": res["recon_loss"],
        "threshold": res["threshold"],
    }
