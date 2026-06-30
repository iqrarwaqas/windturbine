"""CARE-to-Compare dataset loader.

Implements the §2 loader contract. Matches the *actual* download layout:

    <root>/Wind Farm A/datasets/<event_id>.csv   (sep=';')
                       /event_info.csv
                       /feature_description.csv

Each dataset CSV holds one turbine's time series with both a `train` split
(normal behaviour) and a `prediction` split (which may contain one event).
Columns: time_stamp, asset_id, id, train_test, status_type_id, then
`<sensor>_<avg|min|max|std>` feature columns.

The loader discovers feature counts at load time and fails loudly on any
structural mismatch instead of guessing.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import pandas as pd

FARM_IDS = ["A", "B", "C"]
META_COLS = ["time_stamp", "asset_id", "id", "train_test", "status_type_id"]
EXPECTED_FEATURE_COUNTS = {"A": 86, "B": 257, "C": 957}  # sanity reference only


@dataclass
class Subdataset:
    id: str
    df: pd.DataFrame                      # full time-ordered frame (train + prediction)
    status_col: str | None
    anomaly_events: list[tuple[int, int]] # (start_pos, end_pos) inclusive, positional iloc
    is_normal_only: bool
    label: str = "normal"                 # "anomaly" | "normal"

    def split(self, which: str) -> pd.DataFrame:
        """Return the 'train' or 'prediction' rows (positional index preserved)."""
        return self.df[self.df["train_test"] == which]


@dataclass
class FarmData:
    farm_id: str
    feature_names: list[str]              # sensor feature columns, length F_farm
    train: list[Subdataset]               # subdatasets (use their train split)
    eval: list[Subdataset]                # subdatasets (use their prediction split)
    feature_description: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def _farm_dir(root: str, farm_id: str) -> str:
    d = os.path.join(root, f"Wind Farm {farm_id}")
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"Expected farm directory '{d}'. CARE layout requires "
            f"'Wind Farm A/B/C' folders under root={root!r}."
        )
    return d


def _parse_events(event_info: pd.DataFrame, df: pd.DataFrame, event_id: str):
    """Map an event's start/end `id` values to positional indices in `df`."""
    rows = event_info[event_info["event_id"].astype(str) == str(event_id)]
    if rows.empty:
        # No metadata row -> treat as normal-only.
        return [], True, "normal"
    row = rows.iloc[0]
    label = str(row["event_label"]).strip().lower()
    is_normal = label != "anomaly"

    id_to_pos = {v: i for i, v in enumerate(df["id"].to_numpy())}
    start_id, end_id = row["event_start_id"], row["event_end_id"]
    if pd.isna(start_id) or pd.isna(end_id):
        return [], is_normal, label
    start_pos = id_to_pos.get(int(start_id))
    end_pos = id_to_pos.get(int(end_id))
    if start_pos is None or end_pos is None:
        raise ValueError(
            f"Event {event_id}: start/end id ({start_id},{end_id}) not found in "
            f"the dataset's `id` column. Loader/data mismatch."
        )
    if end_pos < start_pos:
        start_pos, end_pos = end_pos, start_pos
    # Only anomaly events define a positive ground-truth window.
    events = [] if is_normal else [(start_pos, end_pos)]
    return events, is_normal, label


def _load_farm(root: str, farm_id: str, sep: str) -> FarmData:
    fdir = _farm_dir(root, farm_id)
    event_info = pd.read_csv(os.path.join(fdir, "event_info.csv"), sep=sep)
    feat_desc_path = os.path.join(fdir, "feature_description.csv")
    feat_desc = pd.read_csv(feat_desc_path, sep=sep, encoding="latin-1") \
        if os.path.exists(feat_desc_path) else pd.DataFrame()

    csv_paths = sorted(
        glob.glob(os.path.join(fdir, "datasets", "*.csv")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        if os.path.splitext(os.path.basename(p))[0].isdigit() else 0,
    )
    if not csv_paths:
        raise FileNotFoundError(f"No dataset CSVs under {fdir}/datasets")

    subdatasets: list[Subdataset] = []
    feature_names: list[str] | None = None

    for path in csv_paths:
        ev_id = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, sep=sep)

        missing_meta = [c for c in ("train_test", "id", "status_type_id") if c not in df.columns]
        if missing_meta:
            raise ValueError(f"{path}: missing required meta columns {missing_meta}")

        feats = [c for c in df.columns if c not in META_COLS]
        if feature_names is None:
            feature_names = feats
        elif feats != feature_names:
            raise ValueError(
                f"{path}: feature columns differ within farm {farm_id} "
                f"({len(feats)} vs {len(feature_names)}). Inconsistent schema."
            )

        events, is_normal, label = _parse_events(event_info, df, ev_id)
        subdatasets.append(
            Subdataset(
                id=ev_id, df=df, status_col="status_type_id",
                anomaly_events=events, is_normal_only=is_normal, label=label,
            )
        )

    return FarmData(
        farm_id=farm_id,
        feature_names=feature_names or [],
        train=subdatasets,
        eval=subdatasets,
        feature_description=feat_desc,
    )


def load_care(root: str, sep: str = ";") -> dict[str, FarmData]:
    """Return {"A": FarmData, "B": FarmData, "C": FarmData}.

    Raises if the directory structure or schema does not match the contract.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"CARE dataset root not found: {root!r}")

    farms = {fid: _load_farm(root, fid, sep) for fid in FARM_IDS}

    counts = {fid: fd.n_features for fid, fd in farms.items()}
    if len(set(counts.values())) != len(counts):
        raise ValueError(f"Farms must differ in feature count; got {counts}")

    return farms


if __name__ == "__main__":  # pragma: no cover - quick manual probe
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "D:/Datasets/Care"
    farms = load_care(root)
    for fid, fd in farms.items():
        n_anom = sum(not s.is_normal_only for s in fd.eval)
        print(f"Farm {fid}: {fd.n_features} features, {len(fd.eval)} datasets, "
              f"{n_anom} anomaly events")
