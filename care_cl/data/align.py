"""Cross-farm feature alignment (§3).

The three farms have different, only-partially-overlapping feature sets. To make
a single latent space comparable across farms we encode a small canonical set of
physically-meaningful signals (`SHARED_SIGNALS`) present in *all three* farms.

The per-farm column mapping below was derived from each farm's
`feature_description.csv` (matched by physical description). Every mapped column
is verified to exist at load time; a missing column fails loudly (§3 caveat).

`align_mode`:
  - "shared_only"  -> encode only the standardized shared signals (default).
  - "adapter"      -> shared core + per-farm linear adapter over the *full* feature
                      vector (richer, optional). Adapters live in models/ae_nbm.py;
                      this module just exposes the full standardized matrix for them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .loader import FarmData, Subdataset

# Canonical cross-farm signals (order is fixed and defines the shared input layout).
SHARED_SIGNALS = ["active_power", "wind_speed", "reactive_power", "rotor_speed", "ambient_temp"]

# canonical signal -> per-farm data column (the 10-min average channel).
FARM_SIGNAL_MAP: dict[str, dict[str, str]] = {
    "A": {
        "active_power":   "power_30_avg",      # Grid power [kW]
        "wind_speed":     "wind_speed_3_avg",  # Windspeed [m/s]
        "reactive_power": "sensor_31_avg",     # Grid reactive power [kVAr]
        "rotor_speed":    "sensor_52_avg",     # Rotor rpm
        "ambient_temp":   "sensor_0_avg",      # Ambient temperature [C]
    },
    "B": {
        "active_power":   "power_62_avg",         # Active power [kW]
        "wind_speed":     "wind_speed_61_avg",    # Wind speed [m/s]
        "reactive_power": "reactive_power_11_avg",# Reactive power [kvar]
        "rotor_speed":    "sensor_25_avg",        # Rotor speed [rpm]
        "ambient_temp":   "sensor_8_avg",         # Outside temperature [C]
    },
    "C": {
        "active_power":   "power_6_avg",            # Active power HV grid [kW]
        "wind_speed":     "wind_speed_235_avg",     # Wind speed 1 [m/s]
        "reactive_power": "reactive_power_122_avg", # Reactive power HV grid [kvar]
        "rotor_speed":    "sensor_144_avg",         # Rotor speed 1 [1/min]
        "ambient_temp":   "sensor_7_avg",           # Ambient temperature [C]
    },
}


@dataclass
class FarmAligner:
    """Standardizes a farm's signals; fit on normal training data only."""
    farm_id: str
    align_mode: str
    shared_cols: list[str]                 # farm columns for SHARED_SIGNALS, in canon order
    feature_cols: list[str]                # full feature column list (for adapter mode)
    normal_status_ids: tuple[int, ...]
    shared_scaler: StandardScaler
    full_scaler: StandardScaler | None

    @property
    def n_shared(self) -> int:
        return len(self.shared_cols)

    @property
    def input_dim(self) -> int:
        return len(self.feature_cols) if self.align_mode == "adapter" else self.n_shared

    def _raw(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        x = df[cols].to_numpy(dtype=np.float64)
        # Anonymized SCADA contains occasional NaNs; impute with column means
        # learned from the (already-seen) frame, then 0 for all-NaN columns.
        if np.isnan(x).any():
            col_mean = np.nanmean(x, axis=0)
            col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
            inds = np.where(np.isnan(x))
            x[inds] = np.take(col_mean, inds[1])
        return x

    def transform_shared(self, df: pd.DataFrame) -> np.ndarray:
        return self.shared_scaler.transform(self._raw(df, self.shared_cols))

    def transform_full(self, df: pd.DataFrame) -> np.ndarray:
        assert self.full_scaler is not None
        return self.full_scaler.transform(self._raw(df, self.feature_cols))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.transform_full(df) if self.align_mode == "adapter" \
            else self.transform_shared(df)


def _resolve_shared_cols(farm: FarmData) -> list[str]:
    mapping = FARM_SIGNAL_MAP.get(farm.farm_id)
    if mapping is None:
        raise ValueError(f"No SHARED_SIGNALS mapping for farm {farm.farm_id!r}")
    cols = []
    for sig in SHARED_SIGNALS:
        col = mapping.get(sig)
        if col is None or col not in farm.feature_names:
            raise ValueError(
                f"Farm {farm.farm_id}: shared signal {sig!r} -> column {col!r} "
                f"not found in data. Available example cols: {farm.feature_names[:8]}... "
                f"Cannot align farms; resolve SHARED_SIGNALS before proceeding (§3)."
            )
        cols.append(col)
    return cols


def normal_training_frame(farm: FarmData, normal_status_ids) -> pd.DataFrame:
    """Concatenate all subdatasets' train-split rows that are in normal operation."""
    parts = []
    for sub in farm.train:
        tr = sub.split("train")
        if sub.status_col and sub.status_col in tr.columns:
            tr = tr[tr[sub.status_col].isin(list(normal_status_ids))]
        parts.append(tr)
    if not parts:
        raise ValueError(f"Farm {farm.farm_id}: no normal training rows found")
    return pd.concat(parts, ignore_index=True)


def fit_aligner(farm: FarmData, align_mode: str, normal_status_ids) -> FarmAligner:
    """Fit StandardScaler(s) on normal training data only (never on eval)."""
    shared_cols = _resolve_shared_cols(farm)
    train_df = normal_training_frame(farm, normal_status_ids)

    shared_scaler = StandardScaler().fit(
        _impute(train_df[shared_cols].to_numpy(dtype=np.float64))
    )
    full_scaler = None
    if align_mode == "adapter":
        full_scaler = StandardScaler().fit(
            _impute(train_df[farm.feature_names].to_numpy(dtype=np.float64))
        )

    return FarmAligner(
        farm_id=farm.farm_id,
        align_mode=align_mode,
        shared_cols=shared_cols,
        feature_cols=list(farm.feature_names),
        normal_status_ids=tuple(normal_status_ids),
        shared_scaler=shared_scaler,
        full_scaler=full_scaler,
    )


def _impute(x: np.ndarray) -> np.ndarray:
    if np.isnan(x).any():
        col_mean = np.nanmean(x, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        inds = np.where(np.isnan(x))
        x[inds] = np.take(col_mean, inds[1])
    return x


def normal_training_matrix(farm: FarmData, aligner: FarmAligner) -> np.ndarray:
    """Standardized matrix of normal training rows (model input space)."""
    df = normal_training_frame(farm, aligner.normal_status_ids)
    return aligner.transform(df)
