from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class StatCfg:
    stat: str
    table: str
    itemids: List[int]
    unit: str
    bounds: Tuple[float, float]


def _dominant_unit(units: pd.Series, expected: str) -> Tuple[str, bool]:
    vals = units.dropna().astype(str).str.strip().str.lower()
    expected_l = (expected or "").strip().lower()
    dominant = expected
    unit_conflict = False
    if len(vals) > 0:
        counts = vals.value_counts()
        dominant_seen = counts.index[0]
        dominant = dominant_seen
        # Keep expected if given but normalize case
        if expected:
            dominant = expected
        # conflict if any unit differs from expected (if provided), else if >1 unique
        uniq = set(vals.unique())
        if expected_l:
            unit_conflict = any(u != expected_l for u in uniq)
        else:
            unit_conflict = len(uniq) > 1
    return dominant, unit_conflict


def _mad(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _robust_slope_per_hour(times: np.ndarray, values: np.ndarray) -> float:
    """
    Approximate Theil–Sen slope: median of pairwise slopes, using subsampling for n>200.
    Returns slope in units per hour.
    Deterministic (no RNG): uses evenly spaced subsampling when needed.
    """
    n = len(values)
    if n < 2:
        return 0.0
    # Convert to hours from first timestamp
    t0 = times[0]
    th = (times - t0).astype('timedelta64[s]').astype(np.float64) / 3600.0
    # Guard: drop zero-time duplicates to avoid divide-by-zero in slope
    # Keep first occurrence at each unique time
    order = np.argsort(th)
    th = th[order]
    v = values[order]
    uniq_mask = np.concatenate([[True], np.diff(th) > 0])
    th = th[uniq_mask]
    v = v[uniq_mask]
    m = len(v)
    if m < 2:
        return 0.0
    # Subsample indices if too many points
    if m > 200:
        k = 100
        idx = np.linspace(0, m - 1, k, dtype=int)
    else:
        idx = np.arange(m)
    # Compute pairwise slopes for i<j among selected indices
    sel_t = th[idx]
    sel_v = v[idx]
    slopes: List[float] = []
    for i in range(len(sel_t)):
        dt = sel_t[i + 1 :] - sel_t[i]
        dv = sel_v[i + 1 :] - sel_v[i]
        mask = dt != 0
        if mask.any():
            s = dv[mask] / dt[mask]
            slopes.append(s)
    if not slopes:
        return 0.0
    all_slopes = np.concatenate(slopes)
    return float(np.median(all_slopes))


def _iso(ts: Optional[pd.Timestamp]) -> Optional[str]:
    if ts is None or pd.isna(ts):
        return None
    # ensure naive ISO format
    return ts.to_pydatetime().isoformat()


def compute_daily_features(
    rows: List[Dict[str, Any]],
    cfg: StatCfg,
    day: str,
    head_n: int = 3,
    tail_n: int = 3,
) -> Dict[str, Any]:
    """
    Compute daily features and flags for a stat given raw rows from SQL.
    Returns a dict payload ready for summarization.
    """
    # Base empty payload for no-data case
    base_payload: Dict[str, Any] = {
        "stat": cfg.stat,
        "units": cfg.unit,
        "day": day,
        "range": {"min": None, "max": None},
        "central": {"mean": None, "median": None},
        "percentiles": {"p05": None, "p95": None},
        "trend": {"slope_per_hr": 0.0, "delta_last_first": None},
        "variability": {"std": None, "mad": None},
        "coverage": {
            "n_obs": 0,
            "prop_day": 0.0,
            "hours_w_obs": 0,
            "minutes_w_obs": 0,
            "n_missing_slots": 24,
        },
        "outliers": {"n": 0, "timestamps": []},
        "flags": {
            "unit_conflicts": False,
            "sparse": True,
            "value_out_of_bounds": False,
            "no_data": True,
            "duplicates": False,
            "approximate": False,
        },
        "head_rows_csv": "",
        "tail_rows_csv": "",
    }

    if len(rows) == 0:
        return base_payload

    df = pd.DataFrame(rows)
    # Ensure expected columns exist
    for col in ["charttime", "valuenum", "valueuom", "itemid", "error"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Parse datetimes
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df = df.dropna(subset=["charttime", "valuenum"]).sort_values("charttime")

    # Keep a copy for duplicates flag
    dup_flag = df.duplicated(subset=["charttime"]).any()

    # Dominant unit and conflicts
    dominant_unit, unit_conflict = _dominant_unit(df["valueuom"], cfg.unit)

    # Coverage metrics
    n_obs = len(df)
    # unique hours with at least one obs
    hours = df["charttime"].dt.floor("h")
    hours_w_obs = int(hours.nunique())
    n_missing_slots = max(0, 24 - hours_w_obs)
    # unique minutes coverage
    minutes = df["charttime"].dt.floor("min").nunique()
    prop_day = float(min(1.0, hours_w_obs / 24.0))

    # Value sanity and initial bounds-based outliers
    low, high = cfg.bounds
    val = pd.to_numeric(df["valuenum"], errors="coerce")
    in_bounds = (val >= low) & (val <= high)
    bounds_outliers_idx = (~in_bounds).to_numpy().nonzero()[0]
    value_out_of_bounds = int((~in_bounds).sum()) > 0

    # Clean series for stats/trend
    clean = df[in_bounds].copy()
    clean_val = pd.to_numeric(clean["valuenum"], errors="coerce")
    clean = clean.loc[clean_val.notna()]
    clean_val = pd.to_numeric(clean["valuenum"])  # aligned

    # Compute stats on clean data
    if len(clean) == 0:
        payload = base_payload.copy()
        payload["units"] = dominant_unit or cfg.unit
        payload["flags"]["unit_conflicts"] = unit_conflict
        payload["flags"]["value_out_of_bounds"] = value_out_of_bounds
        payload["flags"]["duplicates"] = bool(dup_flag)
        payload["coverage"]["n_obs"] = int(n_obs)
        payload["coverage"]["prop_day"] = prop_day
        payload["coverage"]["hours_w_obs"] = hours_w_obs
        payload["coverage"]["n_missing_slots"] = n_missing_slots
        payload["coverage"]["minutes_w_obs"] = int(minutes)
        # outliers are all observations (since none clean)
        payload["outliers"]["n"] = int(n_obs)
        payload["outliers"]["timestamps"] = [
            ts.isoformat() for ts in df["charttime"].iloc[: min(5, n_obs)]
        ]
        # samples
        sample_cols = ["charttime", "valuenum", "valueuom", "itemid"]
        payload["head_rows_csv"] = (
            df[sample_cols].head(3).to_csv(index=False) if n_obs > 0 else ""
        )
        payload["tail_rows_csv"] = (
            df[sample_cols].tail(3).to_csv(index=False) if n_obs > 0 else ""
        )
        return payload

    # Basic stats
    v = clean_val.to_numpy(dtype=float)
    v_min = float(np.min(v))
    v_max = float(np.max(v))
    v_mean = float(np.mean(v))
    v_median = float(np.median(v))
    v_std = float(np.std(v, ddof=0)) if len(v) > 1 else 0.0
    v_p05 = float(np.percentile(v, 5))
    v_p95 = float(np.percentile(v, 95))
    v_mad = float(_mad(v))

    # Trend
    times = clean["charttime"].to_numpy(dtype="datetime64[ns]")
    slope = _robust_slope_per_hour(times, v)
    delta = float(v[-1] - v[0]) if len(v) >= 2 else 0.0

    # Extremes timestamps
    idx_min = int(np.argmin(v))
    idx_max = int(np.argmax(v))
    t_min = _iso(pd.to_datetime(times[idx_min]))
    t_max = _iso(pd.to_datetime(times[idx_max]))

    # Robust outliers (MAD-based) on clean values
    if np.isnan(v_mad) or v_mad == 0:
        rob_mask = np.zeros_like(v, dtype=bool)
    else:
        rob_mask = np.abs(v - v_median) > 3.5 * v_mad
    rob_out_idx = np.where(rob_mask)[0]

    # Combine bounds and robust outliers (map indices to original df)
    # Bounds outliers already from full df; robust outliers refer to clean subset
    rob_ts = clean["charttime"].to_numpy()[rob_out_idx]

    outlier_ts = list(df["charttime"].iloc[bounds_outliers_idx].to_list()) + list(rob_ts)
    outlier_ts_iso: List[str] = []
    for ts in outlier_ts[:5]:
        try:
            ts_dt = pd.to_datetime(ts)
            if pd.isna(ts_dt):
                continue
            outlier_ts_iso.append(ts_dt.to_pydatetime().isoformat())
        except Exception:
            outlier_ts_iso.append(str(ts))

    # Sparse flag: few observations or low coverage
    sparse = bool((n_obs < 6) or (hours_w_obs <= 3) or (prop_day < 0.1))

    # Samples
    sample_cols = ["charttime", "valuenum", "valueuom", "itemid"]
    head_csv = clean[sample_cols].head(head_n).to_csv(index=False)
    tail_csv = clean[sample_cols].tail(tail_n).to_csv(index=False)

    payload: Dict[str, Any] = {
        "stat": cfg.stat,
        "units": dominant_unit or cfg.unit,
        "day": day,
        "range": {"min": v_min, "max": v_max, "t_min": t_min, "t_max": t_max},
        "central": {"mean": v_mean, "median": v_median},
        "percentiles": {"p05": v_p05, "p95": v_p95},
        "trend": {"slope_per_hr": slope, "delta_last_first": delta},
        "variability": {"std": v_std, "mad": v_mad},
        "coverage": {
            "n_obs": int(n_obs),
            "prop_day": prop_day,
            "hours_w_obs": hours_w_obs,
            "minutes_w_obs": int(minutes),
            "n_missing_slots": n_missing_slots,
        },
        "outliers": {"n": int(len(outlier_ts)), "timestamps": outlier_ts_iso},
        "flags": {
            "unit_conflicts": unit_conflict,
            "sparse": sparse,
            "value_out_of_bounds": value_out_of_bounds,
            "no_data": False,
            "duplicates": bool(dup_flag),
            "approximate": False,
        },
        "head_rows_csv": head_csv,
        "tail_rows_csv": tail_csv,
    }
    return payload
