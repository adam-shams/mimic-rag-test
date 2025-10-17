from typing import Any, Dict, List, Optional, Sequence, Tuple
import csv
import io
import json
import os
from datetime import datetime


def _fmt_float(x: Any) -> Any:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return x


def _build_json(sp: Dict[str, Any]) -> str:
    # Construct JSON strictly with DailyStatJSON keys
    js = {
        "stat": sp.get("stat"),
        "units": sp.get("units"),
        "day": sp.get("day"),
        "range": sp.get("range", {}),
        "central": sp.get("central", {}),
        "percentiles": sp.get("percentiles", {}),
        "trend": sp.get("trend", {}),
        "variability": sp.get("variability", {}),
        "coverage": sp.get("coverage", {}),
        "outliers": sp.get("outliers", {}),
        "flags": sp.get("flags", {}),
    }
    # Ensure JSON-serializable (cast numpy types)
    def _py(v: Any) -> Any:
        try:
            import numpy as np  # local import to avoid hard dep here

            if isinstance(v, (np.floating, np.integer)):
                return v.item()
        except Exception:
            pass
        return v

    def _walk(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _walk(_py(v)) for k, v in x.items()}
        if isinstance(x, list):
            return [_walk(_py(v)) for v in x]
        return _py(x)

    return json.dumps(_walk(js))


def _offline_summary(sp: Dict[str, Any]) -> str:
    # Deterministic 2–5 sentence summary using provided stats only
    stat = sp.get("stat", "stat")
    unit = sp.get("units", "")
    cov = sp.get("coverage", {})
    n = cov.get("n_obs", 0)
    prop = float(cov.get("prop_day") or 0.0)
    rng = sp.get("range", {})
    vmin, vmax = rng.get("min"), rng.get("max")
    med = sp.get("central", {}).get("median")
    trend = sp.get("trend", {})
    slope = trend.get("slope_per_hr")
    delta = trend.get("delta_last_first")
    out_n = sp.get("outliers", {}).get("n")

    sents = []
    if n == 0:
        sents.append(f"No {stat} measurements recorded; coverage {prop:.2f} of day.")
    else:
        if vmin is not None and vmax is not None:
            sents.append(f"{stat} ranged {vmin}–{vmax} {unit}.")
        if med is not None:
            sents.append(f"Median {med} {unit}; {n} readings, coverage {prop:.2f}.")
        if slope is not None and delta is not None:
            sents.append(f"Trend {slope} {unit}/hr; Δ {delta}.")
        if out_n is not None:
            sents.append(f"Outliers: {out_n}.")
    text = " ".join([s for s in sents if s][:5])
    return f"{text}\n\n{_build_json(sp)}"


def summarize(stat_payload: Dict[str, Any]) -> str:
    sp = stat_payload
    # Normalize coverage prop_day to float with 2 decimals in text rendering
    try:
        sp["coverage"]["prop_day"] = float(sp["coverage"].get("prop_day") or 0.0)
    except Exception:
        sp.setdefault("coverage", {})["prop_day"] = 0.0
    # Optionally use LLM when available and enabled
    mode = os.getenv("SUMMARIZER_MODE", "auto").lower()
    use_llm = mode in ("llm", "auto") and os.getenv("OPENAI_API_KEY")
    if use_llm:
        try:
            import langroid as lr  # type: ignore
            from .config import get_llm_config

            SUM_CFG = lr.ChatAgentConfig(
                name="DailyStatSummarizer",
                llm=get_llm_config(),
                system_message=(
                    "You are a clinical data summarizer. ONLY use the provided data. "
                    "Every numeric in your prose must appear in the JSON you return. "
                    "Return: 2–5 sentences, then JSON strictly matching the provided schema."
                ),
            )
            agent = lr.ChatAgent(SUM_CFG)
            task = lr.Task(agent, name="SummarizeOneDay")

            user_msg = f"""
STAT: {sp['stat']} (units={sp['units']})
DAY: {sp['day']}
N_OBS={sp['coverage']['n_obs']}, COVERAGE={sp['coverage']['prop_day']:.2f}
MIN={sp['range'].get('min')}, MEDIAN={sp['central'].get('median')}, MAX={sp['range'].get('max')}
P05={sp['percentiles'].get('p05')}, P95={sp['percentiles'].get('p95')}
SLOPE_PER_HR={sp['trend'].get('slope_per_hr')}, DELTA_LAST_FIRST={sp['trend'].get('delta_last_first')}
STD={sp['variability'].get('std')}, MAD={sp['variability'].get('mad')}
OUTLIERS={sp['outliers'].get('n')}, UNIT_CONFLICTS={sp['flags'].get('unit_conflicts')}, SPARSE={sp['flags'].get('sparse')}

First/Last readings (samples):
HEAD:
{sp.get('head_rows_csv','')}

TAIL:
{sp.get('tail_rows_csv','')}

Instructions:
1) Write ≤5 sentences summarizing only range, central tendency, trend, volatility, coverage, outliers/missingness.
2) Then output JSON matching the DailyStatJSON schema exactly with these keys:
   stat, units, day, range, central, percentiles, trend, variability, coverage, outliers, flags.
3) Do not invent numbers or dates. Use only those shown above.
"""
            return task.run(user_msg).content
        except Exception:
            pass
    # Fallback: deterministic offline summary + strict JSON
    return _offline_summary(sp)


def _rows_to_csv(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(fields))
    writer.writeheader()
    for row in rows:
        norm = {f: row.get(f) for f in fields}
        writer.writerow(norm)
    return buf.getvalue().strip()


def _basic_payload_from_rows(
    rows: List[Dict[str, Any]],
    stat: str,
    unit: str,
    day: str,
    bounds: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd

    low, high = (bounds or (None, None))
    base_payload: Dict[str, Any] = {
        "stat": stat,
        "units": unit,
        "day": day,
        "range": {"min": None, "max": None, "t_min": None, "t_max": None},
        "central": {"mean": None, "median": None},
        "percentiles": {"p05": None, "p95": None},
        "trend": {"slope_per_hr": 0.0, "delta_last_first": None},
        "variability": {"std": None, "mad": None},
        "coverage": {"n_obs": 0, "prop_day": 0.0, "hours_w_obs": 0, "n_missing_slots": 24},
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
    if not rows:
        return base_payload

    df = pd.DataFrame(rows)
    if "charttime" not in df.columns or "valuenum" not in df.columns:
        return base_payload

    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df["valuenum"] = pd.to_numeric(df["valuenum"], errors="coerce")
    df = df.dropna(subset=["charttime", "valuenum"]).sort_values("charttime")
    if df.empty:
        payload = base_payload.copy()
        payload["flags"]["no_data"] = True
        payload["flags"]["sparse"] = True
        return payload

    df["error"] = pd.to_numeric(df.get("error"), errors="coerce")
    valid_mask = (df["error"].isna()) | (df["error"] == 0)
    df_valid = df.loc[valid_mask].copy()
    if df_valid.empty:
        payload = base_payload.copy()
        payload["coverage"]["n_obs"] = int(len(df))
        payload["flags"]["no_data"] = True
        payload["flags"]["sparse"] = True
        payload["flags"]["duplicates"] = bool(df["charttime"].duplicated().any())
        return payload

    n_obs = int(len(df_valid))
    hours = df_valid["charttime"].dt.floor("h")
    hours_w_obs = int(hours.nunique())
    n_missing_slots = max(0, 24 - hours_w_obs)
    minutes = int(df_valid["charttime"].dt.floor("min").nunique())
    prop_day = float(min(1.0, minutes / (24 * 60))) if minutes else 0.0

    val = df_valid["valuenum"].to_numpy(dtype=float)
    v_min = float(np.min(val))
    v_max = float(np.max(val))
    v_mean = float(np.mean(val))
    v_median = float(np.median(val))
    v_std = float(np.std(val, ddof=0)) if len(val) > 1 else 0.0
    v_p05 = float(np.percentile(val, 5))
    v_p95 = float(np.percentile(val, 95))
    v_mad = float(np.median(np.abs(val - v_median)))

    times = df_valid["charttime"].to_numpy(dtype="datetime64[ns]")
    t_min = df_valid.loc[val.argmin(), "charttime"]
    t_max = df_valid.loc[val.argmax(), "charttime"]

    delta = float(val[-1] - val[0]) if len(val) >= 2 else 0.0
    time_span_hours = (times[-1] - times[0]).astype("timedelta64[s]").astype(float) / 3600.0 if len(times) >= 2 else 0.0
    slope = float(delta / time_span_hours) if time_span_hours else 0.0

    low = float(low) if low is not None else None
    high = float(high) if high is not None else None
    if low is not None and high is not None:
        bounds_mask = (val < low) | (val > high)
    elif low is not None:
        bounds_mask = val < low
    elif high is not None:
        bounds_mask = val > high
    else:
        bounds_mask = np.zeros_like(val, dtype=bool)

    outlier_idx = np.where(bounds_mask)[0]
    outlier_ts = df_valid.iloc[outlier_idx]["charttime"].head(5).dt.to_pydatetime().tolist()
    outlier_iso = [ts.isoformat() for ts in outlier_ts]

    units = df_valid.get("valueuom")
    unit_conflict = False
    dominant_unit = unit
    if units is not None:
        unit_vals = units.dropna().astype(str).str.strip().str.lower()
        if not unit_vals.empty:
            dominant_unit = unit or unit_vals.mode().iat[0]
            expected_l = (unit or "").strip().lower()
            if expected_l:
                unit_conflict = any(u != expected_l for u in unit_vals.unique())
            else:
                unit_conflict = len(unit_vals.unique()) > 1

    dup_flag = df_valid["charttime"].duplicated().any()

    sample_cols = [c for c in ["charttime", "valuenum", "valueuom", "itemid", "error"] if c in df_valid.columns]
    head_csv = df_valid[sample_cols].head(3).to_csv(index=False)
    tail_csv = df_valid[sample_cols].tail(3).to_csv(index=False)

    payload: Dict[str, Any] = {
        "stat": stat,
        "units": dominant_unit or unit,
        "day": day,
        "range": {
            "min": v_min,
            "max": v_max,
            "t_min": t_min.to_pydatetime().isoformat() if isinstance(t_min, datetime) else None,
            "t_max": t_max.to_pydatetime().isoformat() if isinstance(t_max, datetime) else None,
        },
        "central": {"mean": v_mean, "median": v_median},
        "percentiles": {"p05": v_p05, "p95": v_p95},
        "trend": {"slope_per_hr": slope, "delta_last_first": delta},
        "variability": {"std": v_std, "mad": v_mad},
        "coverage": {
            "n_obs": n_obs,
            "prop_day": prop_day,
            "hours_w_obs": hours_w_obs,
            "n_missing_slots": n_missing_slots,
        },
        "outliers": {"n": int(len(outlier_idx)), "timestamps": outlier_iso},
        "flags": {
            "unit_conflicts": bool(unit_conflict),
            "sparse": bool((n_obs < 6) or (hours_w_obs <= 3) or (prop_day < 0.1)),
            "value_out_of_bounds": bool(len(outlier_idx) > 0),
            "no_data": False,
            "duplicates": bool(dup_flag),
            "approximate": False,
        },
        "head_rows_csv": head_csv,
        "tail_rows_csv": tail_csv,
    }
    return payload


def _offline_summary_from_rows(
    rows: List[Dict[str, Any]],
    stat: str,
    unit: str,
    day: str,
    bounds: Optional[Tuple[float, float]] = None,
) -> str:
    payload = _basic_payload_from_rows(rows, stat, unit, day, bounds=bounds)
    return _offline_summary(payload)


def summarize_from_rows(
    rows: List[Dict[str, Any]],
    *,
    stat: str,
    unit: str,
    day: str,
    bounds: Optional[Tuple[float, float]] = None,
    subject_id: Optional[int] = None,
    max_rows: int = 1000,
) -> str:
    mode = os.getenv("SUMMARIZER_MODE", "auto").lower()
    use_llm = mode in ("llm", "auto") and os.getenv("OPENAI_API_KEY")
    total_rows = len(rows)
    send_rows = rows[:max_rows]
    truncated = total_rows > len(send_rows)
    fields = ("charttime", "valuenum", "valueuom", "itemid", "error")
    rows_csv = _rows_to_csv(send_rows, fields)

    if use_llm:
        try:
            import langroid as lr  # type: ignore
            from .config import get_llm_config

            system_message = (
                "You are a clinical data summarizer. Analyze the provided raw measurements "
                "and compute statistics before writing. Only rely on the table shown. "
                "Every numeric mentioned in your prose must appear in the JSON you return. "
                "Return: 2–5 sentences of analysis, then JSON with keys "
                "stat, units, day, range, central, percentiles, trend, variability, coverage, outliers, flags."
            )
            SUM_CFG = lr.ChatAgentConfig(
                name="DailyStatSummarizerRaw",
                llm=get_llm_config(),
                system_message=system_message,
            )
            agent = lr.ChatAgent(SUM_CFG)
            task = lr.Task(agent, name="SummarizeRawRows")

            bound_text = ""
            if bounds is not None:
                low, high = bounds
                bound_text = f"VALUE_BOUNDS: low={low}, high={high}"

            subject_line = f"SUBJECT_ID: {subject_id}" if subject_id is not None else ""

            user_msg = f"""
STAT: {stat} (units={unit})
DAY: {day}
{subject_line}
TOTAL_ROWS: {total_rows}
ROWS_INCLUDED: {len(send_rows)}
TRUNCATED: {"yes" if truncated else "no"}
{bound_text}

Columns: charttime (iso8601), valuenum (float), valueuom, itemid, error (0 means valid; discard rows with error != 0).

ROWS_CSV:
{rows_csv}

Instructions:
1) Use only rows with numeric valuenum and error null/0. Treat charttime as ISO timestamps.
2) Compute:
   - range.min/max from valuenum; include t_min/t_max timestamps where min/max occur (ISO).
   - central.mean/median using valid valuenum.
   - percentiles.p05/p95.
   - trend.delta_last_first = last_val - first_val (in time order). trend.slope_per_hr = delta_last_first / hours between first and last timestamp (0 if undefined).
   - variability.std (population) and variability.mad (median absolute deviation).
   - coverage.n_obs = count of valid rows; coverage.hours_w_obs = distinct hour buckets; coverage.prop_day = distinct minute buckets/1440 (clip 0–1); coverage.n_missing_slots = 24 - hours_w_obs.
   - outliers.n = count of valid rows where valuenum outside the provided bounds (if bounds missing, use 0). outliers.timestamps = first up to 5 ISO timestamps of outliers.
   - flags.unit_conflicts = true if valueuom differs across rows (case-insensitive) from the expected units; sparse = true if n_obs < 6 or hours_w_obs <= 3 or prop_day < 0.1; value_out_of_bounds = true if outliers present; no_data = false if you have any valid rows; duplicates = true if duplicate charttime appears; approximate = {"true" if truncated else "false"}.
3) If no valid rows, write that coverage is zero and set flags.no_data=true, others null/false as appropriate.
4) Ensure every number in your sentences also appears in the JSON. Do not invent values not present in the table.
"""
            return task.run(user_msg).content
        except Exception:
            pass

    return _offline_summary_from_rows(rows, stat, unit, day, bounds=bounds)
