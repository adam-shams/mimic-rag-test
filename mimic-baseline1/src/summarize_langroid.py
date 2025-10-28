from typing import Any, Dict, List, Optional, Sequence, Tuple
import copy
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


def _payload_block(sp: Dict[str, Any]) -> str:
    cov = sp.get("coverage", {})
    rng = sp.get("range", {})
    central = sp.get("central", {})
    percentiles = sp.get("percentiles", {})
    trend = sp.get("trend", {})
    variability = sp.get("variability", {})
    outliers = sp.get("outliers", {})
    flags = sp.get("flags", {})

    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v}"
        return str(v)

    lines = [
        f"STAT: {sp.get('stat')} (units={sp.get('units')})",
        f"DAY: {sp.get('day')}",
        f"N_OBS={cov.get('n_obs')}, HOURS_W_OBS={cov.get('hours_w_obs')}, MINUTES_W_OBS={cov.get('minutes_w_obs')}, PROP_DAY={cov.get('prop_day')}",
        f"N_MISSING_SLOTS={cov.get('n_missing_slots')}",
        f"MIN={rng.get('min')}, MAX={rng.get('max')}, T_MIN={rng.get('t_min')}, T_MAX={rng.get('t_max')}",
        f"MEAN={central.get('mean')}, MEDIAN={central.get('median')}",
        f"P05={percentiles.get('p05')}, P95={percentiles.get('p95')}",
        f"DELTA_LAST_FIRST={trend.get('delta_last_first')}, SLOPE_PER_HR={trend.get('slope_per_hr')}",
        f"STD={variability.get('std')}, MAD={variability.get('mad')}",
        f"OUTLIERS_N={outliers.get('n')}, OUTLIERS_TIMESTAMPS={_fmt(outliers.get('timestamps'))}",
        f"FLAGS: UNIT_CONFLICTS={flags.get('unit_conflicts')}, SPARSE={flags.get('sparse')}, VALUE_OUT_OF_BOUNDS={flags.get('value_out_of_bounds')}, DUPLICATES={flags.get('duplicates')}, NO_DATA={flags.get('no_data')}, APPROXIMATE={flags.get('approximate')}",
    ]
    return "\n".join(lines)


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
    approx = sp.get("flags", {}).get("approximate")

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
    if approx:
        sents.append("Summary based on truncated data (approximate).")
    text = " ".join([s for s in sents if s][:5])
    return f"{text}\n\n{_build_json(sp)}"


def summarize(stat_payload: Dict[str, Any]) -> str:
    sp = stat_payload
    # Normalize coverage prop_day to float with 2 decimals in text rendering
    cov = sp.setdefault("coverage", {})
    try:
        cov["prop_day"] = float(cov.get("prop_day") or 0.0)
    except Exception:
        cov["prop_day"] = 0.0
    cov.setdefault("minutes_w_obs", 0)
    cov.setdefault("hours_w_obs", 0)
    cov.setdefault("n_missing_slots", 24)
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

            payload_block = _payload_block(sp)
            head_csv = sp.get("head_rows_csv", "") or "<no rows>"
            tail_csv = sp.get("tail_rows_csv", "") or "<no rows>"

            user_msg = f"""
COMPUTED FEATURES (canonical):
{payload_block}

Sample rows (head):
{head_csv}

Sample rows (tail):
{tail_csv}

Instructions:
1) Use ONLY the computed features above for every numeric statement.
2) Write ≤5 sentences touching on coverage (including hours_w_obs, minutes_w_obs, prop_day = hours_w_obs/24), range, central tendency, trend (delta_last_first, slope_per_hr), variability (std, mad), outliers, and flags (sparse, value_out_of_bounds, unit_conflicts, duplicates, approximate).
3) Then output JSON matching the DailyStatJSON schema exactly with keys: stat, units, day, range, central, percentiles, trend, variability, coverage, outliers, flags. Populate the JSON with the same values shown above.
4) Do not invent numbers or recompute statistics from the raw samples; the samples are for qualitative context only.
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


def _split_text_json(content: str) -> Tuple[str, str]:
    """
    Split a string into (text_without_last_json, last_json_str).
    Returns ("", "") if no JSON object is found.
    """
    stack: List[int] = []
    last_start = -1
    last_end = -1
    for idx, ch in enumerate(content):
        if ch == "{":
            stack.append(idx)
        elif ch == "}":
            if stack:
                start = stack.pop()
                if not stack:
                    last_start = start
                    last_end = idx + 1
    if last_start >= 0 and last_end > last_start:
        text_part = content[:last_start].rstrip()
        json_part = content[last_start:last_end].strip()
        return text_part, json_part
    return content, ""


def _compare_payloads(
    canonical: Dict[str, Any],
    candidate: Dict[str, Any],
    tol: float = 1e-2,
) -> List[str]:
    """
    Return list of field paths where candidate differs from canonical beyond tolerance.
    """
    paths: List[str] = []

    def walk(a: Any, b: Any, prefix: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            keys = set(a.keys()) | set(b.keys())
            for key in keys:
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                av = a.get(key) if isinstance(a, dict) else None
                bv = b.get(key) if isinstance(b, dict) else None
                walk(av, bv, next_prefix)
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                paths.append(prefix)
                return
            for av, bv in zip(a, b):
                if isinstance(av, (dict, list)) or isinstance(bv, (dict, list)):
                    if av != bv:
                        paths.append(prefix)
                        return
                else:
                    if av != bv:
                        paths.append(prefix)
                        return
            return
        if a is None and b is None:
            return
        if isinstance(a, (int, float)) or isinstance(b, (int, float)):
            try:
                af = float(a)
                bf = float(b)
                if not (abs(af - bf) <= tol):
                    paths.append(prefix)
            except Exception:
                if a != b:
                    paths.append(prefix)
            return
        if isinstance(a, bool) or isinstance(b, bool):
            if bool(a) != bool(b):
                paths.append(prefix)
            return
        if a != b:
            paths.append(prefix)

    walk(canonical, candidate, "")
    return sorted(set(paths))


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = s[3:]
        if "\n" in s:
            s = s.split("\n", 1)[1]
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


METRIC_DEFINITIONS = """\
Metric definitions (apply to rows where valuenum is numeric and error is NULL or 0):
- n_obs: count of valid rows.
- min/max: minimum and maximum valuenum across valid rows.
- mean: arithmetic mean of valid valuenum.
- median: 50th percentile of valid valuenum.
- p05/p95: 5th and 95th percentiles of valid valuenum.
- delta_last_first: last valuenum minus first valuenum when sorted by charttime ascending.
- slope_per_hr: delta_last_first divided by hours between first and last timestamp (0 if time span ≤ 0).
- std: population standard deviation (ddof=0) of valid valuenum.
- mad: median absolute deviation (unscaled) of valid valuenum.
- hours_w_obs: number of distinct hour buckets (floor charttime to hour) with observations.
- minutes_w_obs: number of distinct minute buckets (floor charttime to minute) with observations.
- prop_day: min(1, hours_w_obs / 24).
- n_missing_slots: max(0, 24 - hours_w_obs).
- outliers.n: count of valid rows with valuenum outside the provided bounds (treat as 0 if bounds missing).
- outliers.timestamps: ISO timestamps (up to 5) for the outlier rows in chronological order.
- flags.sparse: true if prop_day < 0.5, else false.
- flags.unit_conflicts: true if valueuom (case-insensitive) differs from the expected unit for any valid row.
- flags.value_out_of_bounds: true if outliers.n > 0.
- flags.no_data: true if there are no valid rows.
- flags.duplicates: true if any valid rows share the same charttime.
- flags.approximate: true if data was truncated before sending to the model, else false.
- coverage.hours_w_obs, coverage.minutes_w_obs, coverage.prop_day, coverage.n_missing_slots follow the values above.
"""


def _run_llm_task(system_message: str, user_message: str, agent_name: str, task_name: str) -> Optional[str]:
    mode = os.getenv("SUMMARIZER_MODE", "auto").lower()
    if mode not in ("llm", "auto"):
        return None
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        import langroid as lr  # type: ignore
        from .config import get_llm_config

        cfg = lr.ChatAgentConfig(
            name=agent_name,
            llm=get_llm_config(),
            system_message=system_message,
        )
        agent = lr.ChatAgent(cfg)
        task = lr.Task(agent, name=task_name)
        return task.run(user_message).content
    except Exception:
        return None


def _make_rows_prompt(
    stat: str,
    unit: str,
    day: str,
    subject_id: Optional[int],
    bounds: Optional[Tuple[float, float]],
    total_rows: int,
    included_rows: int,
    truncated: bool,
    rows_csv: str,
    include_text: bool,
) -> str:
    subject_line = f"SUBJECT_ID: {subject_id}" if subject_id is not None else "SUBJECT_ID: unknown"
    if bounds is not None:
        low, high = bounds
        bounds_line = f"VALUE_BOUNDS: low={low}, high={high}"
    else:
        bounds_line = "VALUE_BOUNDS: not provided"
    truncated_line = f"TRUNCATED: {'yes' if truncated else 'no'} (sent {included_rows} of {total_rows} rows)"
    base_header = f"""
STAT: {stat} (units={unit})
DAY: {day}
{subject_line}
{bounds_line}
TOTAL_ROWS: {total_rows}
{truncated_line}

{METRIC_DEFINITIONS}

Columns: charttime (ISO 8601), valuenum (numeric), valueuom, itemid, error.
Treat rows with non-numeric valuenum or error not in (NULL, 0) as invalid.
Always sort by charttime ascending before computing statistics.

ROWS_CSV:
{rows_csv or '<no rows>'}
""".strip()

    if include_text:
        instructions = """
Instructions:
1) Compute all metrics exactly as defined, using only the valid rows.
2) Write 2–5 sentences summarizing the day. Every numeric mentioned must appear in the JSON that follows.
3) Output JSON matching the DailyStatJSON schema with keys:
   stat, units, day, range, central, percentiles, trend, variability, coverage, outliers, flags.
4) Populate coverage.hours_w_obs, coverage.minutes_w_obs, coverage.prop_day, coverage.n_missing_slots, outliers.timestamps (first up to 5 ISO timestamps), and all flags per the definitions.
5) Set flags.approximate=true if TRUNCATED=yes, else false.
6) Do not invent rows or refer to data outside the supplied CSV.
"""
    else:
        instructions = """
Instructions:
1) Compute the metrics exactly as defined above.
2) Return ONLY JSON (no prose) matching the DailyStatJSON schema with keys:
   stat, units, day, range, central, percentiles, trend, variability, coverage, outliers, flags.
3) Populate coverage.hours_w_obs, coverage.minutes_w_obs, coverage.prop_day, coverage.n_missing_slots, outliers.timestamps (first up to 5 ISO timestamps), and all flags per the definitions.
4) Ensure every numeric value comes from the computed metrics. Set flags.approximate=true if TRUNCATED=yes, else false.
5) Do not add any additional text outside the JSON object.
"""
    return f"{base_header}\n\n{instructions.strip()}"


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

    hours_w_obs = 0
    n_missing_slots = 24
    minutes = 0
    prop_day = 0.0

    if df_valid.empty:
        payload = base_payload.copy()
        payload["coverage"]["n_obs"] = int(len(df))
        payload["coverage"]["prop_day"] = prop_day
        payload["flags"]["no_data"] = True
        payload["flags"]["sparse"] = True
        payload["flags"]["duplicates"] = bool(df["charttime"].duplicated().any())
        payload["coverage"]["hours_w_obs"] = int(hours_w_obs)
        payload["coverage"]["n_missing_slots"] = int(n_missing_slots)
        payload["coverage"]["minutes_w_obs"] = int(minutes)
        return payload

    n_obs = int(len(df_valid))
    hours = df_valid["charttime"].dt.floor("h")
    hours_w_obs = int(hours.nunique())
    n_missing_slots = max(0, 24 - hours_w_obs)
    minutes = int(df_valid["charttime"].dt.floor("min").nunique())
    prop_day = float(min(1.0, hours_w_obs / 24.0)) if hours_w_obs else 0.0

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
            "n_obs": int(n_obs),
            "prop_day": prop_day,
            "hours_w_obs": int(hours_w_obs),
            "minutes_w_obs": int(minutes),
            "n_missing_slots": int(n_missing_slots),
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


def summarize_from_rows(
    rows: List[Dict[str, Any]],
    *,
    stat: str,
    unit: str,
    day: str,
    bounds: Optional[Tuple[float, float]] = None,
    subject_id: Optional[int] = None,
    max_rows: int = 1000,
    payload: Optional[Dict[str, Any]] = None,
    save_json_path: Optional[str] = None,
) -> str:
    payload_data = copy.deepcopy(payload) if payload is not None else _basic_payload_from_rows(
        rows, stat, unit, day, bounds=bounds
    )
    coverage = payload_data.setdefault("coverage", {})
    coverage.setdefault("minutes_w_obs", 0)
    coverage.setdefault("hours_w_obs", 0)
    coverage.setdefault("n_missing_slots", 24)
    flags = payload_data.setdefault("flags", {})
    flags.setdefault("approximate", False)
    payload_data.setdefault("meta", {})

    total_rows = len(rows)
    rows_limit = max_rows if max_rows is not None else total_rows
    send_rows = rows[:rows_limit]
    truncated = total_rows > len(send_rows)
    if truncated:
        flags["approximate"] = True
    payload_data["meta"]["total_rows"] = total_rows
    payload_data["meta"]["rows_in_prompt"] = len(send_rows)
    if subject_id is not None:
        payload_data["meta"]["subject_id"] = subject_id
    payload_data["meta"]["mode"] = "hybrid"

    fields = ("charttime", "valuenum", "valueuom", "itemid", "error")
    rows_csv = _rows_to_csv(send_rows, fields)

    canonical_content = summarize(payload_data)
    canonical_text, canonical_json = _split_text_json(canonical_content)
    canonical_text = canonical_text.strip()
    if not canonical_text:
        canonical_text = "Canonical summary unavailable."
    if not canonical_json:
        canonical_json = _build_json(payload_data)

    system_message = (
        "You are a clinical data summarizer. Compute statistics from the raw rows and return JSON only."
    )
    user_msg = _make_rows_prompt(
        stat,
        unit,
        day,
        subject_id,
        bounds,
        total_rows,
        len(send_rows),
        truncated,
        rows_csv,
        include_text=False,
    )
    llm_json_str = ""
    llm_obj: Optional[Dict[str, Any]] = None
    llm_response = _run_llm_task(
        system_message,
        user_msg,
        agent_name="DailyStatSummarizerHybrid",
        task_name="SummarizeRawRowsHybrid",
    )
    if llm_response:
        candidate = _strip_code_fence(llm_response)
        if not candidate.strip().startswith("{"):
            _, possible_json = _split_text_json(llm_response)
            if possible_json:
                candidate = possible_json
        try:
            llm_obj = json.loads(candidate)
            llm_json_str = json.dumps(llm_obj)
        except Exception:
            llm_obj = None
            llm_json_str = ""
    if llm_json_str and save_json_path:
        try:
            with open(save_json_path, "w", encoding="utf-8") as f:
                f.write(llm_json_str + "\n")
        except Exception:
            pass

    if llm_obj is not None:
        diffs = _compare_payloads(payload_data, llm_obj)
        if diffs:
            diff_line = "Differences detected in: " + ", ".join(diffs)
        else:
            diff_line = "Differences detected in: none"
        parts = [
            canonical_text,
            "\n\nLLM-computed DailyStatJSON:\n",
            llm_json_str or "<LLM JSON unavailable>",
            "\n\nComparison: ",
            diff_line,
            "\n\nCanonical DailyStatJSON:\n",
            canonical_json.strip(),
        ]
        return "".join(parts)

    return (
        f"{canonical_text}\n\nLLM-computed metrics unavailable; showing canonical metrics only."
        f"\n\n{canonical_json.strip()}"
    )
