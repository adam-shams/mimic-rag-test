"""
Benchmark the SQL Chat agent against deterministic SQL templates.

This script compares the rows fetched by the NL→SQL agent (`nl_fetch_day`)
with the deterministic baseline query (`sql_fetch_day`) for a set of
(subject_id, day, stat_key) cases. It reports row-level recall/precision,
feature deltas, and flag mismatches.

Usage:
    python -m mimic-baseline1.src.benchmark_sql \
        --cases mimic-baseline1/conf/sql_benchmark_cases.yaml \
        --stats-yaml mimic-baseline1/conf/stats.yaml \
        --output mimic-baseline1/data/sql_benchmark_results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set

import yaml

from .config import load_stat_config
from .features import StatCfg, compute_daily_features
from .sql_agent import nl_fetch_day, sql_fetch_day


# -------- Helpers -----------------------------------------------------------------


def _as_iso(dt: Any) -> str:
    if isinstance(dt, datetime):
        return dt.isoformat()
    if dt is None:
        return ""
    return str(dt)


def _canonical_row(row: Dict[str, Any], precision: int = 3) -> Tuple[str, int, float]:
    """
    Represent a SQL result row as a comparable tuple.
    We only use charttime, itemid, and rounded valuenum.
    """
    charttime = _as_iso(row.get("charttime"))
    itemid = int(row.get("itemid") or 0)
    val = row.get("valuenum")
    try:
        val_f = round(float(val), precision)
    except (TypeError, ValueError):
        val_f = float("nan")
    return charttime, itemid, val_f


def _row_set(rows: Sequence[Dict[str, Any]]) -> Set[Tuple[str, int, float]]:
    return {_canonical_row(r) for r in rows}


def _abs_diff(a: Any, b: Any) -> float:
    try:
        return abs(float(a) - float(b))
    except Exception:
        return float("inf")


def _get_nested(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


NUMERIC_PATHS: List[Tuple[Tuple[str, ...], float]] = [
    (("range", "min"), 0.01),
    (("range", "max"), 0.01),
    (("central", "mean"), 0.01),
    (("central", "median"), 0.01),
    (("percentiles", "p05"), 0.01),
    (("percentiles", "p95"), 0.01),
    (("trend", "delta_last_first"), 0.01),
    (("trend", "slope_per_hr"), 0.01),
    (("variability", "std"), 0.01),
    (("variability", "mad"), 0.01),
    (("coverage", "prop_day"), 1e-4),
    (("coverage", "n_obs"), 0.0),
    (("coverage", "hours_w_obs"), 0.0),
    (("coverage", "minutes_w_obs"), 0.0),
]

FLAG_PATHS: List[Tuple[str, ...]] = [
    ("flags", "unit_conflicts"),
    ("flags", "sparse"),
    ("flags", "value_out_of_bounds"),
    ("flags", "no_data"),
    ("flags", "duplicates"),
]


def compare_features(
    agent_payload: Dict[str, Any],
    baseline_payload: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, bool]]:
    numeric_deltas: Dict[str, float] = {}
    numeric_within_tol: Dict[str, bool] = {}
    for path, tol in NUMERIC_PATHS:
        agent_v = _get_nested(agent_payload, *path)
        base_v = _get_nested(baseline_payload, *path)
        key = ".".join(path)
        if agent_v is None or base_v is None:
            numeric_deltas[key] = float("nan")
            numeric_within_tol[key] = False
            continue
        delta = _abs_diff(agent_v, base_v)
        numeric_deltas[key] = delta
        numeric_within_tol[key] = delta <= tol

    flag_matches: Dict[str, bool] = {}
    for path in FLAG_PATHS:
        key = ".".join(path)
        agent_flag = bool(_get_nested(agent_payload, *path))
        base_flag = bool(_get_nested(baseline_payload, *path))
        flag_matches[key] = agent_flag == base_flag

    return numeric_deltas, flag_matches


# -------- Data Loading -------------------------------------------------------------


def load_cases(path: str) -> List[Dict[str, Any]]:
    """
    Expect YAML with either:
        - a list of case dicts
        - or a dict with key 'cases' -> list
    Each case requires subject_id, day, stat_key.
    """
    with open(path, "r") as f:
        doc = yaml.safe_load(f)
    if isinstance(doc, dict) and "cases" in doc:
        cases = doc["cases"]
    else:
        cases = doc
    if not isinstance(cases, list):
        raise ValueError("Cases YAML must be a list or dict with 'cases'.")
    normalized: List[Dict[str, Any]] = []
    for entry in cases:
        if not isinstance(entry, dict):
            raise ValueError("Each case must be a mapping.")
        try:
            normalized.append(
                {
                    "subject_id": int(entry["subject_id"]),
                    "day": str(entry["day"]),
                    "stat_key": str(entry["stat_key"]),
                }
            )
        except KeyError as exc:
            raise ValueError(f"Missing required key in case: {exc}") from exc
    return normalized


# -------- Benchmark Core ----------------------------------------------------------


@dataclass
class BenchmarkCase:
    subject_id: int
    day: str
    stat_key: str


def run_case(
    case: BenchmarkCase,
    stats_cfg: Dict[str, Any],
    max_rows: int,
) -> Dict[str, Any]:
    if case.stat_key not in stats_cfg:
        raise ValueError(f"stat_key {case.stat_key} not found in stats config.")

    stat_def = stats_cfg[case.stat_key]
    cfg = StatCfg(
        stat=case.stat_key,
        table=stat_def["table"],
        itemids=list(map(int, stat_def["itemids"])),
        unit=stat_def["unit"],
        bounds=(float(stat_def["bounds"][0]), float(stat_def["bounds"][1])),
    )

    start_dt = f"{case.day} 00:00:00"
    end_dt = f"{case.day} 24:00:00"

    # Baseline deterministic rows
    baseline_rows = sql_fetch_day(
        cfg.itemids,
        case.subject_id,
        start_dt,
        end_dt,
        max_rows=max_rows,
    )

    # Agent-driven rows and SQL
    agent_rows, used_sql = nl_fetch_day(
        case.stat_key,
        cfg.itemids,
        case.subject_id,
        start_dt,
        end_dt,
        max_rows=max_rows,
    )

    # Feature payloads
    baseline_payload = compute_daily_features(baseline_rows, cfg, case.day)
    agent_payload = compute_daily_features(agent_rows, cfg, case.day)

    # Row comparison
    ref_set = _row_set(baseline_rows)
    agent_set = _row_set(agent_rows)
    intersection = ref_set & agent_set
    recall = len(intersection) / len(ref_set) if ref_set else 1.0
    precision = len(intersection) / len(agent_set) if agent_set else 1.0

    numeric_deltas, flag_matches = compare_features(agent_payload, baseline_payload)
    numeric_within_tol = numeric_within_tol_items(numeric_deltas)
    failures: List[str] = []
    if ref_set and (recall < 1.0):
        failures.append("row_recall")
    if agent_set and (precision < 1.0):
        failures.append("row_precision")
    for key, ok in numeric_within_tol.items():
        if not ok:
            failures.append(f"metric:{key}")
    for key, ok in flag_matches.items():
        if not ok:
            failures.append(f"flag:{key}")

    status = "PASS" if not failures else "FAIL"

    return {
        "subject_id": case.subject_id,
        "day": case.day,
        "stat_key": case.stat_key,
        "baseline_rows": len(baseline_rows),
        "agent_rows": len(agent_rows),
        "row_recall": recall,
        "row_precision": precision,
        "missing_rows": len(ref_set - agent_set),
        "extra_rows": len(agent_set - ref_set),
        "numeric_deltas": numeric_deltas,
        "numeric_within_tolerance": numeric_within_tol,
        "flag_matches": flag_matches,
        "used_sql": used_sql,
        "status": status,
        "failures": failures,
    }


def numeric_within_tol_items(numeric_deltas: Dict[str, float]) -> Dict[str, bool]:
    result: Dict[str, bool] = {}
    for path, tol in NUMERIC_PATHS:
        key = ".".join(path)
        delta = numeric_deltas.get(key, float("inf"))
        if math.isnan(delta):
            result[key] = False
        else:
            result[key] = delta <= tol
    return result


def flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested dicts so we can emit a clean CSV row.
    """
    flat: Dict[str, Any] = {
        "subject_id": record["subject_id"],
        "day": record["day"],
        "stat_key": record["stat_key"],
        "baseline_rows": record["baseline_rows"],
        "agent_rows": record["agent_rows"],
        "row_recall": record["row_recall"],
        "row_precision": record["row_precision"],
        "missing_rows": record["missing_rows"],
        "extra_rows": record["extra_rows"],
        "status": record["status"],
        "failures": ";".join(record["failures"]),
        "used_sql": record["used_sql"],
    }

    for path, _tol in NUMERIC_PATHS:
        key = ".".join(path)
        flat[f"delta_{key}"] = record["numeric_deltas"].get(key)
        flat[f"in_tol_{key}"] = record["numeric_within_tolerance"].get(key)

    for path in FLAG_PATHS:
        key = ".".join(path)
        flat[f"flags_match_{key}"] = record["flag_matches"].get(key)

    return flat


# -------- CLI ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NL→SQL agent against deterministic SQL.")
    parser.add_argument(
        "--cases",
        required=True,
        help="Path to YAML file listing benchmark cases.",
    )
    parser.add_argument(
        "--stats-yaml",
        default=os.path.join(os.path.dirname(__file__), "..", "conf", "stats.yaml"),
        help="Stats configuration YAML (default: repo conf/stats.yaml).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "sql_benchmark_results.csv"),
        help="CSV file to write benchmark results.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=5000,
        help="Row cap for both baseline and agent fetches.",
    )
    args = parser.parse_args()

    cases = [BenchmarkCase(**c) for c in load_cases(args.cases)]
    stats_cfg = load_stat_config(os.path.abspath(args.stats_yaml))

    results: List[Dict[str, Any]] = []
    for case in cases:
        record = run_case(case, stats_cfg, max_rows=args.max_rows)
        results.append(record)

    if not results:
        print("No benchmark cases provided; nothing to do.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flatten_record(results[0]).keys()))
        writer.writeheader()
        for record in results:
            writer.writerow(flatten_record(record))

    failures = [r for r in results if r["status"] != "PASS"]
    print(f"Completed {len(results)} cases → {len(failures)} failures.")
    if failures:
        for fail in failures:
            print(
                f"- {fail['subject_id']} {fail['day']} {fail['stat_key']}: "
                f"{fail['failures']}"
            )


if __name__ == "__main__":
    main()
