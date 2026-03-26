"""MCP server exposing MIMIC RAG tools to Claude."""
from __future__ import annotations

import nest_asyncio
nest_asyncio.apply()

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

from .config import load_stat_config
from .fetch import fetch_day_chartevents
from .features import compute_daily_features, StatCfg
from .summarize_langroid import summarize
from .guideline_rag import get_guideline_rag_context, write_patient_context_file

mcp = FastMCP("MIMIC RAG Server")

_STATS_YAML = Path(__file__).parent.parent / "conf" / "stats.yaml"
_RAG_DIR = Path(__file__).parent.parent.parent / "RAG files"


def _day_window(day: str) -> tuple[str, str]:
    d0 = datetime.fromisoformat(day)
    d1 = d0 + timedelta(days=1)
    return d0.strftime("%Y-%m-%d %H:%M:%S"), d1.strftime("%Y-%m-%d %H:%M:%S")


def _load_cfg(stat_key: str) -> StatCfg:
    stats = load_stat_config(str(_STATS_YAML))
    if stat_key not in stats:
        available = list(stats.keys())
        raise ValueError(f"stat_key '{stat_key}' not found. Available: {available}")
    s = stats[stat_key]
    return StatCfg(
        stat=stat_key,
        table=s["table"],
        itemids=list(map(int, s["itemids"])),
        unit=s["unit"],
        bounds=(float(s["bounds"][0]), float(s["bounds"][1])),
    )


def _json_safe(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable objects."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    try:
        import numpy as np
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
    except ImportError:
        pass
    return obj


@mcp.tool()
def query_patient_data(subject_id: int, day: str, stat_key: str = "heart_rate") -> str:
    """
    Fetch raw vital sign rows for a patient on a given day from the MIMIC database.

    Args:
        subject_id: MIMIC patient subject_id (e.g. 10011)
        day: Date in YYYY-MM-DD format (e.g. "2126-08-20")
        stat_key: Stat name from stats.yaml (default: "heart_rate")

    Returns:
        JSON array of chart event rows with keys: charttime, valuenum, valueuom, itemid, error
    """
    cfg = _load_cfg(stat_key)
    start_dt, end_dt = _day_window(day)
    rows = fetch_day_chartevents(cfg.itemids, subject_id, start_dt, end_dt)
    return json.dumps(_json_safe(rows))


@mcp.tool()
def compute_features(subject_id: int, day: str, stat_key: str = "heart_rate") -> str:
    """
    Compute statistical features for a patient's vital signs on a given day.

    Includes: range, central tendency, percentiles, trend slope, variability,
    coverage, outlier detection, and quality flags.

    Args:
        subject_id: MIMIC patient subject_id
        day: Date in YYYY-MM-DD format
        stat_key: Stat name from stats.yaml (default: "heart_rate")

    Returns:
        JSON object with computed features payload
    """
    cfg = _load_cfg(stat_key)
    start_dt, end_dt = _day_window(day)
    rows = fetch_day_chartevents(cfg.itemids, subject_id, start_dt, end_dt)
    payload = compute_daily_features(rows, cfg, day)
    return json.dumps(_json_safe(payload))


@mcp.tool()
def generate_summary(subject_id: int, day: str, stat_key: str = "heart_rate") -> str:
    """
    Generate a natural language summary of a patient's vital signs for a day.

    Fetches data, computes features, and produces an LLM-generated narrative
    summary with key statistics.

    Args:
        subject_id: MIMIC patient subject_id
        day: Date in YYYY-MM-DD format
        stat_key: Stat name from stats.yaml (default: "heart_rate")

    Returns:
        Text summary of the patient's vital signs
    """
    cfg = _load_cfg(stat_key)
    start_dt, end_dt = _day_window(day)
    rows = fetch_day_chartevents(cfg.itemids, subject_id, start_dt, end_dt)
    payload = compute_daily_features(rows, cfg, day)
    return summarize(payload)


@mcp.tool()
def query_guidelines(
    subject_id: int,
    day: str,
    stat_key: str = "heart_rate",
    question: str = "",
) -> str:
    """
    Retrieve relevant clinical guideline excerpts and interpretation for a patient's vitals.

    Runs the full pipeline (fetch → features → summary) then queries embedded
    clinical guidelines via RAG to provide evidence-based interpretation.

    Args:
        subject_id: MIMIC patient subject_id
        day: Date in YYYY-MM-DD format
        stat_key: Stat name from stats.yaml (default: "heart_rate")
        question: Optional specific clinical question to ask the guidelines

    Returns:
        Clinical guideline interpretation with source citations
    """
    cfg = _load_cfg(stat_key)
    start_dt, end_dt = _day_window(day)
    rows = fetch_day_chartevents(cfg.itemids, subject_id, start_dt, end_dt)
    payload = compute_daily_features(rows, cfg, day)
    summary_text = summarize(payload)

    rag_dir = str(_RAG_DIR) if _RAG_DIR.exists() else None
    patient_context_text = write_patient_context_file(
        summary_text, payload, rag_dir=rag_dir, subject_id=subject_id, stat=stat_key
    )
    result = get_guideline_rag_context(
        summary_text,
        payload,
        rag_dir=rag_dir,
        subject_id=subject_id,
        stat=stat_key,
        question=question or None,
        patient_context_text=patient_context_text,
    )
    return result.text


@mcp.tool()
def analyze_patient(subject_id: int, day: str, stat_key: str = "heart_rate") -> str:
    """
    Run the full MIMIC analysis pipeline for a patient on a given day.

    Executes: data fetch → feature computation → LLM summary → guideline RAG.
    Returns a structured JSON with all results combined.

    Args:
        subject_id: MIMIC patient subject_id
        day: Date in YYYY-MM-DD format
        stat_key: Stat name from stats.yaml (default: "heart_rate")

    Returns:
        JSON with keys: summary (text), features (dict), guidelines (text)
    """
    cfg = _load_cfg(stat_key)
    start_dt, end_dt = _day_window(day)
    rows = fetch_day_chartevents(cfg.itemids, subject_id, start_dt, end_dt)
    payload = compute_daily_features(rows, cfg, day)
    summary_text = summarize(payload)

    rag_dir = str(_RAG_DIR) if _RAG_DIR.exists() else None
    guidelines_text = ""
    try:
        patient_context_text = write_patient_context_file(
            summary_text, payload, rag_dir=rag_dir, subject_id=subject_id, stat=stat_key
        )
        result = get_guideline_rag_context(
            summary_text,
            payload,
            rag_dir=rag_dir,
            subject_id=subject_id,
            stat=stat_key,
            patient_context_text=patient_context_text,
        )
        guidelines_text = result.text
    except Exception as exc:
        guidelines_text = f"Guideline RAG failed: {exc}"

    return json.dumps({
        "subject_id": subject_id,
        "day": day,
        "stat_key": stat_key,
        "summary": summary_text,
        "features": _json_safe(payload),
        "guidelines": guidelines_text,
    })


if __name__ == "__main__":
    mcp.run()
