from typing import Any, Dict
import json
import os


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
