from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any

from .config import load_stat_config
from .features import StatCfg, compute_daily_features
from .sql_agent import nl_fetch_day
from .summarize_langroid import summarize_from_rows
from .eval_faithfulness import check_faithfulness


DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
INT_RE = re.compile(r"\b\d+\b")
DEFAULT_ALIASES = {
    "heart_rate": ["heart rate", "hr", "pulse"],
}


@dataclass
class QuestionRequest:
    subject_id: int
    day: str
    stat_key: str


def _day_window(day: str) -> tuple[str, str]:
    d0 = datetime.fromisoformat(day)
    d1 = d0 + timedelta(days=1)
    return d0.strftime("%Y-%m-%d %H:%M:%S"), d1.strftime("%Y-%m-%d %H:%M:%S")


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _find_stat(question: str, stats: Dict[str, Any]) -> str:
    norm_q = _normalize(question)
    for key in stats:
        aliases = list(DEFAULT_ALIASES.get(key, []))
        aliases.append(key.replace("_", " "))
        stat_def = stats[key]
        extra_aliases = stat_def.get("aliases") if isinstance(stat_def, dict) else None
        if isinstance(extra_aliases, (list, tuple)):
            aliases.extend(str(a) for a in extra_aliases)
        for alias in aliases:
            alias_norm = _normalize(alias)
            if alias_norm and alias_norm in norm_q:
                return key
    if len(stats) == 1:
        return next(iter(stats))
    raise ValueError("Unable to identify stat from question; please mention it explicitly.")


def interpret_question(question: str, stats: Dict[str, Any]) -> QuestionRequest:
    dates = DATE_RE.findall(question)
    if not dates:
        raise ValueError("No date (YYYY-MM-DD) found in question.")
    day = dates[0]

    stripped = question
    for d in dates:
        stripped = stripped.replace(d, " ")
    candidates = [int(tok) for tok in INT_RE.findall(stripped)]
    if not candidates:
        raise ValueError("No subject_id found in question.")
    subject_id = candidates[0]

    stat_key = _find_stat(question, stats)
    return QuestionRequest(subject_id=subject_id, day=day, stat_key=stat_key)


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or not val.strip():
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def answer_question(
    question: str,
    stats_cfg: Dict[str, Any],
    max_rows: int = 5000,
) -> str:
    parsed = interpret_question(question, stats_cfg)
    stat_def = stats_cfg[parsed.stat_key]
    cfg = StatCfg(
        stat=parsed.stat_key,
        table=stat_def["table"],
        itemids=list(map(int, stat_def["itemids"])),
        unit=stat_def["unit"],
        bounds=(float(stat_def["bounds"][0]), float(stat_def["bounds"][1])),
    )

    start_dt, end_dt = _day_window(parsed.day)
    rows, used_sql = nl_fetch_day(parsed.stat_key, cfg.itemids, parsed.subject_id, start_dt, end_dt, max_rows)

    payload = compute_daily_features(rows, cfg, parsed.day)
    payload.setdefault("meta", {})["sql"] = used_sql

    rows_limit = _env_int("SUMMARIZE_ROWS_LIMIT", 1000)
    save_json_path = os.path.join(os.path.dirname(__file__), "sql_direct_results.txt")
    content = summarize_from_rows(
        rows,
        stat=parsed.stat_key,
        unit=cfg.unit,
        day=parsed.day,
        bounds=cfg.bounds,
        subject_id=parsed.subject_id,
        max_rows=rows_limit,
        payload=payload,
        save_json_path=save_json_path,
    )
    try:
        faithfulness = check_faithfulness(content)
    except Exception:
        faithfulness = None

    output_lines = [content]
    if faithfulness is not None:
        output_lines.append("\nFaithfulness: " + str(faithfulness))
    else:
        output_lines.append("\nFaithfulness check failed.")
    if used_sql:
        output_lines.append("\nSQL used:\n" + used_sql)
    return "\n".join(output_lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Answer natural language questions about daily patient stats.")
    ap.add_argument("question", type=str, help="Natural language question (e.g., 'How is 10011 doing on 2126-08-20 heart rate?')")
    default_stats = os.path.join(os.path.dirname(__file__), "..", "conf", "stats.yaml")
    ap.add_argument("--stats-yaml", type=str, default=default_stats)
    ap.add_argument("--max-rows", type=int, default=5000)
    args = ap.parse_args()

    stats_cfg = load_stat_config(os.path.abspath(args.stats_yaml))
    output = answer_question(
        args.question,
        stats_cfg,
        max_rows=args.max_rows,
    )
    print(output)


if __name__ == "__main__":
    main()
