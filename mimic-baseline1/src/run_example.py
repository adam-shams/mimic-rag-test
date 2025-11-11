import argparse
from datetime import datetime, timedelta
import json
import os
from typing import Optional

from .config import load_stat_config
from .fetch import fetch_day_chartevents
from .features import compute_daily_features, StatCfg
from .summarize_langroid import summarize
from .eval_faithfulness import check_faithfulness
from .guideline_rag import (
    interpret_with_guidelines,
    get_guideline_rag_context,
    answer_guideline_question,
    GuidelineRAGResult,
)


def _day_window(day: str) -> tuple[str, str]:
    d0 = datetime.fromisoformat(day)
    d1 = d0 + timedelta(days=1)
    return d0.strftime("%Y-%m-%d %H:%M:%S"), d1.strftime("%Y-%m-%d %H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject_id", type=int, help="MIMIC subject_id")
    ap.add_argument("day", type=str, help="Day YYYY-MM-DD")
    ap.add_argument("stat_key", type=str, default="heart_rate", nargs="?", help="Stat key, e.g., heart_rate")
    ap.add_argument("--stats-yaml", type=str, default=os.path.join(os.path.dirname(__file__), "..", "conf", "stats.yaml"))
    ap.add_argument("--max-rows", type=int, default=5000)
    ap.add_argument(
        "--rag-dir",
        type=str,
        default=None,
        help="Directory containing guideline documents (default: repo-level 'RAG files').",
    )
    ap.add_argument(
        "--no-rag",
        action="store_true",
        help="Skip guideline RAG interpretation even if documents are available.",
    )
    ap.add_argument(
        "--rag-chat",
        action="store_true",
        help="After the initial guideline summary, allow follow-up questions against the same guideline documents.",
    )
    args = ap.parse_args()

    stats = load_stat_config(os.path.abspath(args.stats_yaml))
    if args.stat_key not in stats:
        raise SystemExit(f"stat_key {args.stat_key} not found in stats.yaml")
    s_cfg = stats[args.stat_key]
    cfg = StatCfg(
        stat=args.stat_key,
        table=s_cfg["table"],
        itemids=list(map(int, s_cfg["itemids"])),
        unit=s_cfg["unit"],
        bounds=(float(s_cfg["bounds"][0]), float(s_cfg["bounds"][1])),
    )

    start_dt, end_dt = _day_window(args.day)
    rows = fetch_day_chartevents(cfg.itemids, args.subject_id, start_dt, end_dt, max_rows=args.max_rows)

    payload = compute_daily_features(rows, cfg, args.day)
    content = summarize(payload)

    # Print summary text and JSON
    print(content)

    # Eval numeric reuse
    try:
        result = check_faithfulness(content)
        print("\nFaithfulness:", json.dumps(result))
    except Exception as e:
        print("\nFaithfulness check failed:", str(e))

    rag_context: Optional[GuidelineRAGResult] = None
    if not args.no_rag:
        print("\n[Guideline RAG] Querying guideline documents... (this may take ~30s)", flush=True)
        try:
            rag_context = get_guideline_rag_context(
                content,
                payload,
                rag_dir=args.rag_dir,
                subject_id=args.subject_id,
                stat=args.stat_key,
            )
            rag_result = rag_context.text
        except Exception as exc:
            rag_result = f"Guideline RAG failed: {exc}"
        print("\n--- Guideline interpretation (RAG) ---\n")
        print(rag_result)

        if args.rag_chat and rag_context and rag_context.agent is not None:
            print("\n[Guideline RAG Chat] Ask follow-up questions (blank to finish).\n")
            while True:
                try:
                    follow = input("Guideline question> ").strip()
                except EOFError:
                    break
                if not follow:
                    break
                follow_resp = answer_guideline_question(rag_context.agent, follow)
                print("\n" + follow_resp + "\n")


if __name__ == "__main__":
    main()
