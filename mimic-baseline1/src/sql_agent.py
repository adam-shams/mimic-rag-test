import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from langroid.agent.special.sql.sql_chat_agent import (
    SQLChatAgent,
    SQLChatAgentConfig,
)
from langroid import Task

from .config import get_llm_config, get_mimic_dsn


LLM = get_llm_config()

SQLCFG = SQLChatAgentConfig(
    database_uri=get_mimic_dsn(),
    llm=LLM,
    schema_tools=True,
    system_message=(
        "You are a SQL assistant. Use ONLY the following tables to answer queries: "
        "chartevents, d_items, labevents, d_labitems, admissions, icustays. "
        "Never modify data. Always filter by provided subject_id and time window. "
        "For vitals from chartevents, return columns: charttime, valuenum, valueuom, itemid, error."
    ),
)

sql_agent = SQLChatAgent(SQLCFG)
sql_task = Task(sql_agent)


def _rows_to_dicts(rows: Any) -> List[Dict[str, Any]]:
    """Normalize rows returned by SQLChatAgent.run_query to a list of dicts."""
    out: List[Dict[str, Any]] = []
    if rows is None:
        return out
    # Rows may be list[Row] or list[dict]
    for r in rows:
        if isinstance(r, dict):
            d = dict(r)
        else:
            # SQLAlchemy Row
            try:
                d = dict(r._mapping)
            except Exception:
                d = dict(r)
        # Ensure datetime objects, not strings
        ct = d.get("charttime")
        if isinstance(ct, str):
            try:
                d["charttime"] = datetime.fromisoformat(ct)
            except Exception:
                pass
        out.append(d)
    return out


def nl_fetch_day(
    stat_name: str,
    itemids: List[int],
    subject_id: int,
    start_dt: str,
    end_dt: str,
    max_rows: int = 5000,
) -> List[Dict[str, Any]]:
    """
    NL→SQL mode using the agent; returns rows as list of dicts.
    """
    nl = f"""
    Fetch {stat_name} rows from chartevents for subject_id={subject_id}
    between '{start_dt}' and '{end_dt}' inclusive.
    Use itemid IN ({", ".join(map(str, itemids))}).
    Return columns [charttime, valuenum, valueuom, itemid, error],
    WHERE valuenum IS NOT NULL AND (error IS NULL OR error=0),
    ORDER BY charttime ASC, LIMIT {max_rows}.
    """
    resp = sql_task.run(nl)
    # Best-effort: try to extract a table result directly via the agent
    try:
        rows = sql_agent.run_query(resp.content)
    except Exception:
        # If content isn't SQL, try to extract code fence with SQL
        content = resp.content or ""
        sql_start = content.lower().find("select ")
        sql = content[sql_start:] if sql_start >= 0 else ""
        rows = sql_agent.run_query(sql)
    return _rows_to_dicts(rows)


def sql_fetch_day(
    itemids: List[int],
    subject_id: int,
    start_dt: str,
    end_dt: str,
    max_rows: int = 5000,
) -> List[Dict[str, Any]]:
    """
    Deterministic SQL template mode. Returns list of dict rows.
    """
    ids = ", ".join(map(str, itemids))
    sql = f"""
    SELECT charttime, valuenum, valueuom, itemid, error
    FROM chartevents
    WHERE subject_id = {subject_id}
      AND itemid IN ({ids})
      AND valuenum IS NOT NULL
      AND (error IS NULL OR error = 0)
      AND charttime >= '{start_dt}' AND charttime < '{end_dt}'
    ORDER BY charttime ASC
    LIMIT {max_rows}
    """
    rows = sql_agent.run_query(sql)
    return _rows_to_dicts(rows)
