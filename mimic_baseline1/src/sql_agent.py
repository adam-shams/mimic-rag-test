import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from langroid.agent.special.sql.sql_chat_agent import (
    SQLChatAgent,
    SQLChatAgentConfig,
)
from langroid import Task
from langroid.agent.chat_document import ChatDocument
from sqlalchemy import text

from .config import get_llm_config, get_mimic_dsn


LLM = get_llm_config()

SQLCFG = SQLChatAgentConfig(
    database_uri=get_mimic_dsn(),
    llm=LLM,
    system_message=(
        "You are a SQL assistant. Use ONLY the following tables to answer queries: "
        "chartevents, d_items, labevents, d_labitems, admissions, icustays. "
        "Never modify data. Always filter by provided subject_id and time window. "
        "For vitals from chartevents, return columns: charttime, valuenum, valueuom, itemid, error."
    ),
    stream=False,
)

sql_agent = SQLChatAgent(SQLCFG)
sql_task = Task(sql_agent, interactive=False, llm_delegate=True)


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


def _clip_rows_to_window(
    rows: List[Dict[str, Any]], start_dt: str, end_dt: str
) -> List[Dict[str, Any]]:
    try:
        start = datetime.fromisoformat(start_dt)
        end = datetime.fromisoformat(end_dt)
    except Exception:
        return rows
    clipped: List[Dict[str, Any]] = []
    for row in rows:
        ct = row.get("charttime")
        ct_dt: Optional[datetime] = None
        if isinstance(ct, datetime):
            ct_dt = ct
        elif isinstance(ct, str):
            try:
                ct_dt = datetime.fromisoformat(ct)
            except Exception:
                ct_dt = None
        if ct_dt is None:
            clipped.append(row)
            continue
        if start <= ct_dt < end:
            new_row = dict(row)
            new_row["charttime"] = ct_dt
            clipped.append(new_row)
    return clipped


def _extract_sql(text: Optional[str]) -> str:
    """Extract SQL statement from agent response text."""
    if not text:
        return ""
    content = text.strip()

    # Try to read query from JSON payload (tool call)
    match = re.search(r'query"\s*:\s*"([^\"]+)"', content, flags=re.IGNORECASE)
    if match:
        sql = match.group(1)
        try:
            sql = bytes(sql, "utf-8").decode("unicode_escape")
        except Exception:
            pass
        return sql.strip()

    # Prefer fenced code blocks
    if "```" in content:
        parts = content.split("```")
        if len(parts) >= 2:
            for i in range(1, len(parts), 2):
                header = parts[i - 1].lower()
                code = parts[i].strip()
                if code.lower().startswith("sql\n"):
                    code = code.split("\n", 1)[1]
                elif code.lower() == "sql":
                    continue
                if header.rstrip().endswith("sql"):
                    return code.strip()
            code = parts[1].strip()
            if code.lower().startswith("sql\n"):
                return code.split("\n", 1)[1].strip()
            if code.lower() == "sql":
                return ""
            return code.strip()

    # Otherwise grab from first SELECT onward
    lower = content.lower()
    sel_idx = lower.find("select ")
    if sel_idx >= 0:
        return content[sel_idx:].strip()
    return ""


def _extract_sql_from_history(history: List[ChatDocument]) -> str:
    for doc in reversed(history):
        sql = _extract_sql(getattr(doc, "content", ""))
        if sql.lower().strip().startswith("select"):
            return sql
    return ""


def _run_sql(sql: str, max_rows: int) -> List[Dict[str, Any]]:
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are supported.")
    engine = getattr(sql_agent, "engine", None)
    if engine is None:
        raise RuntimeError("SQL agent engine not initialized")
    with engine.connect() as conn:
        res = conn.execute(text(sql))
        rows = res.fetchall()
    if max_rows and len(rows) > max_rows:
        rows = rows[:max_rows]
    return _rows_to_dicts(rows)


def nl_fetch_day(
    stat_name: str,
    itemids: List[int],
    subject_id: int,
    start_dt: str,
    end_dt: str,
    max_rows: int = 5000,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    NL→SQL mode using the agent; returns rows as list of dicts.
    """
    nl = f"""
    Fetch {stat_name} rows from chartevents for subject_id={subject_id}
    between '{start_dt}' (inclusive) and '{end_dt}' (exclusive).
    Ensure the WHERE clause uses charttime >= '{start_dt}' AND charttime < '{end_dt}'.
    Use itemid IN ({", ".join(map(str, itemids))}).
    Return columns [charttime, valuenum, valueuom, itemid, error],
    WHERE valuenum IS NOT NULL AND (error IS NULL OR error=0),
    ORDER BY charttime ASC, LIMIT {max_rows}.
    """
    resp = sql_task.run(nl)
    used_sql = _extract_sql(resp.content)
    if not used_sql:
        used_sql = _extract_sql_from_history(sql_agent.message_history)
    if not used_sql:
        raise RuntimeError("SQL agent did not produce a query")
    try:
        rows = _run_sql(used_sql, max_rows)
        rows = _clip_rows_to_window(rows, start_dt, end_dt)
    finally:
        sql_agent.clear_history()
    return rows, used_sql


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
    return _run_sql(sql, max_rows)
