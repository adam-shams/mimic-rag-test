from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy import create_engine, text

from .config import get_mimic_dsn


def fetch_day_chartevents(
    itemids: List[int],
    subject_id: int,
    start_dt: str,
    end_dt: str,
    max_rows: int = 5000,
) -> List[Dict[str, Any]]:
    dsn = get_mimic_dsn()
    engine = create_engine(dsn)
    ids = ", ".join(map(str, itemids))
    sql = f"""
        SELECT charttime, valuenum, valueuom, itemid, error
        FROM chartevents
        WHERE subject_id = :subject_id
          AND itemid IN ({ids})
          AND valuenum IS NOT NULL
          AND (error IS NULL OR error = 0)
          AND charttime >= :start_dt AND charttime < :end_dt
        ORDER BY charttime ASC
        LIMIT {max_rows}
    """
    with engine.connect() as conn:
        res = conn.execute(
            text(sql),
            dict(subject_id=subject_id, start_dt=start_dt, end_dt=end_dt),
        )
        rows = res.fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r._mapping) if hasattr(r, "_mapping") else dict(r)
            out.append(d)
        return out

