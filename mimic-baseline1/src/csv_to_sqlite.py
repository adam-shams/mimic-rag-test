from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from sqlalchemy import create_engine


CSV_DIR_CANDIDATES = [
    # repo root typical
    Path("mimic-iii-clinical-database-demo-1.4"),
    # relative to this file
    Path(__file__).resolve().parent.parent / ".." / "mimic-iii-clinical-database-demo-1.4",
]


def find_csv_dir() -> Path:
    for cand in CSV_DIR_CANDIDATES:
        p = Path(cand).resolve()
        if p.exists() and p.is_dir() and (p / "CHARTEVENTS.csv").exists():
            return p
    raise FileNotFoundError(
        "Could not find mimic-iii-clinical-database-demo-1.4 directory with CSVs."
    )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_csv_minimal(path: Path, usecols: Optional[List[str]] = None, chunksize: int = 100000):
    return pd.read_csv(
        path,
        low_memory=False,
        dtype=str,  # keep as string to avoid dtype issues; we will cast selectively in SQLite
        usecols=usecols,
        chunksize=chunksize,
    )


def build_sqlite(db_path: Path, tables: Optional[List[str]] = None) -> str:
    """
    Build a minimal SQLite DB from CSVs for the tables we need.
    Returns a SQLAlchemy sqlite DB URI.
    """
    csv_dir = find_csv_dir()
    _ensure_dir(db_path.parent)
    uri = f"sqlite:///{db_path}"
    engine = create_engine(uri)

    # Default: minimal set for HR and labs
    tables = tables or [
        "chartevents",
        "d_items",
        "labevents",
        "d_labitems",
        "admissions",
        "icustays",
    ]

    # Map table to CSV filename and column subset
    plan = {
        "chartevents": {
            "csv": "CHARTEVENTS.csv",
            "usecols": [
                "row_id",
                "subject_id",
                "hadm_id",
                "icustay_id",
                "itemid",
                "charttime",
                "valuenum",
                "valueuom",
                "error",
            ],
        },
        "d_items": {"csv": "D_ITEMS.csv", "usecols": None},
        "labevents": {
            "csv": "LABEVENTS.csv",
            "usecols": [
                "row_id",
                "subject_id",
                "hadm_id",
                "itemid",
                "charttime",
                "valuenum",
                "valueuom",
                "flag",
            ],
        },
        "d_labitems": {"csv": "D_LABITEMS.csv", "usecols": None},
        "admissions": {"csv": "ADMISSIONS.csv", "usecols": None},
        "icustays": {"csv": "ICUSTAYS.csv", "usecols": None},
    }

    with engine.begin() as conn:
        for tbl in tables:
            spec = plan.get(tbl)
            if spec is None:
                continue
            csv_path = csv_dir / spec["csv"]
            print(f"Loading {tbl} from {csv_path} ...")
            chunks = _load_csv_minimal(csv_path, usecols=spec.get("usecols"))
            first = True
            for chunk in chunks:
                # Standardize column names to lowercase to match SQL in code
                chunk.columns = [c.lower() for c in chunk.columns]
                # Write with append/replace on first
                chunk.to_sql(tbl, conn, if_exists="replace" if first else "append", index=False)
                first = False

            # Add basic indexes for performance on chartevents
            if tbl == "chartevents":
                # Create simple indexes; ignore errors if already exist
                try:
                    conn.exec_driver_sql(
                        "CREATE INDEX IF NOT EXISTS idx_chartevents_subj_time ON chartevents(subject_id, charttime)"
                    )
                    conn.exec_driver_sql(
                        "CREATE INDEX IF NOT EXISTS idx_chartevents_item ON chartevents(itemid)"
                    )
                except Exception:
                    pass

    print(f"SQLite DB created at {db_path}")
    return uri


def ensure_sqlite_db(db_path: Optional[Path] = None) -> str:
    db_path = db_path or (Path(__file__).resolve().parent.parent / "data" / "mimic_demo.db").resolve()
    uri = f"sqlite:///{db_path}"
    if db_path.exists() and db_path.stat().st_size > 0:
        # sanity check: table exists
        try:
            from sqlalchemy import create_engine, text

            eng = create_engine(uri)
            with eng.connect() as conn:
                res = conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' AND name='chartevents'")
                ).fetchone()
                if res:
                    return uri
        except Exception:
            pass
        # else fall through to rebuild
    return build_sqlite(db_path)


if __name__ == "__main__":
    # CLI build
    ensure_sqlite_db()
    print("Done.")
