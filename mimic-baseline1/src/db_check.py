from typing import Optional

from sqlalchemy import create_engine, text

from .config import get_mimic_dsn


def main() -> None:
    try:
        dsn = get_mimic_dsn()
    except Exception as e:
        print("MIMIC_DSN not configured:", e)
        print("Hint: set it in mimic-baseline1/conf/db.env or export MIMIC_DSN in your shell.")
        return

    try:
        engine = create_engine(dsn)
        with engine.connect() as conn:
            res = conn.execute(text("SELECT 1 AS ok"))
            row = res.fetchone()
            print("Connected. Test query:", dict(row._mapping) if hasattr(row, "_mapping") else dict(row))
            # Check expected tables exist (without scanning big tables)
            tbls = conn.execute(
                text(
                    """
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema NOT IN ('pg_catalog','information_schema')
                      AND table_name IN ('chartevents','labevents','d_items','d_labitems','admissions','icustays')
                    ORDER BY table_name
                    """
                )
            ).fetchall()
            names = [r[0] for r in tbls]
            print("Found tables:", names)
    except Exception as e:
        print("DB connection error:", e)
        print("Check host, port, credentials, and that your IP has DB access.")


if __name__ == "__main__":
    main()
