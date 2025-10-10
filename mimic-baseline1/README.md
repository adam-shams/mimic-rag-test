MIMIC Daily Stat Summarizer (Langroid SQLChatAgent)

Overview
- Goal: For a chosen stat (e.g., heart rate), produce a faithful, concise daily summary using only that day’s measurements for one patient.
- Pipeline: SQLChatAgent fetch → features.py compute → summarizer agent returns text + strict JSON → eval for numeric faithfulness.

Install
- Python deps: `pip install -r requirements.txt`

Zero-setup CSV mode (recommended)
- Place or keep the MIMIC-III demo CSVs under `mimic-iii-clinical-database-demo-1.4/` (already present).
- The first run will auto-build a local SQLite DB at `mimic-baseline1/data/mimic_demo.db` from the CSVs.
- No need to set `MIMIC_DSN`.

Run (structured args)
1) Configure stats in `conf/stats.yaml` (already includes heart_rate).
2) Example usage:
   - `python -m mimic-baseline1.src.run_example 10006 2111-10-20 heart_rate`
   - Uses deterministic SQL and feature computation; summaries + JSON are generated offline (no LLM needed).

Run (natural language question)
- Activate your environment and set `OPENAI_API_KEY`.
- Ask a question such as:
  - `python -m mimic-baseline1.src.run_question "How is 10011 doing on 2126-08-20?"`
- The SQL Chat agent (Langroid) turns the question into SQL, rows are summarized by the LLM, and the executed SQL is printed for transparency.

Folder Layout
- conf/db.env                 # optional .env with MIMIC_DSN
- conf/stats.yaml             # stat configs (itemids, units, bounds)
- src/config.py               # LLM and DSN config
- src/sql_agent.py            # SQLChatAgent setup + fetch
- src/features.py             # compute daily features
- src/schema.py               # Pydantic schema for JSON output
- src/summarize_langroid.py   # summarizer agent
- src/eval_faithfulness.py    # numeric reuse checks
- src/run_example.py          # simple end-to-end demo (structured args)
- src/run_question.py         # natural-language entry point using SQL agent

Notes
- No medical knowledge or guidelines used; only aggregates + small table samples are given to the summarizer.
- Deterministic mode uses template SQL and `temperature=0`.
- To use a remote Postgres instead of SQLite, set `MIMIC_DSN` and re-run.
