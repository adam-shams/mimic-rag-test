import os
from typing import Any, Dict
import yaml
from pathlib import Path

from dotenv import load_dotenv
from langroid.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel


def get_llm_config() -> OpenAIGPTConfig:
    return OpenAIGPTConfig(
        chat_model=OpenAIChatModel.O3_MINI,
        temperature=0.0,
        timeout=120,
    )


def get_mimic_dsn() -> str:
    # Try to load env from conf/db.env if present
    here = Path(__file__).resolve().parent
    db_env = (here / ".." / "conf" / "db.env").resolve()
    if db_env.exists():
        load_dotenv(db_env)
    dsn = os.environ.get("MIMIC_DSN", "").strip()
    if dsn:
        return dsn
    # Fall back to local SQLite built from CSVs (zero-setup mode)
    from .csv_to_sqlite import ensure_sqlite_db

    sqlite_uri = ensure_sqlite_db()
    return sqlite_uri


def load_stat_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)
