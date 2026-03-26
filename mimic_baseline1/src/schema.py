from pydantic import BaseModel
from typing import Dict, Any


class DailyStatJSON(BaseModel):
    stat: str
    units: str
    day: str
    range: Dict[str, Any]
    central: Dict[str, Any]
    percentiles: Dict[str, Any]
    trend: Dict[str, Any]
    variability: Dict[str, Any]
    coverage: Dict[str, Any]
    outliers: Dict[str, Any]
    flags: Dict[str, Any]
