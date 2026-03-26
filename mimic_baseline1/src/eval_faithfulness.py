import json
import re
from typing import Any, Dict, List, Tuple

from .schema import DailyStatJSON


def _strip_code_fences(s: str) -> str:
    return re.sub(r"```[a-zA-Z]*\n?|```", "", s)


def _find_json_obj(s: str) -> Tuple[Dict[str, Any], int, int]:
    """
    Find a JSON object in the string `s`. Returns (obj, start_idx, end_idx).
    Chooses the last valid JSON object if multiple exist.
    Raises ValueError if none found.
    """
    s2 = _strip_code_fences(s)
    last_ok = None
    # naive brace-scan attempts for all '{' positions
    for i, ch in enumerate(s2):
        if ch != '{':
            continue
        depth = 0
        for j in range(i, len(s2)):
            if s2[j] == '{':
                depth += 1
            elif s2[j] == '}':
                depth -= 1
                if depth == 0:
                    candidate = s2[i : j + 1]
                    try:
                        obj = json.loads(candidate)
                        # validate shape with pydantic
                        DailyStatJSON(**obj)
                        last_ok = (obj, i, j + 1)
                    except Exception:
                        pass
                    break
    if last_ok is None:
        raise ValueError("No valid JSON object found")
    return last_ok


def _extract_numerics(text: str) -> List[float]:
    # Match integers and decimals, including negatives
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    out: List[float] = []
    for n in nums:
        try:
            out.append(float(n))
        except Exception:
            pass
    return out


def _flatten_numeric_values(obj: Any) -> List[float]:
    acc: List[float] = []
    if isinstance(obj, dict):
        for v in obj.values():
            acc.extend(_flatten_numeric_values(v))
    elif isinstance(obj, list):
        for v in obj:
            acc.extend(_flatten_numeric_values(v))
    else:
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            acc.append(float(obj))
    return acc


def check_faithfulness(content: str, tol: float = 0.01) -> Dict[str, Any]:
    obj, s, e = _find_json_obj(content)
    prose = content[:s]
    nums_prose = _extract_numerics(prose)
    nums_json = _flatten_numeric_values(obj)

    misses: List[float] = []
    for x in nums_prose:
        if not any(abs(x - y) <= tol for y in nums_json):
            misses.append(x)

    return {
        "ok": len(misses) == 0,
        "misses": misses,
        "n_text_nums": len(nums_prose),
        "n_json_nums": len(nums_json),
    }

