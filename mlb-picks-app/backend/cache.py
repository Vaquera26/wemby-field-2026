import json, os
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _path(date_str: str) -> Path:
    return CACHE_DIR / f"picks_{date_str}.json"


def load(date_str: str):
    p = _path(date_str)
    if p.exists():
        return json.loads(p.read_text())
    return None


def save(date_str: str, data):
    _path(date_str).write_text(json.dumps(data, ensure_ascii=False))
