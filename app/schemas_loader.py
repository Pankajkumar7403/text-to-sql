"""
Loads all schema JSONs from disk. Falls back to SCHEMA_DIR env var for deployment.
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

_cache: dict[str, dict] | None = None


def _schema_dir() -> Path:
    env_path = os.getenv("SCHEMA_DIR")
    if env_path:
        return Path(env_path)
    return ROOT / "data" / "raw" / "schemas"


def load_all_schemas() -> dict[str, dict]:
    global _cache
    if _cache is not None:
        return _cache

    schemas: dict[str, dict] = {}
    for f in _schema_dir().glob("*.json"):
        if f.name == "index.json":
            continue
        with open(f) as fh:
            s = json.load(fh)
        if "name" in s and "create_sql" in s:
            schemas[s["name"]] = s

    _cache = schemas
    print(f"[schemas] Loaded {len(schemas)} schemas from {_schema_dir()}")
    return _cache
