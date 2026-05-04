"""
Day 1 Script: Download Spider + BIRD, filter to medium/hard complexity.
Output: data/processed/spider_filtered.jsonl, bird_filtered.jsonl
"""

import json
import os
import sqlparse
import duckdb
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)


# ── Complexity heuristics ───────────────────────────────────────────────────

def count_joins(sql: str) -> int:
    return sql.upper().count(" JOIN ")

def count_subqueries(sql: str) -> int:
    return sql.upper().count("SELECT") - 1  # extra SELECTs = subqueries

def has_aggregation(sql: str) -> bool:
    keywords = ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN(", "GROUP BY", "HAVING"]
    return any(k in sql.upper() for k in keywords)

def has_nested_conditions(sql: str) -> bool:
    return any(k in sql.upper() for k in ["EXISTS", "NOT EXISTS", "IN (SELECT", "NOT IN (SELECT"])

def classify_complexity(sql: str) -> str:
    """
    easy   — single table, no joins, no subqueries
    medium — 1-2 joins OR aggregation
    hard   — 2+ joins, subqueries, nested conditions
    """
    joins = count_joins(sql)
    subs  = count_subqueries(sql)
    agg   = has_aggregation(sql)
    nested = has_nested_conditions(sql)

    if joins >= 2 or subs >= 1 or nested:
        return "hard"
    if joins == 1 or agg:
        return "medium"
    return "easy"

def is_valid_sql(sql: str) -> bool:
    """Check SQL parses without error using sqlparse."""
    try:
        parsed = sqlparse.parse(sql.strip())
        return len(parsed) > 0 and parsed[0].tokens is not None
    except Exception:
        return False


# ── Spider ──────────────────────────────────────────────────────────────────

def process_spider():
    print("\n── Spider ──")
    ds = load_dataset("spider", trust_remote_code=True)

    results = []
    skipped = 0

    for split in ["train", "validation"]:
        for row in tqdm(ds[split], desc=f"spider/{split}"):
            sql   = row.get("query", "").strip()
            nl    = row.get("question", "").strip()
            db_id = row.get("db_id", "")
            schema = row.get("db_schema", "") or ""

            if not sql or not nl:
                skipped += 1
                continue
            if not is_valid_sql(sql):
                skipped += 1
                continue

            complexity = classify_complexity(sql)
            if complexity == "easy":
                skipped += 1
                continue  # skip easy

            results.append({
                "source":     "spider",
                "split":      split,
                "db_id":      db_id,
                "question":   nl,
                "sql":        sql,
                "complexity": complexity,
                "joins":      count_joins(sql),
                "subqueries": count_subqueries(sql),
            })

    out = PROCESSED / "spider_filtered.jsonl"
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"  Kept: {len(results)} | Skipped (easy/invalid): {skipped}")
    print(f"  Saved → {out}")

    # complexity breakdown
    df = pd.DataFrame(results)
    print(df["complexity"].value_counts().to_string())
    return results


# ── BIRD ────────────────────────────────────────────────────────────────────

def process_bird():
    print("\n── BIRD ──")
    try:
        ds = load_dataset("birdsql/bird", trust_remote_code=True)
    except Exception:
        # BIRD requires manual download sometimes — fall back to spider subset
        print("  BIRD not available via HF hub — using Spider hard split only.")
        print("  To add BIRD: download from https://bird-bench.github.io/ and place in data/raw/bird/")
        return []

    results = []
    skipped = 0

    for split in ds.keys():
        for row in tqdm(ds[split], desc=f"bird/{split}"):
            sql   = (row.get("SQL") or row.get("query") or "").strip()
            nl    = (row.get("question") or "").strip()
            db_id = row.get("db_id", "")
            difficulty = row.get("difficulty", "")

            if not sql or not nl:
                skipped += 1
                continue
            if not is_valid_sql(sql):
                skipped += 1
                continue

            # BIRD has its own difficulty labels — keep medium + challenging
            if difficulty == "simple":
                skipped += 1
                continue

            complexity = classify_complexity(sql)

            results.append({
                "source":     "bird",
                "split":      split,
                "db_id":      db_id,
                "question":   nl,
                "sql":        sql,
                "complexity": complexity,
                "difficulty": difficulty,
                "joins":      count_joins(sql),
                "subqueries": count_subqueries(sql),
            })

    out = PROCESSED / "bird_filtered.jsonl"
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"  Kept: {len(results)} | Skipped: {skipped}")
    print(f"  Saved → {out}")
    return results


# ── Merge + stats ───────────────────────────────────────────────────────────

def merge_and_report(spider_data, bird_data):
    all_data = spider_data + bird_data

    out = PROCESSED / "layer1_combined.jsonl"
    with open(out, "w") as f:
        for r in all_data:
            f.write(json.dumps(r) + "\n")

    df = pd.DataFrame(all_data)
    print(f"\n── Combined Layer 1 ──")
    print(f"  Total examples : {len(df)}")
    print(f"  By source      :\n{df['source'].value_counts().to_string()}")
    print(f"  By complexity  :\n{df['complexity'].value_counts().to_string()}")
    print(f"  By joins       :\n{df['joins'].value_counts().sort_index().to_string()}")
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    spider_data = process_spider()
    bird_data   = process_bird()
    merge_and_report(spider_data, bird_data)
    print("\nDay 1 complete. Run scripts/generate_schemas.py next (Day 1 part 2).")