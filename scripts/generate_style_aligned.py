"""
Day 13 — Style-aligned training data generation.

WHY THIS EXISTS:
  v2 eval showed medium/hard accuracy unchanged at 38.5%/33.3% despite 700 hard
  negatives targeting window functions, CTEs, etc. Root cause analysis of the 32
  failures revealed the model KNOWS the SQL constructs — it fails on OUTPUT STYLE:

  1. Missing name/label columns: model writes SELECT user_id, SUM(...) but reference
     has SELECT user_id, full_name, SUM(...). Same data, different columns → eval fails.
  2. Wrong status values: model writes status = 'approved' but domain requires
     status IN ('approved', 'disbursed').
  3. Missing implied filters: drops WHERE status = 'active' or date range clauses.
  4. Missing companion metrics: reference has SUM + COUNT together; model gives SUM only.

  Hard negatives did not fix this because they taught SYNTAX (window functions) not
  STYLE (which columns to include, which status values apply). This script fixes that.

APPROACH:
  - Use 5 passing golden eval examples as few-shot style demonstrations in every prompt.
  - Inject 4 explicit style rules that directly address the failure patterns above.
  - Generate 80 examples per schema (800 total) at medium/hard complexity.
  - Validate every SQL with DuckDB before saving.

HOW TO RUN:
  python scripts/generate_style_aligned.py
  python scripts/generate_style_aligned.py --per-schema 20  # quick test
  python scripts/generate_style_aligned.py --dry-run

OUTPUT:
  data/processed/layer4_style_aligned.jsonl
  Next: python scripts/build_dataset.py  (rebuild with all 4 layers)
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import duckdb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT          = Path(__file__).parent.parent
SCHEMA_DIR    = ROOT / "data" / "raw" / "schemas"
PROC_DIR      = ROOT / "data" / "processed"
EVAL_FILE     = ROOT / "data" / "eval" / "golden_eval.jsonl"
V1_RESULTS    = ROOT / "data" / "eval" / "results" / "finetuned_qwen25_7b_20260505_132721.jsonl"
OUT_FILE      = PROC_DIR / "layer4_style_aligned.jsonl"

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

TARGET_PER_SCHEMA = 80
BATCH_SIZE        = 8
RATE_SLEEP        = 15.0

COMPLEXITY_MIX = {"medium": 0.40, "hard": 0.60}


STYLE_RULES = """\
STYLE RULES — follow all of these exactly:

1. NAME COLUMNS: When GROUP BY includes an entity ID (user_id, seller_id, loan_id, etc.),
   always include the entity's human-readable name/label column in SELECT too.
   WRONG:  SELECT user_id, SUM(amount) ... GROUP BY user_id
   RIGHT:  SELECT u.user_id, u.full_name, SUM(o.total_amount) ... GROUP BY u.user_id, u.full_name

2. STATUS FILTERS: When filtering by status, include ALL semantically relevant values.
   WRONG:  WHERE status = 'approved'
   RIGHT:  WHERE status IN ('approved', 'disbursed')
   Common multi-value statuses: active/pending, approved/disbursed, completed/settled/processed

3. COMPANION METRICS: When the question asks for totals, rates, or rankings by category,
   include both COUNT and SUM (or AVG) — one for volume, one for value.
   WRONG:  SELECT seller_id, SUM(revenue) AS total_revenue
   RIGHT:  SELECT seller_id, shop_name, COUNT(order_id) AS order_count, SUM(revenue) AS total_revenue

4. DATE/STATUS FILTERS: When a question implies a time window ("last 30 days", "this month",
   "active", "recent") or a completion state, always include the WHERE clause.
   WRONG:  SELECT merchant_id, SUM(amount) FROM payments GROUP BY merchant_id
   RIGHT:  SELECT merchant_id, SUM(amount) FROM payments WHERE status = 'settled'
           AND paid_at >= CURRENT_DATE - INTERVAL '30 days' GROUP BY merchant_id\
"""

SYSTEM_PROMPT = f"""\
You are an expert SQL engineer and training data curator.
You generate high-quality question-SQL pairs for fine-tuning a Text-to-SQL model.

{STYLE_RULES}

Rules for questions:
- Write natural business-analyst English: "How many...", "Find the...", "Which...", "Show..."
- Every question in the batch must be unique

Rules for SQL:
- Output ONLY valid DuckDB-compatible SQL (PostgreSQL dialect)
- Always use explicit JOIN syntax with table aliases
- Never use SELECT * — always name columns
- All column and table names must EXACTLY match the schema

Output format — respond with ONLY a valid JSON array, nothing else:
[
  {{"question": "...", "sql": "...", "complexity": "medium|hard"}},
  ...
]\
"""


def load_schemas(filter_names: list[str] | None = None) -> dict[str, dict]:
    schemas: dict[str, dict] = {}
    for f in sorted(SCHEMA_DIR.glob("*.json")):
        if f.name == "index.json":
            continue
        with open(f) as fh:
            s = json.load(fh)
        if filter_names and s["name"] not in filter_names:
            continue
        schemas[s["name"]] = s
    log.info(f"Loaded {len(schemas)} schemas.")
    return schemas


def load_passing_examples() -> list[dict]:
    """Return golden eval examples the v1 model got right — used as style demos."""
    if not V1_RESULTS.exists():
        log.warning("v1 results not found — using random golden eval examples as demos.")
        examples = [json.loads(l) for l in open(EVAL_FILE)]
        return [e for e in examples if e["complexity"] in ("medium", "hard")][:5]

    results  = [json.loads(l) for l in open(V1_RESULTS)]
    passing  = {r["question"] for r in results if r["exec_match"]}
    examples = [json.loads(l) for l in open(EVAL_FILE)]
    return [e for e in examples if e["question"] in passing
            and e["complexity"] in ("medium", "hard")]


def load_existing_questions(path: Path) -> set[str]:
    questions: set[str] = set()
    if not path.exists():
        return questions
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    questions.add(json.loads(line)["question"].strip().lower())
                except Exception:
                    pass
    return questions


def count_existing_per_schema(path: Path) -> dict[str, int]:
    """Count already-written examples per schema — used to skip completed schemas."""
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    name = json.loads(line)["schema_name"]
                    counts[name] = counts.get(name, 0) + 1
                except Exception:
                    pass
    return counts


def _infer_value(col_def: str, row_idx: int) -> str:
    col_lower = col_def.lower()
    col_name  = col_def.split()[0].lower()
    if "bool" in col_lower:
        return "TRUE" if row_idx % 2 == 0 else "FALSE"
    if "timestamp" in col_lower:
        return f"'2024-0{(row_idx % 9)+1}-{(row_idx % 28)+1:02d} 10:{row_idx*7 % 60:02d}:00'"
    if "date" in col_lower:
        return f"'2024-0{(row_idx % 9)+1}-{(row_idx % 28)+1:02d}'"
    if "fk(" in col_lower or col_name.endswith("_id"):
        return str((row_idx % 3) + 1)
    if any(t in col_lower for t in ["int", "serial", "bigint", "smallint"]):
        return str(row_idx + 1)
    if any(t in col_lower for t in ["decimal", "numeric", "float", "double", "real"]):
        return str(round(1000.0 + row_idx * 123.45, 2))
    if any(t in col_lower for t in ["varchar", "char", "text"]):
        domain_samples = {
            "status":   ["active", "pending", "completed", "cancelled", "failed"],
            "type":     ["type_a", "type_b", "type_c"],
            "country":  ["IN", "US", "UK", "SG", "AE"],
            "email":    [f"user{row_idx}@example.com"],
            "name":     [f"Name {row_idx + 1}"],
            "category": ["electronics", "clothing", "food", "services"],
            "currency": ["INR", "USD", "GBP"],
            "city":     ["Mumbai", "Delhi", "Bangalore", "Chennai"],
        }
        for key, vals in domain_samples.items():
            if key in col_name:
                return f"'{vals[row_idx % len(vals)]}'"
        return f"'value_{row_idx + 1}'"
    return "NULL"


def build_sandbox(schema: dict) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    for table_name, table_info in schema["tables"].items():
        col_defs = [
            f"  {col.split(' FK(')[0].replace(' PK', '').strip()}"
            for col in table_info["columns"]
        ]
        try:
            conn.execute(f"CREATE TABLE {table_name} (\n" + ",\n".join(col_defs) + "\n);")
        except Exception:
            pass
        cols = [c.split()[0] for c in table_info["columns"]]
        for row_idx in range(5):
            vals = [_infer_value(c, row_idx) for c in table_info["columns"]]
            try:
                conn.execute(
                    f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({', '.join(vals)});"
                )
            except Exception:
                pass
    return conn


def validate_sql(conn: duckdb.DuckDBPyConnection, sql: str) -> tuple[bool, str]:
    try:
        conn.execute(sql)
        return True, ""
    except Exception as e:
        return False, str(e)


def build_prompt(schema: dict, complexity: str, n: int,
                 demos: list[dict]) -> str:
    """
    Build a generation prompt with:
    - Schema context
    - 3 passing golden eval examples as style demonstrations
    - Explicit complexity and batch size targets
    """
    # Pick demos matching this schema if possible, otherwise use any passing ones
    schema_demos = [d for d in demos if d["schema_name"] == schema["name"]]
    other_demos  = [d for d in demos if d["schema_name"] != schema["name"]]

    # Use up to 2 schema-specific demos + fill from others, max 3 total
    selected = (schema_demos[:2] + other_demos)[:3]

    demo_text = ""
    if selected:
        demo_text = "\n\nEXAMPLE STYLE (from real eval set — match this quality):\n"
        for ex in selected:
            demo_text += f'Q: {ex["question"]}\nSQL: {ex["sql"].strip()}\n\n'

    return (
        f"Database: {schema['name']} ({schema.get('domain', '')} domain)\n"
        f"Schema:\n{schema['create_sql'].strip()}\n"
        f"{demo_text}\n"
        f"Generate exactly {n} UNIQUE {complexity.upper()} complexity question-SQL pairs "
        f"that follow ALL 4 style rules above.\n"
        f"Respond ONLY with a JSON array of {n} objects. No explanation, no markdown."
    )


def call_groq(client: OpenAI, prompt: str, complexity: str,
              retries: int = 3) -> list[dict]:
    max_tokens = 4096 if complexity == "hard" else 2048
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.6,
                max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            pairs = json.loads(raw)
            if isinstance(pairs, list):
                return pairs
        except json.JSONDecodeError as e:
            log.warning(f"  JSON parse error (attempt {attempt+1}): {e}")
        except Exception as e:
            err = str(e)
            wait = 60 if "429" in err or "503" in err else 10
            log.warning(f"  API error (attempt {attempt+1}): {err[:80]}. Wait {wait}s...")
            time.sleep(wait)
    return []


def generate_for_schema(schema: dict, client: OpenAI, target: int,
                        demos: list[dict], existing: set[str],
                        dry_run: bool = False) -> list[dict]:
    conn    = build_sandbox(schema)
    results = []
    seen    = set(existing)

    for complexity, fraction in COMPLEXITY_MIX.items():
        target_n  = round(target * fraction)
        collected = 0
        log.info(f"  {complexity}: targeting {target_n}")

        while collected < target_n:
            remaining = target_n - collected
            batch_n   = min(BATCH_SIZE, remaining + BATCH_SIZE)

            if dry_run:
                log.info(f"  [dry-run] would request {batch_n} {complexity} pairs")
                break

            prompt = build_prompt(schema, complexity, batch_n, demos)
            pairs  = call_groq(client, prompt, complexity)
            time.sleep(RATE_SLEEP)

            for pair in pairs:
                if collected >= target_n:
                    break
                q   = pair.get("question", "").strip()
                sql = pair.get("sql", "").strip()
                if not q or not sql or q.lower() in seen:
                    continue
                valid, err = validate_sql(conn, sql)
                if not valid:
                    log.debug(f"  INVALID: {err[:60]}")
                    continue
                seen.add(q.lower())
                results.append({
                    "schema_name": schema["name"],
                    "question":    q,
                    "sql":         sql,
                    "complexity":  complexity,
                    "source":      "style_aligned",
                })
                collected += 1

            log.info(f"  {complexity}: {collected}/{target_n} validated")

    conn.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate style-aligned training data.")
    parser.add_argument("--per-schema", type=int, default=TARGET_PER_SCHEMA)
    parser.add_argument("--schemas",    nargs="+", default=None)
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    if not GROQ_API_KEY and not args.dry_run:
        raise SystemExit("GROQ_API_KEY not found in .env")

    schemas        = load_schemas(args.schemas)
    demos          = load_passing_examples()
    existing       = load_existing_questions(OUT_FILE)
    schema_counts  = count_existing_per_schema(OUT_FILE)

    log.info(f"Style demos loaded: {len(demos)} passing golden eval examples")
    log.info(f"Existing layer4 questions: {len(existing)}")
    if schema_counts:
        log.info(f"Progress so far: { {k: v for k, v in schema_counts.items()} }")

    client        = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL) if not args.dry_run else None
    total_written = 0

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "a") as out_f:
        for schema_name, schema in schemas.items():
            already_done = schema_counts.get(schema_name, 0)

            if already_done >= args.per_schema:
                log.info(f"\nSchema: {schema_name} — SKIP ({already_done}/{args.per_schema} already done)")
                continue

            remaining_target = args.per_schema - already_done
            log.info(f"\nSchema: {schema_name} — {already_done} done, need {remaining_target} more")

            pairs = generate_for_schema(
                schema=schema, client=client,
                target=remaining_target, demos=demos,
                existing=existing, dry_run=args.dry_run,
            )
            for pair in pairs:
                out_f.write(json.dumps(pair) + "\n")
                existing.add(pair["question"].strip().lower())
            out_f.flush()
            log.info(f"  -> {len(pairs)} new pairs for {schema_name}")
            total_written += len(pairs)

    log.info(f"\nDone. Total written: {total_written}")
    log.info(f"Output: {OUT_FILE}")
    log.info("Next: update build_dataset.py to include layer4, then rebuild + retrain")


if __name__ == "__main__":
    main()
