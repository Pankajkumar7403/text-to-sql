"""
Day 11 — Hard negative mining: generate targeted training examples for weak SQL patterns.

WHAT THIS DOES:
  Reads failure_analysis.json (from analyze_failures.py) to find which SQL
  constructs the fine-tuned model fails on most. Then generates new training
  examples that REQUIRE those exact constructs, validates them with DuckDB,
  and saves them to layer3_hard_negatives.jsonl.

WHY "HARD NEGATIVES":
  In ML, "hard negatives" are examples the model almost gets right but fails on —
  specifically chosen to close the gap. Training on random examples improves
  average accuracy. Training on targeted failures improves the specific weakness.

WHY NOT RE-USE layer2_synthetic.py:
  The original generator had a 20/40/40 easy/medium/hard split and no pattern
  targeting. This script:
    - Forces 0/20/80 medium/hard split (easy is already fine at 85.7%)
    - Adds a REQUIRED PATTERN constraint to each prompt so Groq must use CTEs,
      window functions, etc.
    - Generates fewer but higher-quality examples per schema

SETUP:
  1. Run analyze_failures.py first to produce failure_analysis.json
  2. Ensure GROQ_API_KEY is in .env

HOW TO RUN:
  python scripts/generate_hard_negatives.py
  python scripts/generate_hard_negatives.py --per-schema 50    # 50 pairs per schema (quick test)
  python scripts/generate_hard_negatives.py --schemas lending_v1 payments_v1
  python scripts/generate_hard_negatives.py --dry-run

OUTPUT:
  data/processed/layer3_hard_negatives.jsonl
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT        = Path(__file__).parent.parent
SCHEMA_DIR  = ROOT / "data" / "raw" / "schemas"
PROC_DIR    = ROOT / "data" / "processed"
ANALYSIS_FILE = PROC_DIR / "failure_analysis.json"
OUT_FILE    = PROC_DIR / "layer3_hard_negatives.jsonl"

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

TARGET_PER_SCHEMA = 100    # hard negatives per schema (vs 200 in original run)
BATCH_SIZE        = 10     # smaller batches — complex SQL needs more token budget
RATE_SLEEP        = 15.0   # seconds between API calls (keeps TPM under Groq free limit)

# Hard negatives are 80% hard, 20% medium.
# Easy was already 85.7% accurate — no point adding more easy examples.
COMPLEXITY_MIX: dict[str, float] = {
    "medium": 0.20,
    "hard":   0.80,
}

# Per-complexity token budget — hard SQL with window functions can be verbose
MAX_TOKENS_BY_COMPLEXITY: dict[str, int] = {
    "medium": 2048,
    "hard":   4096,
}

# PATTERN-SPECIFIC PROMPT CONSTRAINTS
# Each pattern maps to an instruction injected into the generation prompt.
# This FORCES Groq to use the specific SQL construct we want more examples of.
PATTERN_CONSTRAINTS: dict[str, str] = {
    "cte": (
        "REQUIRED: Every SQL in this batch MUST use a WITH clause (CTE). "
        "The CTE should name an intermediate result that simplifies the main query. "
        "Example patterns: filtered set, aggregated base, ranked subset."
    ),
    "window_function": (
        "REQUIRED: Every SQL in this batch MUST use a window function with OVER(). "
        "Use PARTITION BY where grouping makes sense. "
        "Examples: running total, moving average, rank within group, previous row comparison."
    ),
    "rank_function": (
        "REQUIRED: Every SQL in this batch MUST use a ranking window function: "
        "ROW_NUMBER(), RANK(), DENSE_RANK(), or NTILE(). "
        "Classic patterns: top-N per group, first record per category, percentile buckets."
    ),
    "multi_join": (
        "REQUIRED: Every SQL in this batch MUST join 3 or more tables. "
        "Use meaningful aliases. Make sure all JOIN conditions reference correct FK columns."
    ),
    "having": (
        "REQUIRED: Every SQL in this batch MUST use HAVING to filter grouped results. "
        "GROUP BY must aggregate on meaningful columns, HAVING must filter on aggregate values."
    ),
    "correlated_sub": (
        "REQUIRED: Every SQL in this batch MUST use a subquery in WHERE or SELECT. "
        "The subquery should reference the outer query (correlated) or produce a scalar/list "
        "for IN/EXISTS/comparison."
    ),
    "aggregation": (
        "REQUIRED: Every SQL in this batch MUST use GROUP BY with at least one aggregate "
        "function (COUNT, SUM, AVG, MAX, MIN). GROUP BY columns must come from a JOIN."
    ),
}

GENERATION_SYSTEM_PROMPT = """\
You are an expert SQL engineer and training data curator.
You generate diverse, realistic question-SQL pairs for fine-tuning a Text-to-SQL model.

Rules for questions:
- Write natural English a business analyst would actually ask
- Vary phrasing: "How many...", "List all...", "Find the...", "What is...", "Which...", "Show me..."
- Every question in the batch must be unique

Rules for SQL:
- Output ONLY valid SQL compatible with DuckDB (PostgreSQL dialect)
- Always use explicit JOIN syntax — never comma joins
- Use table aliases on any multi-table query
- Never use SELECT * — always name columns
- All column and table names must EXACTLY match the schema

Output format — respond with ONLY a valid JSON array, nothing else:
[
  {"question": "...", "sql": "...", "complexity": "medium|hard"},
  ...
]\
"""


# SCHEMA LOADING

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


def load_failure_analysis() -> dict:
    if not ANALYSIS_FILE.exists():
        raise SystemExit(
            f"failure_analysis.json not found at {ANALYSIS_FILE}\n"
            "Run: python scripts/analyze_failures.py"
        )
    with open(ANALYSIS_FILE) as f:
        return json.load(f)


# DUCKDB SANDBOX (same as generate_synthetic.py)

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
                    f"INSERT INTO {table_name} ({', '.join(cols)}) "
                    f"VALUES ({', '.join(vals)});"
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


# GROQ GENERATION

def build_targeted_prompt(
    schema: dict,
    complexity: str,
    n: int,
    target_patterns: list[str],
) -> str:
    """
    Build a generation prompt that FORCES use of the specified SQL patterns.

    WHY FORCE PATTERNS:
    Without constraints, Groq gravitates toward simple SELECT + WHERE + GROUP BY.
    We need window functions and CTEs specifically — the patterns the model fails on.
    Explicit constraints in the prompt reliably produce the required constructs.
    """
    constraints = []
    for p in target_patterns:
        if p in PATTERN_CONSTRAINTS:
            constraints.append(PATTERN_CONSTRAINTS[p])

    constraint_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(constraints))

    return (
        f"Database: {schema['name']} ({schema.get('domain', '')} domain)\n"
        f"Schema:\n{schema['create_sql'].strip()}\n\n"
        f"Generate exactly {n} UNIQUE {complexity.upper()} complexity question-SQL pairs.\n\n"
        f"MANDATORY SQL CONSTRAINTS (all must be satisfied):\n{constraint_text}\n\n"
        f"Respond ONLY with a JSON array of {n} objects. No explanation, no markdown."
    )


def call_groq_batch(
    client: OpenAI,
    prompt: str,
    complexity: str,
    retries: int = 3,
) -> list[dict]:
    """
    Call Groq API and parse the JSON response.
    Returns list of {question, sql, complexity} dicts, or empty list on failure.
    """
    max_tokens = MAX_TOKENS_BY_COMPLEXITY[complexity]
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content.strip()

            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            pairs = json.loads(raw)
            if isinstance(pairs, list):
                return pairs
        except json.JSONDecodeError as e:
            log.warning(f"    JSON parse error (attempt {attempt+1}): {e}")
        except Exception as e:
            err = str(e)
            wait = 60 if "429" in err or "503" in err else 10
            log.warning(f"    API error (attempt {attempt+1}): {err[:80]}. Waiting {wait}s...")
            time.sleep(wait)
    return []


# PER-SCHEMA GENERATION

def load_existing_questions(path: Path) -> set[str]:
    """Load questions already in the output file to avoid duplicates."""
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


def generate_for_schema(
    schema: dict,
    client: OpenAI,
    target_per_schema: int,
    target_patterns: list[str],
    existing_questions: set[str],
    dry_run: bool = False,
) -> list[dict]:
    """Generate targeted hard negatives for one schema. Returns validated pairs."""
    conn = build_sandbox(schema)
    results: list[dict] = []
    seen_questions = set(existing_questions)

    for complexity, fraction in COMPLEXITY_MIX.items():
        target_n = round(target_per_schema * fraction)
        collected = 0
        log.info(f"  {complexity}: targeting {target_n} pairs (patterns: {target_patterns})")

        while collected < target_n:
            remaining = target_n - collected
            batch_n   = min(BATCH_SIZE, remaining + BATCH_SIZE)  # over-request slightly

            if dry_run:
                log.info(f"    [dry-run] would request {batch_n} {complexity} pairs")
                break

            prompt = build_targeted_prompt(schema, complexity, batch_n, target_patterns)
            pairs  = call_groq_batch(client, prompt, complexity)
            time.sleep(RATE_SLEEP)

            for pair in pairs:
                if collected >= target_n:
                    break
                q   = pair.get("question", "").strip()
                sql = pair.get("sql", "").strip()
                if not q or not sql:
                    continue
                if q.lower() in seen_questions:
                    continue  # skip duplicate question

                valid, err = validate_sql(conn, sql)
                if not valid:
                    log.debug(f"    INVALID: {err[:60]}")
                    continue

                seen_questions.add(q.lower())
                results.append({
                    "schema_name": schema["name"],
                    "question":    q,
                    "sql":         sql,
                    "complexity":  complexity,
                    "source":      "hard_negative",
                })
                collected += 1

            log.info(f"    {complexity}: {collected}/{target_n} validated")

    conn.close()
    return results


# ENTRYPOINT

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate targeted hard negative training examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--per-schema", type=int, default=TARGET_PER_SCHEMA,
                        help=f"Hard negative examples per schema (default: {TARGET_PER_SCHEMA}).")
    parser.add_argument("--schemas", nargs="+", default=None,
                        help="Only run these schemas (default: all 10).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip API calls — test the pipeline without generating data.")
    args = parser.parse_args()

    if not GROQ_API_KEY and not args.dry_run:
        raise SystemExit("GROQ_API_KEY not found in .env. Add it and retry.")

    analysis = load_failure_analysis()
    target_patterns = analysis.get("top_patterns_to_target", [])
    if not target_patterns:
        # Fallback if analysis file is missing the field
        target_patterns = ["cte", "window_function", "rank_function", "multi_join"]
    log.info(f"Targeting patterns: {target_patterns}")
    log.info(f"Complexity failures: {analysis.get('complexity_failures', {})}")

    schemas = load_schemas(args.schemas)

    existing_questions = load_existing_questions(OUT_FILE)
    log.info(f"Existing hard negatives: {len(existing_questions)} questions already in output file.")

    total_written = 0
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL) if not args.dry_run else None

    with open(OUT_FILE, "a") as out_f:   # append — don't overwrite existing data
        for schema_name, schema in schemas.items():
            log.info(f"\nSchema: {schema_name}")

            pairs = generate_for_schema(
                schema=schema,
                client=client,
                target_per_schema=args.per_schema,
                target_patterns=target_patterns,
                existing_questions=existing_questions,
                dry_run=args.dry_run,
            )

            for pair in pairs:
                out_f.write(json.dumps(pair) + "\n")
                existing_questions.add(pair["question"].strip().lower())
            out_f.flush()

            log.info(f"  -> {len(pairs)} new pairs written for {schema_name}")
            total_written += len(pairs)

    log.info(f"\nDone. Total hard negatives written: {total_written}")
    log.info(f"Output: {OUT_FILE}")
    log.info(f"Next:   python scripts/build_dataset.py  (rebuild train/val with hard negatives)")


if __name__ == "__main__":
    main()
