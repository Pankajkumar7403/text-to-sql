"""
Day 2, Part 2 — Baseline eval runner.

WHY THIS FILE EXISTS:
  This runs Groq's llama-3.3-70b against your golden eval set and records
  execution accuracy. These are your "competition numbers" — the baseline
  your fine-tuned Qwen2.5-7B must beat to have a story worth telling.

HOW EXECUTION ACCURACY WORKS (step by step):
  1. Load a question + schema from the eval set
  2. Send the question + schema to Groq, get back SQL
  3. Create an in-memory DuckDB from the schema's CREATE TABLE statements
  4. Insert synthetic sample rows so queries have real data to run against
  5. Execute the generated SQL — catch any errors
  6. Execute the reference SQL (the correct answer)
  7. Compare result sets. Match → score 1. No match → score 0.
  8. Aggregate scores by complexity bucket (easy / medium / hard)

WHY GROQ:
  Free tier, no credit card, OpenAI-compatible API.
  llama-3.3-70b-versatile = GPT-4o level quality for SQL tasks.
  Just swap base_url + model — everything else is standard OpenAI client.

WHY IN-MEMORY DUCKDB:
  No files, no server to spin up. CREATE TABLE from schema JSON, insert
  sample rows, run SQL, compare results, done. Each eval example gets
  its own fresh connection — no state bleeds between tests.

SETUP:
  Add to .env:
    GROQ_API_KEY=gsk_...
    GROQ_BASE_URL=https://api.groq.com/openai/v1
    GROQ_MODEL=llama-3.3-70b-versatile

HOW TO RUN:
  python scripts/run_baseline_eval.py                    # full eval
  python scripts/run_baseline_eval.py --limit 10         # quick 10-question test
  python scripts/run_baseline_eval.py --dry-run          # no API calls, tests pipeline

GROQ FREE TIER LIMITS:
  30 requests/min, 6000 tokens/min, 1000 requests/day
  Script auto-sleeps 2s between calls to stay within 30 RPM.
  Full 68-question eval takes ~2.5 minutes and uses ~136 of your 1000 daily requests.

Output:
  data/eval/results/baseline_{model}_{timestamp}.jsonl  — per-question results
  data/eval/results/baseline_summary_{timestamp}.json   — comparison table
"""

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import duckdb
from openai import OpenAI          # Groq uses the openai package — just different base_url
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

ROOT       = Path(__file__).parent.parent
EVAL_DIR   = ROOT / "data" / "eval"
SCHEMA_DIR = ROOT / "data" / "raw" / "schemas"
RESULTS    = EVAL_DIR / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL    = os.getenv("GROQ_MODEL",    "llama-3.3-70b-versatile")

# WHY: Fair comparison requires the same prompt for all models.
# The fine-tune is trained on examples formatted with this exact system prompt.
SYSTEM_PROMPT = """You are an expert SQL engineer. Given a database schema and a natural language question, write a correct and efficient SQL query.

Rules:
- Output ONLY the SQL query, no explanation, no markdown fences
- Use table aliases for readability on multi-join queries
- Use explicit JOIN syntax, never implicit comma joins
- Prefer CTEs over deeply nested subqueries when it improves readability
- Never use SELECT * — always specify column names"""


# DATA LOADING

def load_schemas() -> dict:
    """Load all schema JSONs into a dict keyed by schema name."""
    schemas = {}
    for f in SCHEMA_DIR.glob("*.json"):
        if f.name == "index.json":
            continue
        with open(f) as fh:
            s = json.load(fh)
        schemas[s["name"]] = s
    print(f"Loaded {len(schemas)} schemas.")
    return schemas


def load_eval_set(limit: int | None = None) -> list:
    """Load golden eval set from JSONL, optionally capped at `limit` examples."""
    examples = []
    with open(EVAL_DIR / "golden_eval.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if limit:
        examples = examples[:limit]
    print(f"Loaded {len(examples)} eval examples.")
    return examples


# DUCKDB SANDBOX
#
# WHY SYNTHETIC ROWS:
#   Empty tables return [] for every query — two wrong queries would both
#   return [] and look like a match. A few rows give queries something to
#   disagree on, making execution accuracy a real signal.
#
# WHY DUCKDB NOT SQLITE:
#   SQLite lacks window functions (ROW_NUMBER, LAG, RANK), DATE_TRUNC,
#   and INTERVAL arithmetic. Our hard eval questions use all of these.
#   DuckDB supports the full SQL standard we care about.

def _infer_value(col_def: str, row_idx: int) -> str:
    """
    Auto-generate a SQL literal for a column based on its type definition.

    WHY: We create INSERT rows from schema JSON alone — no hardcoded
    per-table values. This infers a sensible value from the column name
    and type string so we never need to manually seed each schema.

    Args:
        col_def:  Full column definition string e.g. "amount DECIMAL(10,2)"
        row_idx:  Which row we're generating (0-4), used to vary values
    Returns:
        SQL literal string e.g. "'active'" or "1234.50"
    """
    col_lower = col_def.lower()
    col_name  = col_def.split()[0].lower()

    if "bool" in col_lower:
        return "TRUE" if row_idx % 2 == 0 else "FALSE"

    if "timestamp" in col_lower:
        return f"'2024-0{(row_idx % 9)+1}-{(row_idx % 28)+1:02d} 10:{row_idx*7 % 60:02d}:00'"
    if "date" in col_lower:
        return f"'2024-0{(row_idx % 9)+1}-{(row_idx % 28)+1:02d}'"

    # FK references and ID columns — keep values in 1-3 range so JOINs work
    if "fk(" in col_lower or col_name.endswith("_id"):
        return str((row_idx % 3) + 1)

    if any(t in col_lower for t in ["int", "serial", "bigint", "smallint"]):
        return str(row_idx + 1)
    if any(t in col_lower for t in ["decimal", "numeric", "float", "double", "real"]):
        return str(round(1000.0 + row_idx * 123.45, 2))

    if any(t in col_lower for t in ["varchar", "char", "text"]):
        # Domain-aware string values so status/type columns have realistic data
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
    """
    Spin up an in-memory DuckDB with all schema tables and 5 rows each.

    WHY 5 ROWS: Enough for GROUP BY / aggregations to produce non-trivial
    results. Window functions like LAG/LEAD need ≥3 rows to show behaviour.
    Small enough that each eval example takes <5ms to set up.
    """
    conn = duckdb.connect(":memory:")

    for table_name, table_info in schema["tables"].items():
        # Strip PK/FK annotations — they're documentation markers in our schema
        # format, not valid DuckDB syntax
        col_defs = []
        for col in table_info["columns"]:
            clean = col.split(" FK(")[0].replace(" PK", "").strip()
            col_defs.append(f"  {clean}")

        create_sql = f"CREATE TABLE {table_name} (\n" + ",\n".join(col_defs) + "\n);"
        try:
            conn.execute(create_sql)
        except Exception:
            pass  # Log but don't abort — one bad table shouldn't kill the eval

        # Insert 5 rows per table
        cols = [c.split()[0] for c in table_info["columns"]]
        for row_idx in range(5):
            vals = [_infer_value(c, row_idx) for c in table_info["columns"]]
            insert_sql = (
                f"INSERT INTO {table_name} ({', '.join(cols)}) "
                f"VALUES ({', '.join(vals)});"
            )
            try:
                conn.execute(insert_sql)
            except Exception:
                pass  # Type mismatches on some rows are OK — data variety matters

    return conn


def execute_sql(conn: duckdb.DuckDBPyConnection, sql: str) -> tuple[bool, list | None, str]:
    """
    Execute SQL safely, return (success, rows, error_message).

    WHY TUPLE: We need to distinguish three states:
      - Syntax/execution error  → (False, None, "error msg")
      - Valid SQL, empty result → (True, [], "")
      - Valid SQL, has data     → (True, [...rows...], "")
    Only the first state counts as a failure for valid_sql_rate.
    """
    try:
        result = conn.execute(sql).fetchall()
        return True, result, ""
    except Exception as e:
        return False, None, str(e)


def results_match(a: list | None, b: list | None) -> bool:
    """
    Compare two result sets order-insensitively.

    WHY ORDER-INSENSITIVE: The reference SQL might ORDER BY differently
    than generated SQL. If the data content is identical, the query is
    logically correct. We normalize by sorting both result sets before
    comparing.

    WHY ROUND FLOATS: Floating point arithmetic can produce 100.00000001
    vs 100.0 — rounding to 2dp eliminates false negatives on numeric results.
    """
    if a is None or b is None:
        return False
    try:
        def normalize(rows: list) -> list:
            return sorted(
                [tuple(round(float(v), 2) if isinstance(v, float) else v for v in row)
                 for row in rows],
                key=lambda x: [str(i) for i in x]
            )
        return normalize(a) == normalize(b)
    except Exception:
        return a == b


# GROQ INFERENCE

def build_groq_client() -> OpenAI:
    """
    Build an OpenAI client pointed at Groq's API.

    WHY OPENAI CLIENT FOR GROQ: Groq implements the OpenAI API spec exactly.
    Changing base_url is all it takes — no new SDK, no new auth scheme.
    This is also why our fine-tuned model's inference API will be
    OpenAI-compatible: any client can talk to it without changes.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not set.\n"
            "1. Sign up at console.groq.com (free, no credit card)\n"
            "2. Create an API key\n"
            "3. Add GROQ_API_KEY=gsk_... to your .env file"
        )
    return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)


def call_groq(
    schema_sql: str,
    question: str,
    client: OpenAI,
    model: str = GROQ_MODEL,
) -> tuple[str, float]:
    """
    Send a question + schema to Groq, return (generated_sql, latency_seconds).

    WHY temperature=0: We want deterministic SQL generation for eval.
    Temperature > 0 would make the same question produce different SQL on
    each run, making results non-reproducible.

    WHY max_tokens=512: SQL queries rarely exceed 300 tokens. Capping at 512
    saves cost and prevents runaway generation.
    """
    start = time.time()

    response = client.chat.completions.create(
        model=model,
        temperature=0,       # deterministic — same input always gives same output
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Schema:\n{schema_sql}\n\nQuestion: {question}\n\nSQL:"
            },
        ],
    )

    latency = time.time() - start
    sql = response.choices[0].message.content.strip()

    # Strip markdown fences — some models add them despite the system prompt
    sql = sql.replace("```sql", "").replace("```", "").strip()

    return sql, latency


# EVAL LOOP

def run_eval(
    model: str,
    examples: list,
    schemas: dict,
    client: OpenAI | None,
    dry_run: bool,
) -> list:
    """
    Run all eval examples through the model, return per-question result dicts.

    Args:
        model:    Model name string (used for logging and output filenames)
        examples: List of eval examples from golden_eval.jsonl
        schemas:  Dict of schema dicts keyed by schema_name
        client:   Groq/OpenAI client (None in dry_run mode)
        dry_run:  If True, skip API calls and use reference SQL (tests pipeline)
    """
    results = []
    api_errors = 0
    sql_errors = 0

    for ex in tqdm(examples, desc=f"Evaluating {model}"):
        schema = schemas.get(ex["schema_name"])
        if not schema:
            print(f"  WARNING: schema '{ex['schema_name']}' not found, skipping.")
            continue

        schema_sql = schema["create_sql"]
        question   = ex["question"]
        ref_sql    = ex["sql"]

        if dry_run:
            # Use reference SQL so pipeline scores 100% — confirms eval logic works
            gen_sql = ref_sql
            latency = 0.0
        else:
            try:
                gen_sql, latency = call_groq(schema_sql, question, client, model)
            except Exception as e:
                api_errors += 1
                results.append({
                    **ex,
                    "model":         model,
                    "generated_sql": "",
                    "valid_sql":     False,
                    "exec_match":    False,
                    "gen_error":     str(e),
                    "latency_s":     0.0,
                })
                continue

        conn = build_sandbox(schema)
        valid_sql, gen_result, gen_error = execute_sql(conn, gen_sql)
        _,          ref_result, _        = execute_sql(conn, ref_sql)
        conn.close()  # explicitly close — each example gets its own fresh DB

        if not valid_sql:
            sql_errors += 1

        match = results_match(gen_result, ref_result) if valid_sql else False

        results.append({
            "schema_name":   ex["schema_name"],
            "question":      question,
            "complexity":    ex["complexity"],
            "tags":          ex["tags"],
            "model":         model,
            "reference_sql": ref_sql,
            "generated_sql": gen_sql,
            "valid_sql":     valid_sql,
            "exec_match":    match,
            "gen_error":     gen_error,
            "latency_s":     round(latency, 3),
        })

        # Groq free tier ~30 RPM; proactive sleep avoids 429 backoff logic.
        if not dry_run:
            time.sleep(2.0)

    print(f"  Errors → API: {api_errors} | SQL exec: {sql_errors}")
    return results


def compute_summary(results: list, model: str) -> dict:
    """
    Compute eval metrics broken down by complexity bucket.

    Two metrics:
    - valid_sql_rate:  % of outputs that parse and execute without error
                       Measures "did the model produce syntactically correct SQL"
    - exec_accuracy:   % where result set matches reference answer
                       Measures "did the model produce CORRECT SQL"
                       This is the headline metric — the one that goes in README.
    """
    by_complexity: dict = defaultdict(list)
    for r in results:
        by_complexity[r["complexity"]].append(r)

    n = len(results)
    summary = {
        "model":                    model,
        "total":                    n,
        "valid_sql_rate_overall":   round(sum(1 for r in results if r["valid_sql"]) / n * 100, 1) if n else 0,
        "exec_accuracy_overall":    round(sum(1 for r in results if r["exec_match"]) / n * 100, 1) if n else 0,
        "avg_latency_s":            round(sum(r["latency_s"] for r in results) / n, 3) if n else 0,
    }

    for bucket, bucket_results in sorted(by_complexity.items()):
        bn = len(bucket_results)
        summary[f"valid_sql_{bucket}"]      = round(sum(1 for r in bucket_results if r["valid_sql"]) / bn * 100, 1)
        summary[f"exec_accuracy_{bucket}"]  = round(sum(1 for r in bucket_results if r["exec_match"]) / bn * 100, 1)

    return summary


def print_results_table(summaries: list) -> None:
    """
    Print the comparison table — this exact table goes in your README and
    is what interviewers will ask you to explain in interviews.
    """
    print("\n" + "═" * 76)
    print("BASELINE RESULTS — Execution Accuracy by Model and Complexity")
    print("═" * 76)
    print(f"{'Model':<28} {'Easy':>7} {'Medium':>8} {'Hard':>8} {'Overall':>9} {'Latency':>9}")
    print("─" * 76)
    for s in summaries:
        print(
            f"{s['model']:<28}"
            f"{s.get('exec_accuracy_easy',    0):>6.1f}%"
            f"{s.get('exec_accuracy_medium',  0):>7.1f}%"
            f"{s.get('exec_accuracy_hard',    0):>7.1f}%"
            f"{s.get('exec_accuracy_overall', 0):>8.1f}%"
            f"{s.get('avg_latency_s',         0):>8.2f}s"
        )
    print("═" * 76)
    print("\nThese are your competition numbers.")
    print("Your fine-tuned Qwen2.5-7B must beat the 'Hard' column to have a story.")


# ENTRYPOINT

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline eval against Groq models."
    )
    parser.add_argument(
        "--models", nargs="+",
        default=[GROQ_MODEL],
        help=f"Groq model(s) to evaluate. Default: {GROQ_MODEL}\n"
             "Other good options: qwen-qwq-32b, llama-3.1-70b-versatile"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap number of eval examples (useful for quick sanity checks)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip API calls, use reference SQL — tests the eval pipeline itself"
    )
    args = parser.parse_args()

    schemas  = load_schemas()
    examples = load_eval_set(args.limit)

    # Build Groq client once — reused across all models
    client = None
    if not args.dry_run:
        client = build_groq_client()
        print(f"Groq client ready. Base URL: {GROQ_BASE_URL}")
        print(f"Free tier reminder: 30 RPM / 1000 req/day. "
              f"This run will use ~{len(examples)} requests.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries = []

    for model in args.models:
        print(f"\n── Evaluating: {model} ──")
        results = run_eval(model, examples, schemas, client, args.dry_run)

        # Save per-question results — useful for error analysis later
        out_file = RESULTS / f"baseline_{model.replace('/', '_')}_{timestamp}.jsonl"
        with open(out_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"  Per-question results → {out_file}")

        summaries.append(compute_summary(results, model))

    # Save summary JSON
    summary_file = RESULTS / f"baseline_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summaries, f, indent=2)

    print_results_table(summaries)
    print(f"\nSummary JSON → {summary_file}")


if __name__ == "__main__":
    main()