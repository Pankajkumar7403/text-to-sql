"""
Day 3 — Synthetic training data generation.

WHAT THIS BUILDS:
  For each of the 10 domain schemas, asks Groq (llama-3.3-70b-versatile) to
  generate question-SQL pairs in batches, validates each SQL against a
  DuckDB sandbox, and writes passing pairs to data/processed/layer2_synthetic.jsonl.

WHY BATCH GENERATION (20 pairs per API call):
  Groq's free tier allows 1000 requests/day. Asking for 20 pairs per call
  vs 1 means 20x fewer requests consumed. JSON batch output also lets the
  model stay consistent within a batch (same schema, same complexity level).

WHY DUCKDB VALIDATION:
  LLMs hallucinate column names, use wrong syntax, and write logically broken
  queries. DuckDB catches all of these before they corrupt the training set.
  One bad example in training data is worth far more than -1 in dataset size.

WHY temperature=0.7 (not 0.0 like the eval runner):
  We WANT variety in phrasing. Training on diverse question styles helps the
  fine-tuned model generalize. Eval uses 0.0 for reproducibility; generation
  uses 0.7 for creativity.

WHY RESUME SUPPORT:
  If the script hits the 1000 req/day cap mid-run, you can restart and it
  skips schemas that already have enough pairs. Without this, you'd waste
  half your daily quota regenerating data you already have.

GROQ FREE TIER MATH:
  10 schemas × 200 pairs × ~1.5x attempts = ~300 pairs attempted per schema
  300 / 20 per batch = 15 batches per schema × 10 schemas = ~150 API calls total
  At 15s between calls: ~30 minutes wall time. Uses ~150 of your 1000 daily requests.

SETUP:
  Add to .env:  GROQ_API_KEY=gsk_...
  Optional:     WANDB_API_KEY=...

HOW TO RUN:
  python scripts/generate_synthetic.py                             # full run (2000 pairs)
  python scripts/generate_synthetic.py --schemas marketplace_v1   # one schema only
  python scripts/generate_synthetic.py --limit 20                 # 20 pairs per schema (quick test)
  python scripts/generate_synthetic.py --dry-run                  # no API calls — test pipeline
  python scripts/generate_synthetic.py --no-resume                # regenerate even if done

OUTPUT:
  data/processed/layer2_synthetic.jsonl — same format as golden_eval.jsonl
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import duckdb
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT       = Path(__file__).parent.parent
SCHEMA_DIR = ROOT / "data" / "raw" / "schemas"
OUT_DIR    = ROOT / "data" / "processed"
OUT_FILE   = OUT_DIR / "layer2_synthetic.jsonl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")

TARGET_PER_SCHEMA = 200
BATCH_SIZE        = 20    # pairs to request per single API call

# Complexity distribution: what fraction of each schema's 200 pairs are each level
COMPLEXITY_MIX: dict[str, float] = {
    "easy":   0.20,   # single-table filters, simple aggregations — kept for variety
    "medium": 0.40,   # 1-2 joins, GROUP BY, HAVING — the bread and butter
    "hard":   0.40,   # window functions, CTEs, subqueries — where the model needs to shine
}

# Sent to the LLM so it calibrates what each difficulty level means
COMPLEXITY_GUIDANCE: dict[str, str] = {
    "easy": (
        "Single table only. Simple filter, aggregation, or sort. "
        "Examples: WHERE clause, COUNT/SUM/AVG, ORDER BY. No JOINs."
    ),
    "medium": (
        "1 or 2 table JOINs with aggregation, GROUP BY, HAVING, or CASE WHEN. "
        "No subqueries or window functions."
    ),
    "hard": (
        "At least one of: window functions (ROW_NUMBER, RANK, LAG, LEAD, SUM OVER), "
        "CTEs (WITH clause), correlated subqueries, or self-joins. "
        "Examples: top-N per group, running totals, month-over-month delta, rank within group."
    ),
}

# SCHEMA LOADING

def load_schemas(filter_names: list[str] | None = None) -> dict[str, dict]:
    """Load all schema JSONs, optionally limited to the names in filter_names."""
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


# DUCKDB SANDBOX
#
# WHY DUPLICATED FROM run_baseline_eval.py:
#   Both files live in scripts/ which is not a Python package. Importing
#   across scripts would require path manipulation hacks. Day 4 will move
#   shared DuckDB utilities into src/data/db_utils.py once we set up the
#   proper package structure.

def _infer_value(col_def: str, row_idx: int) -> str:
    """
    Auto-generate a SQL literal for a column based on its type definition.
    Produces sensible domain-aware values so JOINs connect and aggregations
    return non-trivial results.
    """
    col_lower = col_def.lower()
    col_name  = col_def.split()[0].lower()

    if "bool" in col_lower:
        return "TRUE" if row_idx % 2 == 0 else "FALSE"
    if "timestamp" in col_lower:
        return f"'2024-0{(row_idx % 9)+1}-{(row_idx % 28)+1:02d} 10:{row_idx*7 % 60:02d}:00'"
    if "date" in col_lower:
        return f"'2024-0{(row_idx % 9)+1}-{(row_idx % 28)+1:02d}'"
    # FK and ID columns: keep in range 1-3 so JOIN conditions actually match
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
    """
    Spin up an in-memory DuckDB with all schema tables populated with 5 rows each.
    Same setup as the eval runner — 5 rows is enough for GROUP BY and window
    functions to produce non-trivial results without slowing down validation.
    """
    conn = duckdb.connect(":memory:")
    for table_name, table_info in schema["tables"].items():
        # Strip PK/FK annotation markers — they're schema documentation, not valid SQL
        col_defs = []
        for col in table_info["columns"]:
            clean = col.split(" FK(")[0].replace(" PK", "").strip()
            col_defs.append(f"  {clean}")

        create_sql = f"CREATE TABLE {table_name} (\n" + ",\n".join(col_defs) + "\n);"
        try:
            conn.execute(create_sql)
        except Exception as e:
            log.warning(f"    CREATE TABLE {table_name} failed: {e}")

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
                pass  # type mismatch on some rows is fine — partial data is enough

    return conn


def validate_sql(conn: duckdb.DuckDBPyConnection, sql: str) -> tuple[bool, str]:
    """
    Execute the SQL against the sandbox, return (valid, error_message).

    WHY EXECUTE NOT JUST PARSE:
    DuckDB's parser accepts queries that reference non-existent columns — those
    only fail at execution time. We need execution-level validation to catch
    hallucinated column names, which is the most common LLM mistake.
    """
    try:
        conn.execute(sql)
        return True, ""
    except Exception as e:
        return False, str(e)


# GROQ BATCH GENERATION

# The system prompt tells the LLM what format we need and what rules to follow.
# Separate from src/data/prompt_template.py because this is a generation prompt
# (asking the model to invent SQL), not an inference prompt (asking it to answer).
GENERATION_SYSTEM_PROMPT = """\
You are an expert SQL engineer and training data curator.
You generate diverse, realistic question-SQL pairs for fine-tuning a Text-to-SQL model.

Rules for questions:
- Write natural English a business analyst would actually ask — no technical jargon
- Vary phrasing: "How many...", "List all...", "Find the...", "What is...", "Which...", "Show me..."
- Reference real column and table names from the schema in the question
- Every question in the batch must be unique — no two should ask the same thing

Rules for SQL:
- Output ONLY valid SQL compatible with DuckDB (PostgreSQL dialect)
- Always use explicit JOIN syntax — never comma joins
- Use table aliases on any multi-table query
- Prefer CTEs (WITH clause) over deeply nested subqueries
- Never use SELECT * — always name the columns you need
- All column and table names must EXACTLY match the schema — no hallucination

Output format — respond with ONLY a valid JSON array, nothing else:
[
  {"question": "...", "sql": "...", "complexity": "easy|medium|hard"},
  ...
]\
"""


def build_batch_prompt(schema: dict, complexity: str, n: int) -> str:
    """
    Build the user-turn message for a single batch generation request.

    WHY INCLUDE create_sql NOT JUST TABLE NAMES:
    The model needs exact column names, data types, and FK relationships to
    write correct SQL. The CREATE TABLE statements are the most unambiguous
    representation — column lists alone lead to type errors and wrong JOINs.
    """
    guidance = COMPLEXITY_GUIDANCE[complexity]
    return (
        f"Database: {schema['name']} ({schema.get('domain', '')} domain)\n"
        f"Description: {schema.get('description', '')}\n\n"
        f"Schema:\n{schema['create_sql'].strip()}\n\n"
        f"Generate exactly {n} UNIQUE {complexity.upper()} complexity question-SQL pairs.\n"
        f"Complexity requirement: {guidance}\n\n"
        f"Respond ONLY with a JSON array of {n} objects. No explanation, no markdown."
    )


# Tokens needed scales with complexity: CTEs and window functions are much longer than simple SELECTs
MAX_TOKENS_BY_COMPLEXITY: dict[str, int] = {
    "easy":   1024,   # simple SELECT/WHERE — short output
    "medium": 2048,   # joins + GROUP BY — moderate length
    "hard":   4096,   # CTEs + window functions can easily hit 300+ tokens per query
}


def call_groq_batch(
    client: OpenAI,
    schema: dict,
    complexity: str,
    n: int,
    retries: int = 3,
) -> list[dict]:
    """
    Request n question-SQL pairs from Groq, return parsed and filtered list.

    JSON parse failures and rate-limit 429s are retried with backoff.
    Returns however many valid items parsed successfully — may be < n.
    """
    prompt = build_batch_prompt(schema, complexity, n)
    max_tokens = MAX_TOKENS_BY_COMPLEXITY[complexity]

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.7,    # diversity: we want varied phrasing, not deterministic output
                max_tokens=max_tokens,   # complexity-aware: hard queries with CTEs need 4096
                messages=[
                    {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown fences — model wraps JSON in ```json despite instructions
            if raw.startswith("```"):
                parts = raw.split("```", 2)
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.rstrip("`").strip()

            pairs = json.loads(raw)
            if not isinstance(pairs, list):
                log.warning(f"    Expected JSON array, got {type(pairs).__name__}. Retrying.")
                continue

            # Accept only well-formed items; overwrite complexity to match what we asked for
            valid_pairs = []
            for p in pairs:
                if isinstance(p, dict) and p.get("question") and p.get("sql"):
                    p["complexity"] = complexity  # enforce requested complexity, don't trust model's label
                    valid_pairs.append(p)

            return valid_pairs

        except json.JSONDecodeError as e:
            wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
            log.warning(f"    JSON parse error (attempt {attempt+1}/{retries}): {e}. Waiting {wait}s.")
            time.sleep(wait)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                # Hit Groq's per-minute limit — back off for a full minute
                log.warning(f"    Rate limit (429). Waiting 60s before retry.")
                time.sleep(60)
            elif "503" in err_str or "overloaded" in err_str.lower():
                wait = 10 * (attempt + 1)
                log.warning(f"    Groq overloaded. Waiting {wait}s.")
                time.sleep(wait)
            else:
                log.error(f"    API error: {err_str}")
                break

    return []


# DRY RUN

def generate_dry_run_pairs(schema: dict, complexity: str, n: int) -> list[dict]:
    """
    Return trivially correct pairs for --dry-run mode.
    WHY: Lets us test validation + file-writing end-to-end without using API quota.
    SQL is a simple COUNT so it definitely passes DuckDB validation.
    """
    table_name = list(schema["tables"].keys())[0]
    first_col  = schema["tables"][table_name]["columns"][0].split()[0]
    return [
        {
            "question": f"[dry-run {i}] How many rows are in {table_name}?",
            "sql":      f"SELECT COUNT({first_col}) AS total FROM {table_name}",
            "complexity": complexity,
        }
        for i in range(n)
    ]


# W&B (optional)

def init_wandb(n_schemas: int, target: int) -> Any | None:
    """
    Start a W&B run if WANDB_API_KEY is available, else return None.
    W&B is strictly optional — the script works identically without it.
    """
    if not WANDB_API_KEY:
        log.info("WANDB_API_KEY not set — skipping W&B logging.")
        return None
    try:
        import wandb  # type: ignore
        run = wandb.init(
            project="text-to-sql",
            name="day3-synthetic-generation",
            config={
                "model":             GROQ_MODEL,
                "target_per_schema": target,
                "batch_size":        BATCH_SIZE,
                "complexity_mix":    COMPLEXITY_MIX,
                "n_schemas":         n_schemas,
            },
        )
        log.info(f"W&B run started: {run.url}")
        return run
    except Exception as e:
        log.warning(f"W&B init failed ({e}) — continuing without it.")
        return None


def log_wandb_schema(run: Any | None, schema_name: str, stats: dict) -> None:
    """Log per-schema stats to W&B. Silently no-ops if no run is active."""
    if run is None:
        return
    try:
        run.log({"schema": schema_name, **stats})
    except Exception:
        pass  # W&B logging is best-effort, never let it abort the main job


# RESUME SUPPORT

def load_completed_schemas(target: int) -> set[str]:
    """
    Read the output file and return schema names that already have >= target pairs.
    Safe to call even if the output file doesn't exist yet.
    """
    if not OUT_FILE.exists():
        return set()

    counts: dict[str, int] = {}
    with open(OUT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                name = ex.get("schema_name", "")
                counts[name] = counts.get(name, 0) + 1
            except json.JSONDecodeError:
                continue

    return {name for name, count in counts.items() if count >= target}


# CORE GENERATION LOOP (per schema)

def generate_for_schema(
    schema: dict,
    client: OpenAI | None,
    target: int,
    dry_run: bool,
    rate_sleep: float = 15.0,
) -> list[dict]:
    """
    Generate `target` DuckDB-validated question-SQL pairs for one schema.

    Per-complexity targets are computed from COMPLEXITY_MIX. For each
    complexity bucket, we issue batches until we hit the target or exhaust
    a patience limit (2x the pairs needed). Validated pairs are appended
    to the collected list; invalid SQL is discarded and logged at DEBUG level.

    Args:
        schema:     Schema dict (loaded from JSON).
        client:     Groq OpenAI client. None in dry-run mode.
        target:     How many valid pairs to collect total.
        dry_run:    If True, skip API calls and use trivial SQL.
        rate_sleep: Seconds to sleep between API calls.
                    WHY 15s NOT 2.5s: Groq free tier has TWO limits — 30 RPM
                    and 6000 tokens/min (TPM). Each call generates ~1500 tokens
                    of output. At 2.5s sleep we make ~24 calls/min = 36000 TPM,
                    blowing the TPM cap after 3-4 calls. At 15s we make 4
                    calls/min = ~6000 TPM, staying just within the limit.

    Returns:
        List of validated example dicts ready for JSONL output.
    """
    conn = build_sandbox(schema)

    # Compute how many pairs of each complexity to collect
    complexity_targets: dict[str, int] = {
        c: int(target * frac) for c, frac in COMPLEXITY_MIX.items()
    }
    # Fix rounding gap: any unallocated pairs go to 'medium'
    total_allocated = sum(complexity_targets.values())
    complexity_targets["medium"] += target - total_allocated

    collected: list[dict] = []
    generated: dict[str, int] = {c: 0 for c in COMPLEXITY_MIX}
    discarded:  dict[str, int] = {c: 0 for c in COMPLEXITY_MIX}
    api_calls = 0

    for complexity, c_target in complexity_targets.items():
        log.info(f"  [{schema['name']}] {complexity}: targeting {c_target} pairs")

        # patience = max individual SQL pairs we'll attempt for this bucket before giving up
        patience  = c_target * 2
        attempted = 0

        while generated[complexity] < c_target and attempted < patience:
            # Request a few extra pairs to absorb expected validation failures
            remaining = c_target - generated[complexity]
            batch_n   = min(BATCH_SIZE, remaining + max(4, remaining // 4))

            if dry_run:
                pairs = generate_dry_run_pairs(schema, complexity, batch_n)
            else:
                pairs = call_groq_batch(client, schema, complexity, batch_n)
                api_calls += 1
                time.sleep(rate_sleep)  # TPM budget: 15s = ~4 calls/min × ~1500 tok/call ≈ 6000 TPM

            for pair in pairs:
                attempted += 1
                if generated[complexity] >= c_target:
                    break  # already hit target — don't over-collect this bucket

                sql = pair.get("sql", "").strip()
                if not sql:
                    continue

                ok, err = validate_sql(conn, sql)
                if ok:
                    collected.append({
                        "schema_name": schema["name"],
                        "question":    pair["question"].strip(),
                        "sql":         sql,
                        "complexity":  complexity,
                        "source":      "synthetic",
                    })
                    generated[complexity] += 1
                else:
                    discarded[complexity] += 1
                    # Debug level so normal runs aren't noisy; use -v to see individual failures
                    log.debug(f"    Discarded ({complexity}): {err[:100]}")

    conn.close()

    total_valid     = sum(generated.values())
    total_discarded = sum(discarded.values())
    pass_rate = total_valid / max(total_valid + total_discarded, 1) * 100
    log.info(
        f"  [{schema['name']}] "
        f"valid={total_valid}  discarded={total_discarded}  "
        f"pass_rate={pass_rate:.0f}%  api_calls={api_calls}"
    )

    return collected


# ENTRYPOINT

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic question-SQL pairs for QLoRA fine-tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--schemas", nargs="+", default=None,
        help="Only generate for these schema names. Default: all 10.",
    )
    parser.add_argument(
        "--limit", type=int, default=TARGET_PER_SCHEMA,
        help=f"Target valid pairs per schema. Default: {TARGET_PER_SCHEMA}.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip API calls — validate and write pipeline only.",
    )
    parser.add_argument(
        "--no-resume", action="store_true", default=False,
        help="Regenerate all schemas, even if already in the output file.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show individual SQL validation failures.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    schemas = load_schemas(args.schemas)
    if not schemas:
        raise SystemExit("No schemas found. Check data/raw/schemas/.")

    # Build Groq client (skip in dry-run)
    client: OpenAI | None = None
    if not args.dry_run:
        if not GROQ_API_KEY:
            raise SystemExit(
                "GROQ_API_KEY not set.\n"
                "Sign up free at console.groq.com → API keys → add to .env as GROQ_API_KEY=gsk_..."
            )
        client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
        est_calls = len(schemas) * max(10, (args.limit // BATCH_SIZE) * 3 * len(COMPLEXITY_MIX))
        log.info(f"Groq client ready. Model: {GROQ_MODEL}")
        log.info(
            f"Estimated API calls: ~{est_calls} "
            f"(free tier: 1000/day, 30/min)"
        )

    # Resume: skip schemas that already have enough pairs in the output file
    completed: set[str] = set()
    if not args.no_resume:
        completed = load_completed_schemas(args.limit)
        if completed:
            log.info(f"Resuming — {len(completed)} schemas already done: {sorted(completed)}")

    run = init_wandb(n_schemas=len(schemas), target=args.limit)

    total_written = 0

    # Append mode so partial runs accumulate rather than overwrite
    with open(OUT_FILE, "a") as out_f:
        for schema_name, schema in tqdm(schemas.items(), desc="Schemas", unit="schema"):
            if schema_name in completed:
                log.info(f"  Skipping {schema_name} (already done)")
                continue

            examples = generate_for_schema(
                schema=schema,
                client=client,
                target=args.limit,
                dry_run=args.dry_run,
            )

            for ex in examples:
                out_f.write(json.dumps(ex) + "\n")
            out_f.flush()  # flush after each schema — don't lose work if the next call crashes

            total_written += len(examples)
            log.info(f"  Wrote {len(examples)} pairs for {schema_name} → {OUT_FILE.name}")

            log_wandb_schema(run, schema_name, {
                "pairs_written": len(examples),
                "total_written": total_written,
            })

    if run:
        run.finish()

    log.info(f"\nDone. Total new pairs written this run: {total_written}")
    log.info(f"Output file: {OUT_FILE}")

    if OUT_FILE.exists():
        counts: dict[str, int] = {}
        total_lines = 0
        with open(OUT_FILE) as f:
            for line in f:
                if line.strip():
                    total_lines += 1
                    try:
                        ex = json.loads(line)
                        c = ex.get("complexity", "?")
                        counts[c] = counts.get(c, 0) + 1
                    except json.JSONDecodeError:
                        pass
        log.info(f"\nFull file stats ({total_lines} total pairs):")
        for c in ["easy", "medium", "hard"]:
            n = counts.get(c, 0)
            pct = n / total_lines * 100 if total_lines else 0
            log.info(f"  {c:<8}: {n:>5}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
