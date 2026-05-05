# =============================================================================
# DAY 5 — Qwen2.5-7B-Instruct Baseline Eval (no fine-tuning)
# Run this on Kaggle with T4 GPU enabled.
#
# SETUP (do once before running):
#   1. Go to kaggle.com → Datasets → New Dataset
#   2. Upload these files from your local project:
#        data/eval/golden_eval.jsonl
#        data/raw/schemas/*.json   (all 10 schema files)
#   3. Name the dataset: text-to-sql-data
#   4. In this notebook → Add Data → search "text-to-sql-data" → attach it
#   5. Enable GPU: Settings → Accelerator → GPU T4 x2
#
# WHAT THIS MEASURES:
#   Execution accuracy of Qwen2.5-7B-Instruct on your 68 golden eval questions
#   with NO fine-tuning. This is your "before" number.
#   After QLoRA training (Day 8-9) you run the same eval to get the "after" number.
#   The delta is your headline result for the resume and README.
#
# EXPECTED RESULTS (rough):
#   Easy:   ~80-90%   (model already knows basic SQL)
#   Medium: ~55-70%   (joins + aggregations — hit or miss on your schemas)
#   Hard:   ~30-50%   (CTEs + window functions on novel schemas — likely weak)
#   This is what you need to beat after fine-tuning.
# =============================================================================


# =============================================================================
# CELL 1 — Install packages
# Runtime: ~2 minutes
# =============================================================================

# %%
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "bitsandbytes>=0.43.0",
    "accelerate>=0.27.0",
    "duckdb>=1.2.2",
    "transformers>=4.40.0",
], check=True)
print("Packages installed.")


# =============================================================================
# CELL 2 — Imports and paths
# =============================================================================

# %%
import json
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Kaggle datasets mount at /kaggle/input/<dataset-slug>/
DATA_DIR   = Path("/kaggle/input/text-to-sql-data")
OUTPUT_DIR = Path("/kaggle/working")

EVAL_FILE   = DATA_DIR / "golden_eval.jsonl"
SCHEMA_DIR  = DATA_DIR  # schema JSONs are in the root of the dataset

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

print(f"Eval file exists: {EVAL_FILE.exists()}")
print(f"GPU available:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print(f"VRAM:             {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# =============================================================================
# CELL 3 — Load schemas and eval set
# =============================================================================

# %%


# =============================================================================
# CELL 4 — Load Qwen2.5-7B-Instruct in 4-bit
# Runtime: ~4-6 minutes (downloads ~15 GB model weights)
# Memory:  ~4.5 GB VRAM
# =============================================================================

# %%
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 is best for LLM weights distribution
    bnb_4bit_compute_dtype=torch.float16, # compute in fp16, store in 4-bit
    bnb_4bit_use_double_quant=True,       # second quantization saves ~0.4 GB more
)

print(f"Loading {MODEL_ID} in 4-bit...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",          # auto-places layers across available GPUs
    trust_remote_code=True,
)
model.eval()  # disable dropout — deterministic inference

vram_used = torch.cuda.memory_allocated() / 1e9
print(f"Model loaded. VRAM used: {vram_used:.1f} GB")


# =============================================================================
# CELL 5 — Prompt builder and inference function
# =============================================================================

# %%
# Same system prompt used in training and eval — must match exactly
SYSTEM_PROMPT = """You are an expert SQL engineer. Given a database schema and a natural language question, write a correct and efficient SQL query.

Rules:
- Output ONLY the SQL query, no explanation, no markdown fences
- Use table aliases for readability on multi-join queries
- Use explicit JOIN syntax, never implicit comma joins
- Prefer CTEs over deeply nested subqueries when it improves readability
- Never use SELECT * — always specify column names"""


def build_messages(schema_sql: str, question: str) -> list[dict]:
    """Build the chat messages list — identical format to training data."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Schema:\n{schema_sql.strip()}\n\nQuestion: {question.strip()}\n\nSQL:"
        },
    ]


def generate_sql(schema_sql: str, question: str, max_new_tokens: int = 256) -> tuple[str, float]:
    """
    Run one inference pass, return (generated_sql, latency_seconds).

    WHY temperature=0: deterministic output — same question always gives same SQL.
    This makes the baseline reproducible. Fine-tuning eval uses the same setting
    so comparison is apples-to-apples.

    WHY max_new_tokens=256: SQL queries rarely exceed 200 tokens. Capping here
    prevents runaway generation and keeps eval fast.
    """
    messages = build_messages(schema_sql, question)

    # apply_chat_template adds Qwen's special tokens (<|im_start|>, etc.)
    # and formats the conversation into a single string the model understands
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # adds the assistant turn opener
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy decoding — deterministic, no temperature
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - t0

    # Slice off the input tokens — we only want the generated SQL
    generated_ids = outputs[0][input_len:]
    sql = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Strip markdown fences if the model added them despite the system prompt
    sql = sql.replace("```sql", "").replace("```", "").strip()

    return sql, latency


# =============================================================================
# CELL 6 — DuckDB sandbox (same as run_baseline_eval.py)
# =============================================================================

# %%
def _infer_value(col_def: str, row_idx: int) -> str:
    """Generate a SQL literal for a column based on its type."""
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
    """Spin up in-memory DuckDB with all tables and 5 rows each."""
    conn = duckdb.connect(":memory:")
    for table_name, table_info in schema["tables"].items():
        col_defs = []
        for col in table_info["columns"]:
            clean = col.split(" FK(")[0].replace(" PK", "").strip()
            col_defs.append(f"  {clean}")
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


def execute_sql(conn: duckdb.DuckDBPyConnection, sql: str) -> tuple[bool, list | None, str]:
    """Execute SQL, return (success, rows, error)."""
    try:
        result = conn.execute(sql).fetchall()
        return True, result, ""
    except Exception as e:
        return False, None, str(e)


def results_match(a: list | None, b: list | None) -> bool:
    """Compare result sets order-insensitively, with float rounding."""
    if a is None or b is None:
        return False
    try:
        def normalize(rows):
            return sorted(
                [tuple(round(float(v), 2) if isinstance(v, float) else v for v in row)
                 for row in rows],
                key=lambda x: [str(i) for i in x]
            )
        return normalize(a) == normalize(b)
    except Exception:
        return a == b


# =============================================================================
# CELL 7 — Run the eval loop
# Runtime: ~15-25 minutes for 68 questions (each inference ~10-15s on T4)
# =============================================================================

# %%
results = []
errors  = {"api": 0, "sql": 0}

print(f"Running eval on {len(examples)} examples...\n")

for i, ex in enumerate(examples, 1):
    schema = schemas.get(ex["schema_name"])
    if not schema:
        print(f"  [{i:02d}] SKIP — schema '{ex['schema_name']}' not found")
        continue

    schema_sql = schema["create_sql"]
    question   = ex["question"]
    ref_sql    = ex["sql"]

    # Generate SQL from the base model (no fine-tuning)
    try:
        gen_sql, latency = generate_sql(schema_sql, question)
    except Exception as e:
        errors["api"] += 1
        print(f"  [{i:02d}] ERROR generating SQL: {e}")
        results.append({**ex, "generated_sql": "", "valid_sql": False,
                        "exec_match": False, "latency_s": 0.0, "error": str(e)})
        continue

    # Validate both SQLs in a fresh DuckDB sandbox
    conn = build_sandbox(schema)
    valid, gen_rows, gen_err = execute_sql(conn, gen_sql)
    _,     ref_rows, _       = execute_sql(conn, ref_sql)
    conn.close()

    if not valid:
        errors["sql"] += 1

    match = results_match(gen_rows, ref_rows) if valid else False

    results.append({
        "schema_name":   ex["schema_name"],
        "question":      question,
        "complexity":    ex["complexity"],
        "reference_sql": ref_sql,
        "generated_sql": gen_sql,
        "valid_sql":     valid,
        "exec_match":    match,
        "gen_error":     gen_err,
        "latency_s":     round(latency, 2),
    })

    status = "OK " if match else ("INVALID" if not valid else "WRONG  ")
    print(f"  [{i:02d}/{len(examples)}] {status} | {ex['complexity']:<6} | "
          f"{latency:.1f}s | {question[:55]}")

print(f"\nDone. Errors -> generation: {errors['api']} | SQL exec: {errors['sql']}")


# =============================================================================
# CELL 8 — Compute and display results
# =============================================================================

# %%
def compute_and_print_results(results: list, model_name: str) -> dict:
    """Compute execution accuracy by complexity, print the results table."""
    by_c = defaultdict(list)
    for r in results:
        by_c[r["complexity"]].append(r)

    n = len(results)
    overall_acc  = sum(1 for r in results if r["exec_match"]) / n * 100
    overall_valid = sum(1 for r in results if r["valid_sql"]) / n * 100
    avg_latency   = sum(r["latency_s"] for r in results) / n

    print("\n" + "=" * 68)
    print(f"BASELINE RESULTS — {model_name}")
    print("=" * 68)
    print(f"{'Complexity':<12} {'N':>4} {'Valid SQL':>10} {'Exec Acc':>10}")
    print("-" * 68)

    summary = {"model": model_name, "total": n}

    for bucket in ["easy", "medium", "hard"]:
        group = by_c.get(bucket, [])
        if not group:
            continue
        bn      = len(group)
        valid   = sum(1 for r in group if r["valid_sql"]) / bn * 100
        acc     = sum(1 for r in group if r["exec_match"]) / bn * 100
        print(f"  {bucket:<10} {bn:>4} {valid:>9.1f}% {acc:>9.1f}%")
        summary[f"valid_sql_{bucket}"]     = round(valid, 1)
        summary[f"exec_accuracy_{bucket}"] = round(acc, 1)

    print("-" * 68)
    print(f"  {'OVERALL':<10} {n:>4} {overall_valid:>9.1f}% {overall_acc:>9.1f}%")
    print("=" * 68)
    print(f"\nAvg latency per query: {avg_latency:.2f}s")
    print("\nThis is your BEFORE number.")
    print("Your fine-tuned Qwen2.5-7B must beat the Hard column to have a story.")

    summary["valid_sql_overall"]     = round(overall_valid, 1)
    summary["exec_accuracy_overall"] = round(overall_acc, 1)
    summary["avg_latency_s"]         = round(avg_latency, 2)
    return summary


summary = compute_and_print_results(results, MODEL_ID)


# =============================================================================
# CELL 9 — Save results (Kaggle saves /kaggle/working/ automatically)
# =============================================================================

# %%
import datetime
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Per-question results — useful for error analysis
results_file = OUTPUT_DIR / f"baseline_qwen25_7b_{ts}.jsonl"
with open(results_file, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

# Summary JSON
summary_file = OUTPUT_DIR / f"baseline_summary_{ts}.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Results saved:")
print(f"  Per-question: {results_file.name}")
print(f"  Summary:      {summary_file.name}")
print("\nDownload these from Kaggle Output tab and save to:")
print("  data/eval/results/  (in your local project)")


# =============================================================================
# CELL 10 — Show failures (optional but useful)
# See which question types the base model gets wrong so you know what to target
# =============================================================================

# %%
failures = [r for r in results if not r["exec_match"]]
print(f"\nFailed questions ({len(failures)}/{len(results)}):\n")
for r in failures[:10]:   # show first 10
    print(f"[{r['complexity']}] {r['question']}")
    print(f"  Reference: {r['reference_sql'][:80]}...")
    print(f"  Generated: {r['generated_sql'][:80]}...")
    if r["gen_error"]:
        print(f"  Error:     {r['gen_error'][:80]}")
    print()
