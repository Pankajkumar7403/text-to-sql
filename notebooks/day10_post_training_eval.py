# =============================================================================
# DAY 10 — Post-Training Eval (QLoRA fine-tuned Qwen2.5-7B)
# Works on both Kaggle (T4 GPU) and locally (CPU/GPU).
#
# WHAT THIS MEASURES:
#   Execution accuracy of the fine-tuned model on the 68 golden eval questions.
#   Baseline (pre-training):  Easy 81.0% | Medium 34.6% | Hard 33.3% | Overall 48.5%
#   v1 fine-tuned (Day 6):    Easy 85.7% | Medium 38.5% | Hard 33.3% | Overall 51.5%
#   The delta vs v1 is the headline portfolio result for Day 12.
#
# KAGGLE SETUP:
#   1. Upload your adapter folder (data/models/qwen25_sql_v2/) as a new Kaggle
#      dataset named "qwen25-sql-v2-adapter".
#   2. Attach both "text-to-sql-data" and "qwen25-sql-v2-adapter" to this notebook.
#   3. Enable T4 GPU, run all cells.
#
# LOCAL SETUP:
#   pip install transformers peft accelerate duckdb torch
#   Adapter must be at: data/models/qwen25_sql_v2/
# =============================================================================


# =============================================================================
# CELL 1 — Install packages (Kaggle only)
# =============================================================================

# %%
import os
import subprocess

ON_KAGGLE = os.path.exists("/kaggle/working")

if ON_KAGGLE:
    subprocess.run(["pip", "install", "--quiet", "unsloth"], check=True)
    subprocess.run(["pip", "install", "--quiet", "--upgrade", "--no-cache-dir", "--no-deps",
                    "git+https://github.com/unslothai/unsloth.git"], check=True)
    subprocess.run(["pip", "install", "--quiet",
                    "duckdb>=1.2.2", "accelerate>=0.34.0"], check=True)
    print("Packages installed.")
else:
    print("Local run — skipping install.")


# =============================================================================
# CELL 2 — Imports and paths
# =============================================================================

# %%
import datetime
import json
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import torch
from unsloth import FastLanguageModel

if ON_KAGGLE:
    OUTPUT_DIR  = Path("/kaggle/working")

    # Adapter: prefer same-session /kaggle/working first, then uploaded dataset
    _adapter_working = Path("/kaggle/working/qwen25_sql_v2_adapter")
    _adapter_input   = Path("/kaggle/input/qwen25-sql-v2-adapter")
    ADAPTER_DIR = _adapter_working if _adapter_working.exists() else _adapter_input

    # Eval data: Kaggle datasets may nest files under a version subfolder
    # Walk /kaggle/input/text-to-sql-data to find golden_eval.jsonl wherever it lands
    _data_root = Path("/kaggle/input/text-to-sql-data")
    _eval_candidates = list(_data_root.rglob("golden_eval.jsonl")) if _data_root.exists() else []
    EVAL_FILE  = _eval_candidates[0] if _eval_candidates else _data_root / "golden_eval.jsonl"
    SCHEMA_DIR = EVAL_FILE.parent  # schemas live next to golden_eval.jsonl
else:
    ROOT        = Path(__file__).parent.parent
    EVAL_FILE   = ROOT / "data" / "eval" / "golden_eval.jsonl"
    SCHEMA_DIR  = ROOT / "data" / "raw" / "schemas"
    OUTPUT_DIR  = ROOT / "data" / "eval" / "results"
    ADAPTER_DIR = ROOT / "data" / "models" / "qwen25_sql_v2"

MODEL_ID    = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LEN = 2048

print(f"Running on:      {'Kaggle' if ON_KAGGLE else 'Local'}")
print(f"Eval file:       {EVAL_FILE.exists()}  ({EVAL_FILE})")
print(f"Schema dir:      {SCHEMA_DIR.exists()}  ({SCHEMA_DIR})")
print(f"Adapter dir:     {ADAPTER_DIR.exists()}  ({ADAPTER_DIR})")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"VRAM:            {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# =============================================================================
# CELL 3 — Load base model + adapter (unsloth — 4-bit, fast inference)
# Runtime: ~5-7 minutes
# =============================================================================

# %%
print(f"Loading model + LoRA adapter from {ADAPTER_DIR}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(ADAPTER_DIR),   # unsloth reads adapter_config.json and loads base model automatically
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)
model.generation_config.max_length = None

print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# =============================================================================
# CELL 4 — Load schemas and eval set
# =============================================================================

# %%
def load_schemas(schema_dir: Path) -> dict[str, dict]:
    schemas: dict[str, dict] = {}
    for f in schema_dir.glob("*.json"):
        if f.name == "index.json":
            continue
        with open(f) as fh:
            try:
                s = json.load(fh)
                # Only accept schema files — must have name + create_sql + tables
                if "name" in s and "create_sql" in s and "tables" in s:
                    schemas[s["name"]] = s
            except Exception:
                pass
    return schemas


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


schemas  = load_schemas(SCHEMA_DIR)
examples = load_jsonl(EVAL_FILE)

print(f"Schemas loaded:  {len(schemas)} -> {sorted(schemas.keys())}")
print(f"Eval examples:   {len(examples)}")

complexity_counts = defaultdict(int)
for ex in examples:
    complexity_counts[ex["complexity"]] += 1
print(f"By complexity:   {dict(complexity_counts)}")


# =============================================================================
# CELL 5 — Inference function
# =============================================================================

# %%
SYSTEM_PROMPT = """You are an expert SQL engineer. Given a database schema and a natural language question, write a correct and efficient SQL query.

Rules:
- Output ONLY the SQL query, no explanation, no markdown fences
- Use table aliases for readability on multi-join queries
- Use explicit JOIN syntax, never implicit comma joins
- Prefer CTEs over deeply nested subqueries when it improves readability
- Never use SELECT * -- always specify column names"""


def generate_sql(schema_sql: str, question: str, max_new_tokens: int = 256) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Schema:\n{schema_sql.strip()}\n\nQuestion: {question.strip()}\n\nSQL:"},
    ]
    text      = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs    = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - t0

    sql = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql, latency


# =============================================================================
# CELL 6 — DuckDB sandbox
# =============================================================================

# %%
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


def execute_sql(conn: duckdb.DuckDBPyConnection, sql: str) -> tuple[bool, list | None, str]:
    try:
        return True, conn.execute(sql).fetchall(), ""
    except Exception as e:
        return False, None, str(e)


def results_match(a: list | None, b: list | None) -> bool:
    if a is None or b is None:
        return False
    try:
        def normalize(rows):
            return sorted(
                [tuple(round(float(v), 2) if isinstance(v, float) else v for v in row)
                 for row in rows],
                key=lambda x: [str(i) for i in x],
            )
        return normalize(a) == normalize(b)
    except Exception:
        return a == b


# =============================================================================
# CELL 7 — Run eval loop
# GPU: ~15-25 min. CPU: several hours (consider running overnight).
# =============================================================================

# %%
results = []
errors  = {"gen": 0, "sql": 0}

print(f"Running eval on {len(examples)} examples...\n")

for i, ex in enumerate(examples, 1):
    schema = schemas.get(ex["schema_name"])
    if not schema:
        print(f"  [{i:02d}] SKIP — schema '{ex['schema_name']}' not found")
        continue

    try:
        gen_sql, latency = generate_sql(schema["create_sql"], ex["question"])
    except Exception as e:
        errors["gen"] += 1
        print(f"  [{i:02d}] ERROR: {e}")
        results.append({**ex, "generated_sql": "", "valid_sql": False,
                        "exec_match": False, "latency_s": 0.0, "gen_error": str(e)})
        continue

    conn = build_sandbox(schema)
    valid, gen_rows, gen_err = execute_sql(conn, gen_sql)
    _,     ref_rows, _       = execute_sql(conn, ex["sql"])
    conn.close()

    if not valid:
        errors["sql"] += 1

    match = results_match(gen_rows, ref_rows) if valid else False

    results.append({
        "schema_name":   ex["schema_name"],
        "question":      ex["question"],
        "complexity":    ex["complexity"],
        "reference_sql": ex["sql"],
        "generated_sql": gen_sql,
        "valid_sql":     valid,
        "exec_match":    match,
        "gen_error":     gen_err,
        "latency_s":     round(latency, 2),
    })

    status = "OK " if match else ("INVALID" if not valid else "WRONG  ")
    print(f"  [{i:02d}/{len(examples)}] {status} | {ex['complexity']:<6} | "
          f"{latency:.1f}s | {ex['question'][:55]}")

print(f"\nDone. Errors -> generation: {errors['gen']} | invalid SQL: {errors['sql']}")


# =============================================================================
# CELL 8 — Compute and display results
# =============================================================================

# %%
# v1 fine-tuned (Day 6 smoke test) — delta shows v1 → v2 improvement.
# Pre-training Qwen2.5-7B baseline: easy 81.0 | medium 34.6 | hard 33.3 | overall 48.5
BASELINE = {
    "easy":    85.7,   # v1: 18/21
    "medium":  38.5,   # v1: 10/26
    "hard":    33.3,   # v1: 7/21
    "overall": 51.5,   # v1: 35/68
}


def compute_and_print_results(results: list, model_tag: str) -> dict:
    by_c = defaultdict(list)
    for r in results:
        by_c[r["complexity"]].append(r)

    n             = len(results)
    overall_acc   = sum(1 for r in results if r["exec_match"]) / n * 100
    overall_valid = sum(1 for r in results if r["valid_sql"]) / n * 100
    avg_latency   = sum(r["latency_s"] for r in results) / n

    print("\n" + "=" * 72)
    print(f"POST-TRAINING RESULTS — {model_tag}")
    print("=" * 72)
    print(f"{'Complexity':<12} {'N':>4} {'Valid SQL':>10} {'Exec Acc':>10} {'vs v1':>10}")
    print("-" * 72)

    summary = {"model": model_tag, "total": n}

    for bucket in ["easy", "medium", "hard"]:
        group = by_c.get(bucket, [])
        if not group:
            continue
        bn    = len(group)
        valid = sum(1 for r in group if r["valid_sql"]) / bn * 100
        acc   = sum(1 for r in group if r["exec_match"]) / bn * 100
        delta = acc - BASELINE.get(bucket, 0)
        delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
        print(f"  {bucket:<10} {bn:>4} {valid:>9.1f}% {acc:>9.1f}% {delta_str:>10}")
        summary[f"valid_sql_{bucket}"]     = round(valid, 1)
        summary[f"exec_accuracy_{bucket}"] = round(acc, 1)
        summary[f"delta_vs_v1_{bucket}"]   = round(delta, 1)

    delta_overall = overall_acc - BASELINE["overall"]
    delta_str = f"+{delta_overall:.1f}%" if delta_overall >= 0 else f"{delta_overall:.1f}%"
    print("-" * 72)
    print(f"  {'OVERALL':<10} {n:>4} {overall_valid:>9.1f}% {overall_acc:>9.1f}% {delta_str:>10}")
    print("=" * 72)
    print(f"\nAvg latency: {avg_latency:.2f}s/query")

    if overall_acc >= 84:
        print("OVERALL TARGET MET: >=84%")
    else:
        print(f"Overall still needs {84 - overall_acc:.1f}pp to hit 84% target.")
    hard_acc = summary.get("exec_accuracy_hard", 0)
    if hard_acc >= 71:
        print("HARD TARGET MET: >=71%")
    else:
        print(f"Hard still needs {71 - hard_acc:.1f}pp to hit 71% target.")

    summary.update({
        "valid_sql_overall":     round(overall_valid, 1),
        "exec_accuracy_overall": round(overall_acc, 1),
        "delta_vs_v1_overall":   round(delta_overall, 1),
        "avg_latency_s":         round(avg_latency, 2),
    })
    return summary


summary = compute_and_print_results(results, "Qwen2.5-7B-QLoRA-v2")


# =============================================================================
# CELL 9 — Save results
# =============================================================================

# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

results_file = OUTPUT_DIR / f"finetuned_v2_qwen25_7b_{ts}.jsonl"
summary_file = OUTPUT_DIR / f"finetuned_v2_summary_{ts}.json"

with open(results_file, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Results -> {results_file}")
print(f"Summary -> {summary_file}")


# =============================================================================
# CELL 10 — Show top failures
# =============================================================================

# %%
failures = [r for r in results if not r["exec_match"]]
print(f"\nFailed: {len(failures)}/{len(results)}\n")
for r in failures[:10]:
    print(f"[{r['complexity']}] {r['question']}")
    print(f"  Ref: {r['reference_sql'][:80]}")
    print(f"  Gen: {r['generated_sql'][:80]}")
    if r["gen_error"]:
        print(f"  Err: {r['gen_error'][:80]}")
    print()
