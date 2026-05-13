# Graph Report - text-to-sql  (2026-05-13)

## Corpus Check
- 31 files · ~28,858 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 294 nodes · 318 edges · 90 communities detected
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 23 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]
- [[_COMMUNITY_Community 86|Community 86]]
- [[_COMMUNITY_Community 87|Community 87]]
- [[_COMMUNITY_Community 88|Community 88]]
- [[_COMMUNITY_Community 89|Community 89]]
- [[_COMMUNITY_Community 90|Community 90]]
- [[_COMMUNITY_Community 91|Community 91]]
- [[_COMMUNITY_Community 92|Community 92]]
- [[_COMMUNITY_Community 93|Community 93]]

## God Nodes (most connected - your core abstractions)
1. `InferenceUserError` - 16 edges
2. `main()` - 10 edges
3. `generate()` - 9 edges
4. `classify_complexity()` - 8 edges
5. `generate_for_schema()` - 7 edges
6. `generate_for_schema()` - 7 edges
7. `run_eval()` - 7 edges
8. `main()` - 7 edges
9. `SchemaInfo` - 6 edges
10. `_map_groq_exception()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `GenerateRequest` --uses--> `InferenceUserError`  [INFERRED]
  hf-space-deploy\app\api.py → hf-space-deploy\app\errors.py
- `GenerateResponse` --uses--> `InferenceUserError`  [INFERRED]
  hf-space-deploy\app\api.py → hf-space-deploy\app\errors.py
- `SchemaInfo` --uses--> `InferenceUserError`  [INFERRED]
  hf-space-deploy\app\api.py → hf-space-deploy\app\errors.py
- `generate()` --calls--> `generate_sql()`  [INFERRED]
  hf-space-deploy\app\api.py → notebooks\day5_baseline_eval.py
- `generate()` --calls--> `generate_sql()`  [INFERRED]
  hf-space-deploy\app\api.py → notebooks\day10_post_training_eval.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.11
Nodes (25): build_groq_client(), build_sandbox(), call_groq(), compute_summary(), execute_sql(), _infer_value(), load_eval_set(), load_schemas() (+17 more)

### Community 1 - "Community 1"
Cohesion: 0.12
Nodes (24): build_batch_prompt(), build_sandbox(), call_groq_batch(), generate_dry_run_pairs(), generate_for_schema(), _infer_value(), init_wandb(), load_completed_schemas() (+16 more)

### Community 2 - "Community 2"
Cohesion: 0.19
Nodes (15): generate(), GenerateRequest, GenerateResponse, get_schema(), health(), list_schemas(), FastAPI inference endpoint.  Endpoints:   GET  /health          -> { status: "ok, SchemaInfo (+7 more)

### Community 3 - "Community 3"
Cohesion: 0.14
Nodes (20): complexity_counts(), deduplicate(), format_example(), inspect_examples(), load_jsonl(), load_schemas(), main(), print_summary() (+12 more)

### Community 4 - "Community 4"
Cohesion: 0.12
Nodes (17): build_messages(), build_sandbox(), compute_and_print_results(), execute_sql(), generate_sql(), _infer_value(), Build the chat messages list — identical format to training data., Run one inference pass, return (generated_sql, latency_seconds).      WHY temper (+9 more)

### Community 5 - "Community 5"
Cohesion: 0.22
Nodes (12): InferenceUserError, User-facing inference errors (safe for UI and API responses)., Raised when SQL generation fails; message is safe to show clients., Exception, generate_sql(), _get_client(), load_model(), _map_groq_exception() (+4 more)

### Community 6 - "Community 6"
Cohesion: 0.2
Nodes (15): build_sandbox(), build_targeted_prompt(), call_groq_batch(), generate_for_schema(), _infer_value(), load_existing_questions(), load_failure_analysis(), load_schemas() (+7 more)

### Community 7 - "Community 7"
Cohesion: 0.21
Nodes (15): build_prompt(), build_sandbox(), call_groq(), count_existing_per_schema(), generate_for_schema(), _infer_value(), load_existing_questions(), load_passing_examples() (+7 more)

### Community 8 - "Community 8"
Cohesion: 0.16
Nodes (6): build_sandbox(), generate_sql(), _infer_value(), apply_chat_template(), Convert messages list to a single string for SFTTrainer.     add_generation_prom, smoke_infer()

### Community 9 - "Community 9"
Cohesion: 0.31
Nodes (11): classify_complexity(), count_joins(), count_subqueries(), has_aggregation(), has_nested_conditions(), is_valid_sql(), process_bird(), process_spider() (+3 more)

### Community 10 - "Community 10"
Cohesion: 0.43
Nodes (6): detect_patterns(), find_latest_results(), load_jsonl(), main(), Day 11 — Analyze fine-tuned model failures to identify SQL patterns to target., Return list of SQL pattern names present in the query.

### Community 11 - "Community 11"
Cohesion: 0.4
Nodes (5): format_prompt(), format_training_example(), Prompt formatting utilities. Single source of truth — used by training, inferen, Returns messages list in Qwen2.5 chat format.     Used identically during train, Full training example with assistant turn included.     Output is a dict with '

### Community 12 - "Community 12"
Cohesion: 0.5
Nodes (4): Day 1 Script: Generate 10 synthetic schemas (5 e-commerce, 5 fintech). These be, Generate CREATE TABLE statements for a schema (used in prompts)., save_schemas(), schema_to_create_sql()

### Community 13 - "Community 13"
Cohesion: 0.67
Nodes (1): HuggingFace Spaces entry point.

### Community 14 - "Community 14"
Cohesion: 0.67
Nodes (1): Day 2, Part 1 — Build the golden eval set.  WHY THIS FILE EXISTS:   The golde

### Community 15 - "Community 15"
Cohesion: 0.67
Nodes (1): Pushes the LoRA adapter to HuggingFace Hub.  Usage:   python scripts/push_adapte

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): Build the chat messages list — identical format to training data.

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (1): Run one inference pass, return (generated_sql, latency_seconds).      WHY temper

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (1): Generate a SQL literal for a column based on its type.

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): Spin up in-memory DuckDB with all tables and 5 rows each.

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): Execute SQL, return (success, rows, error).

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Compare result sets order-insensitively, with float rounding.

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Compute execution accuracy by complexity, print the results table.

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Convert messages list to a single string for SFTTrainer.     add_generation_prom

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Load all schema JSONs keyed by schema name.

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Load a JSONL file into a list of dicts, skipping blank lines.

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Attach schema SQL and format into Qwen2.5 chat messages.      Returns None if th

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Remove examples with duplicate question text (case-insensitive, stripped).

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Split into train/val preserving the complexity distribution in both halves.

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): Count examples per complexity bucket.

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Count examples per schema.

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Print the dataset assembly report — this is what you screenshot for the README.

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Print n random examples in full so you can visually verify the format.

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): easy   — single table, no joins, no subqueries     medium — 1-2 joins OR aggreg

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Check SQL parses without error using sqlparse.

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Build a generation prompt that FORCES use of the specified SQL patterns.      WH

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Call Groq API and parse the JSON response.     Returns list of {question, sql, c

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Load questions already in the output file to avoid duplicates.

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Generate targeted hard negatives for one schema. Returns validated pairs.

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Return golden eval examples the v1 model got right — used as style demos.

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Count already-written examples per schema — used to skip completed schemas.

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Build a generation prompt with:     - Schema context     - 3 passing golden eval

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Load all schema JSONs, optionally limited to the names in filter_names.

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Auto-generate a SQL literal for a column based on its type definition.     Produ

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Spin up an in-memory DuckDB with all schema tables populated with 5 rows each.

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Execute the SQL against the sandbox, return (valid, error_message).      WHY EXE

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Build the user-turn message for a single batch generation request.      WHY INCL

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Request n question-SQL pairs from Groq, return parsed and filtered list.      JS

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Return trivially correct pairs for --dry-run mode.     WHY: Lets us test validat

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Start a W&B run if WANDB_API_KEY is available, else return None.     W&B is stri

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Log per-schema stats to W&B. Silently no-ops if no run is active.

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Read the output file and return schema names that already have >= target pairs.

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Generate `target` DuckDB-validated question-SQL pairs for one schema.      Per-c

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Load all schema JSONs into a dict keyed by schema name.

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Load golden eval set from JSONL, optionally capped at `limit` examples.

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Auto-generate a SQL literal for a column based on its type definition.      WH

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): Spin up an in-memory DuckDB with all schema tables and 5 rows each.      WHY 5

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): Execute SQL safely, return (success, rows, error_message).      WHY TUPLE: We

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): Compare two result sets order-insensitively.      WHY ORDER-INSENSITIVE: The r

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): Build an OpenAI client pointed at Groq's API.      WHY OPENAI CLIENT FOR GROQ:

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Send a question + schema to Groq, return (generated_sql, latency_seconds).

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Run all eval examples through the model, return per-question result dicts.

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): Compute eval metrics broken down by complexity bucket.      Two metrics:

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): Print the comparison table — this exact table goes in your README and     is wh

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (1): # WHY: Fair comparison requires the same prompt for all models.

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (1): No-op — kept so demo.py import doesn't change.

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (1): Return (sql_string, latency_seconds).

### Community 71 - "Community 71"
Cohesion: 1.0
Nodes (1): Day 2, Part 1 — Build the golden eval set.  WHY THIS FILE EXISTS:   The golde

### Community 72 - "Community 72"
Cohesion: 1.0
Nodes (1): Day 1 Script: Download Spider + BIRD, filter to medium/hard complexity. Output:

### Community 73 - "Community 73"
Cohesion: 1.0
Nodes (1): easy   — single table, no joins, no subqueries     medium — 1-2 joins OR aggreg

### Community 74 - "Community 74"
Cohesion: 1.0
Nodes (1): Check SQL parses without error using sqlparse.

### Community 75 - "Community 75"
Cohesion: 1.0
Nodes (1): Day 1 Script: Generate 10 synthetic schemas (5 e-commerce, 5 fintech). These be

### Community 76 - "Community 76"
Cohesion: 1.0
Nodes (1): Generate CREATE TABLE statements for a schema (used in prompts).

### Community 77 - "Community 77"
Cohesion: 1.0
Nodes (1): Day 2, Part 2 — Baseline eval runner.  WHY THIS FILE EXISTS:   This runs GPT-

### Community 78 - "Community 78"
Cohesion: 1.0
Nodes (1): Load all schema JSONs into a dict keyed by schema name.

### Community 79 - "Community 79"
Cohesion: 1.0
Nodes (1): Load golden eval set from JSONL.

### Community 80 - "Community 80"
Cohesion: 1.0
Nodes (1): Given a column definition like 'user_id INT PK' or 'amount DECIMAL(10,2)',

### Community 81 - "Community 81"
Cohesion: 1.0
Nodes (1): Create an in-memory DuckDB, create tables from schema, insert 5 rows each.

### Community 82 - "Community 82"
Cohesion: 1.0
Nodes (1): Execute SQL, return (success, result_rows, error_message).      WHY TUPLE RETU

### Community 83 - "Community 83"
Cohesion: 1.0
Nodes (1): Compare two result sets, order-insensitively.      WHY ORDER-INSENSITIVE: The

### Community 84 - "Community 84"
Cohesion: 1.0
Nodes (1): Call GPT-4o and return (sql_output, latency_seconds).

### Community 85 - "Community 85"
Cohesion: 1.0
Nodes (1): Call Claude Sonnet and return (sql_output, latency_seconds).

### Community 86 - "Community 86"
Cohesion: 1.0
Nodes (1): Dispatch to the right model function.

### Community 87 - "Community 87"
Cohesion: 1.0
Nodes (1): Run all eval examples through the model.      Returns list of result dicts, on

### Community 88 - "Community 88"
Cohesion: 1.0
Nodes (1): Compute the key metrics for the comparison table.      Metrics explained:

### Community 89 - "Community 89"
Cohesion: 1.0
Nodes (1): Print the table that goes in your README.     This is the document interviewers

### Community 90 - "Community 90"
Cohesion: 1.0
Nodes (1): # WHY: We want a fair comparison. The fine-tune is trained with this exact

### Community 91 - "Community 91"
Cohesion: 1.0
Nodes (1): Prompt formatting utilities. Single source of truth — used by training, inferen

### Community 92 - "Community 92"
Cohesion: 1.0
Nodes (1): Returns messages list in Qwen2.5 chat format.     Used identically during train

### Community 93 - "Community 93"
Cohesion: 1.0
Nodes (1): Full training example with assistant turn included.     Output is a dict with '

## Knowledge Gaps
- **139 isolated node(s):** `Raised when SQL generation fails; message is safe to show clients.`, `Prompt formatting utilities. Single source of truth — used by training, inferen`, `Returns messages list in Qwen2.5 chat format.     Used identically during train`, `Full training example with assistant turn included.     Output is a dict with '`, `Build the chat messages list — identical format to training data.` (+134 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 13`** (3 nodes): `app.py`, `HuggingFace Spaces entry point.`, `app.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (3 nodes): `build_eval_set()`, `Day 2, Part 1 — Build the golden eval set.  WHY THIS FILE EXISTS:   The golde`, `build_eval_set.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (3 nodes): `main()`, `Pushes the LoRA adapter to HuggingFace Hub.  Usage:   python scripts/push_adapte`, `push_adapter_to_hub.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `Build the chat messages list — identical format to training data.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `Run one inference pass, return (generated_sql, latency_seconds).      WHY temper`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `Generate a SQL literal for a column based on its type.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `Spin up in-memory DuckDB with all tables and 5 rows each.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `Execute SQL, return (success, rows, error).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `Compare result sets order-insensitively, with float rounding.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Compute execution accuracy by complexity, print the results table.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Convert messages list to a single string for SFTTrainer.     add_generation_prom`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Load all schema JSONs keyed by schema name.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `Load a JSONL file into a list of dicts, skipping blank lines.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Attach schema SQL and format into Qwen2.5 chat messages.      Returns None if th`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Remove examples with duplicate question text (case-insensitive, stripped).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Split into train/val preserving the complexity distribution in both halves.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `Count examples per complexity bucket.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Count examples per schema.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Print the dataset assembly report — this is what you screenshot for the README.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Print n random examples in full so you can visually verify the format.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `easy   — single table, no joins, no subqueries     medium — 1-2 joins OR aggreg`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Check SQL parses without error using sqlparse.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Build a generation prompt that FORCES use of the specified SQL patterns.      WH`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Call Groq API and parse the JSON response.     Returns list of {question, sql, c`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Load questions already in the output file to avoid duplicates.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Generate targeted hard negatives for one schema. Returns validated pairs.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Return golden eval examples the v1 model got right — used as style demos.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Count already-written examples per schema — used to skip completed schemas.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Build a generation prompt with:     - Schema context     - 3 passing golden eval`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Load all schema JSONs, optionally limited to the names in filter_names.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Auto-generate a SQL literal for a column based on its type definition.     Produ`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Spin up an in-memory DuckDB with all schema tables populated with 5 rows each.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Execute the SQL against the sandbox, return (valid, error_message).      WHY EXE`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Build the user-turn message for a single batch generation request.      WHY INCL`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Request n question-SQL pairs from Groq, return parsed and filtered list.      JS`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Return trivially correct pairs for --dry-run mode.     WHY: Lets us test validat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Start a W&B run if WANDB_API_KEY is available, else return None.     W&B is stri`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Log per-schema stats to W&B. Silently no-ops if no run is active.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Read the output file and return schema names that already have >= target pairs.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Generate `target` DuckDB-validated question-SQL pairs for one schema.      Per-c`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Load all schema JSONs into a dict keyed by schema name.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Load golden eval set from JSONL, optionally capped at `limit` examples.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Auto-generate a SQL literal for a column based on its type definition.      WH`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `Spin up an in-memory DuckDB with all schema tables and 5 rows each.      WHY 5`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `Execute SQL safely, return (success, rows, error_message).      WHY TUPLE: We`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `Compare two result sets order-insensitively.      WHY ORDER-INSENSITIVE: The r`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `Build an OpenAI client pointed at Groq's API.      WHY OPENAI CLIENT FOR GROQ:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `Send a question + schema to Groq, return (generated_sql, latency_seconds).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Run all eval examples through the model, return per-question result dicts.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `Compute eval metrics broken down by complexity bucket.      Two metrics:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `Print the comparison table — this exact table goes in your README and     is wh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `# WHY: Fair comparison requires the same prompt for all models.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (1 nodes): `No-op — kept so demo.py import doesn't change.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (1 nodes): `Return (sql_string, latency_seconds).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 71`** (1 nodes): `Day 2, Part 1 — Build the golden eval set.  WHY THIS FILE EXISTS:   The golde`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 72`** (1 nodes): `Day 1 Script: Download Spider + BIRD, filter to medium/hard complexity. Output:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 73`** (1 nodes): `easy   — single table, no joins, no subqueries     medium — 1-2 joins OR aggreg`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 74`** (1 nodes): `Check SQL parses without error using sqlparse.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 75`** (1 nodes): `Day 1 Script: Generate 10 synthetic schemas (5 e-commerce, 5 fintech). These be`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 76`** (1 nodes): `Generate CREATE TABLE statements for a schema (used in prompts).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 77`** (1 nodes): `Day 2, Part 2 — Baseline eval runner.  WHY THIS FILE EXISTS:   This runs GPT-`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 78`** (1 nodes): `Load all schema JSONs into a dict keyed by schema name.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 79`** (1 nodes): `Load golden eval set from JSONL.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 80`** (1 nodes): `Given a column definition like 'user_id INT PK' or 'amount DECIMAL(10,2)',`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 81`** (1 nodes): `Create an in-memory DuckDB, create tables from schema, insert 5 rows each.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 82`** (1 nodes): `Execute SQL, return (success, result_rows, error_message).      WHY TUPLE RETU`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 83`** (1 nodes): `Compare two result sets, order-insensitively.      WHY ORDER-INSENSITIVE: The`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 84`** (1 nodes): `Call GPT-4o and return (sql_output, latency_seconds).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 85`** (1 nodes): `Call Claude Sonnet and return (sql_output, latency_seconds).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 86`** (1 nodes): `Dispatch to the right model function.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 87`** (1 nodes): `Run all eval examples through the model.      Returns list of result dicts, on`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 88`** (1 nodes): `Compute the key metrics for the comparison table.      Metrics explained:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 89`** (1 nodes): `Print the table that goes in your README.     This is the document interviewers`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 90`** (1 nodes): `# WHY: We want a fair comparison. The fine-tune is trained with this exact`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 91`** (1 nodes): `Prompt formatting utilities. Single source of truth — used by training, inferen`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 92`** (1 nodes): `Returns messages list in Qwen2.5 chat format.     Used identically during train`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 93`** (1 nodes): `Full training example with assistant turn included.     Output is a dict with '`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `generate()` connect `Community 2` to `Community 8`, `Community 4`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Why does `InferenceUserError` connect `Community 5` to `Community 2`, `Community 4`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Why does `generate_sql()` connect `Community 4` to `Community 8`, `Community 2`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `InferenceUserError` (e.g. with `GenerateRequest` and `GenerateResponse`) actually correct?**
  _`InferenceUserError` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `generate()` (e.g. with `load_all_schemas()` and `generate_sql()`) actually correct?**
  _`generate()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Raised when SQL generation fails; message is safe to show clients.`, `Prompt formatting utilities. Single source of truth — used by training, inferen`, `Returns messages list in Qwen2.5 chat format.     Used identically during train` to the rest of the system?**
  _139 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.11 - nodes in this community are weakly interconnected._