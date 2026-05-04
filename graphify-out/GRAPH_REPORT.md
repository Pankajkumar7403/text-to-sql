# Graph Report - text-to-sql  (2026-05-04)

## Corpus Check
- 6 files · ~10,144 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 57 nodes · 77 edges · 7 communities detected
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1b4d3d1f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]

## God Nodes (most connected - your core abstractions)
1. `classify_complexity()` - 8 edges
2. `run_eval()` - 7 edges
3. `main()` - 6 edges
4. `process_spider()` - 5 edges
5. `process_bird()` - 5 edges
6. `call_model()` - 5 edges
7. `count_joins()` - 4 edges
8. `count_subqueries()` - 4 edges
9. `is_valid_sql()` - 4 edges
10. `build_sandbox()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `run_eval()` --calls--> `call_model()`  [EXTRACTED]
  scripts/run_baseline_eval.py → scripts/run_baseline_eval.py  _Bridges community 3 → community 2_
- `main()` --calls--> `run_eval()`  [EXTRACTED]
  scripts/run_baseline_eval.py → scripts/run_baseline_eval.py  _Bridges community 2 → community 1_

## Communities (8 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.31
Nodes (11): classify_complexity(), count_joins(), count_subqueries(), has_aggregation(), has_nested_conditions(), is_valid_sql(), process_bird(), process_spider() (+3 more)

### Community 1 - "Community 1"
Cohesion: 0.23
Nodes (11): compute_summary(), load_eval_set(), load_schemas(), main(), print_comparison_table(), Day 2, Part 2 — Baseline eval runner.  WHY THIS FILE EXISTS:   This runs GPT-, Compute the key metrics for the comparison table.      Metrics explained:, Print the table that goes in your README.     This is the document interviewers (+3 more)

### Community 2 - "Community 2"
Cohesion: 0.2
Nodes (10): build_sandbox(), execute_sql(), _infer_value(), Given a column definition like 'user_id INT PK' or 'amount DECIMAL(10,2)',, Create an in-memory DuckDB, create tables from schema, insert 5 rows each., Execute SQL, return (success, result_rows, error_message).      WHY TUPLE RETU, Compare two result sets, order-insensitively.      WHY ORDER-INSENSITIVE: The, Run all eval examples through the model.      Returns list of result dicts, on (+2 more)

### Community 3 - "Community 3"
Cohesion: 0.33
Nodes (6): call_claude(), call_gpt4o(), call_model(), Call GPT-4o and return (sql_output, latency_seconds)., Call Claude Sonnet and return (sql_output, latency_seconds)., Dispatch to the right model function.

### Community 4 - "Community 4"
Cohesion: 0.4
Nodes (5): format_prompt(), format_training_example(), Prompt formatting utilities. Single source of truth — used by training, inferen, Returns messages list in Qwen2.5 chat format.     Used identically during train, Full training example with assistant turn included.     Output is a dict with '

### Community 5 - "Community 5"
Cohesion: 0.5
Nodes (4): Day 1 Script: Generate 10 synthetic schemas (5 e-commerce, 5 fintech). These be, Generate CREATE TABLE statements for a schema (used in prompts)., save_schemas(), schema_to_create_sql()

## Knowledge Gaps
- **23 isolated node(s):** `Day 2, Part 1 — Build the golden eval set.  WHY THIS FILE EXISTS:   The golde`, `Day 1 Script: Download Spider + BIRD, filter to medium/hard complexity. Output:`, `easy   — single table, no joins, no subqueries     medium — 1-2 joins OR aggreg`, `Check SQL parses without error using sqlparse.`, `Day 1 Script: Generate 10 synthetic schemas (5 e-commerce, 5 fintech). These be` (+18 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_eval()` connect `Community 2` to `Community 1`, `Community 3`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Why does `call_model()` connect `Community 3` to `Community 1`, `Community 2`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Why does `build_sandbox()` connect `Community 2` to `Community 1`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._
- **What connects `Day 2, Part 1 — Build the golden eval set.  WHY THIS FILE EXISTS:   The golde`, `Day 1 Script: Download Spider + BIRD, filter to medium/hard complexity. Output:`, `easy   — single table, no joins, no subqueries     medium — 1-2 joins OR aggreg` to the rest of the system?**
  _23 weakly-connected nodes found - possible documentation gaps or missing edges._