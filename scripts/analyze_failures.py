"""
Day 11 — Analyze fine-tuned model failures to identify SQL patterns to target.

Reads the per-question eval results from Day 10, detects which SQL constructs
appear most in wrong answers, and outputs a priority list for hard negative mining.

WHY THIS MATTERS:
  Random synthetic data improves general SQL fluency but misses the specific
  patterns the model gets wrong. Failure analysis tells you exactly which
  SQL constructs need more training examples.

OUTPUTS:
  data/processed/failure_analysis.json  -- machine-readable, read by generate_hard_negatives.py
  Console report                        -- human-readable summary

HOW TO RUN:
  python scripts/analyze_failures.py
  python scripts/analyze_failures.py --results data/eval/results/finetuned_qwen25_7b_TIMESTAMP.jsonl
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "data" / "eval" / "results"
PROC_DIR    = ROOT / "data" / "processed"

# SQL constructs to detect in the reference SQL of each failed example.
# Pattern name -> regex (None means custom logic below).
PATTERNS: dict[str, str | None] = {
    "cte":             r"\bWITH\b",
    "window_function": r"\bOVER\s*\(",
    "rank_function":   r"\b(RANK|DENSE_RANK|ROW_NUMBER|NTILE|LAG|LEAD|FIRST_VALUE|LAST_VALUE)\s*\(",
    "multi_join":      None,   # >= 2 JOINs — custom logic
    "aggregation":     r"\bGROUP\s+BY\b",
    "having":          r"\bHAVING\b",
    "correlated_sub":  r"\(\s*SELECT\b",
    "union":           r"\b(UNION|INTERSECT|EXCEPT)\b",
    "distinct":        r"\bSELECT\s+DISTINCT\b",
    "order_limit":     r"\bORDER\s+BY\b",
}


def detect_patterns(sql: str) -> list[str]:
    """Return list of SQL pattern names present in the query."""
    sql_upper = sql.upper()
    found = []
    for name, pattern in PATTERNS.items():
        if pattern is None:
            if len(re.findall(r"\bJOIN\b", sql_upper)) >= 2:
                found.append(name)
        elif re.search(pattern, sql_upper):
            found.append(name)
    return found


def find_latest_results(results_dir: Path) -> Path | None:
    files = sorted(results_dir.glob("finetuned_qwen25_7b_*.jsonl"), reverse=True)
    return files[0] if files else None


def load_jsonl(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze fine-tuned model failures.")
    parser.add_argument(
        "--results", type=Path, default=None,
        help="Path to finetuned_qwen25_7b_*.jsonl. Defaults to most recent file.",
    )
    args = parser.parse_args()

    results_path = args.results or find_latest_results(RESULTS_DIR)
    if not results_path or not results_path.exists():
        raise SystemExit(
            f"No results file found in {RESULTS_DIR}.\n"
            "Download finetuned_qwen25_7b_*.jsonl from Kaggle Output tab first:\n"
            "  data/eval/results/finetuned_qwen25_7b_TIMESTAMP.jsonl"
        )

    results  = load_jsonl(results_path)
    failures = [r for r in results if not r["exec_match"]]
    correct  = [r for r in results if r["exec_match"]]

    print(f"\nLoaded {len(results)} results from: {results_path.name}")
    print(f"Correct: {len(correct)} | Failed: {len(failures)} "
          f"({len(failures)/len(results)*100:.1f}%)\n")

    fail_pattern_ctr    = Counter()
    correct_pattern_ctr = Counter()
    pattern_fail_examples: dict[str, list[dict]] = defaultdict(list)

    complexity_fails: dict[str, int] = defaultdict(int)
    schema_fails:     dict[str, int] = defaultdict(int)
    invalid_sql_count = sum(1 for r in failures if not r.get("valid_sql", True))

    for r in failures:
        patterns = detect_patterns(r["reference_sql"])
        for p in patterns:
            fail_pattern_ctr[p] += 1
            pattern_fail_examples[p].append(r)
        complexity_fails[r["complexity"]] += 1
        schema_fails[r["schema_name"]]    += 1

    for r in correct:
        for p in detect_patterns(r["reference_sql"]):
            correct_pattern_ctr[p] += 1

    pattern_fail_rates: dict[str, float] = {}
    for p in fail_pattern_ctr:
        total = fail_pattern_ctr[p] + correct_pattern_ctr.get(p, 0)
        pattern_fail_rates[p] = fail_pattern_ctr[p] / total * 100 if total > 0 else 0

    print("=" * 65)
    print("FAILURE PATTERN ANALYSIS")
    print("=" * 65)
    print(f"  {'Pattern':<20} {'#Fail':>6} {'#OK':>6} {'Fail%':>8}")
    print("-" * 65)

    for p in sorted(fail_pattern_ctr, key=lambda x: -pattern_fail_rates[x]):
        print(f"  {p:<20} {fail_pattern_ctr[p]:>6} "
              f"{correct_pattern_ctr.get(p, 0):>6} "
              f"{pattern_fail_rates[p]:>7.0f}%")

    print(f"\n  Invalid SQL (model syntax errors): {invalid_sql_count}/{len(failures)}")

    print(f"\n{'=' * 65}")
    print("FAILURES BY COMPLEXITY")
    print("=" * 65)
    for bucket in ["easy", "medium", "hard"]:
        n_fail  = complexity_fails.get(bucket, 0)
        n_total = sum(1 for r in results if r["complexity"] == bucket)
        pct     = n_fail / n_total * 100 if n_total > 0 else 0
        print(f"  {bucket:<10}: {n_fail:>2}/{n_total} failed ({pct:.0f}%)")

    print(f"\n{'=' * 65}")
    print("FAILURES BY SCHEMA")
    print("=" * 65)
    for schema, count in sorted(schema_fails.items(), key=lambda x: -x[1]):
        print(f"  {schema:<35}: {count}")

    print(f"\n{'=' * 65}")
    print("SAMPLE FAILURES (worst complexity first)")
    print("=" * 65)
    sorted_failures = sorted(
        failures,
        key=lambda r: ["easy", "medium", "hard"].index(r["complexity"]),
        reverse=True,
    )
    for r in sorted_failures[:5]:
        pats = detect_patterns(r["reference_sql"])
        print(f"\n[{r['complexity'].upper()}] {r['schema_name']}")
        print(f"  Q:        {r['question']}")
        print(f"  Ref SQL:  {r['reference_sql'][:100]}...")
        print(f"  Gen SQL:  {r['generated_sql'][:100]}...")
        if r.get("gen_error"):
            print(f"  Error:    {r['gen_error'][:80]}")
        print(f"  Patterns: {pats}")

    # Priority: high fail rate AND enough occurrences (>= 2) to matter
    top_patterns = [
        p for p, rate in sorted(pattern_fail_rates.items(), key=lambda x: -x[1])
        if fail_pattern_ctr[p] >= 2
    ][:5]

    print(f"\n{'=' * 65}")
    print("RECOMMENDED TARGETS FOR HARD NEGATIVE MINING")
    print("=" * 65)
    for p in top_patterns:
        print(f"  {p:<22}: {fail_pattern_ctr[p]} failures, "
              f"{pattern_fail_rates[p]:.0f}% fail rate")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "source_file":            results_path.name,
        "total":                  len(results),
        "correct":                len(correct),
        "failed":                 len(failures),
        "invalid_sql":            invalid_sql_count,
        "pattern_failures":       dict(fail_pattern_ctr),
        "pattern_fail_rates":     {k: round(v, 1) for k, v in pattern_fail_rates.items()},
        "complexity_failures":    dict(complexity_fails),
        "schema_failures":        dict(schema_fails),
        "top_patterns_to_target": top_patterns,
    }

    out_path = PROC_DIR / "failure_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved -> {out_path.relative_to(ROOT)}")
    print(f"Next:  python scripts/generate_hard_negatives.py")


if __name__ == "__main__":
    main()
