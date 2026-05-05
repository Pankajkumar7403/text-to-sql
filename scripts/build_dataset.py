"""
Day 4 — Dataset assembly and train/val split.

WHAT THIS BUILDS:
  Combines the 1,160 synthetic pairs from layer2_synthetic.jsonl with their
  schema CREATE SQL, formats every example into Qwen2.5 chat format, deduplicates,
  shuffles, and writes a stratified 90/10 train/val split.

THE THREE DATA LAYERS:
  Layer 1 — Spider/BIRD filtered examples (data/processed/layer1_spider_bird.jsonl)
             Not used yet (Day 5 will decide whether to include).
  Layer 2 — Synthetic pairs we generated (data/processed/layer2_synthetic.jsonl)
             1,160 DuckDB-validated question-SQL pairs across 10 domain schemas.
  Golden   — Hand-curated eval set (data/eval/golden_eval.jsonl)
             68 questions. NEVER enters training — held out as the final test set.

WHY STRATIFIED SPLIT (not random):
  A plain random 90/10 split might end up with val set being mostly easy questions
  if the shuffle is unlucky. Stratifying by complexity guarantees val has the same
  20/40/40 easy/medium/hard ratio as the full dataset. This matters because we
  track val loss separately per complexity during training.

WHY MESSAGES FORMAT (not raw text):
  Unsloth's SFTTrainer reads datasets with a "messages" key natively and applies
  the model's chat template automatically. This means the same formatting function
  works at training time AND inference time — no hidden prompt-format mismatch.

WHY DEDUPLICATE:
  When we called the generator with --resume, a small number of questions
  could appear twice (once from a partial run, once from a retry). Training
  on exact duplicates wastes capacity and can cause the model to overfit to
  specific phrasing.

OUTPUTS:
  data/processed/train.jsonl   — 90% of synthetic data, in chat format
  data/processed/val.jsonl     — 10% of synthetic data, in chat format
  data/processed/dataset_summary.json — counts, split stats, complexity breakdown

HOW TO RUN:
  python scripts/build_dataset.py             # standard build
  python scripts/build_dataset.py --inspect   # also prints 3 random training examples
  python scripts/build_dataset.py --seed 123  # reproducible shuffle with custom seed
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path so we can import from src/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.prompt_template import format_training_example  # noqa: E402

SCHEMA_DIR  = ROOT / "data" / "raw" / "schemas"
PROC_DIR    = ROOT / "data" / "processed"
EVAL_DIR    = ROOT / "data" / "eval"

SYNTHETIC_FILE = PROC_DIR / "layer2_synthetic.jsonl"
TRAIN_FILE     = PROC_DIR / "train.jsonl"
VAL_FILE       = PROC_DIR / "val.jsonl"
SUMMARY_FILE   = PROC_DIR / "dataset_summary.json"

VAL_FRACTION = 0.10   # 10% held out for validation during training


# ──────────────────────────────────────────────────────────────────────────────
# LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_schemas() -> dict[str, dict]:
    """Load all schema JSONs keyed by schema name."""
    schemas: dict[str, dict] = {}
    for f in SCHEMA_DIR.glob("*.json"):
        if f.name == "index.json":
            continue
        with open(f) as fh:
            s = json.load(fh)
        schemas[s["name"]] = s
    return schemas


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts, skipping blank lines."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# ──────────────────────────────────────────────────────────────────────────────
# FORMATTING
# ──────────────────────────────────────────────────────────────────────────────

def format_example(raw: dict, schemas: dict[str, dict]) -> dict | None:
    """
    Attach schema SQL and format into Qwen2.5 chat messages.

    Returns None if the schema is missing (shouldn't happen but guards against
    a corrupt layer2 file referencing a deleted schema).

    Output fields:
      schema_name, question, sql, complexity, source — original fields preserved
      messages — the chat turns SFTTrainer reads for training
    """
    schema = schemas.get(raw["schema_name"])
    if schema is None:
        print(f"  WARNING: schema '{raw['schema_name']}' not found — skipping.")
        return None

    # format_training_example returns {"messages": [...]} with system+user+assistant turns
    chat = format_training_example(
        schema_sql=schema["create_sql"],
        question=raw["question"],
        sql=raw["sql"],
    )

    return {
        "schema_name": raw["schema_name"],
        "question":    raw["question"],
        "sql":         raw["sql"],
        "complexity":  raw["complexity"],
        "source":      raw.get("source", "synthetic"),
        "messages":    chat["messages"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION
# ──────────────────────────────────────────────────────────────────────────────

def deduplicate(examples: list[dict]) -> tuple[list[dict], int]:
    """
    Remove examples with duplicate question text (case-insensitive, stripped).

    WHY QUESTION NOT SQL: Two different questions might produce identical SQL
    (e.g. "count all users" and "how many users are there?"). That's fine —
    we want both phrasings. But the same question asked twice is dead weight.

    Returns (deduplicated list, number of duplicates removed).
    """
    seen: set[str] = set()
    unique: list[dict] = []
    for ex in examples:
        key = ex["question"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique, len(examples) - len(unique)


# ──────────────────────────────────────────────────────────────────────────────
# STRATIFIED SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def stratified_split(
    examples: list[dict],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """
    Split into train/val preserving the complexity distribution in both halves.

    WHY STRATIFY BY COMPLEXITY:
    Random split risks having the val set skew easy (lots of easy examples,
    easy to overfit). Stratified split guarantees the val set is as challenging
    as the full dataset — making val loss a reliable training signal.

    Returns (train_examples, val_examples).
    """
    rng = random.Random(seed)

    # Group by complexity bucket
    by_complexity: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_complexity[ex["complexity"]].append(ex)

    train: list[dict] = []
    val:   list[dict] = []

    for complexity, group in by_complexity.items():
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction))  # at least 1 val example per bucket
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    # Shuffle final lists so schemas/complexities don't appear in blocks
    rng.shuffle(train)
    rng.shuffle(val)

    return train, val


# ──────────────────────────────────────────────────────────────────────────────
# REPORTING
# ──────────────────────────────────────────────────────────────────────────────

def complexity_counts(examples: list[dict]) -> dict[str, int]:
    """Count examples per complexity bucket."""
    counts: dict[str, int] = defaultdict(int)
    for ex in examples:
        counts[ex["complexity"]] += 1
    return dict(counts)


def schema_counts(examples: list[dict]) -> dict[str, int]:
    """Count examples per schema."""
    counts: dict[str, int] = defaultdict(int)
    for ex in examples:
        counts[ex["schema_name"]] += 1
    return dict(counts)


def print_summary(train: list[dict], val: list[dict], golden_n: int) -> None:
    """Print the dataset assembly report — this is what you screenshot for the README."""
    total = len(train) + len(val)
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"{'Split':<12} {'Total':>7} {'Easy':>7} {'Medium':>8} {'Hard':>7}")
    print("-" * 60)
    for name, split in [("train", train), ("val", val)]:
        cc = complexity_counts(split)
        print(
            f"{name:<12}"
            f"{len(split):>7}"
            f"{cc.get('easy',   0):>7}"
            f"{cc.get('medium', 0):>8}"
            f"{cc.get('hard',   0):>7}"
        )
    print("-" * 60)
    print(
        f"{'total':<12}{total:>7}"
        f"{complexity_counts(train+val).get('easy',   0):>7}"
        f"{complexity_counts(train+val).get('medium', 0):>8}"
        f"{complexity_counts(train+val).get('hard',   0):>7}"
    )
    print("=" * 60)
    print(f"\nGolden eval (held-out test set): {golden_n} examples")
    print("  -> Never enters training. Used only for final benchmark.")
    print(f"\nTotal training-available examples: {total}")
    print(f"  Train: {len(train)} ({len(train)/total*100:.0f}%)")
    print(f"  Val:   {len(val)}   ({len(val)/total*100:.0f}%)")


def inspect_examples(examples: list[dict], n: int = 3, seed: int = 42) -> None:
    """Print n random examples in full so you can visually verify the format."""
    rng = random.Random(seed)
    sample = rng.sample(examples, min(n, len(examples)))
    print("\n" + "=" * 60)
    print(f"SAMPLE TRAINING EXAMPLES (n={len(sample)})")
    print("=" * 60)
    for i, ex in enumerate(sample, 1):
        print(f"\n-- Example {i} [{ex['complexity']}] {ex['schema_name']} --")
        print(f"Q: {ex['question']}")
        print(f"SQL: {ex['sql'][:200]}{'...' if len(ex['sql']) > 200 else ''}")
        print(f"Messages turns: {[m['role'] for m in ex['messages']]}")
        # Show the user turn so we can verify schema is embedded correctly
        user_content = ex['messages'][1]['content']
        print(f"User turn (first 300 chars):\n{user_content[:300]}...")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble and split the fine-tuning dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffle and split (default: 42).",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=VAL_FRACTION,
        help=f"Fraction of data to hold out for validation (default: {VAL_FRACTION}).",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Print 3 sample training examples after building.",
    )
    args = parser.parse_args()

    # ── 1. Load schemas ───────────────────────────────────────────────────────
    schemas = load_schemas()
    print(f"Loaded {len(schemas)} schemas.")

    # ── 2. Load synthetic data ────────────────────────────────────────────────
    if not SYNTHETIC_FILE.exists():
        raise SystemExit(f"Synthetic data not found: {SYNTHETIC_FILE}\nRun scripts/generate_synthetic.py first.")

    raw_examples = load_jsonl(SYNTHETIC_FILE)
    print(f"Loaded {len(raw_examples)} raw synthetic examples.")

    # ── 3. Format into chat messages ──────────────────────────────────────────
    formatted: list[dict] = []
    skipped = 0
    for raw in raw_examples:
        ex = format_example(raw, schemas)
        if ex is not None:
            formatted.append(ex)
        else:
            skipped += 1

    print(f"Formatted {len(formatted)} examples ({skipped} skipped — missing schema).")

    # ── 4. Deduplicate ────────────────────────────────────────────────────────
    formatted, n_dupes = deduplicate(formatted)
    print(f"Deduplicated: removed {n_dupes} duplicate questions. {len(formatted)} remain.")

    # ── 5. Stratified train/val split ─────────────────────────────────────────
    train, val = stratified_split(formatted, args.val_fraction, args.seed)
    print(f"Split: {len(train)} train / {len(val)} val (seed={args.seed}).")

    # ── 6. Write output files ─────────────────────────────────────────────────
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    for path, split in [(TRAIN_FILE, train), (VAL_FILE, val)]:
        with open(path, "w") as f:
            for ex in split:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(split):>5} examples -> {path.name}")

    # ── 7. Save summary JSON ──────────────────────────────────────────────────
    golden_n = 0
    if (EVAL_DIR / "golden_eval.jsonl").exists():
        golden_n = sum(1 for _ in open(EVAL_DIR / "golden_eval.jsonl") if _.strip())

    summary = {
        "seed":           args.seed,
        "val_fraction":   args.val_fraction,
        "total":          len(train) + len(val),
        "train":          len(train),
        "val":            len(val),
        "golden_eval":    golden_n,
        "train_by_complexity": complexity_counts(train),
        "val_by_complexity":   complexity_counts(val),
        "train_by_schema":     schema_counts(train),
        "val_by_schema":       schema_counts(val),
    }
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary -> {SUMMARY_FILE.name}")

    # ── 8. Report ─────────────────────────────────────────────────────────────
    print_summary(train, val, golden_n)

    if args.inspect:
        inspect_examples(train, n=3, seed=args.seed)


if __name__ == "__main__":
    main()
