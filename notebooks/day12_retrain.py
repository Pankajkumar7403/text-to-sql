# =============================================================================
# DAY 12 — Retrain with Hard Negatives (Unsloth + TRL SFTTrainer)
# Run this on Kaggle with T4 GPU enabled.
#
# WHAT CHANGED FROM DAY 6:
#   - Dataset: original 1,023 examples + hard negatives (layer3_hard_negatives.jsonl)
#   - Learning rate: 5e-5 (was 2e-4) — refinement, not relearning from scratch
#   - Epochs: 2 (was 3) — lower LR needs fewer epochs to converge
#   - Fresh LoRA from base model (not continuing from overfitted Day 6 adapter)
#
# WHY FRESH LORA NOT CONTINUE:
#   Day 6 adapter reached loss 0.0962 (very low -> overfitting).
#   Continuing from it at lower LR would just memorize more.
#   Starting fresh with better data + lower LR teaches generalization.
#
# SETUP:
#   1. Add to your Kaggle dataset "text-to-sql-data":
#        data/processed/train.jsonl              (original, already there)
#        data/processed/layer3_hard_negatives.jsonl  (generated locally, new)
#   2. Enable GPU T4 x2
# =============================================================================


# =============================================================================
# CELL 1 — Install packages (same two-step Unsloth install)
# =============================================================================

# %%
import subprocess

subprocess.run(["pip", "install", "--quiet", "unsloth"], check=True)
subprocess.run([
    "pip", "install", "--quiet", "--upgrade", "--no-cache-dir", "--no-deps",
    "git+https://github.com/unslothai/unsloth.git",
], check=True)
subprocess.run([
    "pip", "install", "--quiet",
    "trl", "peft>=0.12.0", "accelerate>=0.34.0",
    "bitsandbytes>=0.43.0", "datasets>=2.19.0",
], check=True)

print("Packages installed.")


# =============================================================================
# CELL 2 — Imports and config
# =============================================================================

# %%
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

DATA_DIR    = Path("/kaggle/input/text-to-sql-data")
OUTPUT_DIR  = Path("/kaggle/working")
ADAPTER_DIR = OUTPUT_DIR / "qwen25_sql_v2_adapter"

TRAIN_FILE       = DATA_DIR / "train.jsonl"
HARD_NEG_FILE    = DATA_DIR / "layer3_hard_negatives.jsonl"

MODEL_ID       = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 2048

print(f"Train file:         {TRAIN_FILE.exists()}")
print(f"Hard negatives:     {HARD_NEG_FILE.exists()}")
print(f"CUDA available:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:                {torch.cuda.get_device_name(0)}")
    print(f"VRAM:               {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# =============================================================================
# CELL 3 — Load model (fresh 4-bit base, no adapter)
# =============================================================================

# %%
print(f"Loading {MODEL_ID} fresh (no adapter)...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

vram_used = torch.cuda.memory_allocated() / 1e9
print(f"Model loaded. VRAM: {vram_used:.1f} GB")


# =============================================================================
# CELL 4 — Apply QLoRA (same config as Day 6)
# =============================================================================

# %%
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")


# =============================================================================
# CELL 5 — Build combined dataset (original + hard negatives)
# =============================================================================

# %%
def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def apply_chat_template(examples: dict) -> dict:
    texts = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}


# Load original training data
original = load_jsonl(TRAIN_FILE)
print(f"Original training examples: {len(original)}")

# Load hard negatives (may not exist if skipping Day 11)
hard_negs = []
if HARD_NEG_FILE.exists():
    raw_hard_negs = load_jsonl(HARD_NEG_FILE)

    # Hard negatives are in raw format (schema_name, question, sql, complexity, source).
    # They need the "messages" field added. Load schemas to do that.
    schemas: dict[str, dict] = {}
    for f in DATA_DIR.glob("*.json"):
        if f.suffix == ".json":
            try:
                with open(f) as fh:
                    s = json.load(fh)
                if "name" in s and "create_sql" in s:
                    schemas[s["name"]] = s
            except Exception:
                pass

    SYSTEM_PROMPT = """You are an expert SQL engineer. Given a database schema and a natural language question, write a correct and efficient SQL query.

Rules:
- Output ONLY the SQL query, no explanation, no markdown fences
- Use table aliases for readability on multi-join queries
- Use explicit JOIN syntax, never implicit comma joins
- Prefer CTEs over deeply nested subqueries when it improves readability
- Never use SELECT * -- always specify column names"""

    for ex in raw_hard_negs:
        schema = schemas.get(ex["schema_name"])
        if not schema:
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Schema:\n{schema['create_sql'].strip()}\n\nQuestion: {ex['question'].strip()}\n\nSQL:"},
            {"role": "assistant", "content": ex["sql"]},
        ]
        hard_negs.append({**ex, "messages": messages})

    print(f"Hard negative examples: {len(hard_negs)}")
else:
    print("No hard negatives file found — training on original data only.")

# Combine and shuffle
combined = original + hard_negs
random.seed(42)
random.shuffle(combined)

# Print complexity breakdown
from collections import Counter
counts = Counter(ex["complexity"] for ex in combined)
print(f"\nCombined dataset: {len(combined)} examples")
print(f"  easy: {counts['easy']} | medium: {counts['medium']} | hard: {counts['hard']}")

# Build HuggingFace dataset
dataset = Dataset.from_list(combined)
dataset = dataset.map(apply_chat_template, batched=True, remove_columns=dataset.column_names)
print(f"Dataset formatted. Sample length: {len(dataset[0]['text'])} chars")


# =============================================================================
# CELL 6 — Configure SFTTrainer
# Key changes vs Day 6: lr=5e-5, epochs=2
# =============================================================================

# %%
training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR / "checkpoints_v2"),
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,           # effective batch = 8
    warmup_steps=20,
    learning_rate=5e-5,                      # 4x lower than Day 6 — refinement not relearning
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=20,
    save_strategy="steps",
    save_steps=100,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    report_to="none",                        # add "wandb" if you set up WANDB_API_KEY in Secrets
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

steps_per_epoch = len(dataset) // (training_args.per_device_train_batch_size *
                                    training_args.gradient_accumulation_steps)
total_steps = steps_per_epoch * training_args.num_train_epochs
print(f"Steps per epoch:  {steps_per_epoch}")
print(f"Total steps:      {total_steps}")
print(f"Estimated time:   {total_steps * 4 / 60:.0f}-{total_steps * 6 / 60:.0f} minutes")


# =============================================================================
# CELL 7 — Train
# Expected: 90-120 minutes for combined dataset, 2 epochs on T4
# Loss should settle around 0.20-0.40 (higher than Day 6's 0.09 = less overfitting)
# =============================================================================

# %%
print("Starting Day 12 retraining...\n")
trainer_stats = trainer.train()

print(f"\nTraining complete.")
print(f"  Runtime:     {trainer_stats.metrics['train_runtime']:.0f}s "
      f"({trainer_stats.metrics['train_runtime']/60:.1f} min)")
print(f"  Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
print(f"  Final loss:  {trainer_stats.metrics['train_loss']:.4f}")
print(f"  Peak VRAM:   {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
print()
# Healthy target: loss 0.20-0.40
# If loss < 0.15: still overfitting, try fewer epochs next time
# If loss > 0.60: underfitting, try more epochs or higher LR


# =============================================================================
# CELL 8 — Save adapter (v2)
# =============================================================================

# %%
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))

adapter_files = list(ADAPTER_DIR.iterdir())
total_mb = sum(f.stat().st_size for f in adapter_files) / 1e6
print(f"Adapter v2 saved: {ADAPTER_DIR.name}/")
print(f"Size: {total_mb:.1f} MB")
print(f"\nDownload from Kaggle Output tab -> data/models/qwen25_sql_v2/")


# =============================================================================
# CELL 9 — Quick inference check
# =============================================================================

# %%
FastLanguageModel.for_inference(model)
model.generation_config.max_length = None

SYSTEM_PROMPT = """You are an expert SQL engineer. Given a database schema and a natural language question, write a correct and efficient SQL query.

Rules:
- Output ONLY the SQL query, no explanation, no markdown fences
- Use table aliases for readability on multi-join queries
- Use explicit JOIN syntax, never implicit comma joins
- Prefer CTEs over deeply nested subqueries when it improves readability
- Never use SELECT * -- always specify column names"""


def infer(schema_sql: str, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Schema:\n{schema_sql.strip()}\n\nQuestion: {question.strip()}\n\nSQL:"},
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    in_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    sql = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()
    return sql.replace("```sql", "").replace("```", "").strip()


# Test a hard example (window function) — this is what Day 6 failed on
test_ex = next(
    (ex for ex in hard_negs or combined if ex.get("complexity") == "hard"), combined[0]
)
user_content = test_ex["messages"][1]["content"]
schema_part  = user_content.split("\n\nQuestion:")[0].replace("Schema:\n", "")
question     = user_content.split("Question:")[1].split("\n\nSQL:")[0].strip()

gen_sql = infer(schema_part, question)
print(f"Hard example test:")
print(f"  Q:   {question[:80]}")
print(f"  Gen: {gen_sql[:150]}")
print(f"  Ref: {test_ex['sql'][:150]}")
print()
print("Next: Run day10_post_training_eval.py to get the v2 accuracy numbers.")
print("Expected improvement over v1: +10-20pp on medium/hard")
