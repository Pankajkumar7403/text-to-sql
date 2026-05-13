# DAY 12 — Retrain on original + hard negatives (Unsloth + TRL SFTTrainer)
# Run this on Kaggle with T4 GPU enabled.
#
# STRATEGY: Fresh LoRA from base model, trained on 1,023 original examples
# + 700 hard negatives = 1,723 total.
#
# WHY COMBINED NOT HARD-NEGATIVES ONLY:
#   V1 adapter is unavailable (Kaggle session expired). Training from base
#   on only 700 hard/medium examples would leave easy queries unlearned.
#   The combined dataset teaches everything in one shot.
#
# WHY lr=2e-4 (same as Day 6):
#   Starting from the raw base model — this is a full fine-tune, not a
#   refinement. 2e-4 is the standard Unsloth/QLoRA starting point.
#
# WHY 3 EPOCHS:
#   1,723 examples at lr=2e-4 needs ~3 epochs to converge. Day 6 used 3
#   epochs on 1,023 examples and reached loss 0.0962 (slight overfit).
#   With 700 harder examples added, the model has more to learn — 3 epochs
#   should land in the healthy 0.15-0.35 range.
#
# SETUP:
#   Kaggle dataset "text-to-sql-data" must contain:
#     train.jsonl                        (original 1,023 — already uploaded)
#     layer3_hard_negatives.jsonl        (700 hard negatives — upload this)
#     data/raw/schemas/*.json            (10 schema files — upload these)
#   Enable GPU T4 x2


# CELL 1 — Install packages

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


# CELL 2 — Imports and config

# %%
import json
import random
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

DATA_DIR    = Path("/kaggle/input/text-to-sql-data")
OUTPUT_DIR  = Path("/kaggle/working")
ADAPTER_DIR = OUTPUT_DIR / "qwen25_sql_v2_adapter"

TRAIN_FILE    = DATA_DIR / "train.jsonl"
HARD_NEG_FILE = DATA_DIR / "layer3_hard_negatives.jsonl"

MODEL_ID       = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 2048

print(f"train.jsonl:            {TRAIN_FILE.exists()}")
print(f"layer3_hard_negatives:  {HARD_NEG_FILE.exists()}")
print(f"CUDA available:         {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:                    {torch.cuda.get_device_name(0)}")
    print(f"VRAM:                   {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# CELL 3 — Load base model + fresh LoRA
# Runtime: ~5-7 minutes

# %%
print(f"Loading {MODEL_ID} in 4-bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

vram_used = torch.cuda.memory_allocated() / 1e9
print(f"Model loaded. VRAM: {vram_used:.1f} GB")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")


# CELL 4 — Build combined dataset (original + hard negatives)

# %%
def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


SYSTEM_PROMPT = """You are an expert SQL engineer. Given a database schema and a natural language question, write a correct and efficient SQL query.

Rules:
- Output ONLY the SQL query, no explanation, no markdown fences
- Use table aliases for readability on multi-join queries
- Use explicit JOIN syntax, never implicit comma joins
- Prefer CTEs over deeply nested subqueries when it improves readability
- Never use SELECT * -- always specify column names"""

original = load_jsonl(TRAIN_FILE)
print(f"Original examples: {len(original)}")

schemas: dict[str, dict] = {}
for f in DATA_DIR.glob("*.json"):
    try:
        with open(f) as fh:
            s = json.load(fh)
        if "name" in s and "create_sql" in s:
            schemas[s["name"]] = s
    except Exception:
        pass
print(f"Schemas loaded: {len(schemas)}")

raw_hard_negs = load_jsonl(HARD_NEG_FILE)
hard_negs = []
skipped = 0
for ex in raw_hard_negs:
    schema = schemas.get(ex["schema_name"])
    if not schema:
        skipped += 1
        continue
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": f"Schema:\n{schema['create_sql'].strip()}\n\nQuestion: {ex['question'].strip()}\n\nSQL:"},
        {"role": "assistant", "content": ex["sql"]},
    ]
    hard_negs.append({**ex, "messages": messages})

print(f"Hard negatives formatted: {len(hard_negs)} ({skipped} skipped — missing schema)")

combined = original + hard_negs
random.seed(42)
random.shuffle(combined)

counts = Counter(ex["complexity"] for ex in combined)
print(f"\nCombined: {len(combined)} examples")
print(f"  easy {counts['easy']} | medium {counts['medium']} | hard {counts['hard']}")


# CELL 5 — Apply chat template and build HuggingFace Dataset

# %%
def apply_chat_template(examples: dict) -> dict:
    texts = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}


dataset = Dataset.from_list(combined)
dataset = dataset.map(apply_chat_template, batched=True, remove_columns=dataset.column_names)
print(f"Dataset ready: {len(dataset)} examples | sample length: {len(dataset[0]['text'])} chars")


# CELL 6 — Configure SFTTrainer
# lr=2e-4 and 3 epochs — same as Day 6, now with better data.

# %%
steps_per_epoch = len(dataset) // (2 * 4)   # batch_size=2, grad_accum=4 → effective=8
total_steps     = steps_per_epoch * 3
print(f"Steps per epoch: {steps_per_epoch}  |  Total steps: {total_steps}")
print(f"Estimated time:  {total_steps * 4 / 60:.0f}-{total_steps * 6 / 60:.0f} minutes")

training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR / "checkpoints_v2"),
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,           # effective batch = 8
    warmup_steps=20,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=20,
    save_strategy="steps",
    save_steps=100,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    report_to="none",
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


# CELL 7 — Train
# Expected: ~90-120 minutes on T4.
# Target loss: 0.15-0.35
#   < 0.10 → overfitting (same as Day 6) — reduce epochs to 2 next time
#   > 0.50 → underfitting — add an epoch

# %%
print("Starting Day 12 training...\n")
trainer_stats = trainer.train()

print(f"\nTraining complete.")
print(f"  Runtime:     {trainer_stats.metrics['train_runtime']:.0f}s "
      f"({trainer_stats.metrics['train_runtime']/60:.1f} min)")
print(f"  Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
print(f"  Final loss:  {trainer_stats.metrics['train_loss']:.4f}")
print(f"  Peak VRAM:   {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
print()
print("Healthy target: loss 0.15-0.35")
print("If loss < 0.10: overfitting — try 2 epochs next run")
print("If loss > 0.50: underfitting — add an epoch")


# CELL 8 — Save adapter (v2)

# %%
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))

total_mb = sum(f.stat().st_size for f in ADAPTER_DIR.iterdir()) / 1e6
print(f"Adapter v2 saved: {ADAPTER_DIR.name}/  ({total_mb:.1f} MB)")
print("Download from Kaggle Output tab -> save locally to data/models/qwen25_sql_v2/")


# CELL 9 — Quick sanity check on a hard example
# Tests a window-function query — the #1 failure pattern (70% fail rate in v1).

# %%
FastLanguageModel.for_inference(model)
model.generation_config.max_length = None


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


test_ex = next((ex for ex in hard_negs if ex.get("complexity") == "hard"), combined[0])
user_content = test_ex["messages"][1]["content"]
schema_part  = user_content.split("\n\nQuestion:")[0].replace("Schema:\n", "")
question     = user_content.split("Question:")[1].split("\n\nSQL:")[0].strip()

gen_sql = infer(schema_part, question)
print(f"Hard example sanity check:")
print(f"  Q:   {question[:80]}")
print(f"  Gen: {gen_sql[:200]}")
print(f"  Ref: {test_ex['sql'][:200]}")
print()
print("Next: Run day10_post_training_eval.py (ADAPTER_DIR = qwen25_sql_v2_adapter)")
print("Expected: +15-25pp on medium/hard vs v1 baseline (51.5% overall)")
