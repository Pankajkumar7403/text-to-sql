— QLoRA Smoke Test Training (Unsloth + TRL SFTTrainer)
# Run this on Kaggle with T4 GPU enabled.
#
# GOAL: Verify the full training pipeline runs without crashing.
#   100 examples, 1 epoch, ~10-15 minutes on T4.
#   If Cell 9 (inference) generates valid SQL -> pipeline is confirmed.
#   Day 8 does the real run: all 1,023 examples, 3 epochs, W&B tracking.
#
# SETUP (before running):
#   1. Edit your Kaggle dataset "text-to-sql-data"
#      Add: data/processed/train.jsonl  (upload from your local project)
#   2. Re-attach the dataset to this notebook
#   3. Enable GPU: Settings -> Accelerator -> GPU T4 x2
#
# WHY GIT INSTALL FOR UNSLOTH:
#   The PyPI release has a bug where it calls .mean() on num_items_in_batch
#   which is now passed as int by newer transformers -> AttributeError.
#   The git HEAD has this fixed. Installing from git gets the patched version.
#
# WHAT QLORA DOES:
#   - Base model frozen and quantized to 4-bit (not updated)
#   - LoRA adapters (rank 16) added to all linear projection layers
#   - Only ~42M params out of 7.6B are trained
#   - Unsloth: 2x faster training, 60% less VRAM than vanilla PEFT


# CELL 1 — Install packages
# Runtime: ~3-4 minutes
# Install Unsloth from git HEAD — has the fix for the 'int has no .mean()' crash
# that affects the PyPI release on Kaggle's transformers version.

# %%
import subprocess

# Step 1: Stable PyPI unsloth — pulls a compatible unsloth_zoo version.
#   Kaggle pre-installs unsloth_zoo as a system package; its version is pinned
#   to match the PyPI unsloth release, not git HEAD. Installing from PyPI first
#   ensures unsloth_zoo is at the right version.
subprocess.run(["pip", "install", "--quiet", "unsloth"], check=True)

# Step 2: Upgrade ONLY unsloth from git with --no-deps.
#   Gets the bug-fix for 'int has no .mean()' without touching unsloth_zoo.
#   --no-deps is critical: without it, pip would try to re-resolve unsloth_zoo
#   from git and break the version match we just established.
subprocess.run([
    "pip", "install", "--quiet", "--upgrade", "--no-cache-dir", "--no-deps",
    "git+https://github.com/unslothai/unsloth.git",
], check=True)

# Step 3: Training stack
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
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig   # SFTConfig required in TRL 0.12+
from unsloth import FastLanguageModel

# Kaggle paths
DATA_DIR    = Path("/kaggle/input/text-to-sql-data")
TRAIN_FILE  = DATA_DIR / "train.jsonl"
OUTPUT_DIR  = Path("/kaggle/working")
ADAPTER_DIR = OUTPUT_DIR / "qwen25_sql_smoke_adapter"

MODEL_ID       = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 2048   # covers all schemas + SQL (~900-1400 tokens worst case)
SMOKE_N        = 100    # smoke test: 100 examples, 1 epoch

print(f"Train file exists: {TRAIN_FILE.exists()}")
print(f"CUDA available:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:               {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM:              {vram:.1f} GB")


# CELL 3 — Load model with Unsloth (4-bit)
# Runtime: ~4-6 minutes (downloads ~15 GB)
# Memory:  ~4.5 GB VRAM — leaves ~10 GB for activations + gradients

# %%
print(f"Loading {MODEL_ID} with Unsloth 4-bit...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,        # auto-detect: fp16 on T4, bf16 on A100
    load_in_4bit=True,
)

vram_used = torch.cuda.memory_allocated() / 1e9
print(f"Model loaded. VRAM used: {vram_used:.1f} GB")


# CELL 4 — Apply QLoRA adapters

# %%
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",    # attention projections
        "gate_proj", "up_proj", "down_proj",         # SwiGLU FFN
    ],
    lora_dropout=0,                          # 0 is faster; works as well with QLoRA
    bias="none",
    use_gradient_checkpointing="unsloth",    # Unsloth custom checkpointing: saves 30% VRAM
    random_state=42,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
# Expected: ~42M / 7.6B = ~0.55%


# CELL 5 — Load dataset (100 stratified examples)

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
    """
    Convert messages list to a single string for SFTTrainer.
    add_generation_prompt=False: training data already has the assistant (SQL) answer.
    """
    texts = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}


all_examples = load_jsonl(TRAIN_FILE)
print(f"Total training examples available: {len(all_examples)}")

# Stratified sample: preserve easy/medium/hard ratio in the smoke set
random.seed(42)
by_complexity: dict[str, list] = {"easy": [], "medium": [], "hard": []}
for ex in all_examples:
    by_complexity[ex["complexity"]].append(ex)

smoke_examples = []
for complexity, group in by_complexity.items():
    n = round(SMOKE_N * len(group) / len(all_examples))
    smoke_examples.extend(random.sample(group, min(n, len(group))))

random.shuffle(smoke_examples)
smoke_examples = smoke_examples[:SMOKE_N]

complexity_counts = {c: sum(1 for e in smoke_examples if e["complexity"] == c)
                     for c in ["easy", "medium", "hard"]}
print(f"Smoke set: {len(smoke_examples)} examples | {complexity_counts}")

dataset = Dataset.from_list(smoke_examples)
dataset = dataset.map(apply_chat_template, batched=True, remove_columns=dataset.column_names)
print(f"Dataset ready. Sample length: {len(dataset[0]['text'])} chars")


# CELL 6 — Configure SFTTrainer
# TRL 0.12+ requires SFTConfig instead of TrainingArguments.
# dataset_text_field and max_seq_length now live in SFTConfig, not SFTTrainer.__init__

# %%
training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,    # effective batch size = 8
    warmup_steps=5,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),   # T4 -> fp16
    bf16=torch.cuda.is_bf16_supported(),        # A100 -> bf16
    logging_steps=5,
    save_strategy="epoch",
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    report_to="none",                 # no W&B for smoke test; Day 8 adds it
    # SFT-specific params (moved out of SFTTrainer.__init__ in TRL 0.12+)
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,       # renamed from 'tokenizer' in TRL 0.12+
    train_dataset=dataset,
    args=training_args,
)

steps = len(dataset) // (training_args.per_device_train_batch_size *
                          training_args.gradient_accumulation_steps)
print(f"Steps per epoch:  {steps}")
print(f"Estimated time:   {steps * 4 / 60:.0f}-{steps * 6 / 60:.0f} minutes on T4 (Unsloth)")


# CELL 7 — Train
# Runtime: ~10-15 minutes for 100 examples on T4 with Unsloth
# Loss should drop from ~2.0 toward ~0.5 over the steps.

# %%
print("Starting smoke test training...\n")
trainer_stats = trainer.train()

print(f"\nTraining complete.")
print(f"  Runtime:     {trainer_stats.metrics['train_runtime']:.0f}s "
      f"({trainer_stats.metrics['train_runtime']/60:.1f} min)")
print(f"  Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
print(f"  Final loss:  {trainer_stats.metrics['train_loss']:.4f}")
print(f"  Peak VRAM:   {torch.cuda.max_memory_allocated()/1e9:.1f} GB")


# CELL 8 — Save the LoRA adapter
# Only saves adapter weights (~64 MB), not the 15 GB base model.

# %%
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))

adapter_files = list(ADAPTER_DIR.iterdir())
total_mb = sum(f.stat().st_size for f in adapter_files) / 1e6
print(f"Adapter saved: {ADAPTER_DIR.name}/")
print(f"Files: {[f.name for f in adapter_files]}")
print(f"Size:  {total_mb:.1f} MB")


# CELL 9 — Quick inference check
# Confirms the adapter generates SQL. Does NOT measure accuracy.

# %%
FastLanguageModel.for_inference(model)   # re-enables Unsloth's inference kernels

SYSTEM_PROMPT = """You are an expert SQL engineer. Given a database schema and a natural language question, write a correct and efficient SQL query.

Rules:
- Output ONLY the SQL query, no explanation, no markdown fences
- Use table aliases for readability on multi-join queries
- Use explicit JOIN syntax, never implicit comma joins
- Prefer CTEs over deeply nested subqueries when it improves readability
- Never use SELECT * -- always specify column names"""


def smoke_infer(schema_sql: str, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{schema_sql.strip()}\n\nQuestion: {question.strip()}\n\nSQL:"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][input_len:]
    sql = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return sql.replace("```sql", "").replace("```", "").strip()


print("Smoke inference (2 examples):\n")
for i, ex in enumerate(smoke_examples[:2], 1):
    user_content = ex["messages"][1]["content"]
    schema_part = user_content.split("\n\nQuestion:")[0].replace("Schema:\n", "")
    question    = user_content.split("Question:")[1].split("\n\nSQL:")[0].strip()

    gen_sql = smoke_infer(schema_part, question)
    print(f"[{i}] {question[:70]}")
    print(f"  Generated: {gen_sql[:120]}")
    print(f"  Reference: {ex['sql'][:120]}")
    print()

print("PASS: both Generated lines look like real SQL.")
print("FAIL: output is empty, garbled, or repeating text.")
print()
print("Day 8: set SMOKE_N=1023, num_train_epochs=3, report_to='wandb' for full run.")


# NOTES FOR DAY 8 (full training run)
#
# Changes from smoke test -> full run:
#   SMOKE_N (in Cell 5)      = 100  ->  remove; use all_examples directly
#   num_train_epochs         = 1    ->  3
#   report_to                = "none" -> "wandb"
#     (add WANDB_API_KEY to Kaggle Secrets: Settings -> Secrets)
#   save_strategy            = "epoch" -> "steps"; save_steps = 100
#   logging_steps            = 5    ->  20
#
# Expected full-run metrics on T4 with Unsloth:
#   Runtime:    60-90 minutes  (within 12-hour Kaggle session limit)
#   Final loss: 0.25-0.40
#   Peak VRAM:  8-11 GB
