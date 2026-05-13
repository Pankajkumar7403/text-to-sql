# Text-to-SQL (QLoRA on Qwen2.5-7B-Instruct)

Fine-tuned **natural language → SQL** across ten synthetic business schemas (fintech + e-commerce),
with **execution-accuracy evaluation** in DuckDB and a **QLoRA adapter** on the Hugging Face Hub.

**Live demo (Gradio):** [Hugging Face Space](https://huggingface.co/spaces/pankaj74/text-to-sql)  
**Adapter weights:** [pankaj74/qwen25-sql-v2](https://huggingface.co/pankaj74/qwen25-sql-v2)

> Do **not** copy [`hf-space-deploy/README.md`](hf-space-deploy/README.md) here as-is: that file includes **YAML frontmatter** (`sdk:`, `app_file:`, etc.) that Hugging Face Spaces uses for build config. The **root** `README.md` is for GitHub and collaborators; the deploy folder keeps its own Space-specific readme.

## Highlights

- Custom dataset pipeline (synthetic + hard negatives), stratified eval, DuckDB sandbox validation  
- QLoRA fine-tuning on **Qwen2.5-7B-Instruct**; reported gains on medium/hard splits vs baseline  
- **Interactive demo** uses the **Groq API** for live SQL generation on free-tier CPU RAM; the Space README documents eval numbers and links to the actual fine-tuned adapter  

## Repository layout

| Path | Purpose |
|------|---------|
| [`app/`](app/) | Gradio demo, Groq inference, FastAPI, schema loader |
| [`scripts/`](scripts/) | Dataset build, eval harness, schema generation, Hub push helpers |
| [`notebooks/`](notebooks/) | Training / eval notebooks (e.g. QLoRA, post-training eval) |
| [`data/`](data/) | Raw schemas, processed JSONL, eval artifacts (large files may be gitignored locally) |
| [`hf-space-deploy/`](hf-space-deploy/) | **Git submodule** — deploy bundle for the Hugging Face Space (same remote as the live demo). Use `git submodule update --init --recursive` after cloning. |

### Hugging Face Space workflow

`hf-space-deploy` points at `https://huggingface.co/spaces/pankaj74/text-to-sql.git`. Edit there and push like any repo:

```bash
cd hf-space-deploy
git pull
git add -A && git commit -m "deploy: …" && git push origin main
```

Then bump the submodule pointer in this repo so GitHub records the new Space commit:

```bash
cd ..
git add hf-space-deploy
git commit -m "chore: bump hf-space-deploy submodule"
```

Fresh clone of **this** repo must include submodules:

```bash
git clone --recurse-submodules <url-to-this-repo>
# or:  git submodule update --init --recursive
```

## Local setup

```bash
pip install -r requirements.txt
set GROQ_API_KEY=your_key   # Windows; use export on Linux/macOS
python app.py               # Gradio UI (same entry pattern as the Space)
```

API (optional):

```bash
uvicorn app.api:app --reload --port 8000
```

Regenerate schema JSON under `data/raw/schemas/` when needed:

```bash
python scripts/generate_schemas.py
```

## Evaluation snapshot

Held-out golden set (68 questions), execution accuracy vs DuckDB:

| Split | Fine-tuned (v2) | vs baseline |
|-------|-----------------|-------------|
| Medium | 50.0% | +11.5pp |
| Hard | 42.9% | +9.6pp |
| Overall | 55.9% | +7.4pp |

Full tables and methodology live in [`hf-space-deploy/README.md`](hf-space-deploy/README.md) and the **Fine-tuned Results** tab in the Space.

## License / attribution

Model base: **Qwen2.5-7B-Instruct** (see model card on the Hub). Training data mixes open benchmarks (filtered) and synthetic examples generated for this project.
