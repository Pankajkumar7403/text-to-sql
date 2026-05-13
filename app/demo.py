"""
Gradio demo — two tabs:
  1. Live SQL generator  (Groq API — fast, no GPU needed)
  2. Fine-tuned results  (shows actual eval numbers from the trained model)

Run locally:  python app/demo.py
HF Spaces:    entry via app.py at repo root
"""

import logging

import duckdb
import gradio as gr

from app.errors import InferenceUserError
from app.model import generate_sql, load_model
from app.schemas_loader import load_all_schemas

logger = logging.getLogger(__name__)

schemas = load_all_schemas()
NAMES   = sorted(schemas.keys())
load_model()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _schema_markdown(name: str) -> str:
    s = schemas.get(name, {})
    lines = [f"**{s.get('description', name)}**  ·  domain: `{s.get('domain', '—')}`\n"]
    for tname, tinfo in s.get("tables", {}).items():
        cols = [c.split(" FK(")[0].replace(" PK", "").split()[0]
                for c in tinfo.get("columns", [])]
        lines.append(f"- **{tname}**: {', '.join(cols)}")
    return "\n".join(lines)


def run(schema_name: str, question: str) -> tuple[str, str]:
    if not question.strip():
        return "", "Enter a question first."
    schema = schemas.get(schema_name)
    if not schema:
        return "", f"Schema '{schema_name}' not found."

    try:
        sql, latency = generate_sql(schema["create_sql"], question)
    except InferenceUserError as e:
        return "", e.message

    try:
        conn = duckdb.connect(":memory:")
        conn.execute(schema["create_sql"])
        conn.execute(sql)
        conn.close()
        status = f"Valid SQL  ·  {latency:.1f}s"
    except Exception as e:
        logger.debug("DuckDB validation failed: %s", e)
        err_one = (str(e).splitlines() or [str(e)])[0][:200]
        status = f"SQL did not run in sandbox: {err_one}  ·  {latency:.1f}s"

    return sql, status


# ── Results tab data ───────────────────────────────────────────────────────────

RESULTS_MD = """
## Fine-tuned Model Evaluation Results

Model: **Qwen2.5-7B-Instruct + QLoRA** (r=16, lora_alpha=16)
Training data: 1,700+ examples across 10 business domains (synthetic + hard negatives)
Eval set: 68 held-out golden questions — **never seen during training**
Metric: **Execution accuracy** via DuckDB sandbox

| Complexity | N  | Pre-training baseline | Fine-tuned (v2) | Delta      |
|------------|----|-----------------------|-----------------|------------|
| Easy       | 21 | 81.0%                 | 76.2%           | −4.8pp     |
| Medium     | 26 | 34.6%                 | 50.0%           | **+15.4pp**|
| Hard       | 21 | 33.3%                 | 42.9%           | **+9.6pp** |
| **Overall**| 68 | 48.5%                 | **55.9%**       | **+7.4pp** |

### What the training fixed

The base model already handled simple single-table SELECT queries well.
Fine-tuning on domain-specific hard examples improved:

- **Window functions** (RANK, ROW_NUMBER, LAG) — model now uses them correctly instead of subqueries
- **Multi-table joins with aggregation** — CTEs instead of deeply nested subqueries
- **Business filter patterns** — `IN ('active', 'verified')` multi-value filters learned from domain data

### Training pipeline

```
Spider/BIRD data (filtered)
       ↓
Synthetic pairs via Groq llama-3.3-70b (1,160 examples, DuckDB-validated)
       ↓
Hard negatives — targeted SQL patterns (700 examples)
       ↓
QLoRA fine-tuning on Kaggle T4 (Qwen2.5-7B-Instruct, 3 epochs, lr=2e-4)
       ↓
Execution-accuracy eval on 68 held-out golden questions
```

### Sample outputs (fine-tuned model)

**Q**: "List top 5 customers by total transaction volume with their account balance"
```sql
SELECT c.full_name, SUM(t.amount) AS total_volume, a.balance
FROM customers c
JOIN accounts a ON c.customer_id = a.customer_id
JOIN transactions t ON a.account_id = t.account_id
WHERE t.status = 'completed'
GROUP BY c.full_name, a.balance
ORDER BY total_volume DESC
LIMIT 5
```

**Q**: "Monthly revenue trend for active subscriptions in the last 6 months"
```sql
WITH monthly AS (
    SELECT DATE_TRUNC('month', payment_date) AS month,
           SUM(amount) AS revenue
    FROM payments p
    JOIN subscriptions s ON p.subscription_id = s.subscription_id
    WHERE s.status = 'active'
      AND payment_date >= CURRENT_DATE - INTERVAL '6 months'
    GROUP BY 1
)
SELECT month, revenue
FROM monthly
ORDER BY month
```

---
*Live demo tab uses Groq API (same prompting format) due to free-tier compute constraints.*
*The fine-tuned adapter weights are available at [pankaj74/qwen25-sql-v2](https://huggingface.co/pankaj74/qwen25-sql-v2).*
"""

# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Text-to-SQL | Qwen2.5-7B") as demo:
    gr.Markdown(
        "## Text-to-SQL — Qwen2.5-7B QLoRA\n"
        "Fine-tuned on 10 business domains. **Live Demo** calls the Groq API (no 7B weights in "
        "this container); **Fine-tuned Results** + Hub adapter document the trained model.\n\n"
        "Select a schema, type a question, get SQL."
    )

    with gr.Tabs():

        # ── Tab 1: Live demo ──────────────────────────────────────────────────
        with gr.Tab("Live Demo"):
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    schema_dd = gr.Dropdown(
                        choices=NAMES, value=NAMES[0],
                        label="Schema", interactive=True,
                    )
                    schema_md = gr.Markdown(_schema_markdown(NAMES[0]))
                    schema_dd.change(_schema_markdown, schema_dd, schema_md)

                with gr.Column(scale=2):
                    question_tb = gr.Textbox(
                        label="Question",
                        placeholder="e.g.  How many customers are in each risk tier?",
                        lines=2,
                    )
                    with gr.Row():
                        run_btn   = gr.Button("Generate SQL", variant="primary")
                        clear_btn = gr.Button("Clear")

                    sql_out    = gr.Code(label="Generated SQL", language="sql")
                    status_out = gr.Textbox(label="Status", interactive=False, max_lines=2)

                    run_btn.click(run, [schema_dd, question_tb], [sql_out, status_out])
                    question_tb.submit(run, [schema_dd, question_tb], [sql_out, status_out])
                    clear_btn.click(lambda: ("", "", ""),
                                    outputs=[question_tb, sql_out, status_out])

            gr.Examples(
                label="Try these",
                examples=[
                    ["neobank_v1",          "How many customers are verified vs pending KYC?"],
                    ["neobank_v1",          "Top 5 customers by total transaction amount"],
                    ["payments_v1",         "Total payment volume by payment method this month"],
                    ["lending_v1",          "Average loan amount by loan type"],
                    ["saas_metrics_v1",     "Monthly recurring revenue by plan tier"],
                    ["retail_analytics_v1", "Top 10 products by revenue in the last 90 days"],
                    ["insurance_v1",        "Count of open claims by policy type"],
                    ["investment_v1",       "Portfolios with more than 5 holdings"],
                    ["marketplace_v1",      "Sellers with average rating above 4.5"],
                    ["subscription_ecom_v1","Churn rate by subscription plan"],
                ],
                inputs=[schema_dd, question_tb],
            )

        # ── Tab 2: Fine-tuned results ─────────────────────────────────────────
        with gr.Tab("Fine-tuned Results"):
            gr.Markdown(RESULTS_MD)

    gr.Markdown(
        "---\n"
        "Adapter: [pankaj74/qwen25-sql-v2](https://huggingface.co/pankaj74/qwen25-sql-v2) · "
        "Base: Qwen2.5-7B-Instruct · QLoRA r=16"
    )


if __name__ == "__main__":
    demo.launch(share=False)
