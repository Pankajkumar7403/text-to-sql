"""
Inference engine — calls Groq API (llama-3.3-70b-versatile) for SQL generation.
No local model weights loaded; runs fine on CPU-only HF Spaces free tier.

Set GROQ_API_KEY as an environment variable / HF Space secret.
"""

import os
import time
from groq import Groq

_client: Groq | None = None

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are an expert SQL engineer. Given a database schema and a natural language "
    "question, write a correct and efficient SQL query.\n\n"
    "Rules:\n"
    "- Output ONLY the SQL query, no explanation, no markdown fences\n"
    "- Use table aliases for readability on multi-join queries\n"
    "- Use explicit JOIN syntax, never implicit comma joins\n"
    "- Prefer CTEs over deeply nested subqueries when it improves readability\n"
    "- Never use SELECT * — always specify column names"
)


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable not set.")
        _client = Groq(api_key=api_key)
    return _client


def load_model():
    """No-op — kept so demo.py import doesn't change."""
    _get_client()
    print("[model] Groq client ready.")


def generate_sql(schema_sql: str, question: str, max_new_tokens: int = 256) -> tuple[str, float]:
    """Return (sql_string, latency_seconds)."""
    client = _get_client()

    t0 = time.time()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Schema:\n{schema_sql.strip()}\n\nQuestion: {question.strip()}\n\nSQL:"},
        ],
        max_tokens=max_new_tokens,
        temperature=0,
    )
    latency = time.time() - t0

    sql = response.choices[0].message.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql, round(latency, 2)
