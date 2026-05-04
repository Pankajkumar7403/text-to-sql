"""
Prompt formatting utilities.
Single source of truth — used by training, inference, and eval.
"""

SYSTEM_PROMPT = """You are an expert SQL engineer. Given a database schema and a natural language question, write a correct and efficient SQL query.

Rules:
- Output ONLY the SQL query, no explanation, no markdown fences
- Use table aliases for readability on multi-join queries
- Use explicit JOIN syntax, never implicit comma joins
- Prefer CTEs over deeply nested subqueries when it improves readability
- Never use SELECT * — always specify column names"""


def format_prompt(schema_sql: str, question: str) -> list[dict]:
    """
    Returns messages list in Qwen2.5 chat format.
    Used identically during training (as prompt+completion) and inference.
    """
    user_content = f"""Schema:
{schema_sql.strip()}

Question: {question.strip()}

SQL:"""

    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_content},
    ]


def format_training_example(schema_sql: str, question: str, sql: str) -> dict:
    """
    Full training example with assistant turn included.
    Output is a dict with 'messages' key — compatible with TRL SFTTrainer.
    """
    messages = format_prompt(schema_sql, question)
    messages.append({"role": "assistant", "content": sql.strip()})
    return {"messages": messages}


if __name__ == "__main__":
    # Smoke test
    example = format_training_example(
        schema_sql="CREATE TABLE orders (order_id INT, user_id INT, amount DECIMAL, status VARCHAR);\nCREATE TABLE users (user_id INT, name VARCHAR, country VARCHAR);",
        question="Find the total order amount for each country, only for delivered orders, ordered by total descending.",
        sql="SELECT u.country, SUM(o.amount) AS total_amount\nFROM orders o\nJOIN users u ON o.user_id = u.user_id\nWHERE o.status = 'delivered'\nGROUP BY u.country\nORDER BY total_amount DESC"
    )
    import json
    print(json.dumps(example, indent=2))