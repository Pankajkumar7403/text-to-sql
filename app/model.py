"""
Inference engine — calls Groq API (llama-3.3-70b-versatile) for SQL generation.
No local model weights loaded; runs fine on CPU-only HF Spaces free tier.

Set GROQ_API_KEY as an environment variable / HF Space secret.
"""

from __future__ import annotations

import logging
import os
import time

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)
from groq import BadRequestError

from app.errors import InferenceUserError

logger = logging.getLogger(__name__)

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

_MSG_AUTH = "Inference API authentication failed. Check server configuration."
_MSG_RATE = "Rate limited; try again in a moment."
_MSG_UNAVAILABLE = "Inference service temporarily unavailable; try again later."
_MSG_BAD_OUTPUT = "Model returned no usable SQL; retry or shorten the question."
_MSG_BAD_REQUEST = "Request rejected by inference API; try rephrasing the question."


def _map_groq_exception(exc: BaseException) -> InferenceUserError:
    """Map Groq/transport errors to user-safe messages; log without leaking secrets."""
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        logger.warning("Groq auth failure: %s", type(exc).__name__)
        return InferenceUserError(_MSG_AUTH)
    if isinstance(exc, RateLimitError):
        logger.warning("Groq rate limit: %s", type(exc).__name__)
        return InferenceUserError(_MSG_RATE)
    if isinstance(exc, (APITimeoutError, APIConnectionError, InternalServerError)):
        logger.warning("Groq connectivity/server: %s", type(exc).__name__)
        return InferenceUserError(_MSG_UNAVAILABLE)
    if isinstance(exc, BadRequestError):
        logger.warning("Groq bad request: %s", type(exc).__name__)
        return InferenceUserError(_MSG_BAD_REQUEST)
    if isinstance(exc, APIStatusError):
        code = getattr(exc.response, "status_code", None)
        logger.warning("Groq APIStatusError status=%s", code)
        if code == 429:
            return InferenceUserError(_MSG_RATE)
        if code is not None and code in (401, 403):
            return InferenceUserError(_MSG_AUTH)
        if code is not None and code >= 500:
            return InferenceUserError(_MSG_UNAVAILABLE)
        return InferenceUserError(_MSG_UNAVAILABLE)
    logger.exception("Unexpected error during Groq inference")
    return InferenceUserError(_MSG_UNAVAILABLE)


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable not set.")
        _client = Groq(api_key=api_key)
    return _client


def load_model() -> None:
    """Validate Groq client config at startup (raises if key missing)."""
    _get_client()
    logger.info("Groq client ready.")


def generate_sql(schema_sql: str, question: str, max_new_tokens: int = 256) -> tuple[str, float]:
    """Return (sql_string, latency_seconds). Raises InferenceUserError on failure."""
    client = _get_client()

    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Schema:\n{schema_sql.strip()}\n\nQuestion: {question.strip()}\n\nSQL:"
                    ),
                },
            ],
            max_tokens=max_new_tokens,
            temperature=0,
        )
        latency = time.time() - t0
    except Exception as exc:
        raise _map_groq_exception(exc) from exc

    try:
        raw = response.choices[0].message.content
        if not raw:
            raise InferenceUserError(_MSG_BAD_OUTPUT)
        sql = raw.strip().replace("```sql", "").replace("```", "").strip()
        if not sql:
            raise InferenceUserError(_MSG_BAD_OUTPUT)
    except InferenceUserError:
        raise
    except (AttributeError, IndexError, TypeError) as exc:
        logger.warning("Malformed Groq response structure: %s", type(exc).__name__)
        raise InferenceUserError(_MSG_BAD_OUTPUT) from exc

    return sql, round(latency, 2)
