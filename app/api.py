"""
FastAPI inference endpoint.

Endpoints:
  GET  /health          -> { status: "ok" }
  GET  /schemas         -> list of { name, domain, description, tables }
  GET  /schemas/{name}  -> single schema info
  POST /generate-sql    -> { sql, latency_s, valid_sql, error }

Run locally:
  uvicorn app.api:app --reload --port 8000
  Docs at: http://localhost:8000/docs
"""

import duckdb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.errors import InferenceUserError
from app.model import generate_sql
from app.schemas_loader import load_all_schemas

app = FastAPI(
    title="Text-to-SQL API",
    description=(
        "Natural language to SQL. On Hugging Face Spaces, live generation uses the Groq API "
        "so the demo fits CPU RAM; the QLoRA-fine-tuned adapter is published on the Hub."
    ),
    version="0.2.0",
)


class GenerateRequest(BaseModel):
    schema_name:    str = Field(..., example="neobank_v1")
    question:       str = Field(..., example="How many customers are in each risk tier?")
    max_new_tokens: int = Field(256, ge=32, le=512)


class GenerateResponse(BaseModel):
    sql:       str
    latency_s: float
    valid_sql: bool
    error:     str = ""


class SchemaInfo(BaseModel):
    name:        str
    domain:      str
    description: str
    tables:      list[str]


def _validate(create_sql: str, sql: str) -> tuple[bool, str]:
    try:
        conn = duckdb.connect(":memory:")
        conn.execute(create_sql)
        conn.execute(sql)
        conn.close()
        return True, ""
    except Exception as e:
        return False, str(e)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schemas", response_model=list[SchemaInfo])
def list_schemas():
    schemas = load_all_schemas()
    return [
        SchemaInfo(
            name=v["name"],
            domain=v.get("domain", ""),
            description=v.get("description", ""),
            tables=list(v.get("tables", {}).keys()),
        )
        for v in sorted(schemas.values(), key=lambda x: x["name"])
    ]


@app.get("/schemas/{name}", response_model=SchemaInfo)
def get_schema(name: str):
    schemas = load_all_schemas()
    s = schemas.get(name)
    if not s:
        raise HTTPException(status_code=404, detail=f"Schema '{name}' not found.")
    return SchemaInfo(
        name=s["name"],
        domain=s.get("domain", ""),
        description=s.get("description", ""),
        tables=list(s.get("tables", {}).keys()),
    )


@app.post("/generate-sql", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    schemas = load_all_schemas()
    schema  = schemas.get(req.schema_name)
    if not schema:
        raise HTTPException(status_code=404, detail=f"Schema '{req.schema_name}' not found.")

    try:
        sql, latency = generate_sql(schema["create_sql"], req.question, req.max_new_tokens)
    except InferenceUserError as e:
        raise HTTPException(status_code=503, detail=e.message) from e

    valid, error = _validate(schema["create_sql"], sql)

    return GenerateResponse(sql=sql, latency_s=latency, valid_sql=valid, error=error)
