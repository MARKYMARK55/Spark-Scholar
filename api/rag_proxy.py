"""
api/rag_proxy.py
================
FastAPI OpenAI-compatible proxy that intercepts chat completions and runs
the full LangGraph RAG pipeline before generating a response.

Endpoints
---------
POST /v1/chat/completions  — RAG-augmented chat (streaming and non-streaming)
GET  /v1/models            — Returns available models (OpenAI-compatible)
GET  /health               — Health check for Docker healthchecks
GET  /                     — Redirect to docs

Usage
-----
# Start the server:
uvicorn api.rag_proxy:app --host 0.0.0.0 --port 8002 --workers 1

# Or via Docker (see api/Dockerfile):
docker run -p 8002:8002 --env-file env/.env arxiv-rag-proxy
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Ensure pipeline package is importable when running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"), override=False)

from pipeline.cache import get_cache
from pipeline.langgraph_pipeline import run_pipeline, VLLM_MODEL
from pipeline.router import get_all_collections

logger = logging.getLogger(__name__)

LITELLM_URL = os.environ.get("LITELLM_URL", "http://localhost:4000")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "simple-api-key")
RAG_PROXY_PORT = int(os.environ.get("RAG_PROXY_PORT", "8002"))

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = VLLM_MODEL
    messages: list[ChatMessage]
    temperature: float = 0.1
    max_tokens: int = 2048
    stream: bool = False
    # RAG-specific extensions (OpenAI-compatible extra fields)
    year_min: Optional[int] = Field(default=None, description="Filter papers by minimum year")
    year_max: Optional[int] = Field(default=None, description="Filter papers by maximum year")
    author_filter: Optional[str] = Field(default=None, description="Filter by author substring")


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "arxiv-rag"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the cache connection and log startup."""
    logger.info("arxiv-rag proxy starting up")
    cache = get_cache()
    logger.info("Redis available: %s", cache.is_available())
    yield
    logger.info("arxiv-rag proxy shutting down")
    cache.flush_namespace()


app = FastAPI(
    title="Arxiv RAG Proxy",
    description="OpenAI-compatible proxy with arXiv retrieval-augmented generation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: extract query from messages
# ---------------------------------------------------------------------------

def _extract_query(messages: list[ChatMessage]) -> str:
    """
    Extract the primary user query from the messages list.
    Uses the last user message as the query.
    """
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content.strip()
    # Fallback: use the last message regardless of role
    return messages[-1].content.strip() if messages else ""


def _build_openai_response(
    request_id: str,
    model: str,
    content: str,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    sources: Optional[list[dict]] = None,
) -> dict:
    """Build a well-formed OpenAI chat completion response dict."""
    created = int(time.time())

    # Append sources as a footnote if available
    full_content = content
    if sources:
        source_lines = []
        for i, src in enumerate(sources[:5], 1):
            arxiv_id = src.get("arxiv_id", "")
            title = src.get("title", "")
            year = src.get("year", "")
            if arxiv_id:
                source_lines.append(f"[{i}] ArXiv:{arxiv_id} — {title} ({year})")
            elif src.get("source_file"):
                source_lines.append(f"[{i}] {src.get('source_file')} — {title}")
        if source_lines:
            full_content = content + "\n\n**Sources:**\n" + "\n".join(source_lines)

    response = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
        },
    }
    return response


async def _stream_rag_response(
    request_id: str,
    model: str,
    content: str,
    sources: Optional[list[dict]] = None,
) -> AsyncIterator[str]:
    """
    Stream the RAG response as SSE chunks.

    Since we have the full response from the pipeline already, we simulate
    streaming by yielding tokens. For true streaming, you would hook into
    the LLM stream before pipeline completion.
    """
    created = int(time.time())

    # Yield the content in chunks of ~20 characters (simulates streaming)
    # In a production setup, the llm_inference node would yield tokens directly
    full_content = content
    if sources:
        source_lines = []
        for i, src in enumerate(sources[:5], 1):
            arxiv_id = src.get("arxiv_id", "")
            title = src.get("title", "")
            year = src.get("year", "")
            if arxiv_id:
                source_lines.append(f"[{i}] ArXiv:{arxiv_id} — {title} ({year})")
        if source_lines:
            full_content = content + "\n\n**Sources:**\n" + "\n".join(source_lines)

    chunk_size = 20
    for i in range(0, len(full_content), chunk_size):
        token = full_content[i : i + chunk_size]
        chunk = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        # Small delay to simulate token-by-token generation
        await asyncio.sleep(0.001)

    # Final chunk with finish_reason
    final_chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "Arxiv RAG Proxy", "docs": "/docs", "health": "/health"}


@app.get("/health")
async def health():
    """Health check endpoint for Docker and load balancers."""
    cache = get_cache()
    return {
        "status": "ok",
        "redis": "connected" if cache.is_available() else "disconnected",
        "version": "1.0.0",
    }


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """
    Return available models in OpenAI-compatible format.
    Open WebUI uses this to populate the model dropdown.
    """
    models = [
        ModelInfo(id="arxiv-rag", owned_by="arxiv-rag"),
        ModelInfo(id=VLLM_MODEL, owned_by="vllm"),
    ]
    return ModelsResponse(data=models)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """
    OpenAI-compatible chat completions endpoint with RAG augmentation.

    The pipeline:
    1. Extract query from the last user message
    2. Run the LangGraph RAG pipeline (retrieval + reranking + generation)
    3. Return response in OpenAI format (streaming or non-streaming)

    Open WebUI compatibility:
    - Preserves all OpenAI response fields
    - Supports streaming via SSE
    - Model name is passed through unchanged
    """
    request_id = str(uuid.uuid4()).replace("-", "")[:16]

    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    query = _extract_query(request.messages)
    if not query:
        raise HTTPException(status_code=400, detail="No user message found in messages")

    logger.info(
        "chat_completions: request_id=%s model=%s query=%r stream=%s",
        request_id,
        request.model,
        query[:80],
        request.stream,
    )

    try:
        # Run the full RAG pipeline
        final_state = await run_pipeline(
            query=query,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream,
            year_min=request.year_min,
            year_max=request.year_max,
            author_filter=request.author_filter,
        )

        response_text = final_state.get("response", "")
        reranked = final_state.get("reranked", [])
        prompt_tokens = final_state.get("prompt_tokens")
        completion_tokens = final_state.get("completion_tokens")

        if request.stream:
            return StreamingResponse(
                _stream_rag_response(
                    request_id=request_id,
                    model=request.model,
                    content=response_text,
                    sources=reranked,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Request-Id": request_id,
                    "X-RAG-Cached": str(final_state.get("cached", False)).lower(),
                    "X-RAG-Sources": str(len(reranked)),
                },
            )
        else:
            resp = _build_openai_response(
                request_id=request_id,
                model=request.model,
                content=response_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                sources=reranked,
            )
            return JSONResponse(
                content=resp,
                headers={
                    "X-Request-Id": request_id,
                    "X-RAG-Cached": str(final_state.get("cached", False)).lower(),
                    "X-RAG-Sources": str(len(reranked)),
                    "X-RAG-Latency-Ms": str(int(final_state.get("latency_ms", 0))),
                },
            )

    except Exception as exc:
        logger.exception("chat_completions: unhandled error for request %s: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Additional utility endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/collections")
async def list_collections():
    """List all Qdrant collections (informational endpoint)."""
    return {"collections": get_all_collections()}


@app.delete("/v1/cache")
async def clear_cache():
    """Clear the Redis RAG cache (admin operation)."""
    cache = get_cache()
    count = cache.flush_namespace()
    return {"deleted_keys": count}


@app.get("/v1/cache/stats")
async def cache_stats():
    """Return Redis cache statistics."""
    cache = get_cache()
    return cache.stats()


# ---------------------------------------------------------------------------
# Direct run entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    host = os.environ.get("RAG_PROXY_HOST", "0.0.0.0")
    port = RAG_PROXY_PORT

    logger.info("Starting arxiv-rag proxy on %s:%d", host, port)
    uvicorn.run(
        "api.rag_proxy:app",
        host=host,
        port=port,
        workers=1,  # Single worker to share the compiled graph
        reload=False,
        log_level="info",
    )
