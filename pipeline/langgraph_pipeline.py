"""
pipeline/langgraph_pipeline.py
===============================
Full LangGraph orchestration for the Spark Scholar pipeline.

Graph structure
---------------
START → check_cache
check_cache → END                    (if cache hit)
check_cache → route_query            (on cache miss)
route_query → embed_query
embed_query → hybrid_retrieve
hybrid_retrieve → should_web_search  (conditional edge)
should_web_search → web_search       (if query is time-sensitive)
should_web_search → merge_results    (otherwise)
web_search → merge_results
merge_results → rerank_results
rerank_results → build_context
build_context → llm_inference
llm_inference → cache_result
cache_result → trace_result
trace_result → END

Each node modifies the RAGState TypedDict in place (returns a partial dict).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .cache import ResultCache, get_cache
from .embeddings import SparseVector, async_encode_both
from .hybrid_search import async_hybrid_search
from .reranker import async_rerank
from .router import route_query as _route_query
from .tracer import RAGTracer, get_tracer

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"), override=False)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LITELLM_URL = os.environ.get("LITELLM_URL", "http://localhost:4000")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "simple-api-key")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "simple-api-key")
VLLM_MODEL = os.environ.get("VLLM_MODEL_NAME", "local-model")
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8888")
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "10"))
RAG_RERANK_TOP_N = int(os.environ.get("RAG_RERANK_TOP_N", "50"))
CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))

SYSTEM_PROMPT = """You are a knowledgeable research assistant specialising in scientific literature.
You have access to a curated set of arXiv papers retrieved specifically for this question.

Guidelines:
- Base your answer primarily on the provided papers
- Cite papers using [ArXiv:XXXX.XXXXX] format inline
- If the papers don't fully answer the question, say so clearly
- Be precise and technical where appropriate
- Distinguish between what the papers say and your own reasoning
- For mathematical content, use LaTeX notation when helpful

Provided papers:
{context}"""

# Time-sensitive keywords that trigger web search
TIME_SENSITIVE_KEYWORDS = [
    "latest", "recent", "current", "today", "this year", "2024", "2025", "2026",
    "new", "just released", "announcement", "preprint",
]


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    """Full state for the Spark Scholar LangGraph pipeline."""
    # Input
    query: str
    model: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]
    stream: Optional[bool]
    # Filters
    year_min: Optional[int]
    year_max: Optional[int]
    author_filter: Optional[str]
    # Routing
    collections: list[str]
    # Embeddings
    dense_vec: Optional[list[float]]
    sparse_vec: Optional[dict]  # {"indices": [...], "values": [...]}
    # Retrieval
    candidates: list[dict]
    # Web search
    web_results: list[dict]
    # Reranking
    reranked: list[dict]
    # Generation
    context: str
    response: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    # Pipeline metadata
    cached: bool
    cache_key: Optional[str]
    trace_id: str
    error: Optional[str]
    latency_ms: Optional[float]
    _start_time: Optional[float]


def _initial_state(query: str, **kwargs) -> RAGState:
    """Create an initialised RAGState with defaults."""
    return RAGState(
        query=query,
        model=kwargs.get("model", VLLM_MODEL),
        temperature=kwargs.get("temperature", 0.1),
        max_tokens=kwargs.get("max_tokens", 2048),
        stream=kwargs.get("stream", False),
        year_min=kwargs.get("year_min"),
        year_max=kwargs.get("year_max"),
        author_filter=kwargs.get("author_filter"),
        collections=[],
        dense_vec=None,
        sparse_vec=None,
        candidates=[],
        web_results=[],
        reranked=[],
        context="",
        response="",
        prompt_tokens=None,
        completion_tokens=None,
        cached=False,
        cache_key=None,
        trace_id="",
        error=None,
        latency_ms=None,
        _start_time=time.monotonic(),
    )


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def node_check_cache(state: RAGState) -> dict:
    """
    Node 1: check_cache
    Check Redis for a cached response. If found, set cached=True and response.
    """
    cache: ResultCache = get_cache()
    tracer: RAGTracer = get_tracer()

    # Start the trace before anything else
    trace_id = tracer.start_trace(
        state["query"],
        metadata={"model": state.get("model"), "stream": state.get("stream")},
    )

    # We need at least the collections for a proper cache key; use a query-only key here
    # (full key with collections is set after routing — this is a best-effort pre-check)
    cache_key = cache.make_key(state["query"], ["__pre_route__"])

    cached_val = cache.get(cache_key)
    if cached_val is not None:
        tracer.log_cache_hit(trace_id, cache_key)
        logger.info("Cache HIT for query: %r", state["query"][:60])
        return {
            "cached": True,
            "response": cached_val.get("response", ""),
            "reranked": cached_val.get("reranked", []),
            "trace_id": trace_id,
            "cache_key": cache_key,
            "latency_ms": (time.monotonic() - (state.get("_start_time") or time.monotonic())) * 1000,
        }

    return {
        "cached": False,
        "trace_id": trace_id,
        "cache_key": cache_key,
    }


async def node_route_query(state: RAGState) -> dict:
    """
    Node 2: route_query
    Determine which Qdrant collections to search based on query content.
    """
    collections = _route_query(state["query"], max_collections=3)
    logger.info("route_query: %r → %s", state["query"][:60], collections)

    # Update cache key now that we know the collections
    cache: ResultCache = get_cache()
    cache_key = cache.make_key(state["query"], collections)

    # Check cache again with the proper key
    cached_val = cache.get(cache_key)
    if cached_val is not None:
        tracer = get_tracer()
        tracer.log_cache_hit(state["trace_id"], cache_key)
        return {
            "collections": collections,
            "cached": True,
            "response": cached_val.get("response", ""),
            "reranked": cached_val.get("reranked", []),
            "cache_key": cache_key,
        }

    return {"collections": collections, "cache_key": cache_key}


async def node_embed_query(state: RAGState) -> dict:
    """
    Node 3: embed_query
    Encode the query to dense and sparse vectors in parallel.
    """
    try:
        dense_arr, sparse_list = await async_encode_both([state["query"]])
        dense_vec = dense_arr[0].tolist()
        sparse = sparse_list[0]
        sparse_dict = {"indices": sparse.indices, "values": sparse.values}

        logger.debug(
            "embed_query: dense_dim=%d, sparse_nnz=%d",
            len(dense_vec),
            len(sparse.indices),
        )
        return {"dense_vec": dense_vec, "sparse_vec": sparse_dict}
    except Exception as exc:
        logger.error("embed_query failed: %s", exc)
        return {"error": f"Embedding failed: {exc}"}


async def node_hybrid_retrieve(state: RAGState) -> dict:
    """
    Node 4: hybrid_retrieve
    Run hybrid search across routed collections.
    """
    if state.get("error"):
        return {}

    try:
        candidates = await async_hybrid_search(
            query=state["query"],
            collections=state["collections"],
            top_k=RAG_TOP_K,
            rerank_n=RAG_RERANK_TOP_N,
            year_min=state.get("year_min"),
            year_max=state.get("year_max"),
            author=state.get("author_filter"),
        )

        tracer = get_tracer()
        tracer.log_retrieval(
            state["trace_id"],
            candidates=candidates,
            collections=state["collections"],
            query=state["query"],
        )

        logger.info("hybrid_retrieve: %d candidates", len(candidates))
        return {"candidates": candidates}
    except Exception as exc:
        logger.error("hybrid_retrieve failed: %s", exc)
        return {"candidates": [], "error": f"Retrieval failed: {exc}"}


def _is_time_sensitive(query: str) -> bool:
    """Heuristic: does the query ask about recent/latest information?"""
    q_lower = query.lower()
    return any(kw in q_lower for kw in TIME_SENSITIVE_KEYWORDS)


async def node_web_search(state: RAGState) -> dict:
    """
    Node 5: web_search (optional)
    Run SearXNG search for time-sensitive queries.
    """
    if state.get("error"):
        return {}

    query = state["query"]
    results = []

    try:
        searxng_url = f"{SEARXNG_URL.rstrip('/')}/search"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                searxng_url,
                params={
                    "q": query,
                    "format": "json",
                    "categories": "science",
                    "language": "en",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        for item in data.get("results", [])[:5]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "source": "web",
            })

        tracer = get_tracer()
        tracer.log_web_search(state["trace_id"], query, results)
        logger.info("web_search: %d results", len(results))

    except Exception as exc:
        logger.warning("web_search failed (non-fatal): %s", exc)

    return {"web_results": results}


async def node_merge_results(state: RAGState) -> dict:
    """
    Node 6: merge_results
    Combine Qdrant candidates and web results into a single candidate list.
    Web results are appended at the end (Qdrant results take priority).
    """
    candidates = list(state.get("candidates", []))
    web_results = list(state.get("web_results", []))

    if web_results:
        # Convert web results to the standard doc format
        for wr in web_results:
            candidates.append({
                "arxiv_id": "",
                "title": wr.get("title", ""),
                "abstract": wr.get("snippet", ""),
                "authors": "",
                "year": None,
                "categories": "",
                "score": 0.1,  # Low score for web results
                "collection": "web",
                "chunk_text": wr.get("snippet", ""),
                "source_file": wr.get("url", ""),
                "type": "web",
            })
        logger.info("merge_results: %d qdrant + %d web = %d total", len(state.get("candidates", [])), len(web_results), len(candidates))

    return {"candidates": candidates}


async def node_rerank_results(state: RAGState) -> dict:
    """
    Node 7: rerank_results
    Run BGE-M3 cross-encoder reranking on candidates.
    """
    if state.get("error"):
        # Fall back to returning raw candidates if reranker fails
        return {"reranked": state.get("candidates", [])[:RAG_TOP_K]}

    candidates = state.get("candidates", [])
    if not candidates:
        return {"reranked": []}

    try:
        reranked = await async_rerank(
            query=state["query"],
            documents=candidates,
            top_n=RAG_TOP_K,
        )

        tracer = get_tracer()
        tracer.log_reranking(state["trace_id"], reranked, num_input=len(candidates))

        logger.info("rerank_results: %d → %d", len(candidates), len(reranked))
        return {"reranked": reranked}
    except Exception as exc:
        logger.error("rerank_results failed: %s — using raw candidates", exc)
        return {"reranked": candidates[:RAG_TOP_K]}


def _format_doc_citation(doc: dict, index: int) -> str:
    """Format a single document for inclusion in the prompt context."""
    parts = []

    arxiv_id = doc.get("arxiv_id", "")
    title = doc.get("title", "Unknown Title")
    authors = doc.get("authors", "")
    year = doc.get("year", "")
    categories = doc.get("categories", "")
    doc_type = doc.get("type", "abstract")

    # Header line
    header = f"[{index}]"
    if arxiv_id:
        header += f" ArXiv:{arxiv_id}"
    if title:
        header += f" — {title}"
    parts.append(header)

    # Metadata line
    meta_parts = []
    if authors:
        author_list = str(authors).split(",")
        if len(author_list) > 3:
            meta = ", ".join(a.strip() for a in author_list[:3]) + " et al."
        else:
            meta = ", ".join(a.strip() for a in author_list)
        meta_parts.append(meta)
    if year:
        meta_parts.append(f"({year})")
    if categories:
        meta_parts.append(f"[{categories}]")
    if meta_parts:
        parts.append("  " + " ".join(meta_parts))

    # Content
    if doc_type == "web":
        url = doc.get("source_file", "")
        snippet = doc.get("abstract", "") or doc.get("chunk_text", "")
        if url:
            parts.append(f"  Source: {url}")
        if snippet:
            parts.append(f"  {snippet[:500]}")
    elif doc.get("chunk_text"):
        parts.append(f"  {doc['chunk_text'][:800]}")
    elif doc.get("abstract"):
        parts.append(f"  {doc['abstract'][:600]}")

    return "\n".join(parts)


async def node_build_context(state: RAGState) -> dict:
    """
    Node 8: build_context
    Format the top reranked results into a structured prompt context.
    """
    reranked = state.get("reranked", [])

    if not reranked:
        context = "No relevant papers found in the arXiv database for this query."
        return {"context": context}

    doc_sections = []
    for i, doc in enumerate(reranked, start=1):
        doc_sections.append(_format_doc_citation(doc, i))

    context = "\n\n---\n\n".join(doc_sections)
    logger.debug("build_context: %d docs, context length=%d chars", len(reranked), len(context))
    return {"context": context}


async def node_llm_inference(state: RAGState) -> dict:
    """
    Node 9: llm_inference
    Call the LLM (via LiteLLM proxy) to generate the final response.
    """
    if state.get("error") and not state.get("context"):
        return {"response": f"Error: {state['error']}"}

    query = state["query"]
    context = state["context"]
    model = state.get("model") or VLLM_MODEL
    temperature = state.get("temperature", 0.1)
    max_tokens = state.get("max_tokens", 2048)

    system_message = SYSTEM_PROMPT.format(context=context)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    llm_url = f"{LITELLM_URL.rstrip('/')}/v1/chat/completions"
    response_text = ""
    prompt_tokens = None
    completion_tokens = None

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(180.0, connect=10.0),
            headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
        ) as client:
            resp = await client.post(llm_url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        response_text = choice["message"]["content"]

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        logger.info(
            "llm_inference: model=%s, prompt_tokens=%s, completion_tokens=%s",
            model, prompt_tokens, completion_tokens,
        )

    except Exception as exc:
        logger.error("llm_inference failed: %s", exc)
        response_text = f"I encountered an error generating the response: {exc}"

    tracer = get_tracer()
    tracer.log_generation(
        state["trace_id"],
        query=query,
        context=context,
        response=response_text,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    return {
        "response": response_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


async def node_cache_result(state: RAGState) -> dict:
    """
    Node 10: cache_result
    Store the final response in Redis for future requests.
    """
    if state.get("error") or not state.get("response"):
        return {}

    cache: ResultCache = get_cache()
    cache_key = state.get("cache_key")

    if not cache_key:
        cache_key = cache.make_key(state["query"], state.get("collections", []))

    cache_val = {
        "response": state["response"],
        "reranked": [
            {k: v for k, v in doc.items() if k != "chunk_text"}
            for doc in (state.get("reranked") or [])[:5]
        ],
        "collections": state.get("collections", []),
    }

    cache.set(cache_key, cache_val, ttl=CACHE_TTL)
    logger.debug("cache_result: stored key=%s", cache_key)

    return {"cache_key": cache_key}


async def node_trace_result(state: RAGState) -> dict:
    """
    Node 11: trace_result
    Finalise Langfuse tracing and compute latency.
    """
    tracer = get_tracer()
    start = state.get("_start_time") or time.monotonic()
    latency_ms = (time.monotonic() - start) * 1000

    success = not bool(state.get("error")) and bool(state.get("response"))
    tracer.end_trace(
        state["trace_id"],
        success=success,
        error=state.get("error"),
    )

    logger.info(
        "Pipeline complete: query=%r, latency=%.0fms, cached=%s, sources=%d",
        state["query"][:60],
        latency_ms,
        state.get("cached", False),
        len(state.get("reranked", [])),
    )

    return {"latency_ms": latency_ms}


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def edge_check_cache(state: RAGState) -> str:
    """Route: if cache hit, go to END; otherwise route_query."""
    if state.get("cached"):
        return "end"
    return "route_query"


def edge_route_query_cached(state: RAGState) -> str:
    """After route_query, check if a cache hit was found with the proper key."""
    if state.get("cached"):
        return "llm_skip"
    return "embed_query"


def edge_should_web_search(state: RAGState) -> str:
    """Decide whether to run a web search based on query characteristics."""
    if _is_time_sensitive(state["query"]):
        logger.debug("Time-sensitive query detected — running web search")
        return "web_search"
    return "merge_results"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """
    Build and compile the LangGraph StateGraph.

    Returns the compiled graph ready for invocation.
    """
    builder = StateGraph(RAGState)

    # Register all nodes
    builder.add_node("check_cache", node_check_cache)
    builder.add_node("route_query", node_route_query)
    builder.add_node("embed_query", node_embed_query)
    builder.add_node("hybrid_retrieve", node_hybrid_retrieve)
    builder.add_node("web_search", node_web_search)
    builder.add_node("merge_results", node_merge_results)
    builder.add_node("rerank_results", node_rerank_results)
    builder.add_node("build_context", node_build_context)
    builder.add_node("llm_inference", node_llm_inference)
    builder.add_node("cache_result", node_cache_result)
    builder.add_node("trace_result", node_trace_result)

    # Entry point
    builder.add_edge(START, "check_cache")

    # Cache check → route or end
    builder.add_conditional_edges(
        "check_cache",
        edge_check_cache,
        {"end": END, "route_query": "route_query"},
    )

    # Route query → embed or short-circuit if cached with proper key
    builder.add_conditional_edges(
        "route_query",
        edge_route_query_cached,
        {"embed_query": "embed_query", "llm_skip": "llm_inference"},
    )

    # Linear path: embed → retrieve
    builder.add_edge("embed_query", "hybrid_retrieve")

    # Retrieve → conditional web search
    builder.add_conditional_edges(
        "hybrid_retrieve",
        edge_should_web_search,
        {"web_search": "web_search", "merge_results": "merge_results"},
    )

    # Web search always feeds merge
    builder.add_edge("web_search", "merge_results")

    # Linear remainder
    builder.add_edge("merge_results", "rerank_results")
    builder.add_edge("rerank_results", "build_context")
    builder.add_edge("build_context", "llm_inference")
    builder.add_edge("llm_inference", "cache_result")
    builder.add_edge("cache_result", "trace_result")
    builder.add_edge("trace_result", END)

    return builder.compile()


# Singleton compiled graph
_graph = None


def get_graph():
    """Return the singleton compiled RAG graph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def run_pipeline(
    query: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    stream: bool = False,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    author_filter: Optional[str] = None,
) -> RAGState:
    """
    Run the full RAG pipeline for a query.

    Parameters
    ----------
    query : str
        The user's question.
    model : str, optional
        LLM model name override.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens to generate.
    stream : bool
        Whether the caller wants streaming (affects how response is used).
    year_min, year_max : int, optional
        Year filter for Qdrant results.
    author_filter : str, optional
        Author substring filter.

    Returns
    -------
    RAGState
        Final pipeline state with response, reranked docs, metadata.
    """
    graph = get_graph()
    initial = _initial_state(
        query=query,
        model=model or VLLM_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        year_min=year_min,
        year_max=year_max,
        author_filter=author_filter,
    )

    final_state = await graph.ainvoke(initial)
    return final_state


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the main approaches to quantum error correction?"

    print(f"Running pipeline for: {query!r}")

    async def _main():
        state = await run_pipeline(query)
        print(f"\n=== Response ===\n{state['response']}")
        print(f"\n=== Metadata ===")
        print(f"  Cached: {state['cached']}")
        print(f"  Collections: {state['collections']}")
        print(f"  Candidates: {len(state['candidates'])}")
        print(f"  Reranked: {len(state['reranked'])}")
        print(f"  Latency: {state.get('latency_ms', 0):.0f}ms")
        print(f"  Trace ID: {state['trace_id']}")

    asyncio.run(_main())
