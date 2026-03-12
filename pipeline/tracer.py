"""
pipeline/tracer.py
==================
Langfuse observability tracing for the Arxiv RAG pipeline.

Traces each RAG request end-to-end:
  - start_trace()        → creates a Langfuse trace, returns trace_id
  - log_retrieval()      → span for Qdrant retrieval
  - log_reranking()      → span for BGE reranking
  - log_generation()     → generation span for LLM call (prompt + response)
  - end_trace()          → marks the trace as succeeded/failed

If LANGFUSE_SECRET_KEY is not set, all methods are no-ops so the pipeline
works identically without Langfuse configured.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"), override=False)

logger = logging.getLogger(__name__)

LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

_langfuse_available = bool(LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY)


def _try_import_langfuse():
    """Lazily import langfuse to avoid hard dependency if not installed."""
    try:
        from langfuse import Langfuse
        return Langfuse
    except ImportError:
        logger.warning("langfuse package not installed; tracing disabled")
        return None


class RAGTracer:
    """
    Langfuse tracing wrapper for the Arxiv RAG pipeline.

    All public methods are no-ops when Langfuse credentials are not configured
    or when the langfuse package is not installed.
    """

    def __init__(self):
        self._lf = None
        self._traces: dict[str, Any] = {}  # trace_id → Langfuse trace object
        self._spans: dict[str, list[Any]] = {}  # trace_id → list of open spans

        if _langfuse_available:
            LangfuseClass = _try_import_langfuse()
            if LangfuseClass is not None:
                try:
                    self._lf = LangfuseClass(
                        secret_key=LANGFUSE_SECRET_KEY,
                        public_key=LANGFUSE_PUBLIC_KEY,
                        host=LANGFUSE_HOST,
                    )
                    logger.info("RAGTracer: Langfuse connected at %s", LANGFUSE_HOST)
                except Exception as exc:
                    logger.warning("RAGTracer: Langfuse init failed: %s — tracing disabled", exc)
                    self._lf = None
        else:
            logger.info("RAGTracer: LANGFUSE_SECRET_KEY not set — tracing disabled (no-op mode)")

    @property
    def enabled(self) -> bool:
        """True if Langfuse is configured and connected."""
        return self._lf is not None

    def start_trace(self, query: str, metadata: Optional[dict] = None) -> str:
        """
        Start a new Langfuse trace for a RAG request.

        Parameters
        ----------
        query : str
            The user's query string.
        metadata : dict, optional
            Extra metadata to attach to the trace (e.g. user_id, session_id).

        Returns
        -------
        str
            trace_id — pass this to subsequent log_* calls.
        """
        trace_id = str(uuid.uuid4())

        if self._lf is None:
            return trace_id

        try:
            trace = self._lf.trace(
                id=trace_id,
                name="arxiv-rag",
                input={"query": query},
                metadata=metadata or {},
                tags=["arxiv", "rag"],
            )
            self._traces[trace_id] = trace
            self._spans[trace_id] = []
            logger.debug("RAGTracer: started trace %s", trace_id)
        except Exception as exc:
            logger.warning("RAGTracer.start_trace failed: %s", exc)

        return trace_id

    def log_retrieval(
        self,
        trace_id: str,
        candidates: list[dict],
        collections: list[str],
        query: Optional[str] = None,
    ) -> None:
        """
        Log a Qdrant retrieval span.

        Parameters
        ----------
        trace_id : str
            From start_trace().
        candidates : list[dict]
            Raw candidate documents from hybrid_search.
        collections : list[str]
            Collections that were searched.
        query : str, optional
            The query string (for the span input).
        """
        if self._lf is None or trace_id not in self._traces:
            return

        try:
            trace = self._traces[trace_id]
            span = trace.span(
                name="qdrant-hybrid-retrieval",
                input={
                    "query": query or "",
                    "collections": collections,
                    "num_collections": len(collections),
                },
                output={
                    "num_candidates": len(candidates),
                    "top_arxiv_ids": [c.get("arxiv_id", "") for c in candidates[:5]],
                    "collections_searched": list({c.get("collection", "") for c in candidates}),
                },
                metadata={
                    "strategy": "dense+sparse+RRF",
                    "vector_dim": 1024,
                },
            )
            span.end()
            self._spans[trace_id].append(span)
        except Exception as exc:
            logger.warning("RAGTracer.log_retrieval failed: %s", exc)

    def log_reranking(
        self,
        trace_id: str,
        reranked: list[dict],
        num_input: Optional[int] = None,
    ) -> None:
        """
        Log a BGE-M3 reranking span.

        Parameters
        ----------
        trace_id : str
            From start_trace().
        reranked : list[dict]
            Reranked documents (with rerank_score field).
        num_input : int, optional
            Number of candidates that went into the reranker.
        """
        if self._lf is None or trace_id not in self._traces:
            return

        try:
            trace = self._traces[trace_id]
            top_scores = [round(d.get("rerank_score", 0.0), 4) for d in reranked[:5]]
            span = trace.span(
                name="bge-reranking",
                input={"num_candidates": num_input or 0},
                output={
                    "num_reranked": len(reranked),
                    "top_scores": top_scores,
                    "top_titles": [d.get("title", "")[:80] for d in reranked[:3]],
                },
                metadata={"model": "bge-reranker-v2-m3"},
            )
            span.end()
            self._spans[trace_id].append(span)
        except Exception as exc:
            logger.warning("RAGTracer.log_reranking failed: %s", exc)

    def log_generation(
        self,
        trace_id: str,
        query: str,
        context: str,
        response: str,
        model: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> None:
        """
        Log an LLM generation span.

        Parameters
        ----------
        trace_id : str
            From start_trace().
        query : str
            The user query.
        context : str
            The formatted context passed to the LLM (from build_context node).
        response : str
            The LLM's generated response.
        model : str
            Model name (e.g. "nemotron-70b").
        prompt_tokens : int, optional
            Number of prompt tokens used.
        completion_tokens : int, optional
            Number of completion tokens generated.
        """
        if self._lf is None or trace_id not in self._traces:
            return

        try:
            trace = self._traces[trace_id]
            usage = {}
            if prompt_tokens is not None:
                usage["promptTokens"] = prompt_tokens
            if completion_tokens is not None:
                usage["completionTokens"] = completion_tokens

            generation = trace.generation(
                name="llm-generation",
                model=model,
                input=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": query},
                ],
                output=response,
                usage=usage if usage else None,
                metadata={"context_length": len(context)},
            )
            generation.end()
            self._spans[trace_id].append(generation)

            # Also update trace output
            trace.update(output={"response": response[:500]})
        except Exception as exc:
            logger.warning("RAGTracer.log_generation failed: %s", exc)

    def log_web_search(
        self,
        trace_id: str,
        query: str,
        results: list[dict],
    ) -> None:
        """Log a SearXNG web search span."""
        if self._lf is None or trace_id not in self._traces:
            return

        try:
            trace = self._traces[trace_id]
            span = trace.span(
                name="searxng-web-search",
                input={"query": query},
                output={
                    "num_results": len(results),
                    "urls": [r.get("url", "") for r in results[:3]],
                },
            )
            span.end()
        except Exception as exc:
            logger.warning("RAGTracer.log_web_search failed: %s", exc)

    def log_cache_hit(self, trace_id: str, key: str) -> None:
        """Log a Redis cache hit event."""
        if self._lf is None or trace_id not in self._traces:
            return

        try:
            trace = self._traces[trace_id]
            trace.event(
                name="cache-hit",
                input={"key": key},
                output={"cached": True},
            )
        except Exception as exc:
            logger.warning("RAGTracer.log_cache_hit failed: %s", exc)

    def end_trace(self, trace_id: str, success: bool = True, error: Optional[str] = None) -> None:
        """
        Finalise and flush the trace.

        Parameters
        ----------
        trace_id : str
            From start_trace().
        success : bool
            Whether the pipeline completed successfully.
        error : str, optional
            Error message if success=False.
        """
        if self._lf is None or trace_id not in self._traces:
            return

        try:
            trace = self._traces[trace_id]
            update_kwargs: dict[str, Any] = {
                "metadata": {
                    "success": success,
                    "num_spans": len(self._spans.get(trace_id, [])),
                }
            }
            if error:
                update_kwargs["metadata"]["error"] = error

            trace.update(**update_kwargs)

            # Flush to Langfuse
            self._lf.flush()
            logger.debug("RAGTracer: ended trace %s (success=%s)", trace_id, success)

        except Exception as exc:
            logger.warning("RAGTracer.end_trace failed: %s", exc)
        finally:
            # Clean up local references
            self._traces.pop(trace_id, None)
            self._spans.pop(trace_id, None)

    def flush(self) -> None:
        """Force-flush all pending events to Langfuse."""
        if self._lf is not None:
            try:
                self._lf.flush()
            except Exception as exc:
                logger.warning("RAGTracer.flush failed: %s", exc)


# Singleton tracer instance
_tracer_instance: Optional[RAGTracer] = None


def get_tracer() -> RAGTracer:
    """Return the singleton RAGTracer instance."""
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = RAGTracer()
    return _tracer_instance


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.DEBUG)

    tracer = RAGTracer()
    print(f"Tracing enabled: {tracer.enabled}")

    trace_id = tracer.start_trace("What is quantum entanglement?")
    print(f"Trace ID: {trace_id}")

    tracer.log_retrieval(
        trace_id,
        candidates=[{"arxiv_id": "2301.00001", "title": "Quantum Entanglement Review", "collection": "arxiv-quantph-grqc"}],
        collections=["arxiv-quantph-grqc"],
        query="quantum entanglement",
    )

    tracer.log_reranking(
        trace_id,
        reranked=[{"arxiv_id": "2301.00001", "title": "Quantum Entanglement Review", "rerank_score": 0.95}],
        num_input=10,
    )

    tracer.log_generation(
        trace_id,
        query="What is quantum entanglement?",
        context="Paper: Quantum Entanglement Review (2023)...",
        response="Quantum entanglement is a physical phenomenon...",
        model="nemotron-70b",
    )

    tracer.end_trace(trace_id, success=True)
    print("Trace ended successfully")
