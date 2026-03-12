"""
query/
======
Command-line search tools for the Spark Scholar stack.

All three scripts call the running embedding **services** — they do NOT load
any model locally — so they are safe to run alongside the production stack.

  dense_search.py   → BGE-M3 dense (1024-dim cosine ANN via HNSW)
  sparse_search.py  → BGE-M3 sparse (SPLADE lexical weights via inverted index)
  hybrid_search.py  → dense + sparse fused by Qdrant RRF + cross-encoder rerank

Quick reference
---------------
    # Best overall results
    python query/hybrid_search.py --query "diffusion models image generation"

    # Semantic concept search only
    python query/dense_search.py  --query "variational inference latent variables"

    # Keyword-exact search only (author names, paper IDs, model names)
    python query/sparse_search.py --query "Vaswani attention all you need 2017"

Service dependencies
--------------------
    dense_search.py   needs: BGE_M3_DENSE_URL (port 8025), QDRANT_URL (port 6333)
    sparse_search.py  needs: BGE_M3_SPARSE_URL (port 8035), QDRANT_URL (port 6333)
    hybrid_search.py  needs: all of the above + BGE_RERANKER_URL (port 8020)
"""
