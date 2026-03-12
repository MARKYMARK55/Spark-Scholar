# Troubleshooting & Known Limitations

← [Back to README](../README.md)

---

## Troubleshooting

### Qdrant returns 0 results
```bash
# Check collection status
curl http://localhost:6333/collections | python3 -m json.tool

# Check point counts per collection
for col in arxiv-cs-ml-ai arxiv-condmat arxiv-astro arxiv-hep; do
    echo "$col: $(curl -s http://localhost:6333/collections/$col | \
        python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["result"]["points_count"])')"
done
```
If all collections show 0 points, run the ingestion pipeline (steps 0–3 in
`docs/ingestion.md`).

---

### BGE-M3 dense embedder not responding
```bash
# Check container logs
docker logs bge-m3-dense-embedder --tail 50

# Test directly
curl http://localhost:8025/health
curl -X POST http://localhost:8025/v1/embeddings \
  -H "Authorization: Bearer simple-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3-embedder", "input": ["test sentence"]}'
```
**Common cause:** First startup downloads the model (~2 GB). Check logs for download
progress. Allow 5 minutes.

---

### Sparse embedder OOM
```bash
# Reduce batch size in env/.env:
SPARSE_BATCH_SIZE=16   # default 32

# Or isolate it on a specific GPU:
# Add 'CUDA_VISIBLE_DEVICES=1' to the service environment in embedding/bge_m3_sparse.yml
```

---

### Redis connection refused
```bash
docker logs redis
docker exec -it redis redis-cli ping   # should return PONG
```
Redis is required for caching but the pipeline works without it (slower, no caching).
Check if port 6379 is in use by another process:
```bash
sudo lsof -i :6379
```

---

### LLM returning very short responses
Check that:
1. `max_tokens` is set to at least 1024 in the request
2. LiteLLM is forwarding the parameter: `docker logs litellm --tail 50`
3. SparkRun's vLLM is not truncating: check `--max-model-len` in the SparkRun config

---

### RAG proxy returns "Embedding failed"
Both the dense and sparse embedders must be running:
```bash
curl http://localhost:8025/health   # dense
curl http://localhost:8035/health   # sparse
docker logs bge-m3-dense-embedder --tail 20
docker logs bge-m3-sparse-embedder --tail 20
```

---

### Open WebUI shows no models
```bash
# 1. Check LiteLLM is healthy
curl http://localhost:4000/health

# 2. Check Open WebUI environment
docker exec openwebui env | grep OPENAI

# 3. List models from LiteLLM directly
curl http://localhost:4000/v1/models \
  -H "Authorization: Bearer simple-api-key" | python3 -m json.tool
```
Verify all URLs in `core_services/core_services.yml` use Docker service names
(`http://litellm:4000`) not `localhost` — containers resolve by service name on `llm-net`.

---

### Langfuse traces not appearing
```bash
# Test Langfuse connectivity
curl http://localhost:3000/api/public/health

# Verify keys are loaded
docker exec rag-proxy env | grep LANGFUSE
```
If `LANGFUSE_SECRET_KEY` is not set, tracing silently no-ops — by design.
After setting the keys in `env/.env`, restart the proxy:
```bash
docker compose -f rag_proxy/rag_proxy.yml restart rag-proxy
```

---

### Dense ingestion is slow

Normal throughput on DGX Spark: ~1,100 abstracts/second with batch_size=256 in
indexing mode. Factors that slow it down:

- **Not in indexing mode** — run `bash scripts/start_indexing_mode.sh` first (bumps from 12% to 50% GPU)
- **Small batch size** — increase to 256 or 512 if VRAM allows
- **JSONL on slow storage** — copy to NVMe first: `cp data/*.jsonl /fast-nvme/`
- **Network latency** — from inside containers use `bge-m3-dense-embedder:8025`, not localhost

---

### Collection routing errors

If papers end up in the wrong collections, debug the router:
```bash
python3 -c "
from pipeline.router import route_paper, route_query
print(route_paper('cs.LG cs.AI'))    # from an arXiv category string
print(route_query('attention mechanism transformers'))   # from a query
"
```
The router uses keyword heuristics from `pipeline/router.py`. Adjust the keyword
lists there if routing is consistently wrong for your domain.

---

### RAG proxy build fails

The build context is the repo root and the Dockerfile is at `rag_proxy/Dockerfile`.
Make sure you run the build from the repo root:
```bash
docker compose -f rag_proxy/rag_proxy.yml build
# NOT: cd rag_proxy && docker build .   ← wrong context
```

---

### LiteLLM fails to start ("db connection error")

LiteLLM depends on its Postgres container being healthy. Check:
```bash
docker logs litellm-db --tail 20
docker compose -f core_services/core_services.yml ps litellm-db
```
If the DB is slow to start, LiteLLM will retry. Allow 30–60 seconds. If it persists:
```bash
docker compose -f core_services/core_services.yml restart litellm-db litellm
```

---

### Phi Mini not appearing in model list

1. Verify SparkRun is serving it: `curl http://localhost:8001/v1/models`
2. Check `VLLM_PHI_URL` in `env/.env` is set to `http://host.docker.internal:8001`
3. Restart LiteLLM to reload the config:
   ```bash
   docker compose -f core_services/core_services.yml restart litellm
   ```
4. Check LiteLLM logs: `docker compose -f core_services/core_services.yml logs litellm --tail 30`

---

## Known Limitations

1. **Figures and diagrams in the arXiv index:** The 2.96M indexed abstracts are
   text-only. `06_caption_figures.py` addresses this for PDF ingestion, but arXiv
   abstracts don't include figures.

2. **Multi-column PDF layouts:** PyMuPDF sometimes produces garbled text from
   two-column academic PDFs (left-to-right reading order assumption). The
   `unstructured` fallback handles most cases but is slower.

3. **No reranking in Open WebUI built-in RAG (Path A):** Open WebUI's native
   document RAG uses cosine similarity only (bi-encoder). Cross-encoder reranking
   only applies when using Path B (RAG proxy connection).

4. **arXiv abstracts only (not full text):** The 2.96M paper index contains titles
   and abstracts. Full-text indexing requires downloading and parsing PDFs
   individually — use `05_ingest_pdfs.py` for that.

5. **Equation rendering:** Mathematical equations in abstracts are stored as plain
   ASCII/Unicode. LaTeX rendering in responses depends on Open WebUI's built-in
   KaTeX support.

6. **Cache invalidation:** Redis has no automatic mechanism for invalidating entries
   when new papers are indexed. Either set a short TTL or flush after ingestion:
   ```bash
   docker exec -it redis redis-cli FLUSHDB
   ```

7. **Single-node Qdrant:** The stack uses a single Qdrant instance. For true HA,
   configure Qdrant cluster mode:
   https://qdrant.tech/documentation/guides/distributed_deployment/

8. **Citation expansion — non-arXiv papers:** `08_expand_citations.py` uses
   Semantic Scholar for reference lookup and arXiv for PDF download. Papers not on
   arXiv are fetched via Unpaywall (requires `--email`). Purely paywalled papers
   cannot be downloaded automatically.

9. **SparkRun model names:** `VLLM_MODEL_NAME` must exactly match what SparkRun/vLLM
   reports at `GET http://localhost:8000/v1/models`. A mismatch causes LiteLLM to
   return 404 on every request.
