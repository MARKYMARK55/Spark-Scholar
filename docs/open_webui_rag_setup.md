# Open WebUI RAG Setup — Manual Document Ingestion Guide

This guide covers manually adding documents to Open WebUI and connecting them to the
Qdrant vector store backed by BGE-M3 embeddings. It also clarifies the distinction
between Open WebUI's built-in RAG pipeline and the custom hybrid RAG pipeline.

---

## Two RAG pipelines — understand the difference

Before uploading anything, it's important to understand that there are **two separate
retrieval paths** in this stack:

| | Open WebUI built-in RAG | Custom hybrid pipeline |
|---|---|---|
| **Trigger** | `#document-name` in chat | Automatic for all queries |
| **Embedding** | BGE-M3 dense via LiteLLM | BGE-M3 dense + sparse |
| **Vector store** | Qdrant `open-webui_files` collection | Subject-area collections |
| **Reranking** | None | BGE-M3 cross-encoder |
| **Chunking** | Open WebUI internal (CHUNK_SIZE=1500) | Custom pipeline |
| **Figures/diagrams** | Dropped | Dropped (roadmap: multimodal) |
| **Best for** | Quick one-off document Q&A | Research corpus queries |

For searching the Arxiv corpus and curated collections, the custom hybrid pipeline
is always used. Open WebUI's built-in RAG is for ad-hoc documents you upload through
the UI.

---

## Prerequisites before uploading documents

Verify all services are healthy:

```bash
# Check embedding service is up
curl -s http://localhost:8025/health
# Expected: {"status":"ok"} or {"object":"health"}

# Check Qdrant is up
curl -s http://localhost:6333/readyz
# Expected: {"title":"qdrant - Ready!"}

# Check LiteLLM can reach the embedder
curl -s http://localhost:4000/v1/models \
  -H "Authorization: Bearer simple-api-key" | jq '.data[].id'
# Should include "bge-m3-embedder"

# Check Open WebUI is up
curl -s http://localhost:8080/health
```

If any of these fail, see the Troubleshooting section at the end of this doc.

---

## Step 1: Configure Open WebUI RAG settings

First time only — configure the RAG settings inside Open WebUI:

1. Open `http://localhost:8080` and sign in as admin
2. Go to **Admin Panel** → **Settings** → **Documents**
3. Set the following:

| Setting | Value |
|---|---|
| Default embedding model | `bge-m3-embedder` |
| Embedding backend | OpenAI-compatible |
| Embedding API URL | `http://litellm:4000/v1` (within Docker) or `http://localhost:4000/v1` |
| Embedding API Key | `simple-api-key` |
| Vector database | Qdrant |
| Qdrant URL | `http://qdrant:6333` |
| Chunk size | `1500` |
| Chunk overlap | `200` |
| Top K results | `5` |

4. Click **Save** and verify with the **Test Embedding** button — it should return a
   vector without error.

---

## Step 2: Upload a document manually

### Via the UI (easiest)

1. In the left sidebar, click the **+** next to **Documents** (or go to the **Workspace**
   tab → **Documents**)
2. Click **Upload Document**
3. Select your PDF, DOCX, or TXT file
4. Open WebUI will:
   - Extract text (PyMuPDF for PDFs)
   - Split into chunks (1500 tokens, 200 overlap)
   - Embed each chunk using BGE-M3 via LiteLLM
   - Store vectors in Qdrant under the `open-webui_files` collection
   - Store metadata in Open WebUI's local SQLite database

Upload time depends on document length — roughly **2–5 seconds per page** for a
clean single-column PDF on the Spark.

### Via the API

```bash
curl -X POST http://localhost:8080/api/v1/documents/upload \
  -H "Authorization: Bearer YOUR_SESSION_TOKEN" \
  -F "file=@/path/to/your/paper.pdf"
```

To get your session token: open DevTools in the browser → Application → Local Storage
→ `token`.

---

## Step 3: Use the document in a chat

Once uploaded, reference the document in any chat with the `#` prefix:

```
#paper-name What are the main contributions of this paper?
```

Open WebUI will retrieve the most relevant chunks from Qdrant and include them as
context. The `#` triggers **retrieval-augmented generation** specifically from that
document's chunks.

To query **all uploaded documents** at once, use the Documents toggle in the chat
input rather than a specific `#reference`.

---

## Indexing time estimates

These are measured on the DGX Spark with BGE-M3 dense embedder at
`gpu-memory-utilization=0.12` (12% VRAM — leaving the rest for inference):

| Document type | Pages | Approx. time |
|---|---|---|
| Short paper (clean PDF) | 8–12 | 15–30 seconds |
| Full paper (clean PDF) | 20–30 | 45–90 seconds |
| Technical report | 50–80 | 2–4 minutes |
| Book / thesis | 200–400 | 8–20 minutes |
| Arxiv full corpus (2.96M abstracts) | — | ~18–22 hours (offline batch) |

The Arxiv corpus batch indexing time is for the dense pipeline running at batch
size 64 with 50% GPU utilisation (indexing mode). The production embedding service
runs at 12% GPU utilisation to leave headroom for inference, which means **ad-hoc
document uploads don't noticeably affect inference latency** for concurrent users.

---

## Limitations of the current pipeline

### Diagrams and figures are dropped

Open WebUI's PDF extraction (like the custom pipeline) extracts text only. Architecture
diagrams, results plots, equations typeset as images, and tables rendered as images
are all discarded. The surrounding caption text is retained if it's in the text layer.

For papers where the core contribution is a visual (e.g. a new model architecture
diagram), the retrieved context will be incomplete.

**Roadmap:** A multimodal extraction pass — running each page image through a vision
model (Qwen2-VL or similar) to caption figures before chunking — is planned but not
yet implemented.

### Multi-column layout

PyMuPDF sometimes reads multi-column PDFs in the wrong order (left column then right
column rather than by reading flow). For badly-affected papers, use the `unstructured`
fallback in the custom pipeline rather than Open WebUI's built-in extraction.

### No reranking in Open WebUI's built-in pipeline

Open WebUI's `#document` retrieval returns the top-K chunks by embedding similarity
alone — there is no cross-encoder reranking step. For higher-precision retrieval use
the custom hybrid pipeline instead.

---

## Troubleshooting

**"Failed to embed document" error in Open WebUI**
- Check the BGE-M3 dense embedder is running: `docker ps | grep bge-m3-dense`
- Check LiteLLM can reach it: `curl http://localhost:4000/v1/models -H "Authorization: Bearer simple-api-key"`
- Check Qdrant is running: `curl http://localhost:6333/readyz`

**Uploads succeed but retrieval returns nothing**
- Verify the `open-webui_files` collection exists in Qdrant dashboard: `http://localhost:6333/dashboard`
- Check the collection has points: it should show a non-zero vector count after upload

**Very slow uploads**
- The dense embedder may be competing for VRAM with the inference model
- Consider switching to indexing mode (`start_bge-m3_embedder_indexing.yml`) which
  uses 50% GPU but stops the production embedder: `scripts/start_embedding_stack.sh`
- After bulk indexing, switch back to the production embedder (12% GPU)
