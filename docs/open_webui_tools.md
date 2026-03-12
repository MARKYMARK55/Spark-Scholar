# Open WebUI Tools — Dynamic RAG & Corpus Expansion

Open WebUI supports **Tools** — Python functions that the LLM can call at runtime
to fetch live data, search the web, query APIs, or trigger backend operations.
This is how you turn a static RAG system into a dynamic one where the model can
actively expand what it knows.

---

## How Open WebUI Tools Work

### Mechanism

When a user sends a message, Open WebUI sends the tool definitions (as JSON Schema)
alongside the conversation to the LLM. The LLM decides whether to call a tool based
on the query. If it calls one, Open WebUI:

1. Executes the Python function server-side
2. Injects the return value back into the conversation as a tool result
3. Sends the enriched context to the LLM for a final answer

The user sees the final answer with a small indicator showing which tools fired.
This all happens in a single chat turn — no manual steps required.

### Tool vs. RAG proxy

| Feature | RAG Proxy (Path B) | Open WebUI Tools |
|---------|-------------------|-----------------|
| Trigger | Every query (automatic) | LLM decides when to call |
| Data source | Your indexed collections | Live APIs, web, filesystem |
| Latency | Adds ~1–3s per query | Variable (depends on tool) |
| Best for | Deep search over indexed corpus | Fresh data, targeted lookups |

**They complement each other.** Use the RAG proxy as your primary model connection
for deep retrieval over indexed papers. Use tools on top of that for live lookups,
corpus expansion, and cross-referencing.

---

## Installing Tools

Tools are Python files installed in Open WebUI's admin interface.

1. Open http://localhost:8080
2. **Workspace → Tools → + Add Tool** (or import from the community hub)
3. Paste the Python source code into the editor
4. The tool appears in **Admin → Settings → Tools** and in the chat toolbar

**Enabling tools per chat:**
- Click the **⚙️ Tools** button in the chat input bar
- Toggle the tools you want active for that session

**Auto-enabling tools:**
- **Admin → Settings → Tools** → set a tool to *Enabled by default*

---

## Core Tool: arXiv Paper Search

Search arXiv directly from chat and optionally download + ingest the PDF to expand
your local corpus.

```python
"""
Tool: arxiv_search
Searches arXiv for papers matching the query and returns structured results.
The LLM can then decide to download specific papers for deeper reading.
"""
import httpx
import json
from datetime import datetime

class Tools:
    def __init__(self):
        self.citation_format = "[ArXiv:{arxiv_id}]"

    def search_arxiv(
        self,
        query: str,
        max_results: int = 10,
        category: str = "",
        date_from: str = "",
    ) -> str:
        """
        Search arXiv for papers matching the query.

        :param query: Natural language or keyword query.
        :param max_results: Maximum number of results to return (1-50).
        :param category: Optional arXiv category filter e.g. cs.LG, cs.AI, q-bio.BM
        :param date_from: Optional start date filter in YYYY-MM-DD format.
        :return: Formatted list of matching papers with titles, authors, abstracts, IDs.
        """
        search_query = query
        if category:
            search_query = f"cat:{category} AND ({query})"
        params = {
            "search_query": f"all:{search_query}",
            "max_results": min(max_results, 50),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        resp = httpx.get("http://export.arxiv.org/api/query", params=params, timeout=15)
        resp.raise_for_status()
        # Parse Atom XML response
        import xml.etree.ElementTree as ET
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        root = ET.fromstring(resp.text)
        results = []
        for entry in root.findall("atom:entry", ns):
            arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")[:400]
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)][:3]
            published = entry.find("atom:published", ns).text[:10]
            results.append(
                f"**[ArXiv:{arxiv_id}]** {title}\n"
                f"  Authors: {', '.join(authors)}\n"
                f"  Published: {published}\n"
                f"  Abstract: {summary}...\n"
                f"  PDF: https://arxiv.org/pdf/{arxiv_id}\n"
            )
        return "\n---\n".join(results) if results else "No results found."
```

---

## Core Tool: Semantic Scholar Search

Semantic Scholar has richer citation graph data, recommendation features, and
covers beyond arXiv (ACL, NeurIPS, ICML, Nature, etc.).

```python
"""
Tool: semantic_scholar_search
Searches Semantic Scholar for papers and returns citation-enriched results.
"""
import httpx
import os

class Tools:
    API_BASE = "https://api.semanticscholar.org/graph/v1"

    def search_semantic_scholar(
        self,
        query: str,
        limit: int = 10,
        fields: str = "title,authors,year,abstract,citationCount,externalIds,openAccessPdf",
        min_citations: int = 0,
    ) -> str:
        """
        Search Semantic Scholar for papers.

        :param query: Search query.
        :param limit: Number of results (max 100).
        :param fields: Comma-separated fields to return.
        :param min_citations: Filter papers with at least this many citations.
        :return: Formatted paper results with citation counts.
        """
        headers = {}
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        if api_key:
            headers["x-api-key"] = api_key

        resp = httpx.get(
            f"{self.API_BASE}/paper/search",
            params={"query": query, "limit": limit, "fields": fields},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        papers = resp.json().get("data", [])

        results = []
        for p in papers:
            if p.get("citationCount", 0) < min_citations:
                continue
            arxiv_id = (p.get("externalIds") or {}).get("ArXiv", "")
            pdf_url = (p.get("openAccessPdf") or {}).get("url", "")
            authors = ", ".join(a["name"] for a in (p.get("authors") or [])[:3])
            abstract = (p.get("abstract") or "")[:350]
            results.append(
                f"**{p['title']}** ({p.get('year', '?')})\n"
                f"  Authors: {authors}\n"
                f"  Citations: {p.get('citationCount', 0)}\n"
                + (f"  ArXiv: [ArXiv:{arxiv_id}]\n" if arxiv_id else "")
                + (f"  PDF: {pdf_url}\n" if pdf_url else "")
                + f"  Abstract: {abstract}...\n"
            )
        return "\n---\n".join(results) if results else "No results found."

    def get_paper_citations(self, arxiv_id: str, limit: int = 20) -> str:
        """
        Get papers that cite the given arXiv paper.

        :param arxiv_id: arXiv ID e.g. 2303.08774
        :param limit: Maximum citations to return.
        :return: List of citing papers.
        """
        headers = {}
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        if api_key:
            headers["x-api-key"] = api_key

        resp = httpx.get(
            f"{self.API_BASE}/paper/arXiv:{arxiv_id}/citations",
            params={"fields": "title,authors,year,citationCount,externalIds", "limit": limit},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        citations = resp.json().get("data", [])
        results = []
        for c in citations:
            p = c.get("citingPaper", {})
            arxiv = (p.get("externalIds") or {}).get("ArXiv", "")
            results.append(
                f"**{p.get('title', 'Unknown')}** ({p.get('year', '?')}) — "
                f"{p.get('citationCount', 0)} citations"
                + (f"  [ArXiv:{arxiv}]" if arxiv else "")
            )
        return "\n".join(results) if results else "No citations found."
```

---

## Core Tool: Ingest PDF to Corpus

This is the **corpus expansion** tool — the LLM can trigger ingestion of a new
arXiv paper or PDF URL directly into your local Qdrant collection.

> **Prerequisites:** The RAG proxy and embedding services must be running.

```python
"""
Tool: ingest_pdf
Downloads an arXiv paper or PDF URL and indexes it into the local Qdrant corpus.
After ingestion, the paper becomes searchable via the RAG proxy.
"""
import httpx
import os
import subprocess
import tempfile
from pathlib import Path

class Tools:
    RAG_PROXY_URL = os.environ.get("RAG_PROXY_URL", "http://localhost:8002")

    def ingest_arxiv_paper(
        self,
        arxiv_id: str,
        collection: str = "docs-python",
        tags: str = "",
    ) -> str:
        """
        Download and index an arXiv paper PDF into a local Qdrant collection.

        :param arxiv_id: arXiv ID e.g. 2303.08774 or full URL https://arxiv.org/abs/2303.08774
        :param collection: Target Qdrant collection name (default: docs-python).
        :param tags: Comma-separated tags to attach to the indexed chunks.
        :return: Status message with number of chunks indexed.
        """
        # Normalise arXiv ID
        if "arxiv.org" in arxiv_id:
            arxiv_id = arxiv_id.rstrip("/").split("/")[-1]
        arxiv_id = arxiv_id.replace("abs/", "").replace("pdf/", "").rstrip(".pdf")

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / f"{arxiv_id.replace('/', '_')}.pdf"
            # Download
            try:
                r = httpx.get(pdf_url, follow_redirects=True, timeout=30)
                r.raise_for_status()
                pdf_path.write_bytes(r.content)
            except Exception as e:
                return f"Failed to download {pdf_url}: {e}"

            # Run the PDF ingest script
            cmd = [
                "python", "ingest/05_ingest_pdfs.py",
                "--input", str(pdf_path),
                "--collection", collection,
            ]
            if tag_list:
                cmd += ["--tags", ",".join(tag_list)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            # Extract chunk count from stdout
            lines = result.stdout.strip().splitlines()
            return f"✓ Ingested [ArXiv:{arxiv_id}] → `{collection}` — {lines[-1] if lines else 'done'}"
        else:
            return f"Ingestion failed:\n{result.stderr[-500:]}"

    def ingest_pdf_url(
        self,
        url: str,
        collection: str = "docs-python",
        tags: str = "",
    ) -> str:
        """
        Download a PDF from any URL and index it into a local Qdrant collection.

        :param url: Direct PDF URL.
        :param collection: Target Qdrant collection name.
        :param tags: Comma-separated tags to attach to indexed chunks.
        :return: Status message.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = url.split("/")[-1].split("?")[0] or "document.pdf"
            pdf_path = Path(tmpdir) / filename
            try:
                r = httpx.get(url, follow_redirects=True, timeout=30)
                r.raise_for_status()
                pdf_path.write_bytes(r.content)
            except Exception as e:
                return f"Download failed: {e}"

            cmd = ["python", "ingest/05_ingest_pdfs.py", "--input", str(pdf_path),
                   "--collection", collection]
            if tags:
                cmd += ["--tags", tags]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        return (f"✓ Ingested {filename} → `{collection}`"
                if result.returncode == 0
                else f"Failed: {result.stderr[-300:]}")
```

---

## Core Tool: Query the RAG Corpus

Sometimes you want the LLM to do a focused Qdrant search as a tool call rather than
routing everything through the RAG proxy. This is useful for multi-step research
where you need targeted lookups against specific collections.

```python
"""
Tool: rag_search
Directly queries the RAG proxy for targeted retrieval from specific collections.
"""
import httpx
import os
import json

class Tools:
    RAG_PROXY_URL = os.environ.get("RAG_PROXY_URL", "http://localhost:8002")
    API_KEY = os.environ.get("LITELLM_API_KEY", "simple-api-key")

    def search_rag_corpus(
        self,
        query: str,
        collections: str = "",
        top_k: int = 5,
        year_min: int = 0,
        year_max: int = 0,
    ) -> str:
        """
        Search the local RAG corpus (arXiv + documentation collections) directly.

        :param query: The search query.
        :param collections: Comma-separated collection names to search (empty = auto-route).
        :param top_k: Number of results to return.
        :param year_min: Optional minimum year filter for arXiv papers.
        :param year_max: Optional maximum year filter for arXiv papers.
        :return: Top matching chunks with source metadata.
        """
        payload = {
            "model": "spark-scholar",
            "messages": [{"role": "user", "content": query}],
            "max_tokens": 1,          # We only want the retrieved context, not a full answer
            "stream": False,
        }
        if collections:
            payload["collections"] = [c.strip() for c in collections.split(",")]
        if year_min:
            payload["year_min"] = year_min
        if year_max:
            payload["year_max"] = year_max

        resp = httpx.post(
            f"{self.RAG_PROXY_URL}/v1/search",   # dedicated search-only endpoint
            json=payload,
            headers={"Authorization": f"Bearer {self.API_KEY}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.text
```

---

## Core Tool: Web Search via SearXNG

Uses the local SearXNG instance for private web search. The results are injected
into context before the LLM answers.

> This tool is already built into Open WebUI — enable it via the 🌐 icon in chat.
> The custom version below gives you more control over search categories.

```python
"""
Tool: searxng_search
Queries the local SearXNG instance for web results.
"""
import httpx
import os

class Tools:
    SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8888")

    def web_search(
        self,
        query: str,
        categories: str = "general",
        max_results: int = 10,
        language: str = "en",
    ) -> str:
        """
        Search the web via the local SearXNG instance (private, no tracking).

        :param query: Search query.
        :param categories: Comma-separated categories: general, science, news, files, images.
        :param max_results: Maximum results to return.
        :param language: Language code (en, de, fr, etc.)
        :return: Formatted search results with titles, URLs, snippets.
        """
        resp = httpx.get(
            f"{self.SEARXNG_URL}/search",
            params={
                "q": query,
                "categories": categories,
                "language": language,
                "format": "json",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("results", [])[:max_results]:
            results.append(
                f"**{r.get('title', 'No title')}**\n"
                f"  URL: {r.get('url', '')}\n"
                f"  {r.get('content', '')[:300]}\n"
            )
        return "\n---\n".join(results) if results else "No results."
```

---

## Dynamic RAG Workflow

The real power comes from combining tools with the RAG proxy. Here is how a
full dynamic research session flows:

```
User: "Find the latest papers on test-time compute scaling and add any
       key ones from 2024 to my local corpus"

  1. LLM calls search_arxiv("test-time compute scaling", date_from="2024-01-01")
     → Returns 10 papers with IDs and abstracts

  2. LLM evaluates abstracts, selects 3 most relevant

  3. LLM calls ingest_arxiv_paper("2408.03314", collection="arxiv-cs-ml-ai")
     LLM calls ingest_arxiv_paper("2501.12345", collection="arxiv-cs-ml-ai")
     LLM calls ingest_arxiv_paper("2412.08765", collection="arxiv-cs-ml-ai")
     → Each paper is downloaded, chunked, embedded, and indexed into Qdrant

  4. LLM summarises: "I've indexed 3 papers on test-time compute scaling.
     Ask me anything about them and I'll retrieve the relevant sections."

  5. User: "What's the key finding in the DeepSeek paper about this?"
     → RAG proxy retrieves the newly indexed chunks + reranks
     → LLM answers with [ArXiv:XXXX.XXXXX] citations
```

This is **live corpus expansion** — the model actively grows the knowledge base
in response to gaps it detects while answering.

---

## Suggested Tool Combinations

### Research assistant workflow

Enable all of these together for a full research session:

| Tool | When the LLM uses it |
|------|---------------------|
| `search_arxiv` | Query contains "find papers on", "latest research", "what's been published" |
| `search_semantic_scholar` | Needs citation counts, influential papers, cross-venue |
| `ingest_arxiv_paper` | User says "add this to my library" or gap detected in corpus |
| `web_search` | Needs documentation, news, non-academic sources |
| `search_rag_corpus` | Targeted search against a specific collection |

### Documentation lookup workflow

For code and technical questions:

| Tool | Purpose |
|------|---------|
| `search_rag_corpus` with `collections="docs-python"` | Python stdlib / library docs |
| `search_rag_corpus` with `collections="docs-anthropic"` | Claude API / MCP / CUA docs |
| `web_search` with `categories="science"` | Latest library releases not yet indexed |
| `ingest_pdf_url` | Pull in a new library's PDF docs on the fly |

---

## Expanding Your Corpus — Ingestion Strategies

### From a chat session (tool-driven)

Ask the model directly:
```
"Search for 5 influential papers on mixture-of-experts architectures published
 in 2024 and index them into arxiv-cs-ml-ai"
```

The model will call `search_arxiv`, then `ingest_arxiv_paper` for each one.

### From the command line (batch)

```bash
# Ingest a specific arXiv paper
python ingest/05_ingest_pdfs.py \
    --input /path/to/paper.pdf \
    --collection arxiv-cs-ml-ai \
    --tags "moe,transformers,2024"

# Ingest an entire HTML documentation site
python ingest/07_ingest_html_docs.py \
    --url https://docs.anthropic.com \
    --collection docs-anthropic \
    --depth 4 \
    --delay 0.5

# Ingest multiple URLs from a file
python ingest/07_ingest_html_docs.py \
    --url-file urls.txt \
    --collection docs-devops \
    --depth 2
```

### From Open WebUI document upload (Path A)

For one-off documents you don't want in the permanent index:
1. Click the 📎 paperclip in chat
2. Upload PDF — Open WebUI chunks and embeds it into a temporary Qdrant collection
3. Ask questions — Open WebUI retrieves from the uploaded doc automatically
4. This collection persists in Qdrant under `openwebui_` prefix until you delete it

---

## Installed Academic Research Tools

The following tools are already installed in your Open WebUI workspace
(`~/vllm/model_stack/openWebUI/workspace_tools/`). Import each JSON via
**Workspace → Tools → Import**.

### Quick Reference — All Sources

| Tool | Source / Database | Coverage | API Key Required |
|------|------------------|----------|-----------------|
| **ArXiv Recent & Keyword Alert** | arXiv.org Atom API | 2.4M+ CS/physics/math preprints; keyword + category + date filter | None (free) |
| **Semantic Scholar Graph** | Semantic Scholar (AI2) | 220M+ papers; search, citations, references, author graph | `SEMANTIC_SCHOLAR_API_KEY` |
| **Ai2 Asta MCP Search** | Allen AI Asta MCP | Full-text semantic search over S2 corpus; ranked snippets | `ASTA_TOOL_KEY` |
| **OpenAlex Academic Search** | OpenAlex | 250M+ works; authors, institutions, funders, open access | `OPENALEX_API_KEY` (optional, boosts rate limits) |
| **CORE UK Tool** | CORE.ac.uk | 200M+ OA papers with direct PDF links; repository aggregator | `CORE_API_KEY` |
| **Connected Papers Graph** | Connected Papers | Similarity graph; prior works, derivative works, visual clusters | `CONNECTED_PAPERS_API_KEY` |
| **NCBI PubMed Fetch** | PubMed / NCBI | 35M+ biomedical papers; search by term or fetch by PMID | `NCBI_API_KEY` (optional) |
| **Europe PMC OA & Preprint Search** | Europe PMC (EBI) | Life sciences OA papers, preprints, grants, books, patents | None (free) |
| **Dimensions Lens Academic Graph** | Dimensions.ai / Lens.org | Publications, grants, patents, researchers, funding networks | `DIMENSIONS_API_KEY` (paid) / `LENS_API_KEY` |
| **ResearchGate / Academia.edu** | ResearchGate, Academia.edu | Paper links, author profiles (no bulk API — returns search URLs) | None (link-based) |
| **Consolidated Metadata Fetcher** | S2 + OpenAlex + CORE + Connected Papers + NCBI | Unified tool: one call, choose service + action | Per-service keys above |
| **Google Gemini** | Google Generative AI | Multimodal reasoning; strong on math, tables, figures, recent papers | `GEMINI_API_KEY` |
| **Grok Research & Reasoning** | xAI Grok-3 | Deep reasoning, math, code, obscure references, step-by-step thinking | `XAI_API_KEY` |
| **OpenRouter Research LLM** | OpenRouter | Route to Claude, Gemini, Grok, Llama, Mistral, 200+ models | `OPENROUTER_API_KEY` |
| **Perplexity Academic Q&A** | Perplexity Sonar | Cited answers; search filtered to arxiv, S2, Nature, PubMed, ScienceDirect | `PERPLEXITY_API_KEY` |

---

### Academic Database Tools (no LLM, pure retrieval)

#### ArXiv Recent and Keyword Alert
Search arXiv by keyword + category + date window. Returns title, abstract, authors,
PDF link. **No API key needed.**

```python
# Example calls from chat:
# "Find papers on test-time compute scaling from the last 2 weeks"
# Tool fires: run(query="test-time compute scaling", category="cs.LG", days_back=14)
```

Best for: monitoring new preprints, staying current in a topic area.

---

#### Semantic Scholar Graph
Direct access to the S2 Graph API. Flexible endpoint-based interface — call any
Graph v1 endpoint by name.

```python
# Search
run(endpoint="paper/search", params={"query": "attention is all you need", "limit": 5,
    "fields": "title,abstract,citationCount,openAccessPdf"})

# Get paper by ID
run(endpoint="paper/2302.13971")

# Get citations for a paper
run(endpoint="paper/arXiv:1706.03762/citations",
    params={"fields": "title,authors,year,citationCount", "limit": 20})

# Get references
run(endpoint="paper/arXiv:2303.08774/references",
    params={"fields": "title,year,citationCount"})

# Author lookup
run(endpoint="author/1741101", params={"fields": "name,hIndex,paperCount,papers.title"})
```

Env var: `SEMANTIC_SCHOLAR_API_KEY` — free at https://www.semanticscholar.org/product/api

---

#### Ai2 Asta MCP Search
Natural-language semantic search across millions of full-text papers via AI2's
Asta MCP endpoint. Returns ranked snippets with relevance scores.

```python
run(query="hybrid dense sparse retrieval with BGE-M3", limit=10, min_score_threshold=0.3)
```

Env var: `ASTA_TOOL_KEY` — request access at https://allenai.org/

---

#### OpenAlex Academic Search
250M+ scholarly works; also covers authors, institutions, and funders. Free and open
with no mandatory key — key only needed for higher rate limits (100 req/s vs 10 req/s).

```python
# Search works
run(endpoint="works?search=mixture of experts transformers&per-page=5")

# Get single work
run(endpoint="works/W2741809807")

# Author profile
run(endpoint="authors/A5023888391")

# Institution
run(endpoint="institutions/I4200000001")
```

Env var: `OPENALEX_API_KEY` (optional) — free at https://openalex.org/

---

#### CORE UK Tool
200M+ open access papers aggregated from repositories worldwide. Returns direct
PDF download URLs — excellent for automated ingestion.

```python
run(query="large language model alignment survey", limit=10)
```

Env var: `CORE_API_KEY` — free at https://core.ac.uk/services/api

---

#### Connected Papers Graph
Build a visual similarity graph around any paper. Returns related works, prior
works (foundational), and derivative works (building on it).

```python
# By arXiv ID, DOI, or S2 ID
run(paper_id="2303.08774", action="graph")      # full similarity graph
run(paper_id="2303.08774", action="prior")      # foundational papers
run(paper_id="2303.08774", action="derivative") # papers that build on it
```

Env var: `CONNECTED_PAPERS_API_KEY` — https://www.connectedpapers.com/api

---

#### NCBI PubMed Fetch
35M+ biomedical papers. Search by keyword or fetch a specific paper by PMID.
Key is optional but gives 10 req/s vs 3 req/s.

```python
# Search
run(term="CRISPR base editing 2024", retmax=10)

# Fetch by PMID
run(pmid="38001050")
```

Env var: `NCBI_API_KEY` (optional) — https://www.ncbi.nlm.nih.gov/account/settings/

---

#### Europe PMC OA and Preprint Search
Strong on European life-sciences literature, grants, and preprints (bioRxiv, medRxiv).
**No API key needed.** Returns PMC full-text links automatically when available.

```python
run(query="protein folding AlphaFold", result_type="core", page_size=10)
run(query="COVID-19 vaccine efficacy", result_type="grant")  # grant search
```

---

#### Dimensions / Lens Academic Graph
Covers funding networks, patents, and grants alongside publications. Dimensions
requires a paid API key; Lens.org has a public rate-limited API.

```python
run(query="quantum computing error correction", service="lens", limit=10)
run(query="CRISPR gene editing", service="dimensions")  # returns search URL without key
```

Env vars: `DIMENSIONS_API_KEY` (paid), `LENS_API_KEY` — https://www.lens.org/lens/user/subscriptions

---

#### Consolidated Academic Paper and Metadata Fetcher
Single tool that wraps Semantic Scholar, OpenAlex, CORE, Connected Papers, and NCBI.
Pass `service` + `action` to route to the right backend.

```python
# S2 paper search
run(service="semanticscholar", action="search", query="RLHF reward model")

# OpenAlex works
run(service="openalex", action="search", query="diffusion models image generation")

# CORE full-text
run(service="core", action="search", query="open access transformer survey")

# S2 citations for a paper
run(service="semanticscholar", action="citations", identifier="2302.13971")

# S2 references
run(service="semanticscholar", action="references", identifier="1706.03762")
```

---

### AI / LLM Research Tools

These route a research question to a frontier LLM and return a synthesised answer.
Use when you need reasoning over complex topics, not just retrieval.

#### Google Gemini Academic and Multimodal Query
Gemini 1.5 Pro — strong on math, tables, figures, and recent papers with grounded
citations. Good for questions involving equations or visual content.

```python
run(query="Explain the mathematical basis of RLHF with KL divergence regularisation",
    model="gemini-1.5-pro-latest")
```

Env var: `GEMINI_API_KEY` — https://aistudio.google.com/app/apikey

---

#### Grok Research and Reasoning Query
xAI Grok-3 — deep step-by-step reasoning, strong on math, code, and obscure
cross-disciplinary references.

```python
run(query="Compare the computational complexity of FlashAttention-2 vs standard attention",
    model="grok-3")
```

Env var: `XAI_API_KEY` — https://console.x.ai/

---

#### OpenRouter Research LLM
Route to any frontier model via a single API key. Default is Claude 3.5 Sonnet.
Good for comparing model answers or accessing models not in LiteLLM.

```python
run(query="What is the state of the art in speculative decoding as of 2024?",
    model="anthropic/claude-3.5-sonnet")

# Try a different model:
run(query="...", model="google/gemini-pro-1.5")
run(query="...", model="meta-llama/llama-3.1-405b-instruct")
```

Env var: `OPENROUTER_API_KEY` — https://openrouter.ai/settings/keys

---

#### Perplexity Academic Q&A
Returns cited answers with sources filtered to academic domains (arXiv, Semantic
Scholar, Nature, ScienceDirect, PubMed). Fast and concise.

```python
run(query="What are the main approaches to long-context LLM inference?",
    model="sonar-medium-online", max_tokens=1024)
```

Env var: `PERPLEXITY_API_KEY` — https://www.perplexity.ai/settings/api

---

### Ai2 Extended Tools (Ai2_tools.json)

Four additional Semantic Scholar / Asta tools are available in `Ai2_tools.json`
but not yet imported into Open WebUI. Import them individually:

| Tool | What it does | Key action |
|------|-------------|-----------|
| **Ai2 Semantic Scholar Paper Fetch** | Full metadata for a known paper by ID/DOI/arXiv | `paper_id="2303.08774"` |
| **Ai2 Semantic Scholar Author Fetch** | Author profile: h-index, paper list, affiliations | `author_id="1741101"` |
| **Ai2 Semantic Scholar Recommendations** | Similar papers seeded from a known paper | `paper_id=..., limit=10` |
| **Ai2 Asta MCP Semantic Search** | Full-text semantic search over S2 corpus | Same as Asta MCP above |

All four use `AI2_API_KEY` (set in `env/.env`).

---

### Choosing the Right Tool for the Job

| Task | Best tool(s) |
|------|-------------|
| Find recent preprints on a topic | ArXiv Recent & Keyword Alert |
| Get citation count + PDF for a known paper | Semantic Scholar Graph |
| Full-text semantic search (natural language) | Ai2 Asta MCP Search |
| Biomedical / clinical literature | NCBI PubMed + Europe PMC |
| Funding, grants, patents | Dimensions / Lens |
| Find all papers similar to a seed paper | Connected Papers Graph |
| Papers that cite X / are cited by X | Semantic Scholar citations/references |
| Synthesise complex topic with reasoning | Perplexity → Grok → Gemini |
| Access a model not in LiteLLM | OpenRouter |
| Download PDF and add to local corpus | ArXiv alert → `ingest_arxiv_paper` |

---

### Required env vars — quick checklist

Add these to `env/.env` (all optional — tools degrade gracefully without a key):

```env
# Academic databases
SEMANTIC_SCHOLAR_API_KEY=    # S2 authenticated: 1 req/s (vs public 100 req/5min)
ASTA_TOOL_KEY=               # Ai2 Asta MCP full-text search
OPENALEX_API_KEY=            # OpenAlex high rate limits
CORE_API_KEY=                # CORE open access papers
NCBI_API_KEY=                # PubMed (optional, increases rate limit)
CONNECTED_PAPERS_API_KEY=    # Connected Papers similarity graph
DIMENSIONS_API_KEY=          # Dimensions.ai (paid)
LENS_API_KEY=                # Lens.org (rate-limited free tier available)
AI2_API_KEY=                 # Ai2_tools.json extended S2 tools

# AI reasoning tools
GEMINI_API_KEY=
XAI_API_KEY=
PERPLEXITY_API_KEY=
OPENROUTER_API_KEY=
```

---

## Getting Community Tools

The Open WebUI community maintains a library of pre-built tools:

- **Official hub:** https://openwebui.com/tools
- **Filter by:** Academic search, productivity, coding, web browsing
- **Install:** Copy the tool URL → Open WebUI → Workspace → Tools → Import from URL

Notable tools worth installing:

| Tool | What it does |
|------|-------------|
| ArXiv Search | Structured arXiv API search |
| Semantic Scholar | Citation graph + paper recommendations |
| GitHub Search | Code search across GitHub repos |
| Python REPL | Execute Python code in-chat |
| URL/Web reader | Fetch and summarise any webpage |
| OpenAlex | Open scholarly literature database |

---

## Tool Configuration in Admin Settings

**Admin → Settings → Tools:**

| Setting | Recommendation |
|---------|---------------|
| Trust tools from community | Disabled (review code before enabling) |
| Tool timeout | 30s (increase to 60s for PDF downloads) |
| Auto-enable tools | Only for tools you use in every session |
| Tool result max length | 4000 tokens (enough for 5-10 paper abstracts) |

**Per-model tool support:**
- `local-model` (SparkRun) — tool calling works if your model supports it
  (Qwen3, Nemotron, Phi-4 all support function calling)
- `phi-mini` — good for tool routing / classification, fast tool-augmented responses
- `spark-scholar` (RAG proxy) — the proxy itself does retrieval; tools add live lookups on top

---

## Debugging Tools

If a tool is not being called when expected:

1. **Check the model supports tool calling:**
   ```bash
   curl http://localhost:4000/v1/models -H "Authorization: Bearer simple-api-key" | \
     python3 -c "import json,sys; [print(m['id'], m.get('capabilities','')) for m in json.load(sys.stdin)['data']]"
   ```

2. **Enable tool call verbose logging:**
   Open WebUI → Admin → Settings → Logs → Enable debug mode

3. **Test the tool function directly:**
   Open WebUI → Workspace → Tools → click your tool → Run in playground

4. **Check tool result in Langfuse:**
   Langfuse → Traces → find the query → look for `tool_call` spans

5. **Verify the environment variable reaches the tool:**
   ```python
   # Add to tool for debugging:
   import os
   print(os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "NOT SET"))
   ```
