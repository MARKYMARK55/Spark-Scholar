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
