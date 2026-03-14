# PDF Collections

Each subdirectory maps to a Qdrant collection of the same name.
Drop PDFs into a directory and run `ingest/05_ingest_pdfs.py --input-dir RAG/pdfs/<dir>/ --collection <dir>` to create/update the collection.

## Demo Collections (tracked in git — freely licensed)

| Directory | Collection | Description | License |
|---|---|---|---|
| `demo-bayesian-statistics/` | demo-bayesian-statistics | Think Bayes, Think Stats | CC-BY-NC (Allen Downey) |
| `demo-computer-science/` | demo-computer-science | Think C++, Think Java, Think OCaml, Think OS, Scala By Example | CC-BY-NC / Free |
| `demo-python-programming/` | demo-python-programming | Think Python | CC-BY-NC (Allen Downey) |

## User Collections (gitignored — add your own PDFs)

| Directory | Description | Example Content |
|---|---|---|
| `ML/` | Machine learning textbooks and guides | Bishop PRML, Murphy ML:APP, Elements of Statistical Learning |
| `Deep-Learning/` | Deep learning textbooks | Goodfellow et al. chapters |
| `Python-Books/` | Python programming books | Python for Finance, ML with Python |
| `Coding/` | General programming books | Think C++, Node.js in Action, Scala |
| `Bayes/` | Bayesian statistics | Naive Bayes, Think Bayes |
| `Statistics/` | Statistics textbooks | Think Stats |
| `Data-Mining/` | Data mining and ML tools | Weka, practical ML techniques |
| `LLM/` | Large language model guides | Compact Guide to LLMs |
| `Cuda/` | GPU programming | CUDA by Example, CUDA C Programming Guide |

## How It Works

1. **Docling** (containerized, port 9099) converts PDFs to structured Markdown
2. **Section-aware chunking** splits on heading boundaries, not blind sliding windows
3. **BGE-M3** embeds chunks (dense 1024-dim + sparse vectors)
4. **HDBSCAN + UMAP** auto-clusters chunks into topics
5. **Qwen3 via vLLM** generates human-readable topic names
6. Points are upserted into the matching Qdrant collection
