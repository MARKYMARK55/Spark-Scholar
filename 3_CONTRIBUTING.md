# Contributing to Spark-Scholar

Thank you for your interest in contributing. This is an early-stage project and all contributions are welcome — from bug reports to new ingestion pipelines.

## Ways to contribute

- **Bug reports** — open a GitHub issue with your hardware, Python version, and the exact error output
- **Documentation fixes** — typos, unclear steps, missing env vars
- **New ingestion sources** — additional academic databases, new document formats
- **Pipeline improvements** — better chunking, routing heuristics, reranking strategies
- **Evaluation scripts** — retrieval@k benchmarks, answer quality measures
- **Alternative hardware configs** — configs for non-DGX setups (A100, 4090, etc.)

## Development setup

```bash
git clone https://github.com/MARKYMARK55/spark-scholar
cd spark-scholar
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example env/.env
```

For pipeline changes, you can run the retrieval stack without the full embedding services by pointing `QDRANT_URL` at an existing Qdrant instance and mocking the embedding endpoints.

## Code style

- Python: **ruff** for linting, **black** for formatting (line length 100)
- Type hints on all public functions in `pipeline/` and `ingest/`
- New scripts should follow the numbered pattern in `ingest/` and include a `--help` docstring

```bash
pip install ruff black
ruff check .
black --line-length 100 .
```

## Running the eval

Before submitting pipeline changes, run the retrieval eval to confirm you haven't regressed:

```bash
# Quick check — 5 queries, hybrid mode
python eval/retrieval_eval.py --limit 5 --quiet

# Full comparison before/after your change
python eval/retrieval_eval.py --mode all --output eval/before.json
# ... make your change ...
python eval/retrieval_eval.py --mode all --output eval/after.json
```

Include the summary table in your PR description if you changed anything in `pipeline/`.

## Pull request checklist

- [ ] Tested on DGX Spark or documented alternative hardware
- [ ] Updated relevant `docs/` section if behaviour changes
- [ ] Added env vars to `.env.example` and `env/README.md` if the PR needs new config
- [ ] New ingest scripts are numbered and placed in `ingest/`

## Scope notes

This project is explicitly optimised for **DGX Spark** and similar high-unified-memory machines. PRs that target this hardware are prioritised. Contributions that add support for other platforms (A100, consumer GPUs, CPU-only) are welcome as long as they don't break the primary DGX Spark path.

## Questions

Open a GitHub Discussion or post in the [NVIDIA DGX Spark forums](https://forums.developer.nvidia.com/c/dgx/).
