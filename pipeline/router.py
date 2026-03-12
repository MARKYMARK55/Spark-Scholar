"""
pipeline/router.py
==================
Collection routing for the Spark Scholar pipeline.

Two entry points:
  - route_query(query)       → list of 1-3 collection names for retrieval
  - route_paper(categories)  → single collection name for indexing

Routing uses a two-layer strategy:
  1. Exact arXiv category → collection mapping (CATEGORY_ROUTING)
  2. Domain keyword heuristics for free-text queries (no explicit category)
  3. Fallback to multi-collection search when ambiguous
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category → collection mapping (exactly as configured in Qdrant)
# ---------------------------------------------------------------------------

CATEGORY_ROUTING: dict[str, str] = {
    # CS — ML / AI / Vision / NLP
    "cs.ai": "arxiv-cs-ml-ai",
    "cs.lg": "arxiv-cs-ml-ai",
    "cs.ne": "arxiv-cs-ml-ai",
    "cs.cv": "arxiv-cs-ml-ai",
    "cs.cl": "arxiv-cs-ml-ai",
    "cs.ir": "arxiv-cs-ml-ai",
    # CS — Systems / Theory
    "cs.ds": "arxiv-cs-systems-theory",
    "cs.cc": "arxiv-cs-systems-theory",
    "cs.dc": "arxiv-cs-systems-theory",
    "cs.pl": "arxiv-cs-systems-theory",
    "cs.os": "arxiv-cs-systems-theory",
    "cs.se": "arxiv-cs-systems-theory",
    "cs.ni": "arxiv-cs-systems-theory",
    "cs.cr": "arxiv-cs-systems-theory",
    # Math — Pure
    "math.ag": "arxiv-math-pure",
    "math.nt": "arxiv-math-pure",
    "math.gr": "arxiv-math-pure",
    "math.gt": "arxiv-math-pure",
    "math.at": "arxiv-math-pure",
    # Math — Applied
    "math.oc": "arxiv-math-applied",
    "math.na": "arxiv-math-applied",
    "math.co": "arxiv-math-applied",
    "math.pr": "arxiv-math-applied",
    "math.st": "arxiv-math-applied",
    # Math-Physics
    "math-ph": "arxiv-math-phys",
    "math.mp": "arxiv-math-phys",
    # Statistics / EESS
    "stat.ml": "arxiv-stat-eess",
    "stat.me": "arxiv-stat-eess",
    "stat.th": "arxiv-stat-eess",
    "eess.sp": "arxiv-stat-eess",
    "eess.sy": "arxiv-stat-eess",
    # Quantum / GR
    "quant-ph": "arxiv-quantph-grqc",
    "gr-qc": "arxiv-quantph-grqc",
    # HEP
    "hep-th": "arxiv-hep",
    "hep-ph": "arxiv-hep",
    "hep-ex": "arxiv-hep",
    "hep-lat": "arxiv-hep",
    # Condensed Matter
    "cond-mat": "arxiv-condmat",
    # Astrophysics
    "astro-ph": "arxiv-astro",
    # Nuclear / Nonlinear / Other Physics
    "nucl-th": "arxiv-nucl-nlin-physother",
    "nucl-ex": "arxiv-nucl-nlin-physother",
    "nlin": "arxiv-nucl-nlin-physother",
    "physics": "arxiv-nucl-nlin-physother",
    # Quantitative Biology / Finance / Economics
    "q-bio": "arxiv-qbio-qfin-econ",
    "q-fin": "arxiv-qbio-qfin-econ",
    "econ": "arxiv-qbio-qfin-econ",
}

ALL_COLLECTIONS: list[str] = [
    "arXiv",
    "arxiv-cs-ml-ai",
    "arxiv-condmat",
    "arxiv-astro",
    "arxiv-hep",
    "arxiv-math-applied",
    "arxiv-math-phys",
    "arxiv-math-pure",
    "arxiv-misc",
    "arxiv-nucl-nlin-physother",
    "arxiv-qbio-qfin-econ",
    "arxiv-quantph-grqc",
    "arxiv-stat-eess",
    "arxiv-cs-systems-theory",
]

# ---------------------------------------------------------------------------
# Domain keyword → collection weight map
# Each keyword adds weight to the specified collection when found in the query.
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    # ML / AI / NLP / Vision
    "transformer": [("arxiv-cs-ml-ai", 2.0)],
    "attention mechanism": [("arxiv-cs-ml-ai", 2.0)],
    "neural network": [("arxiv-cs-ml-ai", 1.5)],
    "deep learning": [("arxiv-cs-ml-ai", 2.0)],
    "machine learning": [("arxiv-cs-ml-ai", 1.5)],
    "large language model": [("arxiv-cs-ml-ai", 2.0)],
    "llm": [("arxiv-cs-ml-ai", 2.0)],
    "gpt": [("arxiv-cs-ml-ai", 1.5)],
    "bert": [("arxiv-cs-ml-ai", 1.5)],
    "fine.?tun": [("arxiv-cs-ml-ai", 1.5)],
    "reinforcement learning": [("arxiv-cs-ml-ai", 1.5)],
    "rlhf": [("arxiv-cs-ml-ai", 2.0)],
    "diffusion model": [("arxiv-cs-ml-ai", 2.0), ("arxiv-condmat", 0.5)],
    "generative model": [("arxiv-cs-ml-ai", 1.5)],
    "image classification": [("arxiv-cs-ml-ai", 1.5)],
    "object detection": [("arxiv-cs-ml-ai", 1.5)],
    "computer vision": [("arxiv-cs-ml-ai", 1.5)],
    "natural language": [("arxiv-cs-ml-ai", 1.5)],
    "embedding": [("arxiv-cs-ml-ai", 1.0)],
    "retrieval augmented": [("arxiv-cs-ml-ai", 2.0)],
    "rag": [("arxiv-cs-ml-ai", 1.5)],
    "knowledge graph": [("arxiv-cs-ml-ai", 1.0)],
    "graph neural": [("arxiv-cs-ml-ai", 1.5)],
    "contrastive learning": [("arxiv-cs-ml-ai", 1.5)],
    "self.?supervised": [("arxiv-cs-ml-ai", 1.5)],
    "zero.?shot": [("arxiv-cs-ml-ai", 1.5)],
    "few.?shot": [("arxiv-cs-ml-ai", 1.5)],
    "prompt": [("arxiv-cs-ml-ai", 1.0)],
    "token": [("arxiv-cs-ml-ai", 0.5)],
    "inference": [("arxiv-cs-ml-ai", 0.5)],
    # CS Systems / Theory
    "distributed": [("arxiv-cs-systems-theory", 1.5)],
    "consensus": [("arxiv-cs-systems-theory", 1.5)],
    "blockchain": [("arxiv-cs-systems-theory", 1.5)],
    "compiler": [("arxiv-cs-systems-theory", 2.0)],
    "operating system": [("arxiv-cs-systems-theory", 1.5)],
    "network protocol": [("arxiv-cs-systems-theory", 1.5)],
    "cryptography": [("arxiv-cs-systems-theory", 1.5)],
    "algorithm": [("arxiv-cs-systems-theory", 1.0)],
    "complexity": [("arxiv-cs-systems-theory", 1.0), ("arxiv-math-pure", 0.5)],
    "scheduling": [("arxiv-cs-systems-theory", 1.0)],
    "fault tolerance": [("arxiv-cs-systems-theory", 1.5)],
    "cache": [("arxiv-cs-systems-theory", 1.0)],
    "parallel computing": [("arxiv-cs-systems-theory", 1.5)],
    "gpu": [("arxiv-cs-systems-theory", 1.0), ("arxiv-cs-ml-ai", 0.5)],
    # Quantum / GR
    "quantum": [("arxiv-quantph-grqc", 2.0)],
    "qubit": [("arxiv-quantph-grqc", 2.0)],
    "entanglement": [("arxiv-quantph-grqc", 2.0)],
    "superposition": [("arxiv-quantph-grqc", 1.5)],
    "quantum circuit": [("arxiv-quantph-grqc", 2.0)],
    "quantum computer": [("arxiv-quantph-grqc", 2.0)],
    "quantum error": [("arxiv-quantph-grqc", 2.0)],
    "variational quantum": [("arxiv-quantph-grqc", 2.0)],
    "vqe": [("arxiv-quantph-grqc", 2.0)],
    "qaoa": [("arxiv-quantph-grqc", 2.0)],
    "general relativity": [("arxiv-quantph-grqc", 2.0)],
    "black hole": [("arxiv-quantph-grqc", 1.5), ("arxiv-astro", 1.0)],
    "gravitational wave": [("arxiv-quantph-grqc", 1.5), ("arxiv-astro", 1.5)],
    "spacetime": [("arxiv-quantph-grqc", 1.5)],
    # HEP
    "higgs": [("arxiv-hep", 2.0)],
    "boson": [("arxiv-hep", 1.5), ("arxiv-condmat", 0.5)],
    "fermion": [("arxiv-hep", 1.5), ("arxiv-condmat", 0.5)],
    "hadron": [("arxiv-hep", 2.0)],
    "quark": [("arxiv-hep", 2.0)],
    "lepton": [("arxiv-hep", 2.0)],
    "standard model": [("arxiv-hep", 2.0)],
    "supersymmetry": [("arxiv-hep", 2.0)],
    "string theory": [("arxiv-hep", 2.0), ("arxiv-math-phys", 0.5)],
    "particle physics": [("arxiv-hep", 2.0)],
    "collider": [("arxiv-hep", 2.0)],
    "lhc": [("arxiv-hep", 2.0)],
    # Condensed Matter
    "superconductor": [("arxiv-condmat", 2.0)],
    "superconductivity": [("arxiv-condmat", 2.0)],
    "topological": [("arxiv-condmat", 1.5), ("arxiv-math-pure", 0.5)],
    "phase transition": [("arxiv-condmat", 1.5)],
    "spin": [("arxiv-condmat", 1.0), ("arxiv-hep", 0.5)],
    "magnetism": [("arxiv-condmat", 1.5)],
    "lattice model": [("arxiv-condmat", 1.5), ("arxiv-hep", 0.5)],
    "band structure": [("arxiv-condmat", 2.0)],
    "fermi": [("arxiv-condmat", 1.5)],
    "phonon": [("arxiv-condmat", 2.0)],
    "bose.?einstein": [("arxiv-condmat", 2.0)],
    "density functional": [("arxiv-condmat", 2.0)],
    "dft": [("arxiv-condmat", 1.5)],
    # Astrophysics
    "galaxy": [("arxiv-astro", 2.0)],
    "star": [("arxiv-astro", 1.0)],
    "stellar": [("arxiv-astro", 1.5)],
    "cosmology": [("arxiv-astro", 2.0)],
    "dark matter": [("arxiv-astro", 2.0), ("arxiv-hep", 0.5)],
    "dark energy": [("arxiv-astro", 2.0)],
    "telescope": [("arxiv-astro", 1.5)],
    "exoplanet": [("arxiv-astro", 2.0)],
    "redshift": [("arxiv-astro", 1.5)],
    "cosmic": [("arxiv-astro", 1.5)],
    "solar": [("arxiv-astro", 1.0)],
    "neutron star": [("arxiv-astro", 2.0), ("arxiv-nucl-nlin-physother", 0.5)],
    "supernova": [("arxiv-astro", 2.0)],
    "pulsar": [("arxiv-astro", 2.0)],
    # Nuclear / nonlinear / other physics
    "nuclear": [("arxiv-nucl-nlin-physother", 2.0)],
    "fission": [("arxiv-nucl-nlin-physother", 2.0)],
    "fusion": [("arxiv-nucl-nlin-physother", 1.5)],
    "chaos": [("arxiv-nucl-nlin-physother", 1.5)],
    "soliton": [("arxiv-nucl-nlin-physother", 1.5)],
    "nonlinear": [("arxiv-nucl-nlin-physother", 1.5)],
    "turbulence": [("arxiv-nucl-nlin-physother", 1.5)],
    "fluid dynamics": [("arxiv-nucl-nlin-physother", 1.5)],
    "plasma": [("arxiv-nucl-nlin-physother", 1.5)],
    # Math — Pure
    "algebraic geometry": [("arxiv-math-pure", 2.0)],
    "number theory": [("arxiv-math-pure", 2.0)],
    "group theory": [("arxiv-math-pure", 2.0)],
    "topology": [("arxiv-math-pure", 2.0)],
    "manifold": [("arxiv-math-pure", 1.5)],
    "cohomology": [("arxiv-math-pure", 2.0)],
    "homology": [("arxiv-math-pure", 2.0)],
    "sheaf": [("arxiv-math-pure", 2.0)],
    "galois": [("arxiv-math-pure", 2.0)],
    "prime": [("arxiv-math-pure", 1.0)],
    "moduli": [("arxiv-math-pure", 1.5)],
    # Math — Applied
    "optimization": [("arxiv-math-applied", 1.5), ("arxiv-cs-ml-ai", 0.5)],
    "numerical method": [("arxiv-math-applied", 2.0)],
    "combinatorics": [("arxiv-math-applied", 1.5)],
    "probability": [("arxiv-math-applied", 1.5)],
    "stochastic": [("arxiv-math-applied", 1.5)],
    "markov": [("arxiv-math-applied", 1.0)],
    "graph theory": [("arxiv-math-applied", 1.5)],
    "finite element": [("arxiv-math-applied", 2.0)],
    "linear programming": [("arxiv-math-applied", 1.5)],
    # Math-Physics
    "quantum field theory": [("arxiv-math-phys", 2.0), ("arxiv-hep", 1.0)],
    "conformal field": [("arxiv-math-phys", 2.0), ("arxiv-hep", 1.0)],
    "symplectic": [("arxiv-math-phys", 2.0)],
    "integrable": [("arxiv-math-phys", 1.5)],
    "gauge theory": [("arxiv-math-phys", 1.5), ("arxiv-hep", 1.0)],
    # Statistics / EESS
    "bayesian": [("arxiv-stat-eess", 1.5)],
    "causal": [("arxiv-stat-eess", 1.5)],
    "regression": [("arxiv-stat-eess", 1.0)],
    "hypothesis test": [("arxiv-stat-eess", 1.5)],
    "signal processing": [("arxiv-stat-eess", 2.0)],
    "control system": [("arxiv-stat-eess", 1.5)],
    "time series": [("arxiv-stat-eess", 1.5)],
    "survival analysis": [("arxiv-stat-eess", 1.5)],
    # Quantitative Bio / Finance / Economics
    "protein": [("arxiv-qbio-qfin-econ", 2.0)],
    "genomics": [("arxiv-qbio-qfin-econ", 2.0)],
    "dna": [("arxiv-qbio-qfin-econ", 1.5)],
    "rna": [("arxiv-qbio-qfin-econ", 1.5)],
    "gene": [("arxiv-qbio-qfin-econ", 1.5)],
    "neural coding": [("arxiv-qbio-qfin-econ", 1.5)],
    "epidemiology": [("arxiv-qbio-qfin-econ", 2.0)],
    "option pricing": [("arxiv-qbio-qfin-econ", 2.0)],
    "portfolio": [("arxiv-qbio-qfin-econ", 1.5)],
    "market": [("arxiv-qbio-qfin-econ", 1.0)],
    "economic": [("arxiv-qbio-qfin-econ", 1.5)],
    "game theory": [("arxiv-qbio-qfin-econ", 1.5)],
    "mechanism design": [("arxiv-qbio-qfin-econ", 1.5)],
}


def _score_collections(query: str) -> dict[str, float]:
    """
    Return a score for each collection based on keyword overlap with the query.
    Uses regex matching so multi-word phrases and wildcards work.
    """
    query_lower = query.lower()
    scores: dict[str, float] = defaultdict(float)

    for pattern, collection_weights in DOMAIN_KEYWORDS.items():
        if re.search(pattern, query_lower):
            for collection, weight in collection_weights:
                scores[collection] += weight

    return dict(scores)


def route_query(query: str, max_collections: int = 3) -> List[str]:
    """
    Determine which Qdrant collections to search for a free-text query.

    Strategy
    --------
    1. Score collections using domain keyword heuristics.
    2. Return the top-scoring collections (up to max_collections).
    3. If no keywords match, fall back to ["arXiv"] (the catch-all collection).
    4. Collections with score >= 50% of the max score are included.

    Parameters
    ----------
    query : str
        The user's search query.
    max_collections : int
        Maximum number of collections to return.

    Returns
    -------
    list[str]
        Ordered list of collection names (most relevant first).
    """
    if not query or not query.strip():
        return ["arXiv"]

    scores = _score_collections(query)

    if not scores:
        logger.debug("route_query: no keyword matches → fallback to arXiv")
        return ["arXiv"]

    # Sort by score descending
    sorted_collections = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    max_score = sorted_collections[0][1]

    # Include collections with at least 50% of top score, up to max_collections
    threshold = max_score * 0.5
    selected = [
        coll
        for coll, score in sorted_collections
        if score >= threshold
    ][:max_collections]

    logger.debug(
        "route_query: query=%r → collections=%s (scores=%s)",
        query[:80],
        selected,
        {c: round(s, 2) for c, s in sorted_collections[:5]},
    )
    return selected


def route_paper(categories_str: str) -> str:
    """
    Route a paper to its primary Qdrant collection based on its arXiv category string.

    The categories_str is typically the 'categories' field from the arXiv metadata,
    which may contain multiple space-separated category codes, e.g. "cs.LG cs.AI stat.ML".

    Strategy
    --------
    1. Split by whitespace and check each category against CATEGORY_ROUTING.
    2. Use the first matching category as the primary assignment.
    3. Fall back to "arXiv" if no category matches.

    Parameters
    ----------
    categories_str : str
        Space-separated arXiv category codes (case-insensitive).

    Returns
    -------
    str
        Qdrant collection name.
    """
    if not categories_str or not categories_str.strip():
        return "arXiv"

    cats = [c.strip().lower() for c in categories_str.split()]

    for cat in cats:
        # Exact match
        if cat in CATEGORY_ROUTING:
            return CATEGORY_ROUTING[cat]

        # Prefix match for compound categories like "cond-mat.mes-hall"
        prefix = cat.split(".")[0]
        if prefix in CATEGORY_ROUTING:
            return CATEGORY_ROUTING[prefix]

    logger.debug("route_paper: no match for categories=%r → arXiv", categories_str)
    return "arXiv"


def get_all_collections() -> List[str]:
    """Return the list of all known collection names."""
    return ALL_COLLECTIONS.copy()


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    tests = [
        "How does the transformer attention mechanism work?",
        "Superconductivity in cuprate materials near phase transition",
        "Galaxy formation and dark matter halos",
        "Quantum error correction codes for fault-tolerant computation",
        "Algebraic geometry and Hodge theory",
        "Protein folding with deep learning",
        "Random query about nothing specific",
    ]

    print("=== route_query ===")
    for q in tests:
        result = route_query(q)
        print(f"  Q: {q[:60]}")
        print(f"  → {result}\n")

    print("=== route_paper ===")
    paper_tests = [
        ("cs.LG cs.AI", "deep learning paper"),
        ("hep-th hep-ph", "string theory paper"),
        ("cond-mat.mes-hall", "mesoscopic physics paper"),
        ("astro-ph.GA astro-ph.CO", "galaxy paper"),
        ("unknown.xx", "unknown category paper"),
    ]
    for cats, desc in paper_tests:
        result = route_paper(cats)
        print(f"  {desc}: categories={cats!r} → {result}")
