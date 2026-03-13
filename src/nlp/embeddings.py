"""
Filing embedding computation using sentence transformers.

Computes document-level embeddings for SEC filings and measures the
cosine similarity between consecutive filings for the same company.
Large changes in embedding space indicate material content changes
even when keyword-level sentiment is flat.

Uses a distilled model (all-MiniLM-L6-v2) that runs on CPU in
reasonable time (~100ms per filing section).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


# Lazy-load sentence-transformers to avoid import cost when not needed
_MODEL = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model():
    """Lazy-load the sentence transformer model."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def compute_embedding(text: str, max_length: int = 10_000) -> np.ndarray:
    """Compute a document-level embedding for a text.

    Truncates to max_length characters before embedding (the model
    has a 512 token limit; we chunk and average for longer texts).

    Returns a 1-D numpy array (the embedding vector).
    """
    if not text or len(text.strip()) < 50:
        return np.zeros(384)  # MiniLM output dimension

    model = _get_model()

    # For long texts, chunk into ~500 word segments and average embeddings
    words = text.split()
    chunk_size = 400  # ~400 words per chunk
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    if not chunks:
        return np.zeros(384)

    # Limit to first 25 chunks (~10K words) for efficiency
    chunks = chunks[:25]
    embeddings = model.encode(chunks, show_progress_bar=False)
    return np.mean(embeddings, axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_embedding_change(
    current_text: str,
    prior_text: str,
) -> dict[str, float]:
    """Compute embedding-based change features between two filings.

    Returns:
        full_similarity: cosine sim of full document embeddings
        embedding_distance: 1 - cosine_similarity (higher = more change)
    """
    current_emb = compute_embedding(current_text)
    prior_emb = compute_embedding(prior_text)

    sim = cosine_similarity(current_emb, prior_emb)
    return {
        "embedding_similarity": sim,
        "embedding_distance": 1.0 - sim,
    }
