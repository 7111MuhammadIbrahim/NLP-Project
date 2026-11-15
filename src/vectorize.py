"""The Phase-3 interface: raw document -> vector.

    from src.vectorize import vectorize
    v = vectorize("I didn't like this coffee at all!!", method="tfidf")

Methods: "bow", "tfidf"            -> scipy.sparse row (1 x V)   (dense=True gives numpy)
         "cbow", "skipgram", "fasttext" -> numpy array (100,), mean of the word vectors

The raw text goes through the SAME Phase-1 clean_text() used for training.
Models are only LOADED here (never trained) and cached after the first call.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

import numpy as np

from .cleaning import clean_text

MODELS = Path(__file__).resolve().parents[1] / "models"
_FILES = {"bow": "bow_vectorizer.joblib", "tfidf": "tfidf_vectorizer.joblib",
          "cbow": "w2v_cbow.model", "skipgram": "w2v_skipgram.model", "fasttext": "fasttext.model"}
METHODS = tuple(_FILES)


@lru_cache(maxsize=None)
def load(method: str):
    path = MODELS / _FILES[method]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m src.build_representations` once.")
    if method in ("bow", "tfidf"):
        import joblib
        return joblib.load(path)
    if method == "fasttext":
        from gensim.models import FastText
        return FastText.load(str(path)).wv          # keeps sub-word buckets for OOV words
    from gensim.models import Word2Vec
    return Word2Vec.load(str(path)).wv


def embed_tokens(tokens: list[str], method: str) -> np.ndarray:
    """Mean of word vectors. Word2Vec skips OOV words; fastText builds them from n-grams."""
    kv = load(method)
    vecs = [kv[t] for t in tokens if method == "fasttext" or t in kv.key_to_index]
    return np.mean(vecs, axis=0) if vecs else np.zeros(kv.vector_size, dtype=np.float32)


def vectorize(doc: str, method: str = "tfidf", dense: bool = False):
    if method not in _FILES:
        raise ValueError(f"method must be one of {METHODS}")
    cleaned = clean_text(doc)
    if method in ("bow", "tfidf"):
        row = load(method).transform([cleaned])
        return row.toarray()[0] if dense else row
    return embed_tokens(cleaned.split(), method)


def vectorize_many(docs, method: str = "tfidf"):
    """Batch version for Phase 3 (sparse matrix or 2-D numpy array)."""
    cleaned = [clean_text(d) for d in docs]
    if method in ("bow", "tfidf"):
        return load(method).transform(cleaned)
    return np.vstack([embed_tokens(c.split(), method) for c in cleaned])


if __name__ == "__main__":
    demo = "I didn't like this coffee at all!! Way too bitter <br />and overpriced."
    for m in METHODS:
        v = vectorize(demo, m)
        print(f"{m:9} shape={v.shape}  non-zero={v.nnz if hasattr(v, 'nnz') else int(np.count_nonzero(v))}")
