"""Phase 2: build and SAVE every representation once, so Phase 3 never retrains.

Usage:
    python -m src.build_representations            # everything (~12 min on 2 CPU cores)
    python -m src.build_representations --only count
    python -m src.build_representations --only embeddings

Input : data/processed/cleaned_reviews.parquet   (Phase 1 output, used unchanged)
Output: models/bow_vectorizer.joblib, models/tfidf_vectorizer.joblib,
        models/w2v_cbow.model, models/w2v_skipgram.model, models/fasttext.model,
        models/training_log.json
"""
from __future__ import annotations
import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "cleaned_reviews.parquet"
MODELS = ROOT / "models"

# ---- Hyper-parameters: IDENTICAL for CBOW, Skip-gram and fastText -------------
EMB_PARAMS = dict(
    vector_size=100,   # dimensions of each word vector
    window=5,          # up to 5 words either side form the context
    min_count=5,       # ignore words seen < 5 times (mostly typos)
    epochs=5,          # passes over the 15.2M-token corpus
    negative=5,        # negative sampling: 5 noise words per positive pair
    sample=1e-3,       # down-sample very frequent words
    workers=2,
    seed=42,
)
FASTTEXT_EXTRA = dict(min_n=3, max_n=6, bucket=500_000)  # character n-grams of length 3..6

# Count-based vectorizers. Text is already cleaned, so we only split on spaces.
COUNT_PARAMS = dict(tokenizer=str.split, token_pattern=None, lowercase=False, min_df=5)


def load_texts() -> pd.Series:
    return pd.read_parquet(DATA, columns=["clean_text"])["clean_text"]


def build_count(texts: pd.Series) -> dict:
    log = {}
    bow = CountVectorizer(**COUNT_PARAMS)
    X = bow.fit_transform(texts)
    joblib.dump(bow, MODELS / "bow_vectorizer.joblib")
    log["bow"] = {"shape": list(X.shape), "nnz": int(X.nnz)}

    # Phase-3 TF-IDF: unigrams + bigrams (keeps "not good"), sublinear tf
    tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, **COUNT_PARAMS)
    X = tfidf.fit_transform(texts)
    joblib.dump(tfidf, MODELS / "tfidf_vectorizer.joblib")
    log["tfidf"] = {"shape": list(X.shape), "nnz": int(X.nnz), "ngram_range": [1, 2]}
    return log


def build_embeddings(texts: pd.Series) -> dict:
    from gensim.models import FastText, Word2Vec
    sentences = [t.split() for t in texts]
    log = {"params": EMB_PARAMS, "fasttext_extra": FASTTEXT_EXTRA,
           "tokens": sum(map(len, sentences)), "documents": len(sentences)}
    for name, cls, extra in [("w2v_cbow", Word2Vec, {"sg": 0}),
                             ("w2v_skipgram", Word2Vec, {"sg": 1}),
                             ("fasttext", FastText, {"sg": 1, **FASTTEXT_EXTRA})]:
        t = time.time()
        model = cls(sentences, **EMB_PARAMS, **extra)
        model.save(str(MODELS / f"{name}.model"))
        log[name] = {"vocab": len(model.wv), "seconds": round(time.time() - t, 1), **extra}
        print(f"{name}: vocab={len(model.wv):,}  {log[name]['seconds']} s", flush=True)
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["count", "embeddings"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    MODELS.mkdir(exist_ok=True)
    texts = load_texts()
    log_path = MODELS / "training_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    if args.only in (None, "count"):
        log["count"] = build_count(texts)
        print("count models:", log["count"], flush=True)
    if args.only in (None, "embeddings"):
        log["embeddings"] = build_embeddings(texts)
    log_path.write_text(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
