"""Run the full Phase-1 pipeline: load raw -> dedupe -> clean -> save.

Usage:
    python -m src.run_pipeline

Output: data/processed/cleaned_reviews.parquet  (+ cleaned_reviews.csv) and
        data/processed/vocab_stats.json
"""
from __future__ import annotations
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from .cleaning import clean_text, surface_features
from .load_data import load_raw

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"

SENTIMENT = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}


def main() -> pd.DataFrame:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    n_raw = len(df)

    # 1. Exact duplicates: the same review is posted under several product variants.
    df = df.drop_duplicates(subset=["UserId", "Time", "Text"], keep="first")
    n_dedup = len(df)

    # 2. Vocabulary BEFORE cleaning: case-sensitive whitespace tokens of raw Text.
    vocab_before = Counter(tok for text in df["Text"] for tok in text.split())

    # 3. Features that cleaning would destroy (case + exclamation marks), from RAW text.
    feats = pd.DataFrame([surface_features(t) for t in df["Text"]], index=df.index)

    # 4. Clean (identical function every run).
    df["clean_text"] = df["Text"].map(clean_text)
    df["clean_summary"] = df["Summary"].fillna("").map(clean_text)
    df = pd.concat([df, feats], axis=1)

    # 5. Drop reviews that are empty after cleaning, add labels.
    df = df[df["clean_text"].str.len() > 0].copy()
    n_final = len(df)
    df["sentiment"] = df["Score"].map(SENTIMENT)

    # 6. Vocabulary AFTER cleaning.
    vocab_after = Counter(tok for text in df["clean_text"] for tok in text.split())

    cols = ["Id", "ProductId", "UserId", "Score", "sentiment", "HelpfulnessNumerator",
            "HelpfulnessDenominator", "Time", "Summary", "Text",
            "clean_summary", "clean_text", "n_exclaim", "caps_ratio"]
    df = df[cols].reset_index(drop=True)
    df.to_parquet(OUT_DIR / "cleaned_reviews.parquet", index=False)
    df.to_csv(OUT_DIR / "cleaned_reviews.csv", index=False)

    stats = {
        "rows_raw": n_raw,
        "rows_after_dedup": n_dedup,
        "duplicates_removed": n_raw - n_dedup,
        "rows_final": n_final,
        "empty_after_cleaning": n_dedup - n_final,
        "vocab_before": len(vocab_before),
        "vocab_after": len(vocab_after),
        "tokens_before": sum(vocab_before.values()),
        "tokens_after": sum(vocab_after.values()),
        "vocab_reduction_pct": round(100 * (1 - len(vocab_after) / len(vocab_before)), 2),
        "seconds": round(time.time() - t0, 1),
    }
    (OUT_DIR / "vocab_stats.json").write_text(json.dumps(stats, indent=2))
    for k, v in stats.items():
        print(f"{k:>22}: {v:,}" if isinstance(v, int) else f"{k:>22}: {v}")
    return df


if __name__ == "__main__":
    main()
