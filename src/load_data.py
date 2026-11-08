"""Load the raw Amazon Fine Food Reviews data.

The raw CSV is NOT committed. Put Reviews.csv in data/raw/ (see README), or let
this script fetch it from Kaggle if `kagglehub` is installed and you are logged in.

Usage:
    python -m src.load_data            # checks the file and prints a summary
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "Reviews.csv"
KAGGLE_SLUG = "snap/amazon-fine-food-reviews"


def download_if_missing() -> Path:
    """Return the path of Reviews.csv, trying a Kaggle download if absent."""
    if RAW_CSV.exists():
        return RAW_CSV
    try:
        import shutil
        import kagglehub  # optional dependency, only needed for auto-download
        src = Path(kagglehub.dataset_download(KAGGLE_SLUG)) / "Reviews.csv"
        RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, RAW_CSV)
        return RAW_CSV
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            f"{RAW_CSV} not found and automatic download failed ({exc}).\n"
            f"Download 'Reviews.csv' from https://www.kaggle.com/datasets/{KAGGLE_SLUG} "
            f"and place it in {RAW_CSV.parent}/"
        ) from exc


def load_raw() -> pd.DataFrame:
    """Load the raw reviews as a DataFrame (no cleaning applied)."""
    return pd.read_csv(download_if_missing())


if __name__ == "__main__":
    df = load_raw()
    print(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")
