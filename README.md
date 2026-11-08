# Data (instructions only — no data is committed)

1. Download **Amazon Fine Food Reviews** from Kaggle:
   https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
2. Unzip and place `Reviews.csv` at `data/raw/Reviews.csv`
   (or install `kagglehub` and run `python -m src.load_data`, which tries to fetch it).
3. `python -m src.run_pipeline` writes the cleaned data to `data/processed/`.

Expected raw file: 568,454 rows x 10 columns, ~300 MB.
