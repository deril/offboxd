"""Loading the community ratings dataset and a Letterboxd ratings export."""

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

DATASET_REPO = "titouann/letterboxd-10m-movies-ratings-2025"
DATASET_FILE = "train.parquet"
CACHE_DIR = Path(__file__).parent / "data"


def load_community() -> pd.DataFrame:
    """Community ratings: user, movie_id, title (slug), rating. Downloaded once, then read from cache."""
    kwargs = dict(repo_id=DATASET_REPO, filename=DATASET_FILE, repo_type="dataset", cache_dir=CACHE_DIR / "hf")
    try:
        path = hf_hub_download(**kwargs, local_files_only=True)
    except Exception:
        print(f"Downloading {DATASET_REPO} (one-time, ~100 MB)...")
        path = hf_hub_download(**kwargs)
    return pd.read_parquet(path)


def load_my_ratings(path: str | Path) -> pd.DataFrame:
    """A Letterboxd export ratings.csv: Date, Name, Year, Letterboxd URI, Rating."""
    df = pd.read_csv(path, dtype={"Name": str}, keep_default_na=False)
    df = df.rename(columns={"Name": "name", "Year": "year", "Rating": "rating", "Letterboxd URI": "uri"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["rating"] = df["rating"].astype(float)
    return df[["name", "year", "rating", "uri"]]


def film_index(community: pd.DataFrame) -> pd.DataFrame:
    """One row per film: movie_id, slug, rating count across the whole dataset."""
    films = community.groupby("movie_id", observed=True).agg(slug=("title", "first"), count=("rating", "size"))
    return films.reset_index()
