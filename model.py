"""SVD collaborative filtering (scikit-surprise) with me added as one extra user, cached to disk."""

import hashlib
import json
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from surprise import SVD, Dataset, Reader

from data import CACHE_DIR, DATASET_FILE, DATASET_REPO

ME = 65535  # community user ids are 0..6518, so the top of uint16 is free
# surprise defaults. reg_all 0.05/0.1 and reg_bi 0.2 all scored worse on a community holdout,
# in every popularity bucket, and did not change which niche films top the list.
SVD_PARAMS = dict(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
# Peak RSS growth while building a surprise trainset and fitting SVD: ~390 B/rating on the full 10.4M.
BYTES_PER_RATING = 450
MEMORY_BUDGET = 0.9  # share of "available" RAM; macOS undercounts it by leaving out compressible pages
MODEL_DIR = CACHE_DIR / "models"


@dataclass
class Model:
    """Everything needed to score films for me: global mean, my bias/factors, every film's bias/factors."""
    movie_ids: np.ndarray
    item_bias: np.ndarray
    item_factors: np.ndarray
    my_bias: float
    my_factors: np.ndarray
    global_mean: float
    meta: dict

    def predict(self) -> pd.Series:
        """Unclipped scores: anything above 5.0 still ranks above a plain 5.0. Clip only for display."""
        scores = self.global_mean + self.my_bias + self.item_bias + self.item_factors @ self.my_factors
        return pd.Series(scores, index=self.movie_ids, name="predicted")


def _peak_rss() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024  # bytes on macOS, KiB on Linux


def sample_users(community: pd.DataFrame, max_ratings: int, seed: int = 42) -> pd.DataFrame:
    """Keep whole users (in random order) until ~max_ratings, so sampled profiles stay complete."""
    if max_ratings >= len(community):
        return community
    counts = community["user"].value_counts().sample(frac=1, random_state=seed)
    keep = counts.index[counts.cumsum() <= max_ratings]
    return community[community["user"].isin(keep)]


def fit_budget(requested: int) -> int:
    """Shrink a requested training size to what fits in available memory."""
    available = psutil.virtual_memory().available
    fits = int(available * MEMORY_BUDGET / BYTES_PER_RATING)
    if requested > fits:
        print(f"  ~{requested * BYTES_PER_RATING / 1e9:.1f} GB needed for {requested:,} ratings but only "
              f"{available / 1e9:.1f} GB available - sampling down to {fits:,}")
        return fits
    return requested


def train(community: pd.DataFrame, mine: pd.DataFrame) -> tuple[SVD, dict]:
    """Fit SVD on community ratings plus mine (as user ME). Retries on a half-size sample if memory runs out."""
    ratings = community[["user", "movie_id", "rating"]]
    my_rows = pd.DataFrame({"user": ME, "movie_id": mine["movie_id"].to_numpy(), "rating": mine["rating"].to_numpy()})
    while True:
        df = pd.concat([ratings, my_rows], ignore_index=True)
        start, rss_before = time.time(), _peak_rss()
        try:
            trainset = Dataset.load_from_df(df, Reader(rating_scale=(0.5, 5.0))).build_full_trainset()
            algo = SVD(**SVD_PARAMS)
            algo.fit(trainset)
        except MemoryError:
            ratings = sample_users(ratings, len(ratings) // 2)
            print(f"  out of memory - retrying on {len(ratings):,} community ratings")
            continue
        stats = dict(train_ratings=len(df), train_seconds=round(time.time() - start, 1),
                     peak_rss_gb=round(_peak_rss() / 1e9, 2), rss_growth_gb=round((_peak_rss() - rss_before) / 1e9, 2))
        return algo, stats


def to_model(algo: SVD, meta: dict) -> Model:
    ts = algo.trainset
    movie_ids = np.array([ts.to_raw_iid(i) for i in range(ts.n_items)], dtype=np.uint32)
    me = ts.to_inner_uid(ME)
    return Model(movie_ids=movie_ids, item_bias=algo.bi.astype(np.float32), item_factors=algo.qi.astype(np.float32),
                 my_bias=float(algo.bu[me]), my_factors=algo.pu[me].astype(np.float32),
                 global_mean=float(ts.global_mean), meta=meta)


def holdout_rmse(community: pd.DataFrame, mine: pd.DataFrame, frac: float = 0.1, seed: int = 0) -> dict:
    """Train without a random 10% of my ratings and score the model on them, next to two naive baselines."""
    held = mine.sample(frac=frac, random_state=seed)
    kept = mine.drop(held.index)
    algo, _ = train(community, kept)
    pred = np.array([algo.predict(ME, mid).est for mid in held["movie_id"]])
    truth = held["rating"].to_numpy()
    rmse = lambda p: round(float(np.sqrt(np.mean((p - truth) ** 2))), 3)
    item_mean = community.groupby("movie_id")["rating"].mean()
    return dict(held_out=len(held), svd_rmse=rmse(pred),
                my_mean_rmse=rmse(np.full_like(truth, kept["rating"].mean())),
                film_mean_rmse=rmse(held["movie_id"].map(item_mean).fillna(kept["rating"].mean()).to_numpy()))


def cache_key(mine: pd.DataFrame, max_ratings: int) -> str:
    """Changes whenever my matched ratings, the training size, the dataset or the SVD settings change."""
    payload = json.dumps(dict(ratings=sorted(zip(mine["movie_id"].tolist(), mine["rating"].tolist())),
                              max_ratings=max_ratings, dataset=f"{DATASET_REPO}/{DATASET_FILE}", svd=SVD_PARAMS))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def save(model: Model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, movie_ids=model.movie_ids, item_bias=model.item_bias, item_factors=model.item_factors,
             my_factors=model.my_factors, scalars=np.array([model.my_bias, model.global_mean]),
             meta=np.array(json.dumps(model.meta)))


def load(path: Path) -> Model:
    z = np.load(path)
    return Model(movie_ids=z["movie_ids"], item_bias=z["item_bias"], item_factors=z["item_factors"],
                 my_bias=float(z["scalars"][0]), my_factors=z["my_factors"], global_mean=float(z["scalars"][1]),
                 meta=json.loads(str(z["meta"])))


def get_model(community: pd.DataFrame, mine: pd.DataFrame, max_ratings: int | None,
              retrain: bool = False, evaluate: bool = True) -> Model:
    """Load the cached model for these ratings, or train (and evaluate) a new one."""
    requested = len(community) if max_ratings is None else min(max_ratings, len(community))
    path = MODEL_DIR / f"svd-{cache_key(mine, requested)}.npz"
    if path.exists() and not retrain:
        print(f"Using cached model {path.name} (pass --retrain to rebuild)")
        return load(path)

    size = fit_budget(requested)
    sample = sample_users(community, size)
    print(f"Training SVD on {len(sample):,} community ratings from {sample['user'].nunique():,} users "
          f"+ your {len(mine)}...")
    algo, stats = train(sample, mine)
    print(f"  done in {stats['train_seconds']}s, peak memory {stats['peak_rss_gb']} GB")
    meta = dict(stats, community_users=int(sample["user"].nunique()))
    if evaluate:
        print("Sanity check: retraining without 10% of your ratings...")
        meta["holdout"] = holdout_rmse(sample, mine)
    model = to_model(algo, meta)
    save(model, path)
    return model
