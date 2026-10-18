"""SVD collaborative filtering (surprise's SGD, compiled with numba) with me added as one extra user, cached to disk."""

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
from numba import njit

from data import CACHE_DIR, DATASET_FILE, DATASET_REPO

ME = 65535  # community user ids are 0..6518, so the top of uint16 is free
# scikit-surprise's SVD defaults. reg_all 0.05/0.1 and reg_bi 0.2 all scored worse on a community holdout,
# in every popularity bucket, and did not change which niche films top the list.
SVD_PARAMS = dict(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
# Peak allocations while building the training arrays and fitting SVD: ~65 B/rating on the full 10.4M.
BYTES_PER_RATING = 80
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


@njit(cache=True)
def _sgd(users, items, ratings, global_mean, pu, qi, n_epochs, lr, reg):
    """scikit-surprise's SVD.sgd: same visit order, update rule and float64 arithmetic. Updates pu/qi in place."""
    # Ported from surprise/prediction_algorithms/matrix_factorization.pyx, (c) 2016 Nicolas Hug, BSD-3-Clause:
    # see LICENSE-surprise.
    bu, bi = np.zeros(pu.shape[0]), np.zeros(qi.shape[0])
    for _ in range(n_epochs):
        for k in range(len(ratings)):
            u, i = users[k], items[k]
            dot = 0.0
            for f in range(pu.shape[1]):
                dot += qi[i, f] * pu[u, f]
            err = ratings[k] - (global_mean + bu[u] + bi[i] + dot)
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])
            for f in range(pu.shape[1]):
                puf, qif = pu[u, f], qi[i, f]
                pu[u, f] += lr * (err * qif - reg * puf)
                qi[i, f] += lr * (err * puf - reg * qif)
    return bu, bi


def fit(users: np.ndarray, movies: np.ndarray, ratings: np.ndarray) -> Model:
    """SVD on (user, movie_id, rating) arrays, which must include ME.

    Ids are numbered by first appearance and pu is drawn before qi from the same seeded RNG, as surprise
    does, so on the same rows this reproduces surprise's factors up to float rounding.
    """
    u, user_ids = pd.factorize(users)
    i, movie_ids = pd.factorize(movies)
    p = SVD_PARAMS
    rng = np.random.RandomState(p["random_state"])
    pu = rng.normal(0, 0.1, (len(user_ids), p["n_factors"]))
    qi = rng.normal(0, 0.1, (len(movie_ids), p["n_factors"]))
    global_mean = float(ratings.mean())
    bu, bi = _sgd(u.astype(np.int32), i.astype(np.int32), ratings, global_mean, pu, qi,
                  p["n_epochs"], p["lr_all"], p["reg_all"])
    me = np.flatnonzero(user_ids == ME)[0]
    return Model(movie_ids=movie_ids.astype(np.uint32), item_bias=bi.astype(np.float32),
                 item_factors=qi.astype(np.float32), my_bias=float(bu[me]), my_factors=pu[me].astype(np.float32),
                 global_mean=global_mean, meta={})


def train(community: pd.DataFrame, mine: pd.DataFrame) -> tuple[Model, dict]:
    """Fit SVD on community ratings plus mine (as user ME). Retries on a half-size sample if memory runs out."""
    while True:
        start, rss_before = time.time(), _peak_rss()
        try:
            model = fit(np.concatenate([community["user"].to_numpy(), np.full(len(mine), ME, dtype=np.uint16)]),
                        np.concatenate([community["movie_id"].to_numpy(), mine["movie_id"].to_numpy(np.uint32)]),
                        np.concatenate([community["rating"].to_numpy(np.float64), mine["rating"].to_numpy(np.float64)]))
        except MemoryError:
            community = sample_users(community, len(community) // 2)
            print(f"  out of memory - retrying on {len(community):,} community ratings")
            continue
        stats = dict(train_ratings=len(community) + len(mine), train_seconds=round(time.time() - start, 1),
                     peak_rss_gb=round(_peak_rss() / 1e9, 2), rss_growth_gb=round((_peak_rss() - rss_before) / 1e9, 2))
        return model, stats


def holdout_rmse(community: pd.DataFrame, mine: pd.DataFrame, frac: float = 0.1, seed: int = 0) -> dict:
    """Train without a random 10% of my ratings and score the model on them, next to two naive baselines."""
    held = mine.sample(frac=frac, random_state=seed)
    kept = mine.drop(held.index)
    model, _ = train(community, kept)
    # As surprise's predict: a film missing from training gets the mean plus my bias, and estimates are clipped.
    pred = model.predict().reindex(held["movie_id"]).fillna(model.global_mean + model.my_bias).clip(0.5, 5.0).to_numpy()
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
    model, stats = train(sample, mine)
    print(f"  done in {stats['train_seconds']}s, peak memory {stats['peak_rss_gb']} GB")
    meta = dict(stats, community_users=int(sample["user"].nunique()))
    if evaluate:
        print("Sanity check: retraining without 10% of your ratings...")
        meta["holdout"] = holdout_rmse(sample, mine)
    model.meta = meta
    save(model, path)
    return model
