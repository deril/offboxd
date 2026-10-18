"""Letterboxd recommendations from SVD on community ratings, with a popularity filter.

    uv run recommend.py --ratings ~/Downloads/ratings.csv --top 50 [--popularity 30 | --mix]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import data
import match
import model

TIERS = ["obscure", "mid", "popular"]


def assign_tiers(films: pd.DataFrame) -> pd.Series:
    """Tiers by thirds of rating *volume*, not of film count.

    Counts are extremely long-tailed (median film: 15 ratings), so equal-size thirds would call a
    30-rating film "popular". Instead, popular = the most-rated films that together hold a third of
    all ratings (~1.4k films), mid = the next third (~4.7k), obscure = everything else.
    """
    share = films["count"].sort_values(ascending=False).cumsum() / films["count"].sum()
    tier = pd.cut(share, [0, 1 / 3, 2 / 3, 1], labels=TIERS[::-1], include_lowest=True).astype(str)
    return tier.reindex(films.index)


def exclude_most_popular(candidates: pd.DataFrame, percent: float) -> pd.DataFrame:
    """Drop the top `percent`% most-rated films, like the original site's popularity slider."""
    if percent <= 0:
        return candidates
    cutoff = candidates["count"].quantile(1 - percent / 100)
    print(f"--popularity {percent:g}: excluding films with {cutoff:,.0f}+ ratings")
    return candidates[candidates["count"] < cutoff]


def shrink(predicted: pd.Series, counts: pd.Series, base: float, k: float) -> pd.Series:
    """Rank score: pull each prediction toward `base` by k / (count + k).

    A heuristic, not a calibrated confidence bound. Titles with a few hundred ratings from a devoted
    fanbase (concert films, TV specials) get extreme predictions; this lets them lose ties to
    better-supported films. k=0 ranks by the raw prediction.
    """
    return base + counts / (counts + k) * (predicted - base)


def mixed(candidates: pd.DataFrame, top: int) -> pd.DataFrame:
    """Best-scoring films from each popularity tier in equal shares, grouped popular -> mid -> obscure."""
    share, extra = divmod(top, len(TIERS))
    order = TIERS[::-1]
    return pd.concat([candidates[candidates["tier"] == t].nlargest(share + (i < extra), "score")
                      for i, t in enumerate(order)])


def readable(slug: str) -> str:
    return " ".join(w.capitalize() for w in slug.split("-"))


def sequence_films(films: pd.DataFrame, m: model.Model) -> pd.DataFrame:
    """Order selected films by SVD-factor similarity, starting with the highest-ranked film.

    The selected set stays fixed: this changes watch order, not which films are recommended.
    A nearest-neighbor path is improved by reversing segments when that increases the sum
    of adjacent cosine similarities.
    """
    if len(films) < 2:
        return films

    positions = pd.Index(m.movie_ids).get_indexer(films["movie_id"])
    if (positions < 0).any():
        raise ValueError("Cannot sequence a film missing from the trained model")
    factors = m.item_factors[positions].astype(np.float64)
    lengths = np.linalg.norm(factors, axis=1, keepdims=True)
    unit = np.divide(factors, lengths, out=np.zeros_like(factors), where=lengths != 0)
    similarity = unit @ unit.T

    path = [0]
    remaining = set(range(1, len(films)))
    while remaining:
        next_film = max(remaining, key=lambda i: (similarity[path[-1], i], -i))
        path.append(next_film)
        remaining.remove(next_film)

    # Keep the best-ranked film first. Reversing a segment changes only its boundary edges.
    for _ in range(10):
        improved = False
        for start in range(1, len(path) - 1):
            for end in range(start + 1, len(path)):
                before = similarity[path[start - 1], path[start]]
                after = similarity[path[start - 1], path[end]]
                if end + 1 < len(path):
                    before += similarity[path[end], path[end + 1]]
                    after += similarity[path[start], path[end + 1]]
                if after > before + 1e-10:
                    path[start:end + 1] = reversed(path[start:end + 1])
                    improved = True
        if not improved:
            break
    return films.iloc[path]


def recommend(films: pd.DataFrame, predictions: pd.Series, mine: pd.DataFrame, args,
              m: model.Model | None = None) -> pd.DataFrame:
    candidates = films[films["count"] >= args.min_count].copy()
    # Tiers come from the whole eligible pool, so a film's tier doesn't depend on what you've seen.
    candidates["tier"] = assign_tiers(candidates)
    candidates = exclude_most_popular(candidates, args.popularity)
    candidates = candidates[~candidates["movie_id"].isin(mine["movie_id"])]
    candidates["predicted"] = candidates["movie_id"].map(predictions)
    candidates = candidates.dropna(subset=["predicted"])
    # Shrink toward your mean over the same matched ratings the model was trained (and cached) on.
    candidates["score"] = shrink(candidates["predicted"], candidates["count"], mine["rating"].mean(), args.shrink)
    if args.mix:
        out = mixed(candidates, args.top)
    else:
        out = candidates.nlargest(args.top, "score")
    if args.sequence:
        if m is None:
            raise ValueError("--sequence requires the trained model's film factors")
        out = sequence_films(out, m)
    out = out.reset_index(drop=True)
    return pd.DataFrame({
        "rank": out.index + 1,
        "title": out["slug"].map(readable),
        "score": out["score"].round(2),
        "predicted": out["predicted"].clip(0.5, 5.0).round(2),
        "ratings": out["count"],
        "tier": out["tier"],
        "url": "https://letterboxd.com/film/" + out["slug"] + "/",
    })


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ratings", required=True, type=Path, help="Letterboxd export ratings.csv")
    p.add_argument("--top", type=int, default=50, help="number of recommendations (default 50)")
    pop = p.add_mutually_exclusive_group()
    pop.add_argument("--popularity", type=float, default=0, metavar="N",
                     help="exclude the top N%% most-rated films, 0-100 (default 0)")
    pop.add_argument("--mix", action="store_true", help="equal share from popular / mid / obscure tiers")
    p.add_argument("--sequence", action="store_true",
                   help="order selected recommendations so adjacent films have similar rating patterns")
    p.add_argument("--min-count", type=int, default=50,
                   help="ignore films with fewer community ratings (default 50, like the original's MOVIE_MIN)")
    p.add_argument("--shrink", type=float, default=300, metavar="K",
                   help="rank by prediction shrunk toward your mean by K/(count+K); 0 = raw prediction (default 300)")
    p.add_argument("--overrides", type=Path, default=Path(__file__).parent / "overrides.csv",
                   help="Name,Year,slug fixes for unmatched titles (default ./overrides.csv)")
    p.add_argument("--sample", type=int, default=None, metavar="N",
                   help="train on ~N community ratings (whole users) instead of all ~10.4M")
    p.add_argument("--retrain", action="store_true", help="ignore the cached model")
    p.add_argument("--no-eval", action="store_true", help="skip the 10%% holdout RMSE check when training")
    p.add_argument("--csv", type=Path, help="also save the results to this CSV")
    p.add_argument("--links", type=Path, help="also save the letterboxd.com links, one per line, to this file")
    args = p.parse_args()
    if not 0 <= args.popularity < 100:
        p.error("--popularity must be in [0, 100)")
    return args


def main():
    args = parse_args()
    community = data.load_community()
    films = data.film_index(community)
    mine = data.load_my_ratings(args.ratings.expanduser())
    matched, unmatched = match.match_ratings(mine, films, args.overrides)
    match.print_report(mine, matched, unmatched)

    m = model.get_model(community, matched, args.sample, retrain=args.retrain, evaluate=not args.no_eval)
    if h := m.meta.get("holdout"):
        print(f"Holdout RMSE on {h['held_out']} of your ratings: SVD {h['svd_rmse']}  "
              f"(baselines: your mean {h['my_mean_rmse']}, film's community mean {h['film_mean_rmse']})")
    print(f"Model: {m.meta['train_ratings']:,} ratings, {m.meta['train_seconds']}s, peak {m.meta['peak_rss_gb']} GB\n")

    recs = recommend(films, m.predict(), matched, args, m)
    with pd.option_context("display.max_rows", None, "display.width", 200, "display.max_colwidth", 60):
        print(recs.to_string(index=False))
    if args.csv:
        recs.to_csv(args.csv, index=False)
        print(f"\nSaved {len(recs)} rows to {args.csv}")
    if args.links:
        args.links.write_text("".join(url + "\n" for url in recs["url"]))
        print(f"Saved {len(recs)} links to {args.links}")


if __name__ == "__main__":
    main()
