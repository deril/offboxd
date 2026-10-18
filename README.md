# Letterboxd recommender

A local replacement for [letterboxd-recommendations](https://github.com/sdl60660/letterboxd_recommendations),
which stopped working once Letterboxd blocked its scraper. Same idea: SVD collaborative filtering
on Letterboxd community ratings, with you added as one more user, plus a popularity filter so you
can dig past the films everyone has already seen.

Community ratings come from the Hugging Face dataset
[`titouann/letterboxd-10m-movies-ratings-2025`](https://huggingface.co/datasets/titouann/letterboxd-10m-movies-ratings-2025)
(10.46M ratings by 6,519 active users across 281k films, December 2025, CC BY-NC-SA 4.0).

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync
```

Export your data from Letterboxd (Settings → Data → Export your data) and unzip it. Only
`ratings.csv` is used.

```bash
uv run recommend.py --ratings ~/Downloads/letterboxd-<you>-<date>/ratings.csv --top 50 --mix
```

The first run downloads the dataset (~100 MB, cached in `data/hf/`). It then trains the model twice,
once for the recommendations and once for the holdout check, which takes ~80 s and ~4.3 GB of RAM on
an M-series Mac. Later runs reuse the cached model in `data/models/` and finish in about a second.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--ratings PATH` | required | Your Letterboxd `ratings.csv`. |
| `--top N` | 50 | Number of recommendations. |
| `--popularity N` | 0 | Exclude the top N% most-rated films (0–100), like the original site's slider. The percentile is over films with at least `--min-count` ratings. The run prints the rating-count cutoff it applied. |
| `--mix` | off | Balanced list: an equal share of the best predictions from each popularity tier. Cannot be combined with `--popularity`. |
| `--min-count N` | 5 | Ignore films with fewer than N community ratings, whose predictions are mostly noise. |
| `--overrides PATH` | `./overrides.csv` | Manual title→slug fixes (see below). |
| `--sample N` | full dataset | Train on ~N community ratings, sampled as whole users so each kept profile stays complete. |
| `--retrain` | off | Ignore the cached model. |
| `--no-eval` | off | Skip the holdout check, which halves training time. |
| `--csv PATH` | – | Also write the results to a CSV file. |

### Popularity tiers

Rating counts are extremely long-tailed: the median eligible film has 15 ratings, the 99th
percentile 1,800. Splitting films into equal thirds would label a 30-rating film "popular". Tiers are
therefore thirds of rating *volume*:

- **popular**: the most-rated films that together hold the first third of all ratings (~1.4k films, 1,478+ ratings)
- **mid**: the next third (~4.7k films, roughly 355–1,478 ratings)
- **obscure**: everything else with at least `--min-count` ratings (~94k films)

That same long tail is why `--popularity` gets aggressive quickly: `--popularity 30` already
excludes every film with 33+ ratings. Try `--popularity 1` to `--popularity 5` to drop just the
blockbusters, or use `--mix`.

## How your ratings are matched

The dataset identifies films only by Letterboxd slug (`zootopia-2`), and the export only has name and
year, so slugs are rebuilt the way Letterboxd builds them: accents folded to ASCII, apostrophes and
periods deleted (`L.A. Confidential` → `la-confidential`), every other run of punctuation or spaces
replaced with one hyphen. The `{slug}-{year}` form is tried first, because Letterboxd only adds the
year when another film already owns the bare slug (`Arrival` is `arrival-2016`).

Every run prints a match report. For anything left unmatched, look up the film on letterboxd.com and
add a row to `overrides.csv`, using the slug from its URL:

```csv
Name,Year,slug
Nymphomaniac: Vol. I,2013,nymphomaniac-volume-i
```

Unmatched films are usually missing from the dataset rather than mis-slugged: it only covers films
that its 6.5k users rated.

## Model

`scikit-surprise` SVD with its default hyperparameters (100 factors, 20 epochs). Stronger
regularization was tested against a 2% community holdout and scored worse in every popularity
bucket. Before training, the expected memory use (~450 B per rating) is checked against available
RAM, and the training set is sampled down if it would not fit.

The cache key is a hash of your *matched* ratings plus the sample size and model settings. Editing a
rating, adding a film, or changing `overrides.csv` triggers a retrain. A fresh export with the same
ratings does not.

The sanity check holds out a random 10% of your ratings, retrains without them, and reports RMSE on
them next to two baselines (always predicting your mean, and predicting each film's community mean).

## Known quirks

- Letterboxd lists TV series, concert films, and "making of" specials alongside films. Their small,
  devoted audiences rate them ~4.8, so they crowd the top of the obscure end. Raising `--min-count`
  (e.g. 100) or using `--mix` keeps them in check. The dataset has no metadata to filter them out.
- Titles are reconstructed from slugs, so punctuation is lost and a year suffix can appear
  (`Poor Things 2023`). The link column always goes to the right page.

## Files

- `data.py`: downloading/caching the dataset, reading the export, per-film rating counts
- `match.py`: slug reconstruction, overrides, match report
- `model.py`: training, memory budget, holdout RMSE, model cache
- `recommend.py`: CLI, popularity filter and tiers, output
