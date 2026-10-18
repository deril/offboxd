"""Matching a Letterboxd export to dataset films by reconstructing Letterboxd slugs."""

import re
import unicodedata
from pathlib import Path

import pandas as pd

# Deleted without leaving a gap: "Schindler's" -> "schindlers", "L.A. Confidential" -> "la-confidential".
SILENT = "'‘’ʼ`."


def slugify(name: str) -> str:
    """Letterboxd-style slug: 'Léon: The Professional' -> 'leon-the-professional'.

    Accents are folded to ASCII, apostrophes and periods are dropped outright,
    and every other run of non-alphanumerics becomes one hyphen.
    """
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    text = re.sub(f"[{re.escape(SILENT)}]", "", text.lower())
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def load_overrides(path: Path | None) -> dict[tuple[str, int | None], str]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"Name": str, "slug": str}, keep_default_na=False)
    years = pd.to_numeric(df["Year"], errors="coerce")
    return {(n, None if pd.isna(y) else int(y)): s.strip() for n, y, s in zip(df["Name"], years, df["slug"])}


def match_ratings(mine: pd.DataFrame, films: pd.DataFrame, overrides_path: Path | None = None):
    """Attach movie_id/slug to each of my ratings. Returns (matched, unmatched).

    Order of preference: override from overrides.csv, then '{slug}-{year}', then '{slug}'.
    The year-suffixed slug wins because Letterboxd only adds it to disambiguate a title
    that another film already owns, so the bare slug is then likely a different film.
    """
    slug_to_id = dict(zip(films["slug"], films["movie_id"]))
    overrides = load_overrides(overrides_path)

    slugs, sources = [], []
    for name, year in zip(mine["name"], mine["year"]):
        year = None if pd.isna(year) else int(year)
        override = overrides.get((name, year))
        base = slugify(name)
        candidates = [("override", override)] if override else []
        if year is not None:
            candidates.append(("name-year", f"{base}-{year}"))
        candidates.append(("name", base))
        hit = next(((src, s) for src, s in candidates if s in slug_to_id), (None, None))
        sources.append(hit[0])
        slugs.append(hit[1])

    out = mine.assign(slug=slugs, match=sources)
    out["movie_id"] = out["slug"].map(slug_to_id).astype("Int64")
    matched = out[out["movie_id"].notna()].copy()
    # Two export rows resolving to one film (e.g. a remake without a year slug) would double-count it.
    matched = matched.drop_duplicates("movie_id", keep="first")
    matched["movie_id"] = matched["movie_id"].astype("uint32")
    unmatched = out[out["movie_id"].isna()]
    return matched, unmatched


def print_report(mine: pd.DataFrame, matched: pd.DataFrame, unmatched: pd.DataFrame) -> None:
    dupes = len(mine) - len(matched) - len(unmatched)
    by_source = matched["match"].value_counts().to_dict()
    print(f"Matched {len(matched)}/{len(mine)} of your ratings ({len(matched) / len(mine):.0%}) "
          f"- {', '.join(f'{k}: {v}' for k, v in by_source.items())}")
    if dupes:
        print(f"  {dupes} rating(s) resolved to an already-matched film and were dropped")
    if len(unmatched):
        print(f"Unmatched {len(unmatched)} (not in the dataset, or slug differs - fix via overrides.csv):")
        for name, year in zip(unmatched["name"], unmatched["year"]):
            print(f"  - {name} ({year})  tried: {slugify(name)}")
    print()
