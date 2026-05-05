"""
01_data_prep.py
Yelp Open Dataset: load, filter, engineer features, compute review embeddings,
split into train/val/test. Produces feature matrices consumed by every
downstream stage.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
RAW_DIR = pathlib.Path("data/yelp_raw")
OUT_DIR = pathlib.Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = ["Philadelphia", "Tampa", "Indianapolis", "Nashville", "Tucson"]
CUISINES = [
    "American (Traditional)", "American (New)", "Italian", "Mexican",
    "Chinese", "Japanese", "Thai", "Indian", "Mediterranean", "Pizza",
]
MIN_REVIEWS = 20
REVIEWS_PER_BUSINESS = 30
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jsonl(path: pathlib.Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Stream a Yelp JSONL file into a DataFrame, optionally projecting columns."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if usecols is not None:
                row = {k: row.get(k) for k in usecols}
            records.append(row)
    return pd.DataFrame.from_records(records)


def load_businesses() -> pd.DataFrame:
    cols = ["business_id", "name", "city", "state", "latitude", "longitude",
            "stars", "review_count", "is_open", "attributes", "categories", "hours"]
    return load_jsonl(RAW_DIR / "yelp_academic_dataset_business.json", cols)


def load_reviews(business_ids: set[str]) -> pd.DataFrame:
    cols = ["review_id", "business_id", "stars", "date", "text"]
    chunks = []
    with open(RAW_DIR / "yelp_academic_dataset_review.json", "r", encoding="utf-8") as f:
        buf = []
        for i, line in enumerate(f):
            row = json.loads(line)
            if row["business_id"] in business_ids:
                buf.append({k: row[k] for k in cols})
            if len(buf) >= 200_000:
                chunks.append(pd.DataFrame.from_records(buf))
                buf = []
        if buf:
            chunks.append(pd.DataFrame.from_records(buf))
    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def is_restaurant(categories: str | None) -> bool:
    return isinstance(categories, str) and "Restaurants" in categories


def primary_cuisine(categories: str | None) -> str | None:
    if not isinstance(categories, str):
        return None
    tags = [t.strip() for t in categories.split(",")]
    for cuisine in CUISINES:
        if cuisine in tags:
            return cuisine
    return None


def filter_businesses(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["city"].isin(CITIES)].copy()
    df = df[df["categories"].apply(is_restaurant)]
    df["cuisine"] = df["categories"].apply(primary_cuisine)
    df = df.dropna(subset=["cuisine"])
    df = df[df["review_count"] >= MIN_REVIEWS]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Structured features
# ---------------------------------------------------------------------------

def _safe_attr(attrs: dict | None, key: str, default=None):
    if not isinstance(attrs, dict):
        return default
    val = attrs.get(key, default)
    if isinstance(val, str) and val.lower() in {"true", "false"}:
        return val.lower() == "true"
    return val


def _parse_price(attrs: dict | None) -> float:
    val = _safe_attr(attrs, "RestaurantsPriceRange2")
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def _parse_parking(attrs: dict | None) -> float:
    raw = _safe_attr(attrs, "BusinessParking")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw.replace("'", '"').replace("False", "false").replace("True", "true"))
        except json.JSONDecodeError:
            return 0.0
    if not isinstance(raw, dict):
        return 0.0
    return float(sum(bool(v) for v in raw.values()))


def _weekend_hours(hours: dict | None) -> float:
    if not isinstance(hours, dict):
        return 0.0
    total = 0.0
    for day in ("Saturday", "Sunday"):
        span = hours.get(day)
        if not span:
            continue
        try:
            start, end = span.split("-")
            sh, sm = map(int, start.split(":"))
            eh, em = map(int, end.split(":"))
            delta = (eh + em / 60) - (sh + sm / 60)
            total += delta if delta > 0 else delta + 24
        except ValueError:
            continue
    return total


def _late_night(hours: dict | None) -> int:
    if not isinstance(hours, dict):
        return 0
    for span in hours.values():
        if not isinstance(span, str) or "-" not in span:
            continue
        try:
            end = span.split("-")[1]
            eh = int(end.split(":")[0])
            if eh >= 22 or eh <= 3:
                return 1
        except ValueError:
            continue
    return 0


def engineer_structured(df: pd.DataFrame, review_min_dates: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["price_range"] = df["attributes"].apply(_parse_price)
    out["price_range"] = out["price_range"].fillna(out["price_range"].median())
    out["outdoor_seating"] = df["attributes"].apply(
        lambda a: float(bool(_safe_attr(a, "OutdoorSeating")))
    )
    out["reservations"] = df["attributes"].apply(
        lambda a: float(bool(_safe_attr(a, "RestaurantsReservations")))
    )
    out["parking_score"] = df["attributes"].apply(_parse_parking)
    noise_map = {"quiet": 1, "average": 2, "loud": 3, "very_loud": 4}
    out["noise_level"] = df["attributes"].apply(
        lambda a: noise_map.get(str(_safe_attr(a, "NoiseLevel", "average")).strip("u'\""), 2)
    ).astype(float)
    out["weekend_open_hours"] = df["hours"].apply(_weekend_hours)
    out["late_night_open"] = df["hours"].apply(_late_night).astype(float)
    out["review_count_log"] = np.log1p(df["review_count"].astype(float))

    snapshot = pd.Timestamp("2024-11-01")
    age_days = (snapshot - pd.to_datetime(review_min_dates, errors="coerce")).dt.days
    out["age_years"] = (age_days / 365.25).fillna(age_days.median() / 365.25)

    return out


# ---------------------------------------------------------------------------
# Review embeddings
# ---------------------------------------------------------------------------

def sample_review_text(reviews: pd.DataFrame, n_per_business: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    def _sample(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) <= n_per_business:
            return group
        idx = rng.choice(len(group), size=n_per_business, replace=False)
        return group.iloc[idx]

    return reviews.groupby("business_id", group_keys=False).apply(_sample)


def embed_reviews(reviews: pd.DataFrame, business_order: list[str]) -> np.ndarray:
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        reviews["text"].tolist(),
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    reviews = reviews.copy()
    reviews["embedding"] = list(embeddings)

    pooled = (
        reviews.groupby("business_id")["embedding"]
        .apply(lambda seq: np.mean(np.stack(seq.values), axis=0))
    )
    return np.stack([pooled.loc[bid] for bid in business_order])


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

@dataclass
class Targets:
    stars: np.ndarray
    is_open: np.ndarray
    composite: np.ndarray


def build_targets(df: pd.DataFrame) -> Targets:
    stars = df["stars"].astype(float).to_numpy()
    is_open = df["is_open"].astype(int).to_numpy()
    z_stars = (stars - stars.mean()) / stars.std()
    log_rc = np.log1p(df["review_count"].astype(float).to_numpy())
    z_log_rc = (log_rc - log_rc.mean()) / log_rc.std()
    composite = 0.5 * z_stars + 0.3 * z_log_rc + 0.2 * is_open
    return Targets(stars=stars, is_open=is_open, composite=composite)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def stratified_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strat = df["city"] + "|" + df["cuisine"] + "|" + df["is_open"].astype(str)
    idx = np.arange(len(df))
    train_idx, rest_idx = train_test_split(
        idx, test_size=0.30, stratify=strat, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.50, stratify=strat.iloc[rest_idx], random_state=SEED
    )
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    businesses = filter_businesses(load_businesses())
    print(f"Retained {len(businesses):,} restaurants across {len(CITIES)} cities")

    reviews = load_reviews(set(businesses["business_id"]))
    print(f"Loaded {len(reviews):,} reviews for these businesses")

    min_dates = reviews.groupby("business_id")["date"].min()
    businesses["first_review_date"] = businesses["business_id"].map(min_dates)

    structured = engineer_structured(businesses, businesses["first_review_date"])
    targets = build_targets(businesses)

    sampled = sample_review_text(reviews, REVIEWS_PER_BUSINESS)
    embeddings = embed_reviews(sampled, businesses["business_id"].tolist())
    print(f"Embedding matrix shape: {embeddings.shape}")

    train_idx, val_idx, test_idx = stratified_split(businesses)

    scaler = StandardScaler().fit(structured.iloc[train_idx])
    X_struct = scaler.transform(structured)

    np.savez_compressed(
        OUT_DIR / "features.npz",
        X_struct=X_struct,
        X_embed=embeddings,
        stars=targets.stars,
        is_open=targets.is_open,
        composite=targets.composite,
        city=businesses["city"].to_numpy(),
        cuisine=businesses["cuisine"].to_numpy(),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        struct_cols=np.array(structured.columns.tolist()),
    )
    print(f"Wrote {OUT_DIR / 'features.npz'}")


if __name__ == "__main__":
    main()