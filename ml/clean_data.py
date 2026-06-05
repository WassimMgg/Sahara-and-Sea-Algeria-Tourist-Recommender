"""
clean_data.py
=============
Turns the messy raw ratings file (data/ratings_raw.csv) into a clean,
analysis-ready file (ml/ratings_clean.csv).

The raw data was collected "from friends", so it contains the kind of
problems real survey data has:

  * the same country written many ways: DZ / Algerie / ALGERIA / Algerie / Algerie
  * gender written as F / Female / female / M / Male / male / Other / (blank)
  * ratings such as "5 stars", "4,5", "5.0", or impossible values like 7 / 10 / 0
  * 35 ratings that are simply missing
  * visit dates in ~10 different formats
  * a few duplicate (user, attraction) pairs

We fix each problem with one small, readable step so the result is easy to
explain in a report.
"""

import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, "..", "data", "ratings_raw.csv")
OUT_PATH = os.path.join(HERE, "ratings_clean.csv")

# Minimum and maximum value our rating scale allows.
RATING_MIN, RATING_MAX = 1.0, 5.0

# Every spelling/abbreviation we saw -> one canonical country name.
# (accents are stripped before lookup, so 'algerie' covers 'Algerie'/'Algerie')
COUNTRY_MAP = {
    "dz": "Algeria", "algerie": "Algeria", "algeria": "Algeria",
    "ma": "Morocco", "maroc": "Morocco", "morocco": "Morocco",
    "tn": "Tunisia", "tunisia": "Tunisia",
    "fr": "France", "france": "France",
    "es": "Spain", "spain": "Spain",
    "de": "Germany",
    "italy": "Italy",
    "england": "United Kingdom", "u.k.": "United Kingdom",
    "turkey": "Turkey", "turkiye": "Turkey",
    "u.s.a.": "United States",
    "ca": "Canada",
}

GENDER_MAP = {
    "f": "Female", "female": "Female",
    "m": "Male", "male": "Male",
    "other": "Other",
}


def normalize_country(value: str) -> str:
    """Map any spelling to a canonical country name (handles accents/case)."""
    if not isinstance(value, str) or value.strip() == "":
        return "Unknown"
    key = value.strip().lower()
    # normalise the accented 'Algerie'/'Algerie' variants to 'algerie'
    key = key.replace("\u00e9", "e").replace("\u00e8", "e")  # e-acute / e-grave -> e
    return COUNTRY_MAP.get(key, value.strip().title())


def normalize_gender(value: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        return "Unknown"
    return GENDER_MAP.get(value.strip().lower(), "Unknown")


def parse_rating(value) -> float:
    """
    Convert a messy rating cell into a float in [1, 5], or NaN if impossible.
    Handles: '', '5 stars', '5.0', '4,5', '7', '10', '0'.
    """
    if value is None:
        return float("nan")
    text = str(value).strip().lower()
    if text == "":
        return float("nan")
    text = text.replace("stars", "").replace("star", "").strip()
    text = text.replace(",", ".")  # European decimal comma -> dot
    try:
        number = float(text)
    except ValueError:
        return float("nan")
    # Anything outside the 1..5 scale (0, 6, 7, 10, ...) is treated as invalid.
    if number < RATING_MIN or number > RATING_MAX:
        return float("nan")
    return number


def parse_date(value):
    """Try several known formats; return an ISO date string or '' if unknown."""
    if not isinstance(value, str) or value.strip() == "":
        return ""
    text = value.strip()
    formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y",
        "%m/%d/%Y", "%b %Y", "%B %Y", "%Y",
    ]
    for fmt in formats:
        try:
            return pd.to_datetime(text, format=fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    # last resort: let pandas guess (day-first), but never crash
    try:
        return pd.to_datetime(text, dayfirst=True, errors="raise").strftime("%Y-%m-%d")
    except Exception:
        return ""


def main():
    df = pd.read_csv(RAW_PATH, dtype=str).fillna("")
    start = len(df)
    print(f"Loaded {start} raw rows")

    # 1. Normalise categorical text fields.
    df["home_country"] = df["home_country"].apply(normalize_country)
    df["gender"] = df["gender"].apply(normalize_gender)

    # 2. Parse the rating and drop rows we cannot use (no valid target value).
    df["rating"] = df["rating"].apply(parse_rating)
    before = len(df)
    df = df.dropna(subset=["rating"])
    print(f"Dropped {before - len(df)} rows with missing/invalid ratings")

    # 3. Clean the IDs (must be integers for the model).
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df["attraction_id"] = pd.to_numeric(df["attraction_id"], errors="coerce")
    df = df.dropna(subset=["user_id", "attraction_id"])
    df["user_id"] = df["user_id"].astype(int)
    df["attraction_id"] = df["attraction_id"].astype(int)

    # 4. Age: numeric, fill blanks with the median age.
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    median_age = int(df["age"].median())
    df["age"] = df["age"].fillna(median_age).astype(int)

    # 5. Dates -> ISO (purely informational, not used by the model).
    df["visit_date"] = df["visit_date"].apply(parse_date)

    # 6. Resolve duplicate (user, attraction) pairs by averaging the ratings.
    before = len(df)
    agg = (
        df.groupby(["user_id", "attraction_id"], as_index=False)
        .agg({
            "user_name": "first",
            "age": "first",
            "gender": "first",
            "home_country": "first",
            "attraction_name": "first",
            "rating": "mean",
            "visit_date": "max",
        })
    )
    agg["rating"] = agg["rating"].round(2)
    print(f"Merged {before - len(agg)} duplicate (user, attraction) ratings")

    # 7. Tidy column order and save.
    cols = ["user_id", "user_name", "age", "gender", "home_country",
            "attraction_id", "attraction_name", "rating", "visit_date"]
    agg = agg[cols].sort_values(["user_id", "attraction_id"]).reset_index(drop=True)
    agg.to_csv(OUT_PATH, index=False)

    print(f"\nClean file written to {OUT_PATH}")
    print(f"Final: {len(agg)} ratings | "
          f"{agg['user_id'].nunique()} users | "
          f"{agg['attraction_id'].nunique()} attractions")
    print(f"Rating range: {agg['rating'].min()} .. {agg['rating'].max()}")


if __name__ == "__main__":
    main()
