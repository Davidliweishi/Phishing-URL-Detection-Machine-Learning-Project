# Combine the three cleaned datasets into one feature table.
#
# Steps:
# 1. Load each dataset and give them all the same 4 columns
# (url, label, subtype, source)
# 2. Fix "&amp;" in URLs and remove URLs that can't be read
# 3. Remove duplicate URLs (inside each dataset, then between datasets)
# 4. Work out the features for every URL and save the table
#
# Run with:  python main.py

from urllib.parse import urlsplit, urlunsplit

import pandas as pd

from config import FEATURES_PATH
from lexical_features import extract_lexical_features

CLEAN_DIR = "dataset/clean"

PHISH_PATH = f"{CLEAN_DIR}/malicious_phish_cleaned.csv"
PHIUSIIL_PATH = f"{CLEAN_DIR}/phiusiil_phishing_url_dataset_cleaned.csv"
URLDATA_PATH = f"{CLEAN_DIR}/urldata_cleaned.csv"

# These types in malicious_phish.csv count as malicious
MALICIOUS_TYPES = {"phishing", "defacement", "malware"}

# If a URL is in more than one dataset, keep the copy from the lowest number.
# We trust malicious_phish.csv the most.
SOURCE_PRIORITY = {"phish_corpus": 0, "phiusiil": 1, "urldata": 2}


def _normalise_url(url: str) -> str | None:
    # Write the URL in a standard way, so duplicates are easy to find.
    #
    # "HTTP://Example.com/" and "example.com" both become "http://example.com".
    #
    # Gives None if the URL can't be read at all (e.g. "example[.]com").
    # Those URLs are removed later.
    if "://" not in url:
        url = f"http://{url}"
    try:
        scheme, netloc, path, query, _fragment = urlsplit(url)
    except ValueError:
        return None
    if path == "/":
        path = ""
    return urlunsplit((scheme.lower(), netloc.lower(), path, query, ""))


# Loaders: each dataset names its columns differently,
# so each one is changed to url / label / subtype / source

def load_phish_corpus() -> pd.DataFrame:
    # Load malicious_phish. Its "type" column says benign/phishing/defacement/malware.
    df = pd.read_csv(PHISH_PATH)
    return pd.DataFrame({
        "url": df["url"],
        "label": df["type"].isin(MALICIOUS_TYPES).astype(int),  # True/False becomes 1/0
        "subtype": df["type"],
        "source": "phish_corpus",
    })


def load_phiusiil() -> pd.DataFrame:
    # Load PhiUSIIL. Its "label" column is already 1 (phishing) or 0 (safe).
    df = pd.read_csv(PHIUSIIL_PATH)
    return pd.DataFrame({
        "url": df["URL"],
        "label": df["label"],
        "subtype": df["label"].map({1: "phishing", 0: "legitimate"}),
        "source": "phiusiil",
    })


def load_urldata() -> pd.DataFrame:
    # Load urldata. Its "label" column says "malicious" or "benign".
    df = pd.read_csv(URLDATA_PATH)
    return pd.DataFrame({
        "url": df["url"],
        "label": (df["label"] == "malicious").astype(int),  # True/False becomes 1/0
        "subtype": df["label"],
        "source": "urldata",
    })


# Removing duplicates

def _dedupe_within_source(name: str, df: pd.DataFrame) -> pd.DataFrame:
    # Remove repeated URLs inside one dataset. Keep the first one.
    rows_before = len(df)
    df = df.drop_duplicates(subset="normalised_url", keep="first")
    print(f"  {name}: {rows_before - len(df)} within-source normalised-URL duplicates removed")
    return df


def _resolve_cross_source(df: pd.DataFrame) -> pd.DataFrame:
    # Keep one row per URL across all three datasets.
    #
    # If the datasets disagree about a URL's label:
    #   - trust malicious_phish if it has the URL
    #   - otherwise we can't tell who is right, so remove the URL
    # If they agree, keep the copy from the most trusted dataset.
    df = df.copy()
    df["_priority"] = df["source"].map(SOURCE_PRIORITY)

    # Is this URL also in malicious_phish?
    phish_urls = set(df.loc[df["source"] == "phish_corpus", "normalised_url"])
    has_phish = df["normalised_url"].isin(phish_urls)

    # Conflict = the same URL has different labels in different datasets
    labels_per_url = df.groupby("normalised_url")["label"].transform("nunique")
    conflict = labels_per_url > 1

    # "~" means "not": a conflict that malicious_phish can't settle
    unresolvable = conflict & ~has_phish
    print(f"  {df.loc[unresolvable, 'normalised_url'].nunique()} phiusiil/urldata label "
          f"conflicts dropped (no malicious_phish row to arbitrate)")
    df = df[~unresolvable]

    resolved_by_phish = conflict & has_phish
    print(f"  {df.loc[resolved_by_phish, 'normalised_url'].nunique()} label conflicts "
          f"resolved in favour of malicious_phish")

    # Put the most trusted dataset first, then keep only the first copy
    rows_before = len(df)
    df = df.sort_values("_priority").drop_duplicates(subset="normalised_url", keep="first")
    print(f"  {rows_before - len(df)} agreeing cross-source duplicates collapsed to one row")

    return df.drop(columns=["_priority"])


def build_merged_dataset() -> pd.DataFrame:
    # Load, clean and combine all three datasets, then add the features.
    sources = {
        "phish_corpus": load_phish_corpus(),
        "phiusiil": load_phiusiil(),
        "urldata": load_urldata(),
    }

    for name, df in sources.items():
        # Some URLs say "&amp;" where they mean "&". This adds extra symbols,
        # so swap it back. (html.unescape would also break things like "&param=".)
        amp_count = df["url"].str.contains("&amp;", regex=False, na=False).sum()
        # "&amp;", "&amp;amp;" and so on all become "&"
        df["url"] = df["url"].str.replace(r"&(?:amp;)+", "&", regex=True)
        print(f"  {name}: {amp_count} URL(s) with '&amp;' decoded to '&'")

        df["normalised_url"] = df["url"].apply(_normalise_url)
        unparseable = df["normalised_url"].isna()
        if unparseable.any():
            print(f"  {name}: {unparseable.sum()} unparseable URL(s) dropped, e.g. "
                  f"{df.loc[unparseable, 'url'].iloc[0]!r}")
            sources[name] = df[~unparseable]

    print("within-source dedup:")
    sources = {name: _dedupe_within_source(name, df) for name, df in sources.items()}

    # Stack the three datasets into one table
    combined = pd.concat(sources.values(), ignore_index=True)
    print("cross-source merge:")
    combined = _resolve_cross_source(combined)
    combined = combined.drop(columns=["normalised_url"]).reset_index(drop=True)

    # Add the feature columns on the right
    features = extract_lexical_features(combined, url_col="url")
    return pd.concat([combined, features], axis=1)


if __name__ == "__main__":
    merged = build_merged_dataset()
    merged.to_csv(FEATURES_PATH, index=False)
    print(f"\nsaved {merged.shape[0]} rows, {merged.shape[1]} columns -> {FEATURES_PATH}")

    print("\nsource x label counts:")
    print(pd.crosstab(merged["source"], merged["label"]))

    print("\noverall label balance:")
    print(merged["label"].value_counts(normalize=True))
