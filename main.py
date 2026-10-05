## Build the merged, deduplicated URL feature dataset from all three cleaned sources.

from urllib.parse import urlsplit, urlunsplit

import pandas as pd

from lexical_features import extract_lexical_features

CLEAN_DIR = "dataset/clean"
PROCESSED_DIR = "dataset/processed"

PHISH_PATH = f"{CLEAN_DIR}/malicious_phish_cleaned.csv"
PHIUSIIL_PATH = f"{CLEAN_DIR}/phiusiil_phishing_url_dataset_cleaned.csv"
URLDATA_PATH = f"{CLEAN_DIR}/urldata_cleaned.csv"
FEATURES_PATH = f"{PROCESSED_DIR}/url_features.csv"

MALICIOUS_TYPES = {"phishing", "defacement", "malware"}

# malicious_phish.csv is treated as the trusted source
SOURCE_PRIORITY = {"phish_corpus": 0, "phiusiil": 1, "urldata": 2}


def _normalise_url(url: str) -> str | None:
    
## Returns None for URLs urlsplit can't parse at all (e.g. defanged
## indicators like "example[.]com" that slipped past the garbled-URL
## regex because brackets are printable ASCII) - callers drop these.

    url = url if "://" in url else f"http://{url}"
    try:
        scheme, netloc, path, query, _fragment = urlsplit(url)
    except ValueError:
        return None
    if path == "/":
        path = ""
    return urlunsplit((scheme.lower(), netloc.lower(), path, query, ""))


# Per-source loaders - each maps a differently-shaped cleaned CSV onto the
# same url/label/subtype/source schema
def load_phish_corpus() -> pd.DataFrame:
    df = pd.read_csv(PHISH_PATH)
    return pd.DataFrame({
        "url": df["url"],
        "label": df["type"].isin(MALICIOUS_TYPES).astype(int),
        "subtype": df["type"],
        "source": "phish_corpus",
    })


def load_phiusiil() -> pd.DataFrame:
    df = pd.read_csv(PHIUSIIL_PATH)
    return pd.DataFrame({
        "url": df["URL"],
        "label": df["label"],
        "subtype": df["label"].map({1: "phishing", 0: "legitimate"}),
        "source": "phiusiil",
    })


def load_urldata() -> pd.DataFrame:
    df = pd.read_csv(URLDATA_PATH)
    return pd.DataFrame({
        "url": df["url"],
        "label": (df["label"] == "malicious").astype(int),
        "subtype": df["label"],
        "source": "urldata",
    })


# Dedup helpers - normalised-URL matching, first within each source then across all three
def _dedupe_within_source(name: str, df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset="normalised_url", keep="first")
    print(f"  {name}: {before - len(df)} within-source normalised-URL duplicates removed")
    return df


def _resolve_cross_source(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per normalised URL across all sources."""
    df = df.copy()
    df["_priority"] = df["source"].map(SOURCE_PRIORITY)

    phish_urls = set(df.loc[df["source"] == "phish_corpus", "normalised_url"])
    has_phish = df["normalised_url"].isin(phish_urls)
    label_nunique = df.groupby("normalised_url")["label"].transform("nunique")
    conflict = label_nunique > 1

    unresolvable = conflict & ~has_phish
    print(f"  {df.loc[unresolvable, 'normalised_url'].nunique()} phiusiil/urldata label "
          f"conflicts dropped (no malicious_phish row to arbitrate)")
    df = df[~unresolvable]

    resolved_by_phish = conflict & has_phish
    print(f"  {df.loc[resolved_by_phish, 'normalised_url'].nunique()} label conflicts "
          f"resolved in favour of malicious_phish")

    before = len(df)
    df = df.sort_values("_priority").drop_duplicates(subset="normalised_url", keep="first")
    print(f"  {before - len(df)} agreeing cross-source duplicates collapsed to one row")

    return df.drop(columns=["_priority"])


# Orchestration - load all sources, dedup, then compute features
def build_merged_dataset() -> pd.DataFrame:
    sources = {
        "phish_corpus": load_phish_corpus(),
        "phiusiil": load_phiusiil(),
        "urldata": load_urldata(),
    }
    for name, df in sources.items():
        # Some scraped URLs contain the HTML-escaped "&amp;" instead of "&",
        # which inflates special-character counts. Replace only that exact
        # string (html.unescape would also corrupt e.g. "&param=" into "¶m=").
        amp_count = df["url"].str.contains("&amp;", regex=False, na=False).sum()
        # Collapse any escaping ("&amp;", etc) to a single "&"
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

    combined = pd.concat(sources.values(), ignore_index=True)
    print("cross-source merge:")
    combined = _resolve_cross_source(combined)
    combined = combined.drop(columns=["normalised_url"]).reset_index(drop=True)

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
