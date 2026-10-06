# Data analysis: four charts saved to outputs/figures/, each
# linked to a decision in the project (feature choices, URL normalisation and class column weighing)

import os

import matplotlib
matplotlib.use("Agg") # helps render the files without opening extra windows
import matplotlib.pyplot as plt
import pandas as pd


FEATURES_PATH = "dataset/processed/url_features.csv"
FIG_DIR = "outputs/figures"
NON_FEATURE_COLS = ("url", "label", "subtype", "source")
SAMPLE_SIZE = 50_000 # keeping sample size small to avoid slow processing. 
RANDOM_STATE = 42 

# Save the current figure and close it
def save(name: str) -> None:
    path = os.path. join(FIG_DIR, name)
    plt,plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"saved {path}")

# Chart 1: rows source split by labels. 
def class_balance_by_source(df: pd.DataFrame) -> None:
    counts = pd.crosstab(df["source"], df["label"]).rename(columns={0: "Benign", 1: "Malicious"})
    ax = counts.plot(kind="bar", stacked=True, figsize=(7,4),color=["#4C72B0", "#C44E52"])
    rates = df.groupby("source")["label"].mean()
    for i, src in enumerate(counts.index):
        ax.text(i, counts.loc[src].sum(), f"{rates[src]:.0%} malicious", ha="center", va="bottom")
    ax.set_title("URLs per source by label")
    ax.set_xlabel("")
    ax.set_ylabel("Number of URLs")
    plt.xticks(rotation=0)
    save("1_class_balance_by_source.png")

#Chart 2: distributions of key features by labels, on a random sample. 
def features_by_label(df: pd.DataFrame) -> None:
    sample = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
    cols = ["url_length", "hostname_length", "num_slashes", "digit_ratio"]
    fig, axes = plt.subplots(1, len(cols), figsize=(12, 4))
    for ax, col in zip(axes, cols):
        groups = [sample.loc[sample["label"] == lab, col] for lab in (0, 1)]
        ax.boxplot(groups, tick_labels=["Benign", "Malicious"], showfliers=False)
        ax.set_title(col)
    fig.suptitle(f"Key features by label (random sample of {SAMPLE_SIZE:,} URLs)")
    save("2_features_by_label.png")

# Chart 3: why URL normalisation was needed. Left: share of raw URLs that
# include a scheme, by source and label. Right: average path depth (slashes,
# after normalisation), showing PhiUSIIL's benign URLs are base referecences.
def source_formatting_differences(df: pd.DataFrame) -> None:
    has_scheme = df["url"].str.contains("://", regex=False)
    scheme_share = has_scheme.groupby([df["source"], df["label"]]).mean().unstack()
    slashes = df.groupby(["source", "label"])["num_slashes"].mean().unstack()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, title in [
        (axes[0], scheme_share, "Share of raw URLs with a scheme (http/https)"),
        (axes[1], slashes, "Mean slashes after normalisation"),
    ]:
        data.rename(columns={0: "Benign", 1: "Malicious"}).plot(
            kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
    save("3_source_formatting_differences.png")

# Chart 4: Correlations between features. Highly correlated pairs are considered redundant. 
def correlation_heatmap(df: pd.DataFrame, feature_cols: list[str]) -> None:
    corr = df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_cols)), feature_cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(feature_cols)), feature_cols, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
    ax.set_title("Feature correlation matrix")
    save("4_correlation_heatmap.png")

if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    class_balance_by_source(df)
    features_by_label(df)
    source_formatting_differences(df)
    correlation_heatmap(df, feature_cols)

    # Strongly correlated feature pairs, for the report text
    corr = df[feature_cols].corr().abs()
    pairs = corr.where(pd.DataFrame(
        [[i < j for j in range(len(corr))] for i in range(len(corr))],
        index=corr.index, columns=corr.columns)).stack()
    print("\nfeature pairs with |correlation| > 0.8:")
    print(pairs[pairs > 0.8].sort_values(ascending=False).round(2).to_string())