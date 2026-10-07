# Make four charts and save them in outputs/figures/.
#
# Each chart backs up a choice we made (which features to use, cleaning the
# URLs, and weighting the two classes).
#
# Run with:  python data_analysis.py

import os

import matplotlib
matplotlib.use("Agg")  # save charts as files instead of opening windows
import matplotlib.pyplot as plt
import pandas as pd

from config import FEATURES_PATH, NON_FEATURE_COLS, RANDOM_STATE

FIG_DIR = "outputs/figures"
SAMPLE_SIZE = 50_000  # chart 2 only uses this many random rows, to stay fast

# Colours for safe and malicious in every chart
BENIGN_COLOUR = "#4C72B0"     # blue
MALICIOUS_COLOUR = "#C44E52"  # red


def save(name: str) -> None:
    # Save the current chart and close it.
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"saved {path}")


def class_balance_by_source(df: pd.DataFrame) -> None:
    # Chart 1: how many safe and malicious URLs each dataset has.
    # Count the rows for each dataset + label pair
    counts = pd.crosstab(df["source"], df["label"]).rename(columns={0: "Benign", 1: "Malicious"})
    ax = counts.plot(kind="bar", stacked=True, figsize=(7, 4),
                     color=[BENIGN_COLOUR, MALICIOUS_COLOUR])

    # Write "% malicious" above each bar.
    # (The average of a 0/1 column = the fraction of 1s.)
    malicious_rate = df.groupby("source")["label"].mean()
    for i, src in enumerate(counts.index):
        bar_height = counts.loc[src].sum()
        ax.text(i, bar_height, f"{malicious_rate[src]:.0%} malicious", ha="center", va="bottom")

    ax.set_title("URLs per source by label")
    ax.set_xlabel("")
    ax.set_ylabel("Number of URLs")
    plt.xticks(rotation=0)
    save("1_class_balance_by_source.png")


def features_by_label(df: pd.DataFrame) -> None:
    # Chart 2: box plots comparing a few features for safe vs malicious URLs.
    sample = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
    cols = ["url_length", "hostname_length", "num_slashes", "digit_ratio"]

    # One small chart per feature, side by side
    fig, axes = plt.subplots(1, len(cols), figsize=(12, 4))
    for ax, col in zip(axes, cols):
        benign_values = sample.loc[sample["label"] == 0, col]
        malicious_values = sample.loc[sample["label"] == 1, col]
        # showfliers=False hides extreme dots so the boxes are easy to read
        ax.boxplot([benign_values, malicious_values],
                   tick_labels=["Benign", "Malicious"], showfliers=False)
        ax.set_title(col)
    fig.suptitle(f"Key features by label (random sample of {SAMPLE_SIZE:,} URLs)")
    save("2_features_by_label.png")


def source_formatting_differences(df: pd.DataFrame) -> None:
    # Chart 3: why we had to clean up (normalise) the URLs.
    #
    # Left: how many URLs start with "http://" or "https://", per dataset and label.
    # Right: average number of slashes. PhiUSIIL's safe URLs are mostly just homepages.
    has_scheme = df["url"].str.contains("://", regex=False)
    # unstack() turns labels 0 and 1 into two columns, ready to plot
    scheme_share = has_scheme.groupby([df["source"], df["label"]]).mean().unstack()
    slashes = df.groupby(["source", "label"])["num_slashes"].mean().unstack()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    panels = [
        (axes[0], scheme_share, "Share of raw URLs with a scheme (http/https)"),
        (axes[1], slashes, "Mean slashes after normalisation"),
    ]
    for ax, data, title in panels:
        data.rename(columns={0: "Benign", 1: "Malicious"}).plot(
            kind="bar", ax=ax, color=[BENIGN_COLOUR, MALICIOUS_COLOUR])
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
    save("3_source_formatting_differences.png")


def correlation_heatmap(df: pd.DataFrame, feature_cols: list[str]) -> None:
    # Chart 4: how closely each pair of features moves together.
    #
    # If two features move together very closely, one of them adds little.
    corr = df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    # Red = go up together, blue = one goes up as the other goes down
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_cols)), feature_cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(feature_cols)), feature_cols, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
    ax.set_title("Feature correlation matrix")
    save("4_correlation_heatmap.png")


def print_strong_correlations(df: pd.DataFrame, feature_cols: list[str],
                              threshold: float = 0.8) -> None:
    # Print feature pairs that move together very closely (used in the report).
    # abs() so strong negative links count too
    corr = df[feature_cols].corr().abs()

    # Check each pair once. "i < j" skips a feature with itself,
    # and skips (B, A) when we've already done (A, B).
    pairs = {}
    for i, feature_a in enumerate(feature_cols):
        for j, feature_b in enumerate(feature_cols):
            if i < j:
                pairs[(feature_a, feature_b)] = corr.loc[feature_a, feature_b]
    pairs = pd.Series(pairs)

    print(f"\nfeature pairs with |correlation| > {threshold}:")
    print(pairs[pairs > threshold].sort_values(ascending=False).round(2).to_string())


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [col for col in df.columns if col not in NON_FEATURE_COLS]

    class_balance_by_source(df)
    features_by_label(df)
    source_formatting_differences(df)
    correlation_heatmap(df, feature_cols)
    print_strong_correlations(df, feature_cols)
