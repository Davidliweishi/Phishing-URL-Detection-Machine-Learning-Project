# Sort the malicious URLs into groups (clusters) to find different kinds of attack.
#
# KMeans only sees the features. It never sees the label, subtype or source.
# Afterwards we check how each group matches up with those columns.
#
# Run with:  python cluster_analysis.py        (picks the number of groups itself)
#            python cluster_analysis.py 5      (uses 5 groups)

import sys

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import FEATURES_PATH, NON_FEATURE_COLS, RANDOM_STATE


def pick_k(X_scaled, k_range=range(2, 9)) -> int:
    # Try different numbers of groups (k) and return the best one.
    #
    # "Best" means the highest silhouette score (from -1 to 1).
    # A high score means the groups are clearly separate from each other.
    scores = {}
    for k in k_range:
        # Run KMeans 10 times from different starting points and keep the best
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(X_scaled)

        # The score is slow to work out, so use 5,000 random rows
        scores[k] = silhouette_score(X_scaled, labels, sample_size=5000, random_state=RANDOM_STATE)
        print(f"  k={k}: silhouette={scores[k]:.3f}")

    # Pick the k with the highest score (if two are equal, the smaller k wins)
    best_k = max(scores, key=scores.get)
    print(f"  -> chose k={best_k} (highest silhouette score)")
    return best_k


def describe_clusters(df: pd.DataFrame, feature_cols: list[str]) -> None:
    # Print a summary of each group.
    #
    # Shows: what subtypes and sources are in it, the 3 features that make it
    # stand out most, and a few example URLs.
    overall_mean = df[feature_cols].mean()
    # How spread out each feature is. Use 1 instead of 0 so we never divide by zero.
    overall_std = df[feature_cols].std().replace(0, 1)

    for cluster_id in sorted(df["cluster"].unique()):
        cluster_rows = df[df["cluster"] == cluster_id]
        share = len(cluster_rows) / len(df)
        print(f"\n=== Cluster {cluster_id} (n={len(cluster_rows)}, {share:.1%} of malicious rows) ===")

        # normalize=True gives fractions (like 0.25) instead of counts
        print("subtype composition:")
        print(cluster_rows["subtype"].value_counts(normalize=True).round(3).to_string())

        print("source composition:")
        print(cluster_rows["source"].value_counts(normalize=True).round(3).to_string())

        # z-score: how far this group's average is from the overall average.
        # Far away (big + or -) means the feature makes this group stand out.
        cluster_mean = cluster_rows[feature_cols].mean()
        z = (cluster_mean - overall_mean) / overall_std
        top_features = z.abs().sort_values(ascending=False).head(3)

        print("top differentiating features (cluster mean vs overall malicious-subset mean):")
        for feature in top_features.index:
            print(f"  {feature}: cluster={cluster_mean[feature]:.3f}  "
                  f"overall={overall_mean[feature]:.3f}  (z={z[feature]:+.2f})")

        # Up to 3 random example URLs (always the same 3, because of random_state=1)
        print("example URLs:")
        n_examples = min(3, len(cluster_rows))
        for url in cluster_rows["url"].sample(n_examples, random_state=1):
            print(f"  {url}")


def main():
    # 1. Load the data and find the feature columns
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [col for col in df.columns if col not in NON_FEATURE_COLS]

    # 2. Keep only the malicious URLs
    malicious = df[df["label"] == 1].reset_index(drop=True)
    print(f"clustering {len(malicious)} malicious rows on {len(feature_cols)} features "
          f"(label not used as input)")

    # 3. Put every feature on the same scale, so big numbers
    #    (like url_length) don't count more than small ones
    X_scaled = StandardScaler().fit_transform(malicious[feature_cols])

    # 4. Choose how many groups to make
    print("\nsilhouette scores by k:")
    k = pick_k(X_scaled)

    # If you typed a number (e.g. "python cluster_analysis.py 5"), use that instead
    if len(sys.argv) > 1:
        k = int(sys.argv[1])
        print(f"  -> overriding to k={k} (set on the command line)")

    # 5. Make the final groups and save each URL's group number
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    malicious["cluster"] = model.fit_predict(X_scaled)

    # 6. Describe each group
    describe_clusters(malicious, feature_cols)


if __name__ == "__main__":
    main()
