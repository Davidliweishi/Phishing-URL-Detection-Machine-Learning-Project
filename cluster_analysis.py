
import pandas as pd
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = "dataset/processed/url_features.csv"


# Try a range of cluster counts, pick the one with the best silhouette score
def pick_k(X_scaled, k_range=range(2, 9)) -> int:
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels, sample_size=5000, random_state=42)
        print(f"  k={k}: silhouette={scores[k]:.3f}")
    best_k = max(scores, key=scores.get)
    print(f"  -> chose k={best_k} (highest silhouette score)")
    return best_k


# For each cluster: subtype composition, top differentiating features, example URLs
def describe_clusters(df: pd.DataFrame, feature_cols: list[str]) -> None:
    overall_mean = df[feature_cols].mean()
    overall_std = df[feature_cols].std().replace(0, 1)

    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        print(f"\n=== Cluster {c} (n={len(sub)}, {len(sub) / len(df):.1%} of malicious rows) ===")

        print("subtype composition:")
        print(sub["subtype"].value_counts(normalize=True).round(3).to_string())

        print("source composition:")
        print(sub["source"].value_counts(normalize=True).round(3).to_string())

        z = (sub[feature_cols].mean() - overall_mean) / overall_std
        top = z.abs().sort_values(ascending=False).head(3)
        print("top differentiating features (cluster mean vs overall malicious-subset mean):")
        for feat in top.index:
            print(f"  {feat}: cluster={sub[feat].mean():.3f}  overall={overall_mean[feat]:.3f}  (z={z[feat]:+.2f})")

        print("example URLs:")
        for u in sub["url"].sample(min(3, len(sub)), random_state=1):
            print(f"  {u}")


def main():
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in ("url", "label", "subtype", "source")]

    malicious = df[df["label"] == 1].reset_index(drop=True)
    print(f"clustering {len(malicious)} malicious rows on {len(feature_cols)} features "
          f"(label not used as input)")

    X_scaled = StandardScaler().fit_transform(malicious[feature_cols])

    print("\nsilhouette scores by k:")
    k = pick_k(X_scaled)
    
    if len(sys.argv) > 1:
        k = int(sys.argv[1])
        print(f"  -> overriding to k={k} (set on the command line)")

    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    malicious["cluster"] = model.fit_predict(X_scaled)

    describe_clusters(malicious, feature_cols)


if __name__ == "__main__":
    main()
