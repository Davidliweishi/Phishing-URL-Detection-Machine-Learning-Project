##Quick baseline checks to sanity-check the merged feature set before deeper modeling.

## Two checks:
## 1. Label baseline - can the lexical features predict phishing at all?

## 2. Source leakage check - can the same features predict which raw dataset a
##   URL came from? 

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = "dataset/processed/url_features.csv"


# Check 1: can the features predict phishing at all?
def run_label_baseline(df: pd.DataFrame, feature_cols: list[str]) -> None:
    X_train, X_test, y_train, y_test, _, src_test = train_test_split(
        df[feature_cols], df["label"], df["source"],
        test_size=0.2, stratify=df["label"], random_state=42,
    )

    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("=== Overall ===")
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))

    for source in src_test.unique():
        mask = (src_test == source).to_numpy()
        print(f"\n=== {source} (n={mask.sum()}) ===")
        print(classification_report(y_test[mask], preds[mask]))


# Check 2: can the same features predict which dataset a URL came from?
def run_source_leakage_check(df: pd.DataFrame, feature_cols: list[str]) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df["source"],
        test_size=0.2, stratify=df["source"], random_state=42,
    )

    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    baseline = y_test.value_counts(normalize=True).max()
    accuracy = (preds == y_test.to_numpy()).mean()
    print(f"majority-class baseline accuracy: {baseline:.3f}")
    print(f"model accuracy: {accuracy:.3f}")

    print(classification_report(y_test, preds))
    print("confusion matrix (rows=true source, cols=predicted source):")
    print(pd.DataFrame(
        confusion_matrix(y_test, preds, labels=model.classes_),
        index=model.classes_, columns=model.classes_,
    ))

    print("\ntop |coefficient| features per source (what the model leans on to tell sources apart):")
    clf = model.named_steps["logisticregression"]
    for cls, coefs in zip(clf.classes_, clf.coef_):
        top = pd.Series(coefs, index=feature_cols).abs().sort_values(ascending=False).head(5)
        print(f"  {cls}: {', '.join(top.index)}")


# Check 3: how much does that leakage actually cost on unseen sources?
def run_held_out_source_split(df: pd.DataFrame, feature_cols: list[str]) -> None:

    for held_out in sorted(df["source"].unique()):
        train_df = df[df["source"] != held_out]
        test_df = df[df["source"] == held_out]

        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        )
        model.fit(train_df[feature_cols], train_df["label"])
        preds = model.predict(test_df[feature_cols])

        print(f"\n=== held out: {held_out} "
              f"(trained on the other two sources only, n_test={len(test_df)}) ===")
        print(classification_report(test_df["label"], preds))

        # Error analysis: pull a few actual misclassified URLs for hand
        # inspection
        inspect_cols = [ "url_length", "num_hyphens"]
        results = test_df.copy()
        results["pred"] = preds
        false_positives = results[(results["label"] == 0) & (results["pred"] == 1)]
        false_negatives = results[(results["label"] == 1) & (results["pred"] == 0)]

        print(f"  false positives (benign predicted malicious): {len(false_positives)}")
        for _, row in false_positives.sample(min(3, len(false_positives)), random_state=1).iterrows():
            details = ", ".join(f"{c}={row[c]}" for c in inspect_cols)
            print(f"    {row['url']!r}  [{details}]")

        print(f"  false negatives (malicious predicted benign): {len(false_negatives)}")
        for _, row in false_negatives.sample(min(3, len(false_negatives)), random_state=1).iterrows():
            details = ", ".join(f"{c}={row[c]}" for c in inspect_cols)
            print(f"    {row['url']!r}  [{details}]")


def main():
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in ("url", "label", "subtype", "source")]

    print("############ Label baseline ############")
    run_label_baseline(df, feature_cols)

    print("\n\n############ Source leakage check ############")
    run_source_leakage_check(df, feature_cols)

    print("\n\n############ Held-out-source split ############")
    run_held_out_source_split(df, feature_cols)


if __name__ == "__main__":
    main()
