# Quick checks on the feature table before training the real models.
#
# Three checks, all with a simple Logistic Regression model:
# 1. Can the features spot phishing at all?
# 2. Can the features guess which dataset a URL came from?
# If so, a model might learn "which dataset" instead of "is it phishing".
# 3. How well does it do on a dataset it has never seen?
#
# Run with:  python sanity_baseline.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import FEATURES_PATH, NON_FEATURE_COLS, RANDOM_STATE


def build_baseline_model():
    # Scale the features, then use Logistic Regression.
    #
    # class_weight="balanced" stops the model from just guessing the most common answer.
    return make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
    )


def run_label_baseline(df: pd.DataFrame, feature_cols: list[str]) -> None:
    # Check 1: can the features spot phishing at all?
    # Also split the source column, so we can show scores per dataset
    X_train, X_test, y_train, y_test, _, src_test = train_test_split(
        df[feature_cols], df["label"], df["source"],
        test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE,
    )

    model = build_baseline_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("=== Overall ===")
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))

    # Scores for each dataset on its own
    for source in src_test.unique():
        mask = (src_test == source).to_numpy()  # True for rows from this dataset
        print(f"\n=== {source} (n={mask.sum()}) ===")
        print(classification_report(y_test[mask], preds[mask]))


def run_source_leakage_check(df: pd.DataFrame, feature_cols: list[str]) -> None:
    # Check 2: can the features guess which dataset a URL came from?
    # Like check 1, but the answer to predict is "source" instead of "label"
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df["source"],
        test_size=0.2, stratify=df["source"], random_state=RANDOM_STATE,
    )

    model = build_baseline_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Baseline = the score you'd get by always guessing the biggest dataset
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

    # Bigger number (ignoring + or -) = the model relies on that feature more
    print("\ntop |coefficient| features per source (what the model leans on to tell sources apart):")
    clf = model.named_steps["logisticregression"]
    for source, coefs in zip(clf.classes_, clf.coef_):
        top = pd.Series(coefs, index=feature_cols).abs().sort_values(ascending=False).head(5)
        print(f"  {source}: {', '.join(top.index)}")


def print_examples(rows: pd.DataFrame, inspect_cols: list[str], n: int = 3) -> None:
    # Print up to n random rows: the URL and a few of its features.
    # random_state=1 means we get the same examples every run
    for _, row in rows.sample(min(n, len(rows)), random_state=1).iterrows():
        details = ", ".join(f"{col}={row[col]}" for col in inspect_cols)
        print(f"    {row['url']!r}  [{details}]")


def run_held_out_source_split(df: pd.DataFrame, feature_cols: list[str]) -> None:
    # Check 3: how well does the model do on a dataset it has never seen?
    #
    # Train on two datasets, test on the third. Each dataset gets a turn.
    for held_out in sorted(df["source"].unique()):
        train_df = df[df["source"] != held_out]
        test_df = df[df["source"] == held_out]

        model = build_baseline_model()
        model.fit(train_df[feature_cols], train_df["label"])
        preds = model.predict(test_df[feature_cols])

        print(f"\n=== held out: {held_out} "
              f"(trained on the other two sources only, n_test={len(test_df)}) ===")
        print(classification_report(test_df["label"], preds))

        # Show a few URLs the model got wrong, to look at by hand
        inspect_cols = ["url_length", "num_hyphens"]
        results = test_df.copy()
        results["pred"] = preds
        false_positives = results[(results["label"] == 0) & (results["pred"] == 1)]
        false_negatives = results[(results["label"] == 1) & (results["pred"] == 0)]

        print(f"  false positives (benign predicted malicious): {len(false_positives)}")
        print_examples(false_positives, inspect_cols)

        print(f"  false negatives (malicious predicted benign): {len(false_negatives)}")
        print_examples(false_negatives, inspect_cols)


def main():
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [col for col in df.columns if col not in NON_FEATURE_COLS]

    print("############ Label baseline ############")
    run_label_baseline(df, feature_cols)

    print("\n\n############ Source leakage check ############")
    run_source_leakage_check(df, feature_cols)

    print("\n\n############ Held-out-source split ############")
    run_held_out_source_split(df, feature_cols)


if __name__ == "__main__":
    main()
