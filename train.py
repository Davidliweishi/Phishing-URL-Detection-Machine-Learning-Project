# Train four models, compare them, and save the best one for the app.
#
# Steps:
# 1. Random split: train on 80% of the data, test on the other 20%
# 2. Unseen dataset: train on two datasets, test on the third (harder test)
# 3. Look at the mistakes the final model (Random Forest) makes
# 4. Save the final model in models/
#
# Run with:  python train.py

# Built into Python
import os
import time

# Loading and saving data
import joblib
import pandas as pd

# The four models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Preparing and splitting the data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Scores for how good a model is
from sklearn.metrics import (
    average_precision_score,  # PR-AUC
    classification_report,    # precision, recall and F1
    confusion_matrix,
    roc_auc_score,
)

from config import FEATURES_PATH, NON_FEATURE_COLS, RANDOM_STATE

MODELS_DIR = "models"


def load_data() -> tuple[pd.DataFrame, list[str]]:
    # Load the data and list the feature columns (what the model learns from).
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [col for col in df.columns if col not in NON_FEATURE_COLS]
    print(f"loaded {len(df)} rows, {len(feature_cols)} features, "
          f"{df['label'].mean():.1%} malicious")
    return df, feature_cols


def build_models() -> dict:
    # Make the four models (not trained yet).
    #
    # There are about twice as many safe URLs as malicious ones. So each model
    # is told to care more about the malicious ones:
    #   - class_weight="balanced" does this for us
    #   - XGBoost uses scale_pos_weight instead (safe count / malicious count = 2.17)
    return {
        # Logistic Regression needs scaled features, so StandardScaler runs first
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,     # number of trees
            min_samples_leaf=5,   # stops trees learning tiny, one-off patterns
            class_weight="balanced",
            n_jobs=-1,            # use every CPU core
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=2.17, eval_metric="logloss",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, num_leaves=63, learning_rate=0.1,
            class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,  # don't print training messages
        ),
    }


def evaluate(name: str, model, X_test, y_test) -> dict:
    # Test a trained model, print its scores, and return the main ones.
    preds = model.predict(X_test)                # 0 or 1 for each URL
    probs = model.predict_proba(X_test)[:, 1]    # chance each URL is malicious

    print(f"\n--- {name} ---")
    print(classification_report(y_test, preds, digits=3))
    print("confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_test, preds))

    # Same report as a dictionary, so we can pick out numbers.
    # report["1"] = scores for the malicious class.
    report = classification_report(y_test, preds, output_dict=True)
    return {
        "model": name,
        "accuracy": report["accuracy"],
        "precision_mal": report["1"]["precision"],
        "recall_mal": report["1"]["recall"],
        "f1_mal": report["1"]["f1-score"],
        "roc_auc": roc_auc_score(y_test, probs),
        "pr_auc": average_precision_score(y_test, probs),
    }


def random_split_experiment(df: pd.DataFrame, feature_cols: list[str]):
    # Normal test: train on a random 80%, test on the other 20%.
    #
    # All four models use the same test set, so the scores are fair to compare.
    # stratify keeps the same mix of safe and malicious URLs in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df["label"],
        test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE,
    )
    print(f"\n########## Random split: {len(X_train)} train / {len(X_test)} test ##########")

    models = build_models()
    rows = []
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        print(f"\n{name} trained in {train_seconds:.1f}s")

        row = evaluate(name, model, X_test, y_test)
        row["train_seconds"] = train_seconds
        rows.append(row)

    results = pd.DataFrame(rows).set_index("model")
    print("\n=== Random split summary ===")
    print(results.round(3).to_string())
    return results, models, X_test, y_test


def held_out_source_experiment(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    # Harder test: train on two datasets, test on the third.
    #
    # Repeat so each dataset gets a turn as the unseen one.
    rows = []
    for held_out in sorted(df["source"].unique()):
        train_df = df[df["source"] != held_out]
        test_df = df[df["source"] == held_out]
        print(f"\n########## Held out: {held_out} "
              f"({len(train_df)} train / {len(test_df)} test) ##########")

        # New, untrained models each time
        for name, model in build_models().items():
            model.fit(train_df[feature_cols], train_df["label"])
            row = evaluate(f"{name} | held out: {held_out}", model,
                           test_df[feature_cols], test_df["label"])
            row["model"] = name
            row["held_out"] = held_out
            rows.append(row)

    results = pd.DataFrame(rows).set_index(["held_out", "model"])
    print("\n=== Held-out-source summary ===")
    print(results.round(3).to_string())
    return results


def error_analysis(df: pd.DataFrame, model, X_test, y_test, n: int = 10) -> None:
    # Look at the mistakes the final model makes.
    #
    # False positive = safe URL called malicious.
    # False negative = malicious URL called safe.
    # Prints how many of each, which datasets they come from, and some examples.
    # X_test remembers each row's original number, so we can look up its url/source/subtype
    results = df.loc[X_test.index, ["url", "source", "subtype"]].copy()
    results["true"] = y_test.values
    results["pred"] = model.predict(X_test)
    results["prob_malicious"] = model.predict_proba(X_test)[:, 1].round(3)

    false_positives = results[(results["true"] == 0) & (results["pred"] == 1)]
    false_negatives = results[(results["true"] == 1) & (results["pred"] == 0)]
    print(f"\n########## Error analysis ##########")
    print(f"false positives (benign flagged malicious): {len(false_positives)}")
    print(f"false negatives (malicious missed): {len(false_negatives)}")

    # The average of True/False values = the fraction that are True
    print("\nerror rate by source:")
    errors = results["true"] != results["pred"]
    print(errors.groupby(results["source"]).mean().round(3).to_string())

    pd.set_option("display.max_colwidth", 100)  # show longer URLs
    print(f"\n--- {n} random false positives ---")
    print(false_positives.sample(min(n, len(false_positives)), random_state=RANDOM_STATE).to_string(index=False))
    print(f"\n--- {n} random false negatives ---")
    print(false_negatives.sample(min(n, len(false_negatives)), random_state=RANDOM_STATE).to_string(index=False))


def save_model(model, feature_cols: list[str]) -> None:
    # Save the model and the feature column order. app.py needs both.
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "final_model.joblib")
    joblib.dump(model, model_path, compress=3)
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.joblib"))
    print(f"\nsaved final model -> {model_path} "
          f"({os.path.getsize(model_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    df, feature_cols = load_data()
    _, models, X_test, y_test = random_split_experiment(df, feature_cols)
    held_out_source_experiment(df, feature_cols)

    # Random Forest is the model we chose
    final_model = models["Random Forest"]
    error_analysis(df, final_model, X_test, y_test)
    save_model(final_model, feature_cols)
