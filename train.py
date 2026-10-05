# Standard Library
import os
import time

# Data handling
import joblib
import pandas as pd

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

#Preprocessing and data splitting
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#Evaluation metrics
from sklearn.metrics import(
    average_precision_score, # PR-AUC
    classification_report,   # precision / recall / F1 per class
    confusion_matrix, 
    roc_auc_score,
)

# Constants
FEATURES_PATH = "dataset/processed/url_features.csv"
MODELS_DIR = "models"
NON_FEATURE_COLS = ("url", "label", "subtype", "source")
RANDOM_STATE = 42

# Load the feature table and identify the feature columns (model inputs),
# separate from the label and the url/subtype/source columns kept for analysis
def load_data() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"loaded {len(df)} rows, {len(feature_cols)} features, "
          f"{df['label'].mean():.1%} malicious")
    return df, feature_cols

# Define the four classifiers to compare. 
def build_models() -> dict:
    return {
        "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=2.17, eval_metric="logloss",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, num_leaves=63, learning_rate=0.1,
            class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE, verbose=-1,
        ),
    }

#Evaluate 
# This scores a model on test data
def evaluate(name: str, model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {name} ---")
    print(classification_report(y_test, preds, digits=3))
    print("confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_test, preds))

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

# Standard evaluation: a stratified 80/20 random split, training and scoring
# all four models on the same test set.
def random_split_experiment(df: pd.DataFrame, feature_cols: list[str]):
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

# Harder evaluation: for each source in turn, train on the other two sources
# only and test on the one the models never saw. A large drop from the
# random-split scores means a model relied on dataset-specific patterns
# rather than signals that transfer to URLs collected a different way.
def held_out_source_experiment(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for held_out in sorted(df["source"].unique()):
        train_df = df[df["source"] != held_out]
        test_df = df[df["source"] == held_out]
        print(f"\n########## Held out: {held_out} "
              f"({len(train_df)} train / {len(test_df)} test) ##########")

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

# Error analysis for the final model: count false positives and false
# negatives, show which sources they come from, and print a random sample of
# each for manual inspection.
def error_analysis(df: pd.DataFrame, model, X_test, y_test, n: int = 10) -> None:
    results = df.loc[X_test.index, ["url", "source", "subtype"]].copy()
    results["true"] = y_test.values
    results["pred"] = model.predict(X_test)
    results["prob_malicious"] = model.predict_proba(X_test)[:, 1].round(3)

    fp = results[(results["true"] == 0) & (results["pred"] == 1)]
    fn = results[(results["true"] == 1) & (results["pred"] == 0)]
    print(f"\n########## Error analysis ##########")
    print(f"false positives (benign flagged malicious): {len(fp)}")
    print(f"false negatives (malicious missed): {len(fn)}")

    print("\nerror rate by source:")
    errors = results["true"] != results["pred"]
    print(errors.groupby(results["source"]).mean().round(3).to_string())

    pd.set_option("display.max_colwidth", 100)
    print(f"\n--- {n} random false positives ---")
    print(fp.sample(min(n, len(fp)), random_state=RANDOM_STATE).to_string(index=False))
    print(f"\n--- {n} random false negatives ---")
    print(fn.sample(min(n, len(fn)), random_state=RANDOM_STATE).to_string(index=False))

# Save the final model and its feature column order
def save_model(model, feature_cols: list[str]) -> None:
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
    final_model = models["Random Forest"]
    error_analysis(df, final_model, X_test, y_test)
    save_model(final_model, feature_cols)