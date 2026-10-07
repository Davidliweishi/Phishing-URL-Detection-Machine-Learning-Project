## Datasets

Download the raw files into `dataset/raw/`:

- `PhiUSIIL_Phishing_URL_Dataset.csv` - Prasad & Chandra (2024), UCI: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
- `malicious_phish.csv` - Sid321axn (2021), Kaggle: https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
- `urldata.csv` - Siddharthkumar25 (2019), Kaggle: https://www.kaggle.com/datasets/siddharthkumar25/malicious-and-benign-urls

## Environment Setup

Python 3.11 with `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`, `lightgbm`, `joblib`, `streamlit` (all listed in `environment.yml`).

```
conda env create -f environment.yml
conda activate phishing
```

Run every command from the repo root (scripts use relative paths).

## Data Processing

1. Run `Assignment_2.ipynb` top to bottom -> cleans `dataset/raw/` into `dataset/clean/`
2. `python main.py` -> merges and dedupes the 3 cleaned sources, extracts features, saves `dataset/processed/url_features.csv`
3. *(Optional)* `python data_analysis.py` -> saves four exploratory charts to `outputs/figures/`
4. *(Optional)* `python sanity_baseline.py` -> prints label, source-leakage and held-out-source checks
5. *(Optional)* `python cluster_analysis.py [k]` -> KMeans on the malicious class; with no `k` it picks k by silhouette score (final run used `k=5`)

## Training

`python train.py` -> compares Logistic Regression, Random Forest, XGBoost and LightGBM, prints error analysis, saves `models/final_model.joblib` and `models/feature_columns.joblib`

## Prediction

`streamlit run app.py` -> opens the URL checker in the browser (requires the model files from Training). Enter a URL to get its phishing probability, trust score and verdict.

## Tests

`python -m unittest test_extract_domains -v`

## Folder Layout

`dataset/raw/` = original files, `dataset/clean/` = per-source cleaned files,
`dataset/processed/` = merged feature table, `models/` = trained model,
`outputs/` = saved console output from each run.
Shared settings (feature table path, non-feature columns, random seed) live in `config.py`.