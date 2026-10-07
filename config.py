# Settings every script shares. Change them here and they change everywhere.

# The feature table. main.py makes it; the other scripts read it.
FEATURES_PATH = "dataset/processed/url_features.csv"

# Columns that are NOT given to the model:
#   url     - the URL itself
#   label   - the answer: 1 = malicious, 0 = safe
#   subtype - a more detailed label, e.g. "phishing" or "malware"
#   source  - which of the three datasets the row came from
NON_FEATURE_COLS = ("url", "label", "subtype", "source")

# Fixed random seed, so every run gives the same results
RANDOM_STATE = 42
