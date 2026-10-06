## Dataset Instructions and Explanation

*************

## Feature Justification 

- `url_length`, `hostname_length`, `num_dots` - malicious URLs are different in length and host/subdomain structure (Ma et al. 2009)
- `has_ip_host` - a raw IP as the hostname is atypical of legitimate sites (Ma et al. 2009)

- `num_slashes`, `num_hyphens`, `num_underscores`, `num_query_params` - frequency of the delimiter characters Ma et al. (2009) use to tokenise URLs

- `has_punycode_label` - flags IDN-encoded labels, the mechanism that enables homograph spoofing (Yazdani et al. 2020)

- `has_non_ascii`, `has_mixed_script` - after decoding punycode, flags hostnames mixing Latin with visually-confusable Cyrillic/Greek characters (Yazdani et al. 2020)

- `is_shortened` - shorteners hide the real destination domain, a common evasion technique

- `num_at_symbols` - an `@` makes browsers ignore everything before it as userinfo, a classic spoof (`http://trusted.com@evil.com`).

- `subdomain_depth` - deep subdomain nesting is used to bury a fake brand name behind something familiar-looking.

- `has_explicit_port` - legitimate sites rarely expose a non-standard port. Also validated against the UCI Phishing Websites Dataset's `Shortining_Service`, `having_At_Symbol`, `having_Sub_Domain`, `port` (Mohammad & McCluskey 2012)

- `letter_ratio`/`num_digits`/`digit_ratio` - auto-generated malicious URLs skew toward more digits and fewer letters

- `num_special_chars`/`special_char_ratio` - excess special characters are used to pad or obfuscate a URL. Also validated against PhiUSIIL's own raw columns `IsHTTPS`, `LetterRatioInURL`, `NoOfDegitsInURL`/`DegitRatioInURL`, `NoOfOtherSpecialCharsInURL`/`SpacialCharRatioInURL` (Prasad & Chandra 2024)

**Removed:** `has_https` - the source-leakage check in `sanity_baseline.py` showed it was one of the
top features for identifying which *source dataset* a row came from (the sources differ in whether
URLs include a scheme), not whether the URL is malicious. The scheme (`http://`, `https://`) and a
leading `www.` are now stripped before any feature is computed, so no feature depends on them.


**Excluded:** host-based features (WHOIS dates/registrant, DNS TTL, IP blacklist, AS/prefix, geographic IP) - Ma et al. (2009) and Mohammad & McCluskey (2012) both define these; live-page-content features (favicon, forms, popups, redirects, page title/text matching) - Mohammad & McCluskey (2012) and PhiUSIIL's own raw columns (Prasad & Chandra 2024) both include these. All excluded here because each needs a live network lookup or page fetch, and this app must run offline.


## References

### Datasets

Mohammad, R. and McCluskey, L. (2012) *Phishing Websites* [Dataset]. UCI Machine Learning
Repository. Available at: https://doi.org/10.24432/C51W2X (Accessed: 25 September 2026).
Used as a feature-design reference only (see Feature Justification) - it ships pre-computed
feature values with no raw URL column, so it is not merged into `url_features.csv` and is
not included in `dataset/raw/`.

Prasad, A. and Chandra, S. (2024) *PhiUSIIL Phishing URL (Website)* [Dataset]. UCI Machine
Learning Repository. Available at: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
Raw file: `dataset/raw/PhiUSIIL_Phishing_URL_Dataset.csv`.

Sid321axn (2021) *Malicious URLs dataset* [Dataset]. Kaggle. Available at:
https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
Raw file: `dataset/raw/malicious_phish.csv`.

Siddharthkumar25 (2019) *Malicious and benign URLs* [Dataset]. Kaggle. Available at:
https://www.kaggle.com/datasets/siddharthkumar25/malicious-and-benign-urls 
Raw file: `dataset/raw/urldata.csv`.

### Literature

Ma, J., Saul, L.K., Savage, S. and Voelker, G.M. (2009) 'Beyond blacklists: learning to
detect malicious web sites from suspicious URLs', *Proceedings of the 15th ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining (KDD '09)*, pp. 1245-1254.
doi: 10.1145/1557019.1557153.

Prasad, A. and Chandra, S. (2024) 'PhiUSIIL: a diverse security profile empowered phishing
URL detection framework based on similarity index and incremental learning', *Computers &
Security*, 136, 103545. doi: 10.1016/j.cose.2023.103545.

Yazdani, R., van der Toorn, O. and Sperotto, A. (2020) 'A case of identity: detection of
suspicious IDN homograph domains using active DNS measurements', *2020 IEEE European
Symposium on Security and Privacy Workshops (EuroS&PW)*, pp. 559-564.
doi: 10.1109/EuroSPW51379.2020.00082.

*******

## How to Run

### Setup

Python 3.11 with `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`, `lightgbm`, `joblib`, `streamlit`.

```
conda env create -f environment.yml
conda activate phishing
pip install pandas numpy matplotlib scikit-learn xgboost lightgbm joblib streamlit
```

Run every command from the repo root (scripts use relative paths).

### Pipeline

1. Run `Assignment_2.ipynb` top to bottom -> cleans `dataset/raw/` into `dataset/clean/`
2. Run `python3 main.py` -> merges + dedupes the 3 cleaned sources into `dataset/processed/url_features.csv`
3. Run `python3 data_analysis.py` -> saves four PNG exploratory charts to `outputs/figures/`
4. Run `python3 sanity_baseline.py` -> prints sanity-check results to the console
5. Run `python3 train.py` -> compares models, prints error analysis, saves `models/final_model.joblib` + `models/feature_columns.joblib`
6. Run `python3 cluster_analysis.py [k]` -> KMeans on the malicious class; with no `k` it picks k by silhouette score (final run used `python3 cluster_analysis.py 7`)
7. Run `streamlit run app.py` -> opens the URL checker in the browser (needs step 4's model files)

Tests: `python3 -m unittest test_extract_domains -v`

`dataset/raw/` = original files, `dataset/clean/` = per-source cleaned files,
`dataset/processed/` = merged feature table, `models/` = trained model,
`outputs/` = saved console output from each run (`main_output.txt`, `train_output.txt`,
`cluster_output_k5.txt`, `sanity_output_after.txt`, `train_output_before_normalisation.txt`)

***** 

## Repo Guide

- `Assignment_2.ipynb` - cleans each raw source (dupes, garbled URLs, label fixes) and saves to `dataset/clean/`

- `extract_domains.py` - parses a URL into scheme/hostname/path/subdomain/IP-host pieces

- `lexical_features.py` - computes the 22 features in `url_features.csv` from those parsed pieces

- `main.py` - loads the 3 cleaned sources, unifies their schema, dedupes within/across sources, runs feature extraction, saves `dataset/processed/url_features.csv`

- `data_analysis.py` - loads 4 PNG exploratory charts, analyses different trends between the datasets. 

- `sanity_baseline.py` - trains a baseline classifier and runs 3 diagnostics: label prediction, source-leakage check, held-out-source split

- `cluster_analysis.py` - unsupervised KMeans clustering within the malicious class only, to find structure not already in the label

- `train.py` - trains and compares Logistic Regression, Random Forest, XGBoost and LightGBM, runs error analysis, saves the final model to `models/`

- `app.py` - Streamlit web app: enter a URL, get a phishing probability, trust score and verdict from the saved model

- `test_extract_domains.py` - unit tests for `extract_domains.py`'s awkward cases (IPv6, malformed ports, multi-label suffixes, etc.)

*************

## Data Dictionary

`dataset/processed/url_features.csv` columns (4 info columns + 22 features). All features
are numeric; binary ones are 1/0. Features are computed after stripping the scheme and a leading `www.`.

- `url` - the URL
- `label` - 1 = malicious/phishing, 0 = benign
- `subtype` - original label from the source dataset
- `source` - which raw dataset the row came from
- `url_length` - length of the URL
- `hostname_length` - length of the hostname
- `num_digits` - digit count
- `digit_ratio` - digit count / URL length
- `num_dots` - `.` count
- `num_hyphens` - `-` count
- `num_underscores` - `_` count
- `num_slashes` - `/` count
- `num_at_symbols` - `@` count
- `num_special_chars` - special character count
- `num_query_params` - number of query parameters
- `has_explicit_port` - 1 if a port is specified
- `is_shortened` - 1 if domain is a known URL shortener
- `has_ip_host` - 1 if hostname is an IP address
- `subdomain_depth` - number of subdomain labels
- `subdomain_hyphen_count` - `-` count in subdomain
- `letter_ratio` - proportion of letters in the URL
- `special_char_ratio` - proportion of special characters
- `longest_char_run` - longest repeated-character run
- `has_punycode_label` - 1 if hostname has a punycode label
- `has_non_ascii` - 1 if decoded hostname has non-ASCII characters
- `has_mixed_script` - 1 if decoded hostname mixes Latin + Cyrillic/Greek
