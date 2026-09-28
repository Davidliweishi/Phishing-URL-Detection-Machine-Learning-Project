## Dataset Citation Sources: 

Sid321axn (2021) Malicious URLs dataset [Dataset]. Kaggle. Available at:
https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
(Accessed: 25 September 2026).

Siddharthkumar25 (2019) Malicious and benign URLs [Dataset]. Kaggle. Available at:
https://www.kaggle.com/datasets/siddharthkumar25/malicious-and-benign-urls
(Accessed: 25 September 2026).

Prasad, A. and Chandra, S. (2024) PhiUSIIL phishing URL (website) [Dataset].
UCI Machine Learning Repository. Available at: https://doi.org/10.1016/j.cose.2023.103545
(Accessed: 25 September 2026).

Mohammad, R.M. and McCluskey, L. (2015) Phishing Websites Dataset [Dataset].
UCI Machine Learning Repository. Available at: https://archive.ics.uci.edu/dataset/327
(Accessed: 25 September 2026). Used as a feature-design reference only (see Feature
Justification) - it ships pre-computed feature values with no raw URL column, so it is
not merged into `url_features.csv`.

*************

## Feature Justification (put this into the report as well)

- `url_length`, `hostname_length`, `num_dots` - malicious URLs differ structurally in length and host/subdomain structure (Ma et al. 2009)
- `has_ip_host` - a raw IP as the hostname is atypical of legitimate sites (Ma et al. 2009)

- `num_slashes`, `num_hyphens`, `num_underscores`, `num_query_params` - frequency of the delimiter characters Ma et al. (2009) use to tokenise URLs

- `has_punycode_label` - flags IDN-encoded labels, the mechanism that enables homograph spoofing (Yazdani et al. 2020)

- `has_non_ascii`, `has_mixed_script` - after decoding punycode, flags hostnames mixing Latin with visually-confusable Cyrillic/Greek characters (Yazdani et al. 2020)

- `is_shortened` - shorteners hide the real destination domain, a common evasion technique

- `num_at_symbols` - an `@` makes browsers ignore everything before it as userinfo, a classic spoof (`http://trusted.com@evil.com`).

- `subdomain_depth` - deep subdomain nesting is used to bury a fake brand name behind something familiar-looking.

- `has_explicit_port` - legitimate sites rarely expose a non-standard port. Also validated against the UCI Phishing Websites Dataset's `Shortining_Service`, `having_At_Symbol`, `having_Sub_Domain`, `port` (Mohammad & McCluskey 2015)

- `has_https` - phishing pages have historically skipped HTTPS more often than legitimate sites.

- `letter_ratio`/`num_digits`/`digit_ratio` - auto-generated malicious URLs skew toward more digits and fewer letters

- `num_special_chars`/`special_char_ratio` - excess special characters are used to pad or obfuscate a URL. Also validated against PhiUSIIL's own raw columns `IsHTTPS`, `LetterRatioInURL`, `NoOfDegitsInURL`/`DegitRatioInURL`, `NoOfOtherSpecialCharsInURL`/`SpacialCharRatioInURL` (Prasad & Chandra 2024)



**Excluded:** host-based features (WHOIS dates/registrant, DNS TTL, IP blacklist, AS/prefix, geographic IP) - Ma et al. (2009) and Mohammad & McCluskey (2015) both define these; live-page-content features (favicon, forms, popups, redirects, page title/text matching) - Mohammad & McCluskey (2015) and PhiUSIIL's own raw columns (Prasad & Chandra 2024) both include these. All excluded here because each needs a live network lookup or page fetch, and this app must run offline.



Ma, J., Saul, L.K., Savage, S. and Voelker, G.M. (2009) 'Beyond blacklists: learning to
detect malicious web sites from suspicious URLs', in Proceedings of the 15th ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining (KDD '09), pp. 1245-1254.
doi: 10.1145/1557019.1557153.

Yazdani, K., van der Toorn, R. and Sperotto, A. (2020) 'A case of identity: detection of
suspicious IDN homograph domains using active DNS measurements', in 2020 IEEE European
Symposium on Security and Privacy Workshops (EuroS&PW), pp. 559-564.
doi: 10.1109/EuroSPW51379.2020.00082.

*******

## How to Run

Requirements: Python 3 with `pandas`, `numpy`, `matplotlib`, `scikit-learn`.

1. Run `Assignment_2.ipynb` top to bottom → cleans `dataset/raw/` into `dataset/clean/`
2. Run `python3 main.py` → merges + dedupes the 3 cleaned sources into `dataset/processed/url_features.csv`
3. Run `python3 sanity_baseline.py` → prints sanity-check results to the console

`dataset/raw/` = original files, `dataset/clean/` = per-source cleaned files

***** 

## Repo Guide

- `Assignment_2.ipynb` - cleans each raw source (dupes, garbled URLs, label fixes) and saves to `dataset/clean/`

- `extract_domains.py` - parses a URL into scheme/hostname/path/subdomain/IP-host pieces

- `lexical_features.py` - computes the 23 features in `url_features.csv` from those parsed pieces

- `main.py` - loads the 3 cleaned sources, unifies their schema, dedupes within/across sources, runs feature extraction, saves `dataset/processed/url_features.csv`
- `sanity_baseline.py` - trains a baseline classifier and runs 3 diagnostics: label prediction, source-leakage check, held-out-source split
- `cluster_analysis.py` - unsupervised KMeans clustering within the malicious class only, to find structure not already in the label
- `test_extract_domains.py` - unit tests for `extract_domains.py`'s awkward cases (IPv6, malformed ports, multi-label suffixes, etc.)

*************

## Data Dictionary

`dataset/processed/url_features.csv` columns. All features are numeric;
binary ones are 1/0.

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
- `has_https` - 1 if scheme is https
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
 