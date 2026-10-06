import pandas as pd

from extract_domains import extract_domain_features

_SPECIAL_CHARS = "-_.=&?%~+#@!$,;:*'\"<>^`{}|\\[]()/"
_SHORTNERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly"}
_CONFUSABLE_SCRIPT_RANGES = ((0x0400, 0x04FF), (0x0370, 0x03FF))


# Finds the longest line of same character repetition
def _longest_char_run(s: str) -> int:
    if not s:
        return 0
    longest = current = 1
    for prev, cur in zip(s, s[1:]):
        current = current + 1 if cur == prev else 1
        longest = max(longest, current)
    return longest

# Decode punycode ('xn--') labels back to Unicode so homoglyph checks see the
# characters a browser would actually display.
def _decode_hostname_labels(hostname: str) -> str:
    decoded_labels = []
    for label in hostname.split("."):
        if label.lower().startswith("xn--"):
            try:
                decoded_labels.append(label.encode("ascii").decode("idna"))
            except UnicodeError:
                decoded_labels.append(label)
        else:
            decoded_labels.append(label)
    return ".".join(decoded_labels)

# Checks if the URL mixes normal Latin letters with Greek ones (a feature of URL Phishing)
def _has_mixed_script(s: str) -> bool:
    has_latin = any(ch.isascii() and ch.isalpha() for ch in s)
    has_confusable = any(
        any(lo <= ord(ch) <= hi for lo, hi in _CONFUSABLE_SCRIPT_RANGES) for ch in s
    )
    return has_latin and has_confusable

# Function and features setup

def extract_lexical_features(df: pd.DataFrame, url_col: str = "url") -> pd.DataFrame:
    # Strip the scheme ("http://", "https://") and a leading "www." before
    # computing any feature. 
    urls = (df[url_col].fillna("")
            .str.replace(r"^[A-Za-z][A-Za-z0-9+.-]*://", "", regex=True)
            .str.replace(r"^www\.", "", case=False, regex=True))
    domain_parts = extract_domain_features(urls)

    feats = pd.DataFrame(index=df.index)

# General URL structure
    feats["url_length"] = urls.str.len()
    feats["hostname_length"] = domain_parts["hostname"].str.len()

    feats["num_digits"] = urls.str.count(r"\d")
    feats["digit_ratio"] = feats["num_digits"] / feats["url_length"].replace(0, 1)
    feats["num_dots"] = urls.str.count(r"\.")
    feats["num_hyphens"] = urls.str.count("-")
    feats["num_underscores"] = urls.str.count("_")
    feats["num_slashes"] = urls.str.count("/")
    feats["num_at_symbols"] = urls.str.count("@")
    feats["num_special_chars"] = urls.apply(lambda u: sum(ch in _SPECIAL_CHARS for ch in u))
    feats["num_query_params"] = domain_parts["query"].apply(lambda q: q.count("&") + 1 if q else 0)
    feats["has_explicit_port"] = domain_parts["explicit_port"].notna().astype(int)
    feats["is_shortened"] = domain_parts["domain"].isin(_SHORTNERS).astype(int)

# IP-host features
    hostname = domain_parts["hostname"]
    feats["has_ip_host"] = domain_parts["is_ip_host"].astype(int)
    # has_numeric_host, has_hex_host, has_ipv6_host, ip_octet_out_of_range were
    # cut - 0 or near-0 variance across all 1.3M rows 

# Subdomain features
    subdomain = domain_parts["subdomain"]
    feats["subdomain_depth"] = domain_parts["subdomain_depth"]
    feats["subdomain_hyphen_count"] = subdomain.str.count("-")

#Character Composition
    feats["letter_ratio"] = (
        urls.str.count(r"[a-zA-Z]") / feats["url_length"].replace(0, 1)
    )
    feats["special_char_ratio"] = feats["num_special_chars"] / feats["url_length"].replace(0, 1)
    feats["longest_char_run"] = urls.apply(_longest_char_run)

# Homoglyphs - checked on the decoded hostname.
    feats["has_punycode_label"] = hostname.str.contains(r"(?:^|\.)xn--", regex=True).astype(int)
    decoded_hostname = hostname.apply(_decode_hostname_labels)
    feats["has_non_ascii"] = decoded_hostname.apply(lambda h: int(not h.isascii()))
    feats["has_mixed_script"] = decoded_hostname.apply(lambda h: int(_has_mixed_script(h)))

    return feats




