# Turn each URL into numbers the model can learn from.
#
# Every feature comes from the URL text only (length, digits, symbols, ...).
# We never visit the website.

import pandas as pd

from extract_domains import extract_domain_features

# Symbols we count as "special"
_SPECIAL_CHARS = "-_.=&?%~+#@!$,;:*'\"<>^`{}|\\[]()/"

# Link shorteners hide where a link really goes
_SHORTNERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly"}

# Cyrillic and Greek letters. Some look just like English letters.
_CONFUSABLE_SCRIPT_RANGES = ((0x0400, 0x04FF), (0x0370, 0x03FF))


def _longest_char_run(s: str) -> int:
    # Longest run of the same character in a row. "aaab" gives 3.
    if not s:
        return 0
    longest = 1
    current = 1
    # Compare each character with the one before it
    for prev, cur in zip(s, s[1:]):
        if cur == prev:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
    return longest


def _decode_hostname_labels(hostname: str) -> str:
    # Turn "xn--" web addresses back into the letters a browser shows.
    #
    # Example: "xn--pple-43d" is shown as "аpple", but the first letter is Cyrillic.
    decoded_labels = []
    for label in hostname.split("."):
        if label.lower().startswith("xn--"):
            try:
                decoded_labels.append(label.encode("ascii").decode("idna"))
            except UnicodeError:
                # Can't decode it, so keep it as it is
                decoded_labels.append(label)
        else:
            decoded_labels.append(label)
    return ".".join(decoded_labels)


def _has_mixed_script(s: str) -> bool:
    # True if English letters are mixed with Cyrillic or Greek ones.
    #
    # Phishers swap in a look-alike letter to fake a real brand name.
    has_latin = any(ch.isascii() and ch.isalpha() for ch in s)
    has_confusable = any(
        any(lo <= ord(ch) <= hi for lo, hi in _CONFUSABLE_SCRIPT_RANGES) for ch in s
    )
    return has_latin and has_confusable


def extract_lexical_features(df: pd.DataFrame, url_col: str = "url") -> pd.DataFrame:
    # Make a table of features, with one row per URL.
    #
    # main.py and app.py both use this, so training and the app see the same features.
    # Remove "http://", "https://" and "www." first so they don't change the counts
    urls = (df[url_col].fillna("")
            .str.replace(r"^[A-Za-z][A-Za-z0-9+.-]*://", "", regex=True)
            .str.replace(r"^www\.", "", case=False, regex=True))
    domain_parts = extract_domain_features(urls)

    feats = pd.DataFrame(index=df.index)

    # URL length, but 1 instead of 0 so we never divide by zero
    safe_length = urls.str.len().replace(0, 1)

    # Basic counts
    feats["url_length"] = urls.str.len()
    feats["hostname_length"] = domain_parts["hostname"].str.len()

    feats["num_digits"] = urls.str.count(r"\d")
    feats["digit_ratio"] = feats["num_digits"] / safe_length
    feats["num_dots"] = urls.str.count(r"\.")
    feats["num_hyphens"] = urls.str.count("-")
    feats["num_underscores"] = urls.str.count("_")
    feats["num_slashes"] = urls.str.count("/")
    feats["num_at_symbols"] = urls.str.count("@")
    feats["num_special_chars"] = urls.apply(lambda u: sum(ch in _SPECIAL_CHARS for ch in u))
    # "a=1&b=2" has one "&" but two parameters, so add 1
    feats["num_query_params"] = domain_parts["query"].apply(lambda q: q.count("&") + 1 if q else 0)
    feats["has_explicit_port"] = domain_parts["explicit_port"].notna().astype(int)
    feats["is_shortened"] = domain_parts["domain"].isin(_SHORTNERS).astype(int)

    # Is the host an IP address instead of a name?
    hostname = domain_parts["hostname"]
    feats["has_ip_host"] = domain_parts["is_ip_host"].astype(int)
    # Removed: has_numeric_host, has_hex_host, has_ipv6_host, ip_octet_out_of_range.
    # They were (almost) always 0, so they told the model nothing.

    # Subdomain features
    subdomain = domain_parts["subdomain"]
    feats["subdomain_depth"] = domain_parts["subdomain_depth"]
    feats["subdomain_hyphen_count"] = subdomain.str.count("-")

    # What the URL is made of
    feats["letter_ratio"] = urls.str.count(r"[a-zA-Z]") / safe_length
    feats["special_char_ratio"] = feats["num_special_chars"] / safe_length
    feats["longest_char_run"] = urls.apply(_longest_char_run)

    # Look-alike letters (checked on the decoded hostname)
    feats["has_punycode_label"] = hostname.str.contains(r"(?:^|\.)xn--", regex=True).astype(int)
    decoded_hostname = hostname.apply(_decode_hostname_labels)
    feats["has_non_ascii"] = decoded_hostname.apply(lambda h: int(not h.isascii()))
    feats["has_mixed_script"] = decoded_hostname.apply(lambda h: int(_has_mixed_script(h)))

    return feats
