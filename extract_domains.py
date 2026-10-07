# Split a URL into its parts.
#
# Example: "http://login.paypal.co.uk:8080/a?b=1" gives
# scheme="http", hostname="login.paypal.co.uk", subdomain="login",
# domain="paypal.co.uk", subdomain_depth=1, path="/a", query="b=1",
# explicit_port=8080

import ipaddress
import re
from urllib.parse import urlparse

import pandas as pd

# Finds "http://", "https://" etc. at the start of a URL
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")

# Endings made of two parts, like ".co.uk".
# For "shop.example.co.uk" the real domain is "example.co.uk".
_MULTI_LABEL_SUFFIXES = {
    "co.uk", "org.uk", "gov.uk", "ac.uk",
    "com.au", "net.au", "org.au",
    "com.br", "net.br",
    "co.jp", "co.in", "co.nz", "co.za",
    "com.cn", "net.cn", "org.cn",
}


def _ensure_scheme(url: str) -> str:
    # Add "http://" if it's missing. Without it, Python can't find the hostname.
    if _SCHEME_RE.match(url):
        return url
    return f"http://{url}"


def _is_ip_host(host: str) -> bool:
    # True if the host is an IP address like "192.168.0.1".
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _split_domain(hostname: str) -> tuple[str, str, int]:
    # Split a hostname into (subdomain, domain, number of subdomain parts).
    #
    # "a.b.example.com" gives ("a.b", "example.com", 2)
    # "example.co.uk"   gives ("", "example.co.uk", 0)
    if not hostname or _is_ip_host(hostname):
        return "", hostname, 0

    labels = hostname.split(".")
    if len(labels) <= 2:
        return "", hostname, 0

    last_two = ".".join(labels[-2:])
    if last_two in _MULTI_LABEL_SUFFIXES:
        # Like ".co.uk": the domain is the last 3 parts
        registrable = ".".join(labels[-3:])
        subdomain_labels = labels[:-3]
    else:
        # Like ".com": the domain is the last 2 parts
        registrable = last_two
        subdomain_labels = labels[:-2]

    return ".".join(subdomain_labels), registrable, len(subdomain_labels)


def _safe_port(parsed) -> int | None:
    # Get the port number. Gives None if there isn't one or it's broken.
    try:
        return parsed.port
    except ValueError:
        return None


def extract_domain_features(urls: pd.Series) -> pd.DataFrame:
    # Make a table with one row per URL and one column per URL part.
    parsed = urls.fillna("").apply(_ensure_scheme).apply(urlparse)
    hostname = parsed.apply(lambda p: p.hostname or "")

    # Each item is (subdomain, domain, depth)
    split = hostname.apply(_split_domain)

    return pd.DataFrame({
        "scheme": parsed.apply(lambda p: p.scheme),
        "hostname": hostname,
        "path": parsed.apply(lambda p: p.path),
        "query": parsed.apply(lambda p: p.query),
        "explicit_port": parsed.apply(_safe_port),
        "is_ip_host": hostname.apply(_is_ip_host),
        "subdomain": split.apply(lambda t: t[0]),
        "domain": split.apply(lambda t: t[1]),
        "subdomain_depth": split.apply(lambda t: t[2]),
    }, index=urls.index)
