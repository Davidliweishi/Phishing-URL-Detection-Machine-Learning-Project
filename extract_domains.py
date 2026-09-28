import ipaddress
import re
## Splits URL strings into pieces(scheme, hostname, path,etc)
from urllib.parse import urlparse

import pandas as pd

## how we detect a URL that has http:// (for example)
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")

## All suffixes are from general knowledge based on public domains
_MULTI_LABEL_SUFFIXES = {
    "co.uk","co.uk", "org.uk", "gov.uk", "ac.uk",
    "com.au", "net.au", "org.au",
    "com.br", "net.br",
    "co.jp", "co.in", "co.nz", "co.za",
    "com.cn", "net.cn", "org.cn",
}

## check if hostname is an actual IP address
def _ensure_scheme(url: str) -> str:
    return url if _SCHEME_RE.match(url) else f"http://{url}"

def _is_ip_host(host: str) ->bool:
    try: 
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False

## Split hostname into subdomain/domain/depth
def _split_domain(hostname: str) -> tuple[str, str, int]:
    if not hostname or _is_ip_host(hostname):
        return "", hostname, 0 

    labels = hostname.split(".")
    if len(labels) <=2:
        return "", hostname, 0

    last_two = ".".join(labels[-2:])
    if last_two in _MULTI_LABEL_SUFFIXES:
        registrable = ".".join(labels[-3:])
        subdomain_labels = labels[:-3]
    else:
        registrable = last_two
        subdomain_labels = labels[:-2]

    return ".".join(subdomain_labels), registrable, len(subdomain_labels)

# Safely read the port
def _safe_port(parsed) -> int | None:
    try: return parsed.port
    except ValueError:
        return None

# Main Function
def extract_domain_features(urls: pd.Series) -> pd.DataFrame:
    parsed = urls.fillna("").apply(_ensure_scheme).apply(urlparse)
    hostname = parsed.apply(lambda p: p.hostname or "")

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