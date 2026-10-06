## Tests for the awkward URL-parsing cases in extract_domains.py.

## Run with command: python -m unittest test_extract_domains -v

import unittest

import pandas as pd

from extract_domains import extract_domain_features


def _row(url: str) -> dict:
    return extract_domain_features(pd.Series([url])).iloc[0].to_dict()


class TestExtractDomainFeatures(unittest.TestCase):
    def test_missing_scheme_gets_default_http(self):
        row = _row("example.com/path")
        self.assertEqual(row["scheme"], "http")
        self.assertEqual(row["hostname"], "example.com")

    def test_scheme_already_present_is_not_doubled(self):
        row = _row("https://example.com/path")
        self.assertEqual(row["scheme"], "https")

    def test_ipv4_host_detected(self):
        row = _row("http://192.168.1.1/admin")
        self.assertTrue(row["is_ip_host"])
        self.assertEqual(row["domain"], "192.168.1.1")
        self.assertEqual(row["subdomain_depth"], 0)

    def test_ipv6_host_detected_with_port(self):
        row = _row("http://[::1]:8080/path")
        self.assertTrue(row["is_ip_host"])
        self.assertEqual(row["hostname"], "::1")
        self.assertEqual(row["explicit_port"], 8080)

    def test_two_label_domain_has_no_subdomain(self):
        row = _row("http://example.com")
        self.assertEqual(row["subdomain"], "")
        self.assertEqual(row["domain"], "example.com")
        self.assertEqual(row["subdomain_depth"], 0)

    def test_multi_label_suffix_splits_correctly(self):
        # "co.uk" is a two-label public suffix - the  domain
        # should be "example.co.uk", not just "co.uk"
        row = _row("https://www.example.co.uk/login")
        self.assertEqual(row["subdomain"], "www")
        self.assertEqual(row["domain"], "example.co.uk")
        self.assertEqual(row["subdomain_depth"], 1)

    def test_nested_subdomain_with_multi_label_suffix(self):
        row = _row("http://a.b.example.co.uk/x")
        self.assertEqual(row["subdomain"], "a.b")
        self.assertEqual(row["domain"], "example.co.uk")
        self.assertEqual(row["subdomain_depth"], 2)

    def test_explicit_port_extracted(self):
        row = _row("http://example.com:8080/path")
        self.assertEqual(row["explicit_port"], 8080)

    def test_missing_port_is_none(self):
        row = _row("http://example.com/path")
        self.assertTrue(pd.isna(row["explicit_port"]))

    def test_malformed_port_does_not_crash(self):
        row = _row("http://example.com:abc/path")
        self.assertTrue(pd.isna(row["explicit_port"]))

    def test_missing_url_handled_gracefully(self):
        row = extract_domain_features(pd.Series([None])).iloc[0].to_dict()
        self.assertEqual(row["hostname"], "")
        self.assertFalse(row["is_ip_host"])
        self.assertEqual(row["subdomain_depth"], 0)

    def test_defanged_bracket_url_raises(self):
        with self.assertRaises(ValueError):
            _row("http://ladiesfirst-privileges[.]com/x")

    def test_uppercase_url(self):
        row = _row("HTTP://EXAMPLE.COM")
        self.assertEqual(row["hostname"], "example.com")


if __name__ == "__main__":
    unittest.main()
