"""Cross-check tusk's email and URL transform primitives against featuretools.

Each test builds one primitive on both sides over the shared table in
``tests/transform_cases.py`` and compares the one resulting column. tusk's
``domain`` and ``top_level_domain`` read both URLs and email addresses, so
they stand in for featuretools' ``email_address_to_domain``,
``url_to_domain`` and ``url_to_tld``, and agree with each only on
well-formed values.

Run with: uv run --group validation pytest -m differential

Verified against featuretools 1.31.0.
"""

import pandas as pd
import pytest

from differential.transform_matrices import (
    assert_agree,
    featuretools_values,
    tusk_values,
)

featuretools = pytest.importorskip("featuretools")

pytestmark = pytest.mark.differential

EMAIL_DOMAIN_DIVERGENCES = {3: ("no-at-sign", None), 4: ("c.com", "b"), 5: (None, "")}
URL_DOMAIN_DIVERGENCES = {
    5: ("files.example.org", "ftp"),
    6: (None, "http"),
    7: (None, "www."),
}
URL_TOP_LEVEL_DOMAIN_DIVERGENCES = {4: ("company", "com"), 5: ("org", None)}


def test_url_to_protocol_matches_featuretools():
    """Covers a null, no protocol and a protocol other than http or https."""
    ours = tusk_values("url_to_protocol", "URL_TO_PROTOCOL__url")
    theirs = featuretools_values("url_to_protocol", "URL_TO_PROTOCOL(url)")
    assert_agree(ours, theirs)


def test_domain_matches_email_address_to_domain_on_well_formed_addresses():
    """Covers a null, padding, mixed case and a blank string."""
    ours = tusk_values("domain", "DOMAIN__email")
    theirs = featuretools_values(
        "email_address_to_domain",
        "EMAIL_ADDRESS_TO_DOMAIN(email)",
    )
    assert_agree_except(ours, theirs, EMAIL_DOMAIN_DIVERGENCES)


def test_domain_diverges_from_email_address_to_domain_on_malformed_addresses():
    """No `@` is read as a bare domain, the last of two `@` counts, and a
    trailing `@` leaves no domain.
    """
    ours = tusk_values("domain", "DOMAIN__email")
    theirs = featuretools_values(
        "email_address_to_domain",
        "EMAIL_ADDRESS_TO_DOMAIN(email)",
    )
    assert_diverge(ours, theirs, EMAIL_DOMAIN_DIVERGENCES)


def test_domain_matches_url_to_domain_on_http_urls():
    """Covers a null, user info, a port, a query and `www.`."""
    ours = tusk_values("domain", "DOMAIN__url")
    theirs = featuretools_values("url_to_domain", "URL_TO_DOMAIN(url)")
    assert_agree_except(ours, theirs, URL_DOMAIN_DIVERGENCES)


def test_domain_diverges_from_url_to_domain_on_other_schemes_and_empty_hosts():
    """featuretools only knows http and https, and where no host is left it
    reads the scheme or `www.` as the host instead.
    """
    ours = tusk_values("domain", "DOMAIN__url")
    theirs = featuretools_values("url_to_domain", "URL_TO_DOMAIN(url)")
    assert_diverge(ours, theirs, URL_DOMAIN_DIVERGENCES)


def test_top_level_domain_matches_url_to_tld_where_the_last_label_is_a_listed_tld():
    """featuretools only knows a fixed list of common TLDs; every other row agrees."""
    ours = tusk_values("top_level_domain", "TOP_LEVEL_DOMAIN__url")
    theirs = featuretools_values("url_to_tld", "URL_TO_TLD(url)")
    assert_agree_except(ours, theirs, URL_TOP_LEVEL_DOMAIN_DIVERGENCES)


def test_top_level_domain_keeps_the_whole_last_label():
    """``mysite.company`` is `company`, where featuretools prefix-matches `com`,
    and ``ftp://files.example.org`` is `org`, where featuretools finds no host.
    """
    ours = tusk_values("top_level_domain", "TOP_LEVEL_DOMAIN__url")
    theirs = featuretools_values("url_to_tld", "URL_TO_TLD(url)")
    assert_diverge(ours, theirs, URL_TOP_LEVEL_DOMAIN_DIVERGENCES)


def assert_agree_except(ours, theirs, divergences):
    """Assert that tusk and featuretools agree on every row but the divergent ones.

    Args:
        ours: tusk's column in id order.
        theirs: featuretools' column in id order.
        divergences: The row positions where the two differ, as keys.
    """
    kept = [row for row in range(len(ours)) if row not in divergences]
    assert_agree([ours[row] for row in kept], [theirs[row] for row in kept])


def assert_diverge(ours, theirs, divergences):
    """Assert the value each side gives on every divergent row.

    Args:
        ours: tusk's column in id order.
        theirs: featuretools' column in id order.
        divergences: Row position to the pair of tusk's and featuretools'
            value there; a featuretools None stands for any missing value.
    """
    for row, (our, their) in divergences.items():
        assert ours[row] == our
        if their is None:
            assert pd.isna(theirs[row])
        else:
            assert theirs[row] == their
