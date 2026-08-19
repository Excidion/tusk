"""Cross-check tusk's values against featuretools on synthetic data.

Temporary scaffolding: this tier validates the algorithm during development and
is removed once the differential suite has passed unchanged across two
consecutive releases that added primitives. See the spec, section 11.

Run with: uv run --group validation pytest -m differential

Verified against featuretools 1.31.0. That version's ``EntitySet.add_dataframe``
takes ``dataframe_name=``, ``dataframe=``, ``index=`` keywords, and
``add_relationship`` takes four positional arguments:
``(parent_dataframe_name, parent_column_name, child_dataframe_name,
child_column_name)``. Both match what this file assumed.

Two fixtures, because depth-1 aggregation over a single relationship is the
easy part and the bugs did not live there:

* ``synthetic`` -- two tables at ``max_depth=1``, covering empty groups.
* ``deep`` -- three tables at ``max_depth=2``, covering *stacked* aggregation
  (``MEAN(sessions.SUM(transactions.amount))``), nested empty groups (a
  session with no transactions inside a customer), ``std``, and ``n_unique``
  over a column containing nulls.

On synthetic data with empty groups (customers with no sessions), this suite
originally caught tusk and featuretools *disagreeing* on SUM: tusk left an
empty group's SUM null while featuretools filled it with 0.0. That was
adjudicated: sum over an empty set is 0, the additive identity, and tusk
already reports COUNT = 0 for the same rows -- asserting a known-zero count
alongside an unknown total was internally inconsistent. ``Sum`` now declares
``default_value = 0`` (see ``tusk/primitives/aggregation.py``), so SUM agrees
with featuretools on empty groups too; that agreement is now a guarded
invariant below rather than a documented divergence. MEAN/MIN/MAX remain
null on empty groups on both sides, since those are genuinely undefined over
an empty set (0/0, and min/max of nothing).

Tusk and featuretools also *name* the same feature differently. Featuretools
uses the conventional ``MEAN(transactions.amount)``; tusk's column names are
plain SQL identifiers (``MEAN__transactions__amount``) so that they survive a
backend which generates SQL, where dots and parentheses parse as table
qualifiers and function calls. Tusk keeps the conventional form on
``Feature.display_name``. The parameters below are written in featuretools'
notation and translated with :func:`_as_tusk`, so this file reads as "what
featuretools calls it" throughout.

Two *deliberate* differences keep some columns out of the comparison rather
than being disagreements about values:

* featuretools flattens through intermediate tables, emitting
  ``MEAN(transactions.amount)`` directly onto ``customers``. Tusk's depth rule
  (spec section 6) counts every primitive application, so the same information
  arrives as ``MEAN(sessions.MEAN(transactions.amount))``. Only names present
  on both sides are compared.
* featuretools refuses to stack ``sum``/``min``/``max``/``std`` on themselves,
  or ``sum`` on ``count``; tusk allows it (``stack_on_self`` defaults to True).
  Those columns exist only in tusk, so they are likewise not compared.

One live *value* disagreement is recorded rather than papered over:
``N_UNIQUE`` of an empty group is 0 in tusk and NaN in featuretools. See
``test_n_unique_of_an_empty_group_diverges_from_featuretools`` for the
argument; tusk is not changed to match.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

import tusk

featuretools = pytest.importorskip("featuretools")

pytestmark = pytest.mark.differential

PRIMITIVES = {
    "count": "count",
    "sum": "sum",
    "mean": "mean",
    "min": "min",
    "max": "max",
}

# tusk name -> featuretools name. Only n_unique differs.
DEEP_PRIMITIVES = ["count", "sum", "mean", "min", "max", "std", "n_unique"]
DEEP_FEATURETOOLS_PRIMITIVES = [
    "count",
    "sum",
    "mean",
    "min",
    "max",
    "std",
    "num_unique",
]


@pytest.fixture
def synthetic():
    """A two-table dataset with empty groups and nulls."""
    rng = np.random.default_rng(0)
    customers = pd.DataFrame({"id": np.arange(1, 21)})
    sessions = pd.DataFrame(
        {
            "id": np.arange(1, 61),
            "customer_id": rng.integers(1, 18, size=60),  # customers 18-20 get none
            "value": rng.normal(size=60),
        }
    )
    return customers, sessions


def _featuretools_matrix(customers, sessions):
    es = featuretools.EntitySet("s")
    es = es.add_dataframe(dataframe_name="customers", dataframe=customers, index="id")
    es = es.add_dataframe(dataframe_name="sessions", dataframe=sessions, index="id")
    es = es.add_relationship("customers", "id", "sessions", "customer_id")
    matrix, _ = featuretools.dfs(
        entityset=es,
        target_dataframe_name="customers",
        agg_primitives=list(PRIMITIVES),
        trans_primitives=[],
        max_depth=1,
    )
    return matrix.sort_index()


def _tusk_matrix(customers, sessions):
    es = (
        tusk.Database("s")
        .add_table("customers", pl.from_pandas(customers).lazy(), primary_key="id")
        .add_table("sessions", pl.from_pandas(sessions).lazy(), primary_key="id")
        .add_relationship(
            parent="customers", child="sessions", foreign_key="customer_id"
        )
    )
    matrix, _ = tusk.dfs(
        entityset=es,
        target_dataframe_name="customers",
        agg_primitives=list(PRIMITIVES),
        trans_primitives=[],
        max_depth=1,
    )
    return matrix.collect().sort("id").to_pandas().set_index("id")


@pytest.fixture
def matrices(synthetic):
    """Both tools' feature matrices over the same synthetic tables."""
    customers, sessions = synthetic
    return _tusk_matrix(customers, sessions), _featuretools_matrix(customers, sessions)


def _as_tusk(name: str) -> str:
    """Translate a featuretools feature name into tusk's column name.

    Every construct featuretools spells with punctuation -- application,
    argument separator, path step, multi-output index, groupby suffix --
    tusk spells with ``__``.

    Args:
        name: A featuretools feature name, e.g. ``MEAN(sessions.value)``.

    Returns:
        The equivalent tusk column name, e.g. ``MEAN__sessions__value``.
    """
    for old, new in (
        (" by ", "__by__"),
        (", ", "__"),
        ("(", "__"),
        (")", ""),
        ("[", "__"),
        ("]", ""),
        (".", "__"),
    ):
        name = name.replace(old, new)
    return name


@pytest.mark.parametrize(
    "featuretools_name",
    [
        "COUNT(sessions)",
        "SUM(sessions.value)",
        "MEAN(sessions.value)",
        "MIN(sessions.value)",
        "MAX(sessions.value)",
    ],
)
def test_values_match_featuretools(matrices, featuretools_name):
    """These primitives agree on every row, including empty groups.

    COUNT and SUM both default to 0 for an empty group, on both sides.
    MEAN/MIN/MAX are undefined on an empty group and both tools represent
    that as NaN/null, which ``assert_series_equal`` treats as equal.
    """
    ours, theirs = matrices
    pd.testing.assert_series_equal(
        ours[_as_tusk(featuretools_name)].reset_index(drop=True).astype(float),
        theirs[featuretools_name].reset_index(drop=True).astype(float),
        check_names=False,
    )


def test_empty_groups_agree_on_count_and_sum(matrices, synthetic):
    """Both tools default COUNT and SUM to 0 for a customer with no sessions.

    This is the guarded invariant that replaced a real, adjudicated
    disagreement: tusk's ``Sum`` primitive previously had no declared
    ``default_value`` and left empty groups null, while featuretools filled
    them with 0.0. tusk now declares ``default_value = 0`` on ``Sum`` to
    match -- sum over an empty set is 0, the additive identity, and it would
    be inconsistent to report a known-zero COUNT alongside an unknown SUM.
    """
    customers, sessions = synthetic
    ours, theirs = matrices
    childless = sorted(set(customers["id"]) - set(sessions["customer_id"]))
    assert childless
    assert (ours.loc[childless, _as_tusk("COUNT(sessions)")] == 0).all()
    assert (theirs.loc[childless, "COUNT(sessions)"] == 0).all()
    assert (ours.loc[childless, _as_tusk("SUM(sessions.value)")] == 0).all()
    assert (theirs.loc[childless, "SUM(sessions.value)"] == 0).all()


def test_empty_groups_stay_null_for_mean_min_max(matrices, synthetic):
    """MEAN/MIN/MAX are genuinely undefined over an empty set on both sides."""
    customers, sessions = synthetic
    ours, theirs = matrices
    childless = sorted(set(customers["id"]) - set(sessions["customer_id"]))
    assert childless
    for column in (
        "MEAN(sessions.value)",
        "MIN(sessions.value)",
        "MAX(sessions.value)",
    ):
        assert ours.loc[childless, _as_tusk(column)].isna().all()
        assert theirs.loc[childless, column].isna().all()


@pytest.fixture
def deep():
    """Three tables with empty groups at both levels and nulls in a category.

    Customers 13-15 have no sessions; sessions 36-40 have no transactions, and
    every session belongs to some customer, so at least one customer holds a
    session with an empty transaction group. ``kind`` carries nulls so that
    ``n_unique``'s null handling is cross-checked rather than assumed.
    """
    rng = np.random.default_rng(7)
    customers = pd.DataFrame({"id": np.arange(1, 16)})
    sessions = pd.DataFrame(
        {
            "id": np.arange(1, 41),
            "customer_id": rng.integers(1, 13, size=40),
            "kind": rng.choice(["a", "b", "c", None], size=40),
        }
    )
    transactions = pd.DataFrame(
        {
            "id": np.arange(1, 121),
            "session_id": rng.integers(1, 36, size=120),
            "amount": rng.normal(size=120).round(4),
        }
    )
    return customers, sessions, transactions


def _featuretools_deep(customers, sessions, transactions):
    es = featuretools.EntitySet("d")
    es = es.add_dataframe(dataframe_name="customers", dataframe=customers, index="id")
    es = es.add_dataframe(dataframe_name="sessions", dataframe=sessions, index="id")
    es = es.add_dataframe(
        dataframe_name="transactions", dataframe=transactions, index="id"
    )
    es = es.add_relationship("customers", "id", "sessions", "customer_id")
    es = es.add_relationship("sessions", "id", "transactions", "session_id")
    matrix, _ = featuretools.dfs(
        entityset=es,
        target_dataframe_name="customers",
        agg_primitives=DEEP_FEATURETOOLS_PRIMITIVES,
        trans_primitives=[],
        max_depth=2,
    )
    return matrix.sort_index()


def _tusk_deep(customers, sessions, transactions):
    es = (
        tusk.Database("d")
        .add_table("customers", pl.from_pandas(customers).lazy(), primary_key="id")
        .add_table("sessions", pl.from_pandas(sessions).lazy(), primary_key="id")
        .add_table(
            "transactions", pl.from_pandas(transactions).lazy(), primary_key="id"
        )
        .add_relationship(
            parent="customers", child="sessions", foreign_key="customer_id"
        )
        .add_relationship(
            parent="sessions", child="transactions", foreign_key="session_id"
        )
    )
    matrix, _ = tusk.dfs(
        entityset=es,
        target_dataframe_name="customers",
        agg_primitives=DEEP_PRIMITIVES,
        trans_primitives=[],
        max_depth=2,
    )
    return matrix.collect().sort("id").to_pandas().set_index("id")


@pytest.fixture
def deep_matrices(deep):
    """Both tools' depth-2 feature matrices over the same three tables."""
    customers, sessions, transactions = deep
    return _tusk_deep(*deep), _featuretools_deep(customers, sessions, transactions)


STACKED = [
    f"{outer}(sessions.{inner})"
    for outer in ("MEAN", "MIN", "MAX", "STD", "SUM")
    for inner in (
        "COUNT(transactions)",
        "MEAN(transactions.amount)",
        "MIN(transactions.amount)",
        "MAX(transactions.amount)",
        "STD(transactions.amount)",
        "SUM(transactions.amount)",
    )
]


@pytest.mark.parametrize("name", ["COUNT(sessions)", *STACKED])
def test_stacked_values_match_featuretools(deep_matrices, name):
    """Every depth-2 feature both tools generate holds the same values.

    Names featuretools does not generate are skipped rather than failed: it
    flattens through intermediate tables and it refuses to stack sum/min/max
    on themselves, both of which are feature-set differences documented in
    this module's docstring, not value disagreements.
    """
    ours, theirs = deep_matrices
    if name not in theirs.columns:
        pytest.skip(f"featuretools does not generate {name}")
    assert _as_tusk(name) in ours.columns
    pd.testing.assert_series_equal(
        ours[_as_tusk(name)].reset_index(drop=True).astype(float),
        theirs[name].reset_index(drop=True).astype(float),
        check_names=False,
    )


def test_n_unique_matches_num_unique_over_a_column_with_nulls(deep_matrices, deep):
    """tusk's N_UNIQUE and featuretools' NUM_UNIQUE both ignore nulls.

    ``kind`` contains nulls, and polars counts null as a distinct value while
    pandas' ``nunique`` does not. This is the cross-check behind ``NUnique``
    subtracting a null indicator.

    Customers with no sessions at all are excluded here and asserted on
    separately below: that row is a live disagreement about the empty-group
    default, not about null handling.
    """
    customers, sessions, _ = deep
    ours, theirs = deep_matrices
    with_sessions = sorted(set(customers["id"]) & set(sessions["customer_id"]))
    pd.testing.assert_series_equal(
        ours.loc[with_sessions, _as_tusk("N_UNIQUE(sessions.kind)")]
        .reset_index(drop=True)
        .astype(float),
        theirs.loc[with_sessions, "NUM_UNIQUE(sessions.kind)"]
        .reset_index(drop=True)
        .astype(float),
        check_names=False,
    )


def test_n_unique_of_an_empty_group_diverges_from_featuretools(deep_matrices, deep):
    """DIVERGENCE, recorded deliberately: empty group gives tusk 0, them NaN.

    For a customer with no sessions, tusk reports
    ``N_UNIQUE(sessions.kind) == 0`` and featuretools reports
    ``NUM_UNIQUE(sessions.kind) == NaN``. featuretools' ``NumUnique`` declares
    ``default_value = nan`` while its ``Count`` declares ``0``.

    Tusk keeps 0, on exactly the argument already adjudicated for ``Sum``:
    both tools agree ``COUNT(sessions) == 0`` for these rows, which asserts we
    know there were zero child rows. The number of distinct values among zero
    known rows is zero, not unknown -- reporting a known-zero count beside an
    unknown distinct-count is internally inconsistent. MEAN/MIN/MAX/STD stay
    null on both sides because those are genuinely undefined over an empty
    set; a count is not.

    Featuretools is a reference, not an oracle. This test pins the divergence
    so it cannot drift unnoticed.
    """
    customers, sessions, _ = deep
    ours, theirs = deep_matrices
    childless = sorted(set(customers["id"]) - set(sessions["customer_id"]))
    assert childless
    assert (ours.loc[childless, _as_tusk("N_UNIQUE(sessions.kind)")] == 0).all()
    assert theirs.loc[childless, "NUM_UNIQUE(sessions.kind)"].isna().all()
    # Both agree there were zero child rows, which is what makes 0 the right
    # answer for the distinct count.
    assert (ours.loc[childless, _as_tusk("COUNT(sessions)")] == 0).all()
    assert (theirs.loc[childless, "COUNT(sessions)"] == 0).all()


def test_nested_empty_group_agrees(deep_matrices, deep):
    """A session with no transactions contributes 0 to SUM and COUNT, not null.

    The nested empty group is the case a depth-1 suite cannot reach, and the
    one where a default_value applied at the wrong level goes unnoticed.
    """
    customers, sessions, transactions = deep
    ours, theirs = deep_matrices
    childless_sessions = set(sessions["id"]) - set(transactions["session_id"])
    assert childless_sessions
    affected = sorted(
        set(sessions.loc[sessions["id"].isin(childless_sessions), "customer_id"])
    )
    assert affected
    for name in (
        "MIN(sessions.COUNT(transactions))",
        "MIN(sessions.SUM(transactions.amount))",
    ):
        pd.testing.assert_series_equal(
            ours.loc[affected, _as_tusk(name)].reset_index(drop=True).astype(float),
            theirs.loc[affected, name].reset_index(drop=True).astype(float),
            check_names=False,
        )
    nested = _as_tusk("MIN(sessions.COUNT(transactions))")
    assert (ours.loc[affected, nested] == 0).all()
