import datetime as dt
from dataclasses import dataclass

import polars as pl
import pytest

import tusk
from tusk.synthesis import synthesize


def names(features):
    return {f.name for f in features}


def test_depth_one_aggregations(es):
    got = synthesize(
        es,
        "customers",
        agg_primitives=["count", "mean"],
        trans_primitives=[],
        groupby_trans_primitives=[],
        max_depth=1,
    )
    assert names(got) == {"age", "COUNT(sessions)"}


def test_depth_two_stacks_through_two_relationships(es):
    got = synthesize(
        es,
        "customers",
        agg_primitives=["count", "mean"],
        trans_primitives=[],
        groupby_trans_primitives=[],
        max_depth=2,
    )
    assert "MEAN(sessions.MEAN(transactions.amount))" in names(got)
    assert "MEAN(sessions.COUNT(transactions))" in names(got)
    assert "COUNT(sessions)" in names(got)


def test_target_keys_are_not_emitted_as_features(es):
    got = synthesize(
        es,
        "customers",
        agg_primitives=["count"],
        trans_primitives=[],
        groupby_trans_primitives=[],
        max_depth=1,
    )
    assert "id" not in names(got)
    assert "signed_up_at" not in names(got)


def test_never_traverses_back_so_target_columns_do_not_return(es):
    got = synthesize(
        es,
        "customers",
        agg_primitives=["mean"],
        trans_primitives=[],
        groupby_trans_primitives=[],
        max_depth=3,
    )
    assert not any("customers.age" in n for n in names(got))


def test_direct_features_come_from_parents(es):
    got = synthesize(
        es,
        "sessions",
        agg_primitives=[],
        trans_primitives=[],
        groupby_trans_primitives=[],
        max_depth=1,
    )
    assert "customers.age" in names(got)


def test_transforms_respect_dtype_families():
    """A temporal primitive takes only the temporal column, and vice versa.

    The table deliberately carries both a numeric and a temporal non-key
    column, so the candidate set is non-empty for both primitives and the
    assertions can only pass if ``dtypes.matches`` actually discriminates.
    An earlier version of this test used the ``sessions`` fixture, where every
    column was excluded as a key: the candidate set was empty, so it stayed
    green even with dtype matching disabled entirely.
    """
    es = tusk.EntitySet("dtypes").add_dataframe(
        "events",
        pl.LazyFrame(
            {
                "id": [1, 2],
                "amount": [-1.5, 2.5],
                "occurred_at": [dt.datetime(2024, 3, 4), dt.datetime(2024, 4, 5)],
            }
        ),
        primary_key="id",
    )
    got = names(
        synthesize(
            es,
            "events",
            agg_primitives=[],
            trans_primitives=["month", "absolute"],
            groupby_trans_primitives=[],
            max_depth=1,
        )
    )
    # month requires TEMPORAL: it takes occurred_at and not amount.
    assert "MONTH(occurred_at)" in got
    assert "MONTH(amount)" not in got
    # absolute requires NUMERIC: it takes amount and not occurred_at.
    assert "ABSOLUTE(amount)" in got
    assert "ABSOLUTE(occurred_at)" not in got


def test_multi_output_feature_is_an_output_but_never_an_input(es):
    """QUANTILES(x) materializes only ``...[0]``/``[1]``/``[2]``, never the stem.

    Anything stacking on the un-indexed stem compiles to ``nw.col(stem)`` and
    fails at collect with ColumnNotFoundError, so no feature may take a
    multi-output feature as a base.
    """
    from tusk.primitives import Quantiles

    got = synthesize(
        es,
        "customers",
        agg_primitives=["mean", Quantiles(qs=(0.25, 0.5, 0.75))],
        trans_primitives=["absolute"],
        groupby_trans_primitives=[],
        max_depth=2,
    )
    n = names(got)
    # It is still generated as an output.
    assert "QUANTILES(sessions.MEAN(transactions.amount))" in n
    # But nothing consumes it.
    assert "MEAN(sessions.QUANTILES(transactions.amount))" not in n
    assert not any("(QUANTILES(" in name for name in n)

    seen = set()
    stack = list(got)
    while stack:
        feature = stack.pop()
        if feature in seen:
            continue
        seen.add(feature)
        for base in feature.base_features:
            assert not base.is_multi_output, f"{feature.name} stacks on {base.name}"
            stack.append(base)


def test_transform_stacks_on_aggregation(es):
    got = synthesize(
        es,
        "customers",
        agg_primitives=["mean"],
        trans_primitives=["absolute"],
        groupby_trans_primitives=[],
        max_depth=2,
    )
    assert "ABSOLUTE(MEAN(sessions.MEAN(transactions.amount)))" not in names(got)
    assert "MEAN(sessions.MEAN(transactions.amount))" in names(got)


def test_groupby_transform_features(es):
    got = synthesize(
        es,
        "transactions",
        agg_primitives=[],
        trans_primitives=[],
        groupby_trans_primitives=["cum_sum"],
        max_depth=1,
    )
    assert "CUM_SUM(amount) by session_id" in names(got)


def test_stack_on_self_is_respected():
    # count is zero-arity and short-circuits in _aggregations before
    # _combinations is ever reached, so it cannot exercise the stack_on_self
    # filter. n_unique is non-zero-arity (input_dtypes=(F.ANY,)) with
    # stack_on_self=False, so it actually routes through _combinations.
    es = (
        tusk.EntitySet("nu")
        .add_dataframe("a", pl.LazyFrame({"id": [1]}), primary_key="id")
        .add_dataframe(
            "b", pl.LazyFrame({"id": [1], "a_id": [1], "x": [1.0]}), primary_key="id"
        )
        .add_dataframe(
            "c", pl.LazyFrame({"id": [1], "b_id": [1], "y": [1.0]}), primary_key="id"
        )
        .add_relationship(parent="a", child="b", foreign_key="a_id")
        .add_relationship(parent="b", child="c", foreign_key="b_id")
    )
    got = synthesize(
        es,
        "a",
        agg_primitives=["n_unique"],
        trans_primitives=[],
        groupby_trans_primitives=[],
        max_depth=2,
    )
    n = names(got)
    # b.x is a plain (non-derived) column, so N_UNIQUE(b.x) is unaffected.
    assert "N_UNIQUE(b.x)" in n
    # N_UNIQUE(c.y) aggregated onto b is itself an n_unique output; stacking
    # n_unique on top of it must be filtered out.
    assert not any(name.startswith("N_UNIQUE(b.N_UNIQUE") for name in n)


def test_multi_slot_combinations_dedup_commutative_and_forbid_self_pairs():
    # add_numeric/subtract_numeric are the only built-ins with two input
    # slots, so they are the only way to exercise the else branch of
    # `if len(per_slot) == 1` in _combinations: itertools.product, the
    # duplicate-feature filter, and the commutative frozenset dedup.
    es = tusk.EntitySet("arith").add_dataframe(
        "t", pl.LazyFrame({"id": [1], "a": [1.0], "b": [2.0]}), primary_key="id"
    )
    got = synthesize(
        es,
        "t",
        agg_primitives=[],
        trans_primitives=["add_numeric", "subtract_numeric"],
        groupby_trans_primitives=[],
        max_depth=1,
    )
    n = names(got)
    # commutative=True: only one argument order is generated.
    assert "ADD_NUMERIC(a, b)" in n
    assert "ADD_NUMERIC(b, a)" not in n
    # commutative=False: both argument orders are generated.
    assert "SUBTRACT_NUMERIC(a, b)" in n
    assert "SUBTRACT_NUMERIC(b, a)" in n
    # a feature is never paired with itself.
    assert "ADD_NUMERIC(a, a)" not in n
    assert "ADD_NUMERIC(b, b)" not in n
    assert "SUBTRACT_NUMERIC(a, a)" not in n
    assert "SUBTRACT_NUMERIC(b, b)" not in n


def test_self_referential_schema_terminates():
    es = (
        tusk.EntitySet("hr")
        .add_dataframe(
            "employees",
            pl.LazyFrame({"id": [1, 2], "manager_id": [None, 1], "salary": [1.0, 2.0]}),
            primary_key="id",
        )
        .add_relationship(
            parent="employees", child="employees", foreign_key="manager_id"
        )
    )
    got = synthesize(
        es,
        "employees",
        agg_primitives=["mean"],
        trans_primitives=[],
        groupby_trans_primitives=[],
        max_depth=3,
    )
    assert "MEAN(employees.salary)" in names(got)


def test_diamond_schema_terminates():
    es = (
        tusk.EntitySet("d")
        .add_dataframe("a", pl.LazyFrame({"id": [1], "v": [1.0]}), primary_key="id")
        .add_dataframe(
            "b", pl.LazyFrame({"id": [1], "a_id": [1], "v": [1.0]}), primary_key="id"
        )
        .add_dataframe(
            "c", pl.LazyFrame({"id": [1], "a_id": [1], "v": [1.0]}), primary_key="id"
        )
        .add_dataframe(
            "d",
            pl.LazyFrame({"id": [1], "b_id": [1], "c_id": [1], "v": [1.0]}),
            primary_key="id",
        )
        .add_relationship(parent="a", child="b", foreign_key="a_id")
        .add_relationship(parent="a", child="c", foreign_key="a_id")
        .add_relationship(parent="b", child="d", foreign_key="b_id")
        .add_relationship(parent="c", child="d", foreign_key="c_id")
    )
    got = synthesize(
        es,
        "a",
        agg_primitives=["mean"],
        trans_primitives=[],
        groupby_trans_primitives=[],
        max_depth=3,
    )
    assert "MEAN(b.MEAN(d.v))" in names(got)


def test_features_are_deduplicated(es):
    got = synthesize(
        es,
        "customers",
        agg_primitives=["count", "mean"],
        trans_primitives=[],
        groupby_trans_primitives=[],
        max_depth=2,
    )
    assert len(got) == len(set(got))


def test_categorical_column_skipped_by_string_primitive_warns():
    """A Categorical column skipped by a STRING primitive must say so."""
    import pyarrow  # noqa: F401  (dev dep; polars Categorical is enough here)

    from tusk.dtypes import DtypeFamily as F
    from tusk.exceptions import CategoricalDtypeWarning
    from tusk.primitives.base import TransformPrimitive
    from tusk.primitives.registry import register

    @register
    @dataclass(frozen=True)
    class Shout(TransformPrimitive):
        """Uppercase a string column."""

        name = "shout"
        input_dtypes = (F.STRING,)

        def build(self, expr):
            """Build the uppercase expression.

            Args:
                expr: The column to uppercase.

            Returns:
                A narwhals expression.
            """
            return expr.str.to_uppercase()

    es = tusk.EntitySet("x").add_dataframe(
        "t",
        pl.LazyFrame(
            {
                "id": [1, 2],
                "plain": ["a", "b"],
                "cat": pl.Series(["a", "b"], dtype=pl.Categorical),
            }
        ),
        primary_key="id",
    )
    with pytest.warns(CategoricalDtypeWarning, match="cat"):
        got = synthesize(
            es,
            "t",
            agg_primitives=[],
            trans_primitives=["shout"],
            groupby_trans_primitives=[],
            max_depth=1,
        )
    # The String column is still used; only the Categorical one is skipped.
    assert "SHOUT(plain)" in names(got)
    assert "SHOUT(cat)" not in names(got)


def test_no_categorical_warning_when_no_string_primitive_requested(recwarn):
    """A Categorical column is present, but no STRING primitive is requested.

    Using the shared ``es`` fixture here would not distinguish "no warning
    because no STRING primitive was requested" from "no warning because
    there was nothing categorical to warn about" -- it has no Categorical
    columns at all. This schema has one, so the assertion actually tests
    what the test name claims.
    """
    from tusk.exceptions import CategoricalDtypeWarning

    es = tusk.EntitySet("x").add_dataframe(
        "t",
        pl.LazyFrame(
            {
                "id": [1, 2],
                "n": [1.0, 2.0],
                "cat": pl.Series(["a", "b"], dtype=pl.Categorical),
            }
        ),
        primary_key="id",
    )
    synthesize(
        es,
        "t",
        agg_primitives=[],
        trans_primitives=["absolute"],
        groupby_trans_primitives=[],
        max_depth=1,
    )
    assert not [w for w in recwarn if issubclass(w.category, CategoricalDtypeWarning)]


def test_order_dependent_transform_without_row_creation_time_fails_in_phase_one():
    es = tusk.EntitySet("x").add_dataframe(
        "t", pl.LazyFrame({"id": [1], "v": [1.0]}), primary_key="id"
    )
    with pytest.raises(tusk.exceptions.PrimitiveError, match="row_creation_time"):
        synthesize(
            es,
            "t",
            agg_primitives=[],
            trans_primitives=["cum_sum"],
            groupby_trans_primitives=[],
            max_depth=1,
        )


def test_unknown_target_raises(es):
    with pytest.raises(tusk.exceptions.SchemaError, match="nope"):
        synthesize(
            es,
            "nope",
            agg_primitives=[],
            trans_primitives=[],
            groupby_trans_primitives=[],
            max_depth=1,
        )


def test_no_frames_are_touched(es, monkeypatch):
    def explode(_name):
        raise AssertionError("synthesis touched a frame")

    monkeypatch.setattr(es, "frame", explode)
    synthesize(
        es,
        "customers",
        agg_primitives=["count", "mean"],
        trans_primitives=["month"],
        groupby_trans_primitives=[],
        max_depth=2,
    )
