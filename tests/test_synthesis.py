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


def test_transforms_respect_dtype_families(es):
    got = synthesize(
        es,
        "sessions",
        agg_primitives=[],
        trans_primitives=["month", "absolute"],
        groupby_trans_primitives=[],
        max_depth=1,
    )
    # started_at is the row_creation_time, a key column, so it is not an input;
    # there is no other temporal or numeric column on sessions.
    assert not any(n.startswith("MONTH") for n in names(got))


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


def test_stack_on_self_is_respected(es):
    got = synthesize(
        es,
        "customers",
        agg_primitives=["count"],
        trans_primitives=[],
        groupby_trans_primitives=[],
        max_depth=2,
    )
    assert "COUNT(sessions)" in names(got)
    assert not any(n.startswith("COUNT(sessions.COUNT") for n in names(got))


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


def test_no_categorical_warning_when_no_string_primitive_requested(es, recwarn):
    """Default primitives require no STRING input, so nothing is skipped."""
    from tusk.exceptions import CategoricalDtypeWarning

    synthesize(
        es,
        "customers",
        agg_primitives=["count"],
        trans_primitives=[],
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
