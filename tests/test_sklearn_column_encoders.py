import datetime as dt

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pytest

from tusk.exceptions import EncoderError
from tusk.sklearn import (
    DateEncoder,
    DatetimeEncoder,
    DurationEncoder,
    EnumEncoder,
    StringEncoder,
    TimeEncoder,
)

ENUM = pl.DataFrame({"e": pl.Series(["b", "a", None], dtype=pl.Enum(["b", "a"]))})
TEMPORAL = pl.DataFrame(
    {
        "d": [dt.date(2024, 3, 5), None],
        "t": [dt.time(9, 30), None],
        "w": [dt.datetime(2024, 3, 5, 9, 30), None],
        "u": [dt.timedelta(minutes=90), None],
    }
)


def test_enum_codes_follow_the_category_order():
    encoder = EnumEncoder().fit(ENUM)
    np.testing.assert_array_equal(encoder.transform(ENUM).ravel(), [0.0, 1.0, np.nan])
    assert list(encoder.get_feature_names_out()) == ["e_code"]


def test_a_pandas_ordered_categorical_gets_the_same_codes():
    ordered = pd.Categorical(["b", "a", None], categories=["b", "a"], ordered=True)
    codes = EnumEncoder().fit_transform(pd.DataFrame({"e": ordered}))
    np.testing.assert_array_equal(codes.ravel(), [0.0, 1.0, np.nan])


def test_enum_codes_are_computed_on_duckdb():
    connection = duckdb.connect()
    connection.execute("create type mood as enum ('b', 'a')")
    relation = connection.sql(
        "select * from (values ('b'::mood), ('a'::mood), (null::mood)) t(e)"
    )
    codes = EnumEncoder().fit_transform(relation)
    np.testing.assert_array_equal(codes.ravel(), [0.0, 1.0, np.nan])


def test_the_enum_encoder_rejects_another_dtype():
    with pytest.raises(EncoderError, match="EnumEncoder encodes Enum columns"):
        EnumEncoder().fit(pl.DataFrame({"s": ["a"]}))


@pytest.mark.parametrize(
    ("encoder", "column", "expected", "names"),
    [
        (DateEncoder(), "d", [3, 5], ["d_month", "d_day"]),
        (TimeEncoder(), "t", [9, 30], ["t_hour", "t_minute"]),
        (
            DatetimeEncoder(),
            "w",
            [3, 5, 9, 30],
            ["w_month", "w_day", "w_hour", "w_minute"],
        ),
        (DurationEncoder(), "u", [5400], ["u_total_seconds"]),
    ],
)
def test_the_default_components(encoder, column, expected, names):
    out = encoder.fit_transform(TEMPORAL.select(column))
    assert out[0].tolist() == expected
    assert np.isnan(out[1]).all()
    assert list(encoder.get_feature_names_out()) == names


def test_components_can_be_chosen():
    encoder = DateEncoder(components=["year", "weekday", "ordinal_day"])
    assert encoder.fit_transform(TEMPORAL.select("d"))[0].tolist() == [2024, 2, 65]


def test_an_unrecognized_component_lists_the_allowed_ones():
    with pytest.raises(ValueError, match="choose from"):
        DateEncoder(components=["hour"]).fit(TEMPORAL.select("d"))


def test_a_string_components_is_rejected():
    with pytest.raises(TypeError, match=r"a list, such as \['year'\]"):
        DateEncoder(components="month").fit(TEMPORAL.select("d"))


def test_a_temporal_encoder_rejects_another_dtype():
    with pytest.raises(EncoderError, match="DateEncoder encodes Date columns"):
        DateEncoder().fit(TEMPORAL.select("w"))


CITIES = pl.DataFrame(
    {"c": ["Zurich", "Zürich", "Zurich HB", "London", "Londres", None] * 3}
)


def test_the_string_encoder_output_has_n_components_columns():
    encoder = StringEncoder(n_components=4).fit(CITIES)
    assert encoder.transform(CITIES).shape == (18, 4)
    assert list(encoder.get_feature_names_out()) == [f"c_svd_{i}" for i in range(4)]


def test_shared_n_grams_are_closer_than_none():
    out = StringEncoder(n_components=4).fit_transform(CITIES)
    assert np.linalg.norm(out[0] - out[2]) < np.linalg.norm(out[0] - out[3])


def test_a_value_first_seen_at_transform_is_encoded():
    encoder = StringEncoder(n_components=4).fit(CITIES)
    unseen = encoder.transform(pl.DataFrame({"c": ["Zurichberg"]}))
    assert unseen.shape == (1, 4)
    assert np.abs(unseen).sum() > 0


@pytest.mark.parametrize(
    "values",
    [["ab", "cd"], ["a", "a"], [None, ""]],
    ids=["few rows", "one n-gram", "no n-grams"],
)
def test_a_small_vocabulary_is_padded_to_n_components(values):
    table = pl.DataFrame({"c": values}, schema={"c": pl.String})
    assert StringEncoder().fit_transform(table).shape == (2, 30)


def test_no_n_grams_encode_as_zeros():
    table = pl.DataFrame({"c": [None, ""]}, schema={"c": pl.String})
    assert not StringEncoder().fit_transform(table).any()


def test_the_string_encoder_rejects_another_dtype():
    with pytest.raises(EncoderError, match="StringEncoder encodes String columns"):
        StringEncoder().fit(pl.DataFrame({"n": [1]}))
