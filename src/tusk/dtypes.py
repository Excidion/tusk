"""Dtype families used to decide which primitives apply to which columns.

tusk matches columns to families using narwhals dtypes alone. It has no
logical types and no semantic tags. This costs the distinction between an
integer that is a quantity and one that is a category. It gains identical
behaviour on every backend with no extra dependencies.
"""

from __future__ import annotations

from enum import Enum

import narwhals as nw


class DtypeFamily(Enum):
    """A group of narwhals dtypes that a primitive can accept.

    ``STRING`` and ``CATEGORICAL`` are disjoint. A ``String`` column is not
    a ``Categorical`` or ``Enum`` one, even though all three hold
    text-like values. :class:`~tusk.exceptions.CategoricalDtypeWarning`
    reports this distinction.

    ``TEMPORAL`` matches every dtype the narrower temporal families match,
    plus every dtype that is temporal. It stays broad for
    ``dtype_selector``. ``HAS_DATE`` matches ``Datetime`` and ``Date``,
    the dtypes a calendar position can be read from. ``HAS_TIME`` matches
    ``Datetime`` and ``Time``, the dtypes an hour or minute can be read
    from. Both families match ``Datetime``. ``DURATION`` is elapsed time.
    Neither ``HAS_DATE`` nor ``HAS_TIME`` matches it, even though it is
    temporal.
    """

    NUMERIC = "numeric"
    TEMPORAL = "temporal"
    HAS_DATE = "has_date"
    HAS_TIME = "has_time"
    DURATION = "duration"
    STRING = "string"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    ANY = "any"


def matches(dtype: nw.dtypes.DType, family: DtypeFamily) -> bool:
    """Report whether a dtype belongs to a family.

    Args:
        dtype: The narwhals dtype to test.
        family: The family to test membership of.

    Returns:
        True if the dtype belongs to the family.

    Raises:
        ValueError: If family is not a recognized DtypeFamily member.
    """
    if family is DtypeFamily.ANY:
        return True
    if family is DtypeFamily.NUMERIC:
        return bool(dtype.is_numeric())
    if family is DtypeFamily.TEMPORAL:
        return bool(dtype.is_temporal())
    if family is DtypeFamily.HAS_DATE:
        return dtype == nw.Datetime or dtype == nw.Date
    if family is DtypeFamily.HAS_TIME:
        return dtype == nw.Datetime or dtype == nw.Time
    if family is DtypeFamily.DURATION:
        return dtype == nw.Duration
    if family is DtypeFamily.STRING:
        return dtype == nw.String
    if family is DtypeFamily.CATEGORICAL:
        return dtype == nw.Categorical or dtype == nw.Enum
    if family is DtypeFamily.BOOLEAN:
        return dtype == nw.Boolean
    raise ValueError(f"Unrecognized DtypeFamily: {family}")
