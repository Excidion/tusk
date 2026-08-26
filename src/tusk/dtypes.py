"""Dtype families used to decide which primitives apply to which columns.

Matching is done on narwhals dtypes alone. There are no logical types and no
semantic tags: the cost is losing the distinction between an integer that is a
quantity and one that is a category; the benefit is identical behaviour on
every backend with no extra dependencies.
"""

from __future__ import annotations

from enum import Enum

import narwhals as nw


class DtypeFamily(Enum):
    """A group of narwhals dtypes that a primitive can accept.

    ``STRING`` and ``CATEGORICAL`` are disjoint: a ``String`` column is not a
    ``Categorical`` or ``Enum`` one, even though all three hold text-like
    values. That distinction is deliberate -- it is what
    :class:`~tusk.exceptions.CategoricalDtypeWarning` reports -- and it is why
    both families exist rather than one wider one.
    """

    NUMERIC = "numeric"
    TEMPORAL = "temporal"
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
    if family is DtypeFamily.STRING:
        return dtype == nw.String
    if family is DtypeFamily.CATEGORICAL:
        return dtype == nw.Categorical or dtype == nw.Enum
    if family is DtypeFamily.BOOLEAN:
        return dtype == nw.Boolean
    raise ValueError(f"Unrecognized DtypeFamily: {family}")
