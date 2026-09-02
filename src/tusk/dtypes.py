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

    ``TEMPORAL``, ``HAS_DATE``, ``HAS_TIME``, and ``DURATION`` exist as four
    separate families for the same kind of reason: ``TEMPORAL`` matches
    everything the narrow families do, plus every dtype that is temporal at
    all, and is kept broad for ``dtype_selector``. ``HAS_DATE`` and
    ``HAS_TIME`` are named for the operation a primitive needs rather than
    for a single dtype, because no single narwhals dtype carries exactly what
    a calendar primitive like ``year`` or a time-of-day primitive like
    ``hour`` requires: ``HAS_DATE`` matches ``Datetime`` and ``Date``, the
    dtypes a calendar position can be read from; ``HAS_TIME`` matches
    ``Datetime`` and ``Time``, the dtypes an hour or minute can be read from.
    They deliberately overlap on ``Datetime`` rather than partition the
    temporal dtypes, since a ``Datetime`` genuinely supports both operations.
    ``DURATION`` is elapsed time, which neither operation applies to even
    though it is temporal.
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
