"""Primitives: the expression builders DFS composes into features."""

from __future__ import annotations

# ruff: noqa: I001 (file-wide) - imports must be ordered: base, registry, aggregation.
# Importing aggregation before base/registry causes circular import and prevents
# @register decorators from executing after registry initialization.
from tusk.primitives.base import AggregationPrimitive, Primitive, TransformPrimitive
from tusk.primitives.registry import register, resolve, resolve_all, transform_primitive
from tusk.primitives.aggregation import (
    AGG_DEFAULTS,
    Count,
    Max,
    Mean,
    Median,
    Min,
    NUnique,
    PercentTrue,
    Quantiles,
    Std,
    Sum,
)
from tusk.primitives.transform import TRANS_DEFAULTS, AddNumeric, MultiplyNumeric

__all__ = [
    "AGG_DEFAULTS",
    "AddNumeric",
    "AggregationPrimitive",
    "Count",
    "Max",
    "Mean",
    "Median",
    "Min",
    "MultiplyNumeric",
    "NUnique",
    "PercentTrue",
    "Primitive",
    "Quantiles",
    "Std",
    "Sum",
    "TRANS_DEFAULTS",
    "TransformPrimitive",
    "register",
    "resolve",
    "resolve_all",
    "transform_primitive",
]
