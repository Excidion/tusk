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

__all__ = [
    "AGG_DEFAULTS",
    "AggregationPrimitive",
    "Count",
    "Max",
    "Mean",
    "Median",
    "Min",
    "NUnique",
    "PercentTrue",
    "Primitive",
    "Quantiles",
    "Std",
    "Sum",
    "TransformPrimitive",
    "register",
    "resolve",
    "resolve_all",
    "transform_primitive",
]
