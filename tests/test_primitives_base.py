import pickle
from dataclasses import FrozenInstanceError, dataclass, is_dataclass

import narwhals as nw
import polars as pl
import pytest

import tusk.primitives  # noqa: F401  -- registers the built-in primitives
from tusk.dtypes import DtypeFamily as F
from tusk.exceptions import PrimitiveError
from tusk.primitives.base import AggregationPrimitive, TransformPrimitive
from tusk.primitives.registry import _REGISTRY, register, resolve, resolve_all


@register
@dataclass(frozen=True)
class Doubled(TransformPrimitive):
    name = "doubled"
    input_dtypes = (F.NUMERIC,)

    def build(self, expr):
        return expr * 2


@register
@dataclass(frozen=True)
class Spread(AggregationPrimitive):
    name = "spread"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64

    def build(self, expr):
        return expr.max() - expr.min()


@dataclass(frozen=True)
class Pair(AggregationPrimitive):
    name = "pair"
    input_dtypes = (F.NUMERIC,)
    output_dtype = nw.Float64
    scale: float = 1.0

    @property
    def number_of_outputs(self):
        return 2

    def build(self, expr):
        return [expr.min() * self.scale, expr.max() * self.scale]


def test_name_generation():
    assert Doubled().generate_name(("amount",)) == "DOUBLED__amount"


def test_single_output_names():
    assert Spread().output_names("SPREAD__amount") == ("SPREAD__amount",)


def test_multi_output_names_are_indexed():
    assert Pair().output_names("PAIR__amount") == ("PAIR__amount__0", "PAIR__amount__1")


def test_outputs_is_always_a_tuple():
    assert len(Spread().outputs(nw.col("a"))) == 1
    assert len(Pair().outputs(nw.col("a"))) == 2


def test_return_dtype_preserves_input_by_default():
    assert Doubled().return_dtype((nw.Int64(),)) == nw.Int64()
    assert Spread().return_dtype((nw.Int64(),)) == nw.Float64


def test_parameters_are_dataclass_fields_and_affect_equality():
    assert Pair(scale=2.0) != Pair(scale=1.0)
    assert Pair(scale=2.0) == Pair(scale=2.0)


def test_registry_resolves_names_and_passes_instances_through():
    assert resolve("doubled") == Doubled()
    instance = Pair(scale=3.0)
    assert resolve(instance) is instance
    assert resolve_all(["doubled", "spread"]) == (Doubled(), Spread())


def test_unknown_primitive_lists_alternatives():
    with pytest.raises(PrimitiveError, match="doubled"):
        resolve("dubbled")


def test_expressions_actually_evaluate():
    lf = nw.from_native(pl.LazyFrame({"g": [1, 1], "a": [2.0, 5.0]}))
    got = lf.group_by("g").agg(Spread().outputs(nw.col("a"))[0].alias("s")).collect()
    assert got.to_native()["s"][0] == 3.0


def test_every_registered_primitive_is_a_frozen_dataclass():
    for name in _REGISTRY:
        primitive = resolve(name)
        assert is_dataclass(primitive), f"{name} is not a dataclass"
        with pytest.raises(FrozenInstanceError):
            setattr(primitive, "name", "mutated")  # noqa: B010


def test_every_registered_primitive_round_trips_through_pickle():
    # Primitives built by a metaprogramming helper rather than a class body
    # land in the wrong module and silently break any process-parallel or
    # cached use. Instantiating via the registry keeps this honest.
    for name in _REGISTRY:
        primitive = resolve(name)
        assert pickle.loads(pickle.dumps(primitive)) == primitive
