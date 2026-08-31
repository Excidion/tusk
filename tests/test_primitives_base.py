import pickle
from dataclasses import FrozenInstanceError, dataclass, is_dataclass

import narwhals as nw
import polars as pl
import pytest

import tusk.primitives  # noqa: F401  -- registers the built-in primitives
from tusk.dtypes import DtypeFamily as F
from tusk.exceptions import PrimitiveError
from tusk.primitives.base import (
    AggregationPrimitive,
    NeedsCutoffTime,
    TransformPrimitive,
)
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


@pytest.fixture
def restore_registry():
    """Undo any registration a test performs.

    test_every_registered_primitive_is_a_frozen_dataclass asserts that every
    entry in _REGISTRY complies, so a deliberately-bad primitive must not
    outlive the test that needs it.
    """
    before = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(before)


class PlainClassPrimitive(TransformPrimitive):
    """A primitive written the ordinary way: no dataclass, identity equality."""

    name = "plain_class_primitive"
    input_dtypes = (F.NUMERIC,)

    def __init__(self, k=2.0):
        self.k = k

    def build(self, expr):
        return expr * self.k


@dataclass
class UnfrozenPrimitive(TransformPrimitive):
    """A dataclass that forgot frozen=True, so it is unhashable."""

    name = "unfrozen_primitive"
    input_dtypes = (F.NUMERIC,)
    k: float = 2.0

    def build(self, expr):
        return expr * self.k


def test_a_plain_class_primitive_instance_is_rejected():
    with pytest.raises(PrimitiveError, match="PlainClassPrimitive"):
        resolve(PlainClassPrimitive(3.0))


def test_a_plain_class_primitive_is_rejected_by_registered_name(restore_registry):
    register(PlainClassPrimitive)
    with pytest.raises(PrimitiveError, match="PlainClassPrimitive"):
        resolve("plain_class_primitive")


@dataclass(frozen=True)
class CutoffAggregation(NeedsCutoffTime, AggregationPrimitive):
    """An aggregation that wrongly claims to need the cutoff time."""

    name = "cutoff_aggregation"
    input_dtypes = (F.NUMERIC,)

    def build(self, expr):
        return expr.max()


def test_needs_cutoff_time_is_rejected_on_an_aggregation_primitive(restore_registry):
    # The compiler only binds cutoff_time onto row-wise features, so an
    # aggregation combining NeedsCutoffTime would silently build against None.
    register(CutoffAggregation)
    with pytest.raises(PrimitiveError, match="CutoffAggregation"):
        resolve("cutoff_aggregation")


def test_an_unregistered_cutoff_time_aggregation_instance_is_rejected():
    # Passing an instance straight to deep_feature_synthesis never registers
    # it, and that is a supported path -- so the check cannot live in
    # register().
    with pytest.raises(PrimitiveError, match="CutoffAggregation"):
        resolve(CutoffAggregation())


def test_an_unfrozen_dataclass_primitive_is_rejected():
    # Without the check this reaches synthesis and dies with an unhelpful
    # "unhashable type" much further downstream.
    with pytest.raises(PrimitiveError, match="UnfrozenPrimitive"):
        resolve(UnfrozenPrimitive(3.0))


@dataclass(frozen=True, eq=False)
class NoEqPrimitive(TransformPrimitive):
    """A frozen dataclass that opted out of value equality."""

    name = "no_eq_primitive"
    input_dtypes = (F.NUMERIC,)
    k: float = 2.0

    def build(self, expr):
        return expr * self.k


def test_a_frozen_dataclass_with_eq_false_is_rejected():
    # frozen=True alone is not enough: eq=False keeps object.__eq__ /
    # object.__hash__, i.e. identity semantics, which is exactly what this
    # guard exists to reject.
    with pytest.raises(PrimitiveError, match="NoEqPrimitive"):
        resolve(NoEqPrimitive(3.0))


def test_the_rejection_names_the_fix():
    with pytest.raises(PrimitiveError, match=r"@dataclass\(frozen=True\)"):
        resolve(PlainClassPrimitive(3.0))


def test_a_compliant_primitive_still_resolves_unchanged():
    instance = Pair(scale=3.0)
    assert resolve(instance) is instance
    assert resolve("doubled") == Doubled()


def test_the_rejection_reaches_a_user_through_the_public_entry_point(db):
    with pytest.raises(PrimitiveError, match="PlainClassPrimitive"):
        tusk.deep_feature_synthesis(
            database=db,
            target_table="customers",
            agg_primitives=[],
            trans_primitives=[PlainClassPrimitive(3.0)],
            features_only=True,
        )
