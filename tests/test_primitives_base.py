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


def test_a_flat_input_dtypes_is_one_signature():
    """The 25 primitives written before alternatives existed keep working."""
    assert Doubled().signatures == ((F.NUMERIC,),)


def test_a_tuple_of_tuples_is_read_as_alternatives():
    @dataclass(frozen=True)
    class Comparable(TransformPrimitive):
        name = "comparable"
        input_dtypes = ((F.NUMERIC, F.NUMERIC), (F.HAS_DATE, F.HAS_DATE))

        def build(self, left, right):
            return left > right

    assert Comparable().signatures == (
        (F.NUMERIC, F.NUMERIC),
        (F.HAS_DATE, F.HAS_DATE),
    )


def test_no_input_dtypes_is_no_signatures():
    """A zero-arity aggregation such as COUNT declares nothing."""

    @dataclass(frozen=True)
    class Tally(AggregationPrimitive):
        name = "tally"

        def build(self):
            return nw.len()

    assert Tally().signatures == ()


def test_signatures_of_differing_arity_are_rejected():
    """build() has a fixed parameter list, so a mixed arity cannot run."""

    @dataclass(frozen=True)
    class Ragged(TransformPrimitive):
        name = "ragged"
        input_dtypes = ((F.NUMERIC,), (F.NUMERIC, F.NUMERIC))

        def build(self, expr):
            return expr

    with pytest.raises(PrimitiveError, match="same number of inputs"):
        _ = Ragged().signatures


def test_a_family_stranded_beside_a_nested_alternative_is_rejected():
    """A stray family must not be read as a one-family flat shape.

    ``input_dtypes[0]`` being a ``DtypeFamily`` used to be enough to call the
    whole declaration flat, so a tuple sitting in a later slot escaped
    validation and only failed much later inside dtype matching.
    """

    @dataclass(frozen=True)
    class Mistyped(TransformPrimitive):
        name = "mistyped"
        input_dtypes = (F.NUMERIC, (F.NUMERIC, F.NUMERIC))

        def build(self, left, right):
            return left

    with pytest.raises(PrimitiveError, match="mistyped"):
        _ = Mistyped().signatures


def test_a_bare_family_among_alternatives_is_rejected():
    """The mirror-image typo must not reach ``len()`` on a ``DtypeFamily``."""

    @dataclass(frozen=True)
    class AlsoMistyped(TransformPrimitive):
        name = "also_mistyped"
        input_dtypes = ((F.NUMERIC,), F.NUMERIC)

        def build(self, left, right):
            return left

    with pytest.raises(PrimitiveError, match="also_mistyped"):
        _ = AlsoMistyped().signatures


def test_an_empty_alternative_is_rejected():
    """``((),)`` is not ``()`` and must not silently take its no-input path."""

    @dataclass(frozen=True)
    class EmptyAlternative(TransformPrimitive):
        name = "empty_alternative"
        input_dtypes = ((),)

        def build(self):
            return nw.lit(1)

    with pytest.raises(PrimitiveError, match="empty_alternative"):
        _ = EmptyAlternative().signatures
