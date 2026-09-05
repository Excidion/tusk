# Binary Transform Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every binary-transform row in `docs/guide/primitive-coverage.md` that is not marked ⛔, by adding alternative input signatures to the primitive protocol and ten new primitives on top of it.

**Architecture:** `Primitive.input_dtypes` may now hold either a tuple of dtype families (one signature, as today) or a tuple of such tuples (alternatives). A read-only `signatures` property normalizes the two shapes, and `synthesis._combinations` unions the combinations each signature produces. This lets a comparison primitive accept a numeric pair or a datetime pair without ever pairing a number against a string, which is what `itertools.product` over independent slots would do.

**Tech Stack:** Python 3.10+, narwhals (expression layer), polars (default backend), duckdb (SQL backend, portability tests), featuretools 1.31.0 (differential tests only), pytest.

**Spec:** `agents/superpowers/specs/2026-09-05-binary-transform-parity-design.md`

## Global Constraints

- Plans and specs live under `agents/superpowers/`, never `docs/` — zensical builds every markdown file under `docs/` into the published site (`agents/AGENTS.md`).
- Primitives build narwhals expressions and never see a value (`src/tusk/primitives/base.py:1-8`). No `.to_pandas()`, no materialization, no Python-level branching on data.
- Every public function, method and module needs a docstring with `Args:`/`Returns:`/`Raises:` sections — `pydoclint` and `interrogate` run in pre-commit and will reject the commit otherwise.
- Comments explain **why**, never **what** (`agents/style/CODESTYLE.md`). Commented-out code is never allowed.
- Callers sit above callees in a module; detail increases downward.
- Run `just lint` before each commit; it runs ruff, ty, interrogate and pydoclint over the tracked files.
- The default suite is `just test`. The differential suite is `just test-differential` and needs `--group validation`.
- Commit message trailers, on every commit:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0153Kdi2FVgeNUJMiBPfpLpq
  ```
- Work happens on branch `feature/binary-transform-parity`. Do not merge to `main`.

---

### Task 1: Alternative signatures on `Primitive`

**Files:**
- Modify: `src/tusk/primitives/base.py:20-60`
- Test: `tests/test_primitives_base.py`

**Interfaces:**
- Consumes: `tusk.dtypes.DtypeFamily`, `tusk.exceptions.PrimitiveError`.
- Produces: `Primitive.signatures -> tuple[tuple[DtypeFamily, ...], ...]`, a read-only property. Every later task reads signatures through it and never inspects `input_dtypes` directly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_primitives_base.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_primitives_base.py -k signature -v`
Expected: FAIL with `AttributeError: 'Doubled' object has no attribute 'signatures'`

- [ ] **Step 3: Implement the property**

In `src/tusk/primitives/base.py`, widen the annotation on `input_dtypes` and add the property directly beneath `number_of_outputs`:

```python
    input_dtypes: ClassVar[
        tuple[DtypeFamily, ...] | tuple[tuple[DtypeFamily, ...], ...]
    ] = ()
```

```python
    @property
    def signatures(self) -> tuple[tuple[DtypeFamily, ...], ...]:
        """Every input shape this primitive accepts.

        Returns:
            One tuple of dtype families per accepted shape. A primitive that
            takes no column input has none.

        Raises:
            PrimitiveError: If the shapes do not all take the same number of
                inputs.
        """
        if not self.input_dtypes:
            return ()
        if isinstance(self.input_dtypes[0], DtypeFamily):
            return (self.input_dtypes,)
        signatures = tuple(self.input_dtypes)
        if len({len(signature) for signature in signatures}) > 1:
            raise PrimitiveError(
                f"primitive {self.name!r} declares input shapes taking a "
                f"different number of inputs; every shape must take the same "
                f"number, because build() has one parameter list.",
            )
        return signatures
```

Update the `input_dtypes` line of the class docstring's `Attributes:` block to:

```
        input_dtypes: One dtype family per input, or several such tuples if
            the primitive accepts alternative input shapes. Empty means the
            primitive takes no column input, e.g. ``count``. Read it through
            :attr:`signatures`.
```

Add the import: `from tusk.exceptions import PrimitiveError`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_primitives_base.py -v`
Expected: PASS, including the pre-existing tests in the file.

- [ ] **Step 5: Check for an import cycle**

Run: `uv run python -c "import tusk; print(tusk.__version__)"`
Expected: prints the version. `exceptions.py` imports nothing from `primitives`, so this is a one-way edge, but confirm it rather than assume.

- [ ] **Step 6: Commit**

```bash
git add src/tusk/primitives/base.py tests/test_primitives_base.py
git commit -m "$(cat <<'EOF'
feat: let a primitive declare alternative input shapes

A comparison primitive accepts a numeric pair or a datetime pair, but never
a number against a string. Independent per-slot matching cannot say that.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0153Kdi2FVgeNUJMiBPfpLpq
EOF
)"
```

---

### Task 2: Synthesis unions the signatures

**Files:**
- Modify: `src/tusk/synthesis.py:199`, `src/tusk/synthesis.py:355-375`, `src/tusk/synthesis.py:425-480`
- Test: `tests/test_synthesis.py`

**Interfaces:**
- Consumes: `Primitive.signatures` from Task 1.
- Produces: `_Context._combinations(primitive, candidates, table)` keeps its signature and return type (`list[tuple[Feature, ...]]`); a new private `_Context._combinations_for_signature(primitive, candidates, signature)` holds the old body. Tasks 3-5 depend on the behavior, not on these names.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_synthesis.py`:

```python
@dataclass(frozen=True)
class Comparable(TransformPrimitive):
    """A two-slot primitive accepting a numeric pair or a datetime pair."""

    name = "comparable"
    input_dtypes = ((F.NUMERIC, F.NUMERIC), (F.HAS_DATE, F.HAS_DATE))
    output_dtype = nw.Boolean

    def build(self, left, right):
        return left > right


def test_each_signature_contributes_its_own_combinations():
    db = tusk.Database("mixed").add_table(
        "events",
        pl.LazyFrame(
            {
                "id": [1, 2],
                "amount": [1.0, 2.0],
                "quantity": [3, 4],
                "started_at": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
                "ended_at": [dt.datetime(2024, 1, 3), dt.datetime(2024, 1, 4)],
            },
        ),
        primary_key="id",
    )
    names = {
        f.name
        for f in synthesize(
            database=db,
            target_table="events",
            agg_primitives=[],
            trans_primitives=[Comparable()],
            max_depth=1,
        )
    }
    assert "COMPARABLE__amount__quantity" in names
    assert "COMPARABLE__started_at__ended_at" in names


def test_a_signature_never_pairs_across_its_slots():
    """The whole point: amount > started_at is not a feature anyone can run."""
    db = tusk.Database("mixed").add_table(
        "events",
        pl.LazyFrame(
            {
                "id": [1, 2],
                "amount": [1.0, 2.0],
                "started_at": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
            },
        ),
        primary_key="id",
    )
    names = {
        f.name
        for f in synthesize(
            database=db,
            target_table="events",
            agg_primitives=[],
            trans_primitives=[Comparable()],
            max_depth=1,
        )
    }
    assert not any("amount__started_at" in name for name in names)
    assert not any("started_at__amount" in name for name in names)


def test_overlapping_signatures_generate_one_feature():
    """A Datetime column matches HAS_DATE and TEMPORAL both."""

    @dataclass(frozen=True)
    class Twice(TransformPrimitive):
        name = "twice"
        input_dtypes = ((F.HAS_DATE,), (F.TEMPORAL,))
        output_dtype = nw.Int32

        def build(self, expr):
            return expr.dt.year()

    db = tusk.Database("times").add_table(
        "events",
        pl.LazyFrame(
            {"id": [1], "started_at": [dt.datetime(2024, 1, 1)]},
        ),
        primary_key="id",
    )
    names = [
        f.name
        for f in synthesize(
            database=db,
            target_table="events",
            agg_primitives=[],
            trans_primitives=[Twice()],
            max_depth=1,
        )
    ]
    assert names.count("TWICE__started_at") == 1
```

`tests/test_synthesis.py` already imports `dt`, `dataclass`, `pl`, `pytest`, `tusk` and `synthesize` at module level, but imports `DtypeFamily as F` *inside* the two functions that need it (lines 259 and 504). The test primitives above are module-level, so add two module-level imports to the file:

```python
import narwhals as nw
from tusk.dtypes import DtypeFamily as F
from tusk.primitives.base import TransformPrimitive
```

Leave the two existing function-local `F` imports alone; removing them is unrelated cleanup.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_synthesis.py -k "signature or combinations or pairs or overlapping" -v`
Expected: FAIL. `_combinations` reads `primitive.input_dtypes` directly, so a tuple-of-tuples is passed to `matches()` as if it were a family, raising `ValueError: Unrecognized DtypeFamily`.

- [ ] **Step 3: Move the three read sites onto `signatures`**

At `src/tusk/synthesis.py:199`, in `_aggregations`:

```python
                if not primitive.signatures:
```

In `_warn_categorical` (`src/tusk/synthesis.py:355-375`), replace the early return:

```python
        if not any(
            DtypeFamily.STRING in signature for signature in primitive.signatures
        ):
            return
```

- [ ] **Step 4: Split `_combinations`**

Rename the existing method to `_combinations_for_signature`, and give it the signature as a parameter instead of reading `primitive.input_dtypes`. Delete its `_warn_categorical` call and its matched/unmatched bookkeeping — both move up to the new caller, which is the only place that sees the whole result.

```python
    def _combinations(
        self,
        primitive: Primitive,
        candidates: Sequence[Feature],
        table: str,
    ) -> list[tuple[Feature, ...]]:
        """Enumerate input tuples a primitive accepts, across all its shapes.

        Args:
            primitive: The primitive to match inputs for.
            candidates: Available features.
            table: Table the candidates live on, recorded so an unmatched
                primitive can be named alongside it once the walk finishes.

        Returns:
            One tuple per valid input combination.
        """
        # A multi-output feature materializes only its indexed columns
        # (``QUANTILES(x)[0]`` ...), never the bare stem, so nothing can read
        # it as an input. It stays a valid output; it is just not stackable.
        candidates = [f for f in candidates if not f.is_multi_output]
        self._warn_categorical(primitive, candidates)

        # Dtype families overlap -- a Datetime column matches both HAS_DATE
        # and TEMPORAL -- so two shapes can yield the same combination.
        combos = list(
            dict.fromkeys(
                combo
                for signature in primitive.signatures
                for combo in self._combinations_for_signature(
                    primitive,
                    candidates,
                    signature,
                )
            ),
        )

        # Only a primitive that actually produced a feature here counts as
        # matched: dtype-compatible slots are not enough on their own (e.g. a
        # commutative pair primitive with exactly one eligible column has a
        # non-empty slot but zero valid combos). Recording every table with
        # zero combos as unmatched -- not just the empty-slot case -- is what
        # lets warn_unmatched catch a primitive that never produced a single
        # feature anywhere.
        if combos:
            self._matched.add(primitive.name)
        else:
            self._unmatched.setdefault((primitive.name, table), None)
        return combos

    def _combinations_for_signature(
        self,
        primitive: Primitive,
        candidates: Sequence[Feature],
        signature: tuple[DtypeFamily, ...],
    ) -> list[tuple[Feature, ...]]:
        """Enumerate input tuples matching one of a primitive's input shapes.

        Args:
            primitive: The primitive to match inputs for.
            candidates: Available features, already filtered of multi-output
                ones.
            signature: One dtype family per input slot.

        Returns:
            One tuple per valid input combination for this shape.
        """
        per_slot = [
            [f for f in candidates if matches(f.dtype, family)] for family in signature
        ]

        combos: list[tuple[Feature, ...]]
        if len(per_slot) == 1:
            combos = [(f,) for f in per_slot[0]]
        else:
            combos = [c for c in itertools.product(*per_slot) if len(set(c)) == len(c)]
            if primitive.commutative:
                seen: set[frozenset[Feature]] = set()
                deduped = []
                for combo in combos:
                    key = frozenset(combo)
                    if key not in seen:
                        seen.add(key)
                        deduped.append(combo)
                combos = deduped

        if not primitive.stack_on_self:
            combos = [c for c in combos if not any(_uses(f, primitive) for f in c)]

        return combos
```

`_combinations` stays above `_combinations_for_signature`: caller above callee.

- [ ] **Step 5: Run the new tests**

Run: `uv run pytest tests/test_synthesis.py -v`
Expected: PASS, all of them — the pre-existing tests are the regression check that flat `input_dtypes` still behaves identically.

- [ ] **Step 6: Run the whole default suite**

Run: `just test`
Expected: PASS. This is the real gate on Task 2 — every existing primitive now routes through `signatures`.

- [ ] **Step 7: Commit**

```bash
git add src/tusk/synthesis.py tests/test_synthesis.py
git commit -m "$(cat <<'EOF'
feat: synthesize a feature for each of a primitive's input shapes

Combinations are enumerated per shape and unioned, so overlapping dtype
families cannot emit the same feature twice.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0153Kdi2FVgeNUJMiBPfpLpq
EOF
)"
```

---

### Task 3: The six comparison primitives

**Files:**
- Modify: `src/tusk/primitives/transform.py` (append after `MultiplyNumeric`, before `Not`)
- Modify: `src/tusk/primitives/__init__.py` (imports and `__all__`)
- Test: `tests/test_primitives_transform.py`

**Interfaces:**
- Consumes: `Primitive.signatures` (Task 1), signature-aware synthesis (Task 2).
- Produces: registered names `greater_than`, `greater_than_equal_to`, `less_than`, `less_than_equal_to`, `equal`, `not_equal`; classes `GreaterThan`, `GreaterThanEqualTo`, `LessThan`, `LessThanEqualTo`, `Equal`, `NotEqual` exported from `tusk.primitives`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_primitives_transform.py`:

```python
@pytest.fixture
def comparable():
    return nw.from_native(
        pl.LazyFrame(
            {
                "left": [1.0, 2.0, 3.0, None],
                "right": [3.0, 2.0, 1.0, 1.0],
                "earlier": [
                    dt.datetime(2024, 1, 1),
                    dt.datetime(2024, 1, 2),
                    dt.datetime(2024, 1, 3),
                    None,
                ],
                "later": [
                    dt.datetime(2024, 1, 2),
                    dt.datetime(2024, 1, 2),
                    dt.datetime(2024, 1, 1),
                    dt.datetime(2024, 1, 1),
                ],
            },
        ),
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("greater_than", [False, False, True, None]),
        ("greater_than_equal_to", [False, True, True, None]),
        ("less_than", [True, False, False, None]),
        ("less_than_equal_to", [True, True, False, None]),
        ("equal", [False, True, False, None]),
        ("not_equal", [True, False, True, None]),
    ],
)
def test_comparisons_compare_numbers(comparable, name, expected):
    assert _apply(comparable, name, "left", "right") == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("greater_than", [False, False, True, None]),
        ("less_than", [True, False, False, None]),
        ("equal", [False, True, False, None]),
    ],
)
def test_comparisons_compare_datetimes(comparable, name, expected):
    assert _apply(comparable, name, "earlier", "later") == expected


def test_comparisons_take_numeric_or_datetime_pairs():
    for name in ("greater_than", "greater_than_equal_to", "less_than",
                 "less_than_equal_to"):
        assert resolve(name).signatures == (
            (DtypeFamily.NUMERIC, DtypeFamily.NUMERIC),
            (DtypeFamily.HAS_DATE, DtypeFamily.HAS_DATE),
        )
        assert resolve(name).output_dtype == nw.Boolean
        assert resolve(name).commutative is False


def test_equality_also_takes_booleans_and_strings():
    """Equality is meaningful for every type that compares on both backends."""
    for name in ("equal", "not_equal"):
        assert resolve(name).signatures == (
            (DtypeFamily.NUMERIC, DtypeFamily.NUMERIC),
            (DtypeFamily.HAS_DATE, DtypeFamily.HAS_DATE),
            (DtypeFamily.BOOLEAN, DtypeFamily.BOOLEAN),
            (DtypeFamily.STRING, DtypeFamily.STRING),
        )
        assert resolve(name).commutative is True


def test_comparing_a_number_with_a_string_is_never_synthesized():
    db = tusk.Database("mixed").add_table(
        "events",
        pl.LazyFrame({"id": [1], "amount": [1.0], "label": ["x"]}),
        primary_key="id",
    )
    names = {
        f.name
        for f in synthesize(
            database=db,
            target_table="events",
            agg_primitives=[],
            trans_primitives=["greater_than"],
            max_depth=1,
        )
    }
    assert not any("label" in name for name in names)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_primitives_transform.py -k "comparison or comparable or equality or comparing" -v`
Expected: FAIL with `PrimitiveError: unknown primitive 'greater_than'`

- [ ] **Step 3: Implement the comparisons**

In `src/tusk/primitives/transform.py`, after `MultiplyNumeric` and before `Not`. `_COMPARABLE_PAIRS` is a module-level constant near `TRANS_DEFAULTS`, since six classes share it and repeating the tuple six times is the duplication the codestyle rules out:

```python
_COMPARABLE_PAIRS: tuple[tuple[F, ...], ...] = (
    (F.NUMERIC, F.NUMERIC),
    (F.HAS_DATE, F.HAS_DATE),
)
_EQUATABLE_PAIRS: tuple[tuple[F, ...], ...] = (
    *_COMPARABLE_PAIRS,
    (F.BOOLEAN, F.BOOLEAN),
    (F.STRING, F.STRING),
)
```

```python
@register
@dataclass(frozen=True)
class GreaterThan(TransformPrimitive):
    """Whether the first value exceeds the second. A null gives a null."""

    name = "greater_than"
    input_dtypes = _COMPARABLE_PAIRS
    output_dtype = nw.Boolean

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the greater-than expression.

        Args:
            left: First comparable expression.
            right: Second comparable expression.

        Returns:
            A narwhals expression that is true where left exceeds right.
        """
        return left > right
```

Repeat that shape for the remaining five, changing only the class name, `name`, docstring, `input_dtypes` (`_EQUATABLE_PAIRS` for the two equality ones), `commutative` (True on `Equal` and `NotEqual`), and the operator:

| Class | `name` | Operator | `input_dtypes` | `commutative` |
| --- | --- | --- | --- | --- |
| `GreaterThan` | `greater_than` | `left > right` | `_COMPARABLE_PAIRS` | default |
| `GreaterThanEqualTo` | `greater_than_equal_to` | `left >= right` | `_COMPARABLE_PAIRS` | default |
| `LessThan` | `less_than` | `left < right` | `_COMPARABLE_PAIRS` | default |
| `LessThanEqualTo` | `less_than_equal_to` | `left <= right` | `_COMPARABLE_PAIRS` | default |
| `Equal` | `equal` | `left == right` | `_EQUATABLE_PAIRS` | `True` |
| `NotEqual` | `not_equal` | `left != right` | `_EQUATABLE_PAIRS` | `True` |

Export all six from `src/tusk/primitives/__init__.py`, in the alphabetical position the existing import block and `__all__` use.

`test_comparing_a_number_with_a_string_is_never_synthesized` calls `synthesize`, which `tests/test_primitives_transform.py` does not import yet. Add `from tusk.synthesis import synthesize` to its imports — it is not exported from the `tusk` package.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_primitives_transform.py -v`
Expected: PASS

- [ ] **Step 5: Lint and run the suite**

Run: `just lint && just test`
Expected: PASS. `interrogate` and `pydoclint` will fail the commit if any of the six is missing its docstring sections.

- [ ] **Step 6: Commit**

```bash
git add src/tusk/primitives/transform.py src/tusk/primitives/__init__.py tests/test_primitives_transform.py
git commit -m "$(cat <<'EOF'
feat: add the six comparison transforms

Each accepts a numeric pair or a datetime pair; equality also accepts
booleans and strings. A null input gives a null answer, as in SQL.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0153Kdi2FVgeNUJMiBPfpLpq
EOF
)"
```

---

### Task 4: `modulo_numeric` and `multiply_numeric_boolean`

**Files:**
- Modify: `src/tusk/primitives/transform.py` (after `MultiplyNumeric`, before the comparisons)
- Modify: `src/tusk/primitives/__init__.py`
- Test: `tests/test_primitives_transform.py`

**Interfaces:**
- Produces: registered names `modulo_numeric`, `multiply_numeric_boolean`; classes `ModuloNumeric`, `MultiplyNumericBoolean` exported from `tusk.primitives`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_primitives_transform.py`:

```python
@pytest.fixture
def signed():
    return nw.from_native(
        pl.LazyFrame(
            {
                "v": [7.0, -7.0, 7.0, -7.0, 1.0],
                "w": [2.0, 2.0, -2.0, -2.0, 0.0],
                "flag": [True, False, None, True, False],
            },
        ),
    )


def test_modulo_floors_rather_than_truncating(signed):
    """Python's rule, not C's: the result takes the divisor's sign.

    polars floors and duckdb truncates, so a plain % would mean two different
    things depending on the backend.
    """
    assert _apply(signed, "modulo_numeric", "v", "w")[:4] == [1.0, 1.0, -1.0, -1.0]


def test_modulo_by_zero_is_not_a_number(signed):
    result = _apply(signed, "modulo_numeric", "v", "w")[4]
    assert result != result  # NaN


def test_multiply_numeric_boolean_masks_the_number(signed):
    assert _apply(signed, "multiply_numeric_boolean", "v", "flag") == [
        7.0,
        -0.0,
        None,
        -7.0,
        0.0,
    ]


def test_modulo_and_masking_declare_their_inputs():
    assert resolve("modulo_numeric").signatures == (
        (DtypeFamily.NUMERIC, DtypeFamily.NUMERIC),
    )
    assert resolve("multiply_numeric_boolean").signatures == (
        (DtypeFamily.NUMERIC, DtypeFamily.BOOLEAN),
    )
```

Note on the `-0.0` expectation: `-7.0 * 0` is negative zero in IEEE 754, and `-0.0 == 0.0` is True in Python, so this assertion passes either way. Do not "fix" it to `0.0`.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_primitives_transform.py -k "modulo or masking or multiply_numeric_boolean" -v`
Expected: FAIL with `PrimitiveError: unknown primitive 'modulo_numeric'`

- [ ] **Step 3: Implement both**

```python
@register
@dataclass(frozen=True)
class ModuloNumeric(TransformPrimitive):
    """Remainder after division, taking the sign of the divisor."""

    name = "modulo_numeric"
    input_dtypes = (F.NUMERIC, F.NUMERIC)

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the floored-modulo expression.

        Args:
            left: The dividend.
            right: The divisor.

        Returns:
            A narwhals expression of the remainder.
        """
        # Doubled rather than a plain %, because polars floors and duckdb
        # truncates: -7 % 2 is 1 on one and -1 on the other. This forces the
        # floored answer everywhere.
        return ((left % right) + right) % right


@register
@dataclass(frozen=True)
class MultiplyNumericBoolean(TransformPrimitive):
    """A number where the flag is true, zero where it is false."""

    name = "multiply_numeric_boolean"
    input_dtypes = (F.NUMERIC, F.BOOLEAN)

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the masking expression.

        Args:
            left: A numeric expression.
            right: A boolean expression.

        Returns:
            A narwhals expression of the number masked by the flag.
        """
        return left * right.cast(nw.Int8)
```

`MultiplyNumericBoolean` sets no `output_dtype`, so `return_dtype` falls through to the numeric input's dtype — which is why the numeric slot is first.

Export both from `src/tusk/primitives/__init__.py`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_primitives_transform.py -v`
Expected: PASS

- [ ] **Step 5: Verify the modulo claim on the second backend now, not later**

Run:

```bash
uv run python -c "
import duckdb, narwhals as nw
con = duckdb.connect()
con.execute('CREATE TABLE t AS SELECT * FROM (VALUES (-7.0, 2.0)) v(a, b)')
df = nw.from_native(con.table('t'))
print(df.with_columns(m=((nw.col('a') % nw.col('b')) + nw.col('b')) % nw.col('b')).to_native())
"
```

Expected: `m` is `1.0`. If it is `-1.0`, the doubled expression is not doing its job on duckdb and Task 4 is not done — stop and report rather than continuing to Task 5.

- [ ] **Step 6: Commit**

```bash
git add src/tusk/primitives/transform.py src/tusk/primitives/__init__.py tests/test_primitives_transform.py
git commit -m "$(cat <<'EOF'
feat: add modulo_numeric and multiply_numeric_boolean

Modulo is built floored rather than as a plain %, so it means the same
thing on polars and duckdb.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0153Kdi2FVgeNUJMiBPfpLpq
EOF
)"
```

---

### Task 5: Categorical equality

**Files:**
- Modify: `src/tusk/primitives/transform.py` (after `NotEqual`)
- Modify: `src/tusk/primitives/__init__.py`
- Modify: `src/tusk/synthesis.py:365-375` (the warning message)
- Test: `tests/test_primitives_transform.py`, `tests/test_synthesis.py`

**Interfaces:**
- Produces: registered names `equal_categorical`, `not_equal_categorical`; classes `EqualCategorical`, `NotEqualCategorical` exported from `tusk.primitives`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_primitives_transform.py`:

```python
@pytest.fixture
def labels():
    """Two Enum columns whose member lists differ.

    That difference is the case the cast exists for: polars raises
    ``SchemaError: Enum mismatch`` on a direct comparison of these two.
    """
    return nw.from_native(
        pl.LazyFrame(
            {
                "status": pl.Series(
                    ["open", "closed", "open"],
                    dtype=pl.Enum(["open", "closed", "pending"]),
                ),
                "tier": pl.Series(
                    ["open", "open", None],
                    dtype=pl.Enum(["open", "closed"]),
                ),
            },
        ),
    )


def test_categorical_equality_compares_enums_with_different_members(labels):
    assert _apply(labels, "equal_categorical", "status", "tier") == [
        True,
        False,
        None,
    ]


def test_categorical_inequality_compares_enums_with_different_members(labels):
    assert _apply(labels, "not_equal_categorical", "status", "tier") == [
        False,
        True,
        None,
    ]


def test_comparing_the_enums_without_the_cast_would_raise(labels):
    """Guards the reason the primitive casts at all."""
    with pytest.raises(Exception, match="Enum"):
        labels.with_columns(
            (nw.col("status") == nw.col("tier")).alias("o"),
        ).collect()


def test_categorical_equality_takes_categorical_pairs():
    for name in ("equal_categorical", "not_equal_categorical"):
        assert resolve(name).signatures == (
            (DtypeFamily.CATEGORICAL, DtypeFamily.CATEGORICAL),
        )
        assert resolve(name).output_dtype == nw.Boolean
        assert resolve(name).commutative is True
```

Append to `tests/test_synthesis.py`:

```python
def test_the_categorical_warning_names_the_primitive_that_handles_labels():
    db = tusk.Database("labelled").add_table(
        "events",
        pl.LazyFrame(
            {
                "id": [1, 2],
                "status": pl.Series(["a", "b"], dtype=pl.Categorical),
                "note": ["x", "y"],
            },
        ),
        primary_key="id",
    )
    with pytest.warns(CategoricalDtypeWarning, match="equal_categorical"):
        synthesize(
            database=db,
            target_table="events",
            agg_primitives=[],
            trans_primitives=["equal"],
            max_depth=1,
        )
```

`tests/test_synthesis.py` imports its exception classes inside the functions that use them (see line 20). Follow that: `from tusk.exceptions import CategoricalDtypeWarning` as the first line of this test.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_primitives_transform.py tests/test_synthesis.py -k "categorical or labels or enums" -v`
Expected: FAIL with `PrimitiveError: unknown primitive 'equal_categorical'`. `test_comparing_the_enums_without_the_cast_would_raise` should already PASS — it asserts polars' behavior, not tusk's. If it fails, polars has changed and the whole cast rationale needs re-checking before continuing.

- [ ] **Step 3: Implement both primitives**

```python
@register
@dataclass(frozen=True)
class EqualCategorical(TransformPrimitive):
    """Whether two label columns carry the same label."""

    name = "equal_categorical"
    input_dtypes = (F.CATEGORICAL, F.CATEGORICAL)
    output_dtype = nw.Boolean
    commutative = True

    def build(self, left: nw.Expr, right: nw.Expr) -> nw.Expr:
        """Build the label-equality expression.

        Args:
            left: First categorical expression.
            right: Second categorical expression.

        Returns:
            A narwhals expression that is true where the labels match.
        """
        # Compared as text because two Enum columns with different member
        # lists cannot be compared directly on polars, though duckdb allows
        # it. The label is the value; its encoding is not.
        return left.cast(nw.String) == right.cast(nw.String)
```

`NotEqualCategorical` is the same with `name = "not_equal_categorical"`, the docstring "Whether two label columns carry different labels.", and `!=`.

- [ ] **Step 4: Point the warning at them**

In `_warn_categorical` (`src/tusk/synthesis.py`), extend the message's last sentence:

```python
                f"column {feature.name!r} on {feature.table!r} has dtype "
                f"{feature.dtype}, so primitive {primitive.name!r} (which requires "
                f"a string input) will not be applied to it. Cast the column to "
                f"String if you want text primitives to use it, or use "
                f"'equal_categorical' / 'not_equal_categorical' to compare "
                f"labels directly.",
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_primitives_transform.py tests/test_synthesis.py -v`
Expected: PASS

- [ ] **Step 6: Lint and run the suite**

Run: `just lint && just test`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/tusk/primitives/transform.py src/tusk/primitives/__init__.py src/tusk/synthesis.py tests/test_primitives_transform.py tests/test_synthesis.py
git commit -m "$(cat <<'EOF'
feat: compare labels with equal_categorical

Two Enum columns with different member lists cannot be compared directly on
polars. Casting both to text makes the answer the same on every backend.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0153Kdi2FVgeNUJMiBPfpLpq
EOF
)"
```

---

### Task 6: The duckdb portability test

**Files:**
- Modify: `tests/test_backend_duckdb.py` (append, following `test_boolean_transforms_hold_three_valued_logic_on_duckdb` at line 334)

**Interfaces:**
- Consumes: every primitive from Tasks 3-5.

- [ ] **Step 1: Write the failing test**

```python
def test_binary_transforms_translate_to_sql(duck_db):
    """Comparisons, floored modulo and masking survive the trip to SQL.

    Modulo is the reason this test exists: duckdb's ``%`` truncates toward
    zero, so a plain remainder would answer -1 here where polars answers 1.

    Args:
        duck_db: The duckdb-backed database.
    """
    _, con = duck_db
    con.execute(
        "CREATE TABLE readings AS SELECT * FROM (VALUES "
        "(1, -7.0, 2.0, TRUE), (2, 7.0, -2.0, FALSE), (3, 5.0, NULL, NULL)) "
        "t(id, v, w, flag)",
    )
    database = tusk.Database("sensors").add_table(
        "readings",
        con.table("readings"),
        primary_key="id",
    )
    matrix = tusk.deep_feature_synthesis(
        database=database,
        target_table="readings",
        max_depth=1,
        agg_primitives=[],
        trans_primitives=[
            "greater_than",
            "equal",
            "modulo_numeric",
            "multiply_numeric_boolean",
        ],
    )[0].pl()
    row = {r["id"]: r for r in matrix.to_dicts()}
    assert row[1]["MODULO_NUMERIC__v__w"] == 1.0
    assert row[2]["MODULO_NUMERIC__v__w"] == -1.0
    assert row[3]["MODULO_NUMERIC__v__w"] is None
    assert row[1]["GREATER_THAN__v__w"] is False
    assert row[2]["GREATER_THAN__v__w"] is True
    assert row[3]["GREATER_THAN__v__w"] is None
    assert row[1]["EQUAL__v__w"] is False
    assert row[1]["MULTIPLY_NUMERIC_BOOLEAN__v__flag"] == -7.0
    assert row[2]["MULTIPLY_NUMERIC_BOOLEAN__v__flag"] == 0.0
    assert row[3]["MULTIPLY_NUMERIC_BOOLEAN__v__flag"] is None
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_backend_duckdb.py::test_binary_transforms_translate_to_sql -v`
Expected: PASS. The primitives already exist, so this is a portability check rather than a red-green cycle — its value is that it fails loudly if the floored-modulo expression does not translate.

If `MODULO_NUMERIC__v__w` comes back `-1.0` for row 1, the expression is being folded by duckdb. Do not weaken the assertion; report it.

- [ ] **Step 3: Run the full duckdb file**

Run: `uv run pytest tests/test_backend_duckdb.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_backend_duckdb.py
git commit -m "$(cat <<'EOF'
test: floored modulo and the comparisons survive translation to SQL

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0153Kdi2FVgeNUJMiBPfpLpq
EOF
)"
```

---

### Task 7: The differential suite

**Files:**
- Create: `tests/differential/test_binary_transforms.py`

**Interfaces:**
- Consumes: `differential._as_tusk`, every primitive from Tasks 3-5.

This task is what turns ❓ and ❌ into ✅ in the coverage table. Model the file on `tests/differential/test_boolean_transforms.py`, which is the closest existing example: one fixture, `_both_matrices`, `_featuretools_matrix`, `_tusk_matrix`.

- [ ] **Step 1: Write the file**

```python
"""Cross-check tusk's binary transform primitives against featuretools.

The arithmetic four agree everywhere. The comparisons agree wherever both
inputs are known; where one is null, tusk answers null because that is what
SQL answers, and this file asserts whatever featuretools does rather than
excluding the rows.

``modulo_numeric`` is built floored rather than as a plain ``%`` (polars
floors, duckdb truncates), which is also what featuretools' pandas
implementation does, so the two agree on negative operands.

Run with: uv run --group validation pytest -m differential

Verified against featuretools 1.31.0.
"""

import pytest

import tusk
from differential import _as_tusk

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")
featuretools = pytest.importorskip("featuretools")

pytestmark = pytest.mark.differential

ARITHMETIC = ["add_numeric", "subtract_numeric", "multiply_numeric", "divide_numeric"]
COMPARISONS = [
    "greater_than",
    "greater_than_equal_to",
    "less_than",
    "less_than_equal_to",
    "equal",
    "not_equal",
]


@pytest.fixture
def rows():
    """Numbers covering both signs, a zero divisor, equality, and nulls."""
    left = [7.0, -7.0, 2.0, 2.0, 1.0, None, 3.0]
    right = [2.0, 2.0, 2.0, -2.0, 0.0, 4.0, None]
    frame = pd.DataFrame(
        {
            "id": np.arange(1, len(left) + 1),
            "left": left,
            "right": right,
        },
    )
    _assert_rows_invariants(frame)
    return frame


def _assert_rows_invariants(frame):
    """Guard the cases ``rows`` is built to cover.

    Losing any one of these would leave a test passing over nothing.

    Args:
        frame: The table built by ``rows``.
    """
    both_known = frame["left"].notna() & frame["right"].notna()
    assert both_known.any()
    assert (~both_known).any()
    assert (frame["left"] < 0).any()
    assert (frame["right"] == 0).any()
    assert (frame["left"] == frame["right"]).any()


@pytest.mark.parametrize("primitive_name", ARITHMETIC)
def test_arithmetic_transforms_match_featuretools(rows, primitive_name):
    """The four that shipped untested, now held to featuretools' values."""
    ours, theirs = _both_matrices(rows, primitive_name)
    column = f"{primitive_name.upper()}(left, right)"
    assert _numbers(ours[_as_tusk(column)]) == _numbers(theirs[column])


@pytest.mark.parametrize("primitive_name", COMPARISONS)
def test_comparisons_match_featuretools_where_nothing_is_null(rows, primitive_name):
    ours, theirs = _both_matrices(rows, primitive_name)
    known = rows.set_index("id")[["left", "right"]].notna().all(axis=1)
    column = f"{primitive_name.upper()}(left, right)"
    assert _nullable(ours[_as_tusk(column)][known]) == _nullable(theirs[column][known])


def test_modulo_matches_featuretools_on_negative_operands(rows):
    """Both floor, so -7 % 2 is 1 and 7 % -2 is -1 on each side."""
    ours, theirs = _both_matrices(rows, "modulo_numeric")
    column = "MODULO_NUMERIC(left, right)"
    assert _numbers(ours[_as_tusk(column)]) == _numbers(theirs[column])


def _numbers(series):
    """Render a numeric column as a comparable list.

    NaN is mapped to None so that a NaN on one side equals a NaN on the
    other; ``float('nan') != float('nan')`` would otherwise fail the
    comparison against itself.

    Args:
        series: A numeric column from either matrix.

    Returns:
        One float or None per row.
    """
    return [None if pd.isna(value) else float(value) for value in series]


def _nullable(series):
    """Render a boolean column as a comparable list, with nulls as None.

    Args:
        series: A boolean column from either matrix.

    Returns:
        One True, False or None per row.
    """
    return [None if pd.isna(value) else bool(value) for value in series]


def _both_matrices(frame, primitive_name):
    """Build the same one-primitive matrix on both sides.

    Args:
        frame: The table to build features over.
        primitive_name: The primitive's name, spelled the same way by tusk
            and featuretools.

    Returns:
        A tuple of the tusk matrix and the featuretools matrix, both indexed
        by the table's primary key.
    """
    return (
        _tusk_matrix(frame, primitive_name),
        _featuretools_matrix(frame, primitive_name),
    )


def _featuretools_matrix(frame, primitive_name):
    """Run one transform primitive through featuretools and return its matrix.

    Args:
        frame: The table to build features over.
        primitive_name: The primitive's featuretools name, the only entry in
            ``trans_primitives`` so exactly one feature is built.

    Returns:
        The feature matrix, sorted by the table's index.
    """
    es = featuretools.EntitySet("rows")
    es = es.add_dataframe(dataframe_name="rows", dataframe=frame, index="id")
    matrix, _ = featuretools.dfs(
        entityset=es,
        target_dataframe_name="rows",
        agg_primitives=[],
        trans_primitives=[primitive_name],
        max_depth=1,
    )
    return matrix.sort_index()


def _tusk_matrix(frame, primitive_name):
    """Run one transform primitive through tusk and return its matrix.

    Args:
        frame: The table to build features over.
        primitive_name: The primitive's name, the only entry in
            ``trans_primitives`` so exactly one feature is built.

    Returns:
        The feature matrix, collected and sorted by the table's primary key.
    """
    db = tusk.Database("rows_db").add_table(
        "rows",
        pl.from_pandas(frame).lazy(),
        primary_key="id",
    )
    matrix, _ = tusk.deep_feature_synthesis(
        database=db,
        target_table="rows",
        agg_primitives=[],
        trans_primitives=[primitive_name],
        max_depth=1,
    )
    return matrix.collect().sort("id").to_pandas().set_index("id")
```

- [ ] **Step 2: Run it**

Run: `uv run --group validation pytest tests/differential/test_binary_transforms.py -v`
Expected: unknown. This is the point of the task — it is a measurement, not a formality.

- [ ] **Step 3: Adjudicate every failure rather than deleting it**

For each failing case, decide which side is right and record the decision:

- **Both defensible, tusk follows SQL** (the likely shape for `EQUAL` where one input is null): keep the agreement test restricted to known-input rows, and add a test asserting *both* behaviors explicitly, the way `test_binary_boolean_transforms_diverge_where_one_input_is_null` does in `test_boolean_transforms.py`. Add the reason to the module docstring.
- **tusk is wrong**: fix the primitive, in its own commit, and note it in the task's commit message.
- **Column names differ** (featuretools may name a comparison differently from `GREATER_THAN(left, right)`): fix the test's column name. Read the real name off `theirs.columns` rather than guessing.

Do not weaken an assertion to make it pass, and do not drop a primitive from the parametrize list.

- [ ] **Step 4: Run the whole differential suite**

Run: `just test-differential`
Expected: PASS, including the pre-existing files.

- [ ] **Step 5: Commit**

```bash
git add tests/differential/test_binary_transforms.py
git commit -m "$(cat <<'EOF'
test: cross-check the binary transforms against featuretools

Covers the four arithmetic transforms that shipped untested, the six
comparisons, and floored modulo on negative operands.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0153Kdi2FVgeNUJMiBPfpLpq
EOF
)"
```

---

### Task 8: Documentation

**Files:**
- Modify: `docs/api/primitives.md`
- Modify: `docs/guide/primitives.md`
- Modify: `docs/guide/primitive-coverage.md:83-107`

**Interfaces:**
- Consumes: the final, measured behavior from Task 7. Write this task last, because the coverage table must record what the differential suite actually found, not what this plan predicted.

- [ ] **Step 1: Add the API entries**

In `docs/api/primitives.md`, after `::: tusk.primitives.DivideNumeric` and before `::: tusk.primitives.Not`:

```markdown
::: tusk.primitives.ModuloNumeric

::: tusk.primitives.MultiplyNumericBoolean

::: tusk.primitives.GreaterThan

::: tusk.primitives.GreaterThanEqualTo

::: tusk.primitives.LessThan

::: tusk.primitives.LessThanEqualTo

::: tusk.primitives.Equal

::: tusk.primitives.NotEqual

::: tusk.primitives.EqualCategorical

::: tusk.primitives.NotEqualCategorical
```

- [ ] **Step 2: Add the guide section**

In `docs/guide/primitives.md`, after the `## Nulls in \`and\` and \`or\`` section (which ends at line 92) and before `## What can go in \`groupby_trans_primitives\``:

```markdown
## Comparing two columns

The comparison primitives accept a pair of numbers or a pair of datetimes,
never one of each — a primitive may declare several input shapes, and each
shape is matched as a whole:

```py
input_dtypes = ((F.NUMERIC, F.NUMERIC), (F.HAS_DATE, F.HAS_DATE))
```

`equal` and `not_equal` also accept a pair of booleans or a pair of strings.
A null on either side gives a null answer, as in SQL: an unknown value cannot
be shown equal to anything, nor greater than it.

Labels are a separate case. `Categorical` and `Enum` columns are compared with
`equal_categorical` and `not_equal_categorical`, which compare the labels
themselves rather than their encodings — two `Enum` columns with different
member lists cannot be compared directly on polars at all.

`modulo_numeric` takes the sign of the divisor, so `MODULO_NUMERIC(-7, 2)` is
`1`. That is Python's rule and featuretools' rule, but not every SQL engine's:
duckdb's `%` truncates toward zero and would answer `-1`. tusk builds the
floored form explicitly so the feature means one thing on every backend.

There is no `multiply_boolean`. Multiplying two booleans is `and`, which
already ships under the name that says what it does.
```

Check the surrounding heading levels and fenced-code style before pasting; match what the file already does.

- [ ] **Step 3: Update the coverage table**

Rewrite lines 83-107 of `docs/guide/primitive-coverage.md`. Four ❓ rows become ✅; `equal`, `greater_than`, `greater_than_equal_to`, `less_than`, `less_than_equal_to`, `modulo_numeric`, `multiply_numeric_boolean` and `not_equal` become ✅ with a tusk link; `multiply_boolean` becomes ⚠️. Leave every ⛔ row untouched.

The test-column link for the new rows, matching the style already used in the table:

```
[`test_values_match_featuretools`](https://github.com/Excidion/tusk/blob/main/tests/differential/test_binary_transforms.py)
```

Use the real test function names from the file written in Task 7.

Comments to carry:
- On the four comparison-ordering rows: `featuretools also accepts Ordinal columns; tusk has no logical types.`
- On `equal` and `not_equal`: `Compares numbers, datetimes, booleans and strings. Labels have their own primitives.`
- On `multiply_boolean`: `Boolean multiplication is `and`, which tusk already ships.` and status ⚠️.
- On `modulo_numeric`: `Floored, so the sign follows the divisor on every backend.`

Add two ➕ rows for the tusk-only primitives, placed in the group's alphabetical order:

```markdown
| Binary transform | [`equal_categorical`][tusk.primitives.EqualCategorical] | — | ➕ | [`test_categorical_equality_compares_enums_with_different_members`](https://github.com/Excidion/tusk/blob/main/tests/test_primitives_transform.py) | Compares labels, not encodings. |
| Binary transform | [`not_equal_categorical`][tusk.primitives.NotEqualCategorical] | — | ➕ | [`test_categorical_inequality_compares_enums_with_different_members`](https://github.com/Excidion/tusk/blob/main/tests/test_primitives_transform.py) | Compares labels, not encodings. |
```

If Task 7 found a divergence, that row is ⚠️ rather than ✅, with the reason in the comment column and the divergence test linked.

- [ ] **Step 4: Verify the mkdocstrings references resolve**

Run: `just docs`
Expected: builds with no warning about an unresolved `[tusk.primitives.X]` reference. Every `:::` entry added in Step 1 and every `[tusk.primitives.X][]` link added in Step 3 must name a class actually exported from `tusk.primitives`.

- [ ] **Step 5: Check nothing else in the coverage table went stale**

Run: `grep -c "❓" docs/guide/primitive-coverage.md`
Expected: the count dropped by exactly 4. If any ❓ remains in the Binary transform group, a row was missed.

- [ ] **Step 6: Commit**

```bash
git add docs/api/primitives.md docs/guide/primitives.md docs/guide/primitive-coverage.md
git commit -m "$(cat <<'EOF'
docs: cover the binary transforms and their input shapes

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0153Kdi2FVgeNUJMiBPfpLpq
EOF
)"
```

---

### Task 9: Full verification and the pull request

**Files:** none

- [ ] **Step 1: Run everything CI runs**

Run: `just check`
Expected: PASS — lint, the default suite, and the differential suite. Do not open the PR on a partial run.

- [ ] **Step 2: Read the diff for scope**

Run: `git diff main...HEAD --stat` then `git diff main...HEAD`
Expected: only `src/tusk/primitives/base.py`, `src/tusk/primitives/transform.py`, `src/tusk/primitives/__init__.py`, `src/tusk/synthesis.py`, the four test files, the three docs files, the spec and this plan. `docs/relbench.md` is untracked and predates this branch — leave it alone.

- [ ] **Step 3: Confirm no existing primitive was edited**

Run: `git diff main...HEAD -- src/tusk/primitives/transform.py | grep "^-" | grep -v "^---"`
Expected: no deletions except where the new module-level constants were inserted. Every existing primitive's declaration should be untouched, which is the claim the whole backward-compatible `input_dtypes` design rests on.

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin feature/binary-transform-parity
gh pr create --title "Binary transform primitives: comparisons, modulo, and label equality" --body "$(cat <<'EOF'
Closes every binary-transform row in the coverage table that is not out of scope.

## Alternative input signatures

`Primitive.input_dtypes` may now hold several dtype-family tuples rather than
one. Synthesis matches each shape as a whole and unions the results, so
`greater_than` accepts a numeric pair or a datetime pair but never a number
against a string — which independent per-slot matching would have generated,
and which no backend can run.

A flat tuple is still a single shape, so none of the 25 existing primitives
changed.

## New primitives

- `greater_than`, `greater_than_equal_to`, `less_than`, `less_than_equal_to` —
  numeric or datetime pairs.
- `equal`, `not_equal` — those, plus boolean and string pairs.
- `modulo_numeric` — built floored, because polars floors and duckdb truncates.
- `multiply_numeric_boolean` — a number masked by a flag.
- `equal_categorical`, `not_equal_categorical` — tusk-only; compare labels as
  text, since two `Enum` columns with different member lists cannot be
  compared directly on polars.

`multiply_boolean` is deliberately absent: it is `and` under another name.

## Tests

The four arithmetic transforms that shipped untested are now held to
featuretools' values, alongside the new primitives, in
`tests/differential/test_binary_transforms.py`. A duckdb test holds the floored
modulo and the comparisons through translation to SQL.

Design: `agents/superpowers/specs/2026-09-05-binary-transform-parity-design.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_0153Kdi2FVgeNUJMiBPfpLpq
EOF
)"
```

- [ ] **Step 5: Report the PR URL**

Do not merge. The constitution reserves that for the user.
