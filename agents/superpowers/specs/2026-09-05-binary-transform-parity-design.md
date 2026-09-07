# Binary transform parity: alternative input signatures

Design for the comparison, modulo and boolean-multiply transforms, and for the
matching rule they all need.

## Problem

`docs/guide/primitive-coverage.md` lists thirteen binary transforms that are not
out of scope. Four (`add_numeric`, `subtract_numeric`, `multiply_numeric`,
`divide_numeric`) are ❓ — implemented, but with no test to back the values.
The other nine are ❌, unimplemented.

Six of the nine are comparisons, and they run into a limitation in how tusk
matches columns to primitives. A primitive declares one dtype family per input
slot, and synthesis fills the slots independently:

```py
# src/tusk/synthesis.py:446
per_slot = [
    [f for f in candidates if matches(f.dtype, family)]
    for family in primitive.input_dtypes
]
combos = itertools.product(*per_slot)
```

There is no way to express a relationship *between* slots. A `greater_than`
declaring `(ANY, ANY)` therefore generates `GREATER_THAN(amount, name)` on any
table holding a number and a string, and that feature cannot run:

- polars: `ComputeError: cannot compare string with numeric type (f64)`
- duckdb: `Binder Error: Cannot compare values of type DOUBLE and type VARCHAR`

Restricting to `(NUMERIC, NUMERIC)` avoids the crash by giving up datetime
comparison, which featuretools supports.

featuretools has the capability tusk lacks: `input_types` holds a *list of
alternative signatures*, each internally consistent.

```
greater_than -> [[numeric, numeric], [Datetime, Datetime], [Ordinal, Ordinal]]
```

It picks one at a time, so a numeric slot is never paired with a text one.

## Decisions

These were settled with the maintainer.

1. **`input_dtypes` gains alternative signatures.** It may hold a tuple of
   families, as today, or a tuple of such tuples. This is the smallest concept
   that removes the collision, and it is what featuretools has proven works.
2. **One primitive per operation.** With alternative signatures there is no
   name collision to route around, so `greater_than` covers numeric and
   datetime rather than splitting into `greater_than` and `after`. Names map
   1:1 onto featuretools.
3. **`modulo_numeric` forces floored semantics.** polars floors and duckdb
   truncates, so `-7 % 2` is `1` on one backend and `-1` on the other. Every
   other tusk primitive means the same thing on every backend, and this one
   will too.
4. **`multiply_boolean` is not implemented.** Boolean multiplication is
   `left & right`, which is exactly the existing `and`. A synonym would emit a
   duplicate column under a second name.
5. **Categorical inputs are not cast to String in the compiler.** The cast is
   viable and would let `equal` accept categoricals, but it is a permanent
   rule about how every primitive receives its inputs, and it is not what this
   change is about.
6. **Label equality gets its own primitive instead.** `equal_categorical` and
   `not_equal_categorical` declare categorical inputs and so may cast
   unconditionally — the ambiguity that rules the cast out inside `equal` does
   not exist when every input is categorical by declaration.

## Architecture

### Alternative signatures

`input_dtypes` keeps its name and its meaning. A flat tuple is shorthand for a
single signature, so none of the 25 existing primitives is edited.

```py
input_dtypes = (F.NUMERIC, F.NUMERIC)                          # unchanged
input_dtypes = ((F.NUMERIC, F.NUMERIC), (F.HAS_DATE, F.HAS_DATE))   # new
```

A read-only property on `Primitive` normalizes the two shapes, so nothing
outside it inspects the declaration's form:

```py
@property
def signatures(self) -> tuple[tuple[DtypeFamily, ...], ...]:
    """Every input shape this primitive accepts, one tuple of families each."""
```

Every signature must have the same arity. `build()` has a fixed parameter
list, so a two-slot signature beside a three-slot one is a primitive that
cannot run; `signatures` raises `PrimitiveError` rather than leaving it to a
`TypeError` from `build` at compile time.

### Synthesis

Three sites read `input_dtypes` today, and all three move to `signatures`.

| Site | Change |
| --- | --- |
| `synthesis.py:199` | Zero-arity test; `not primitive.signatures` reads the same |
| `synthesis.py:360` | `_warn_categorical` checks every signature for `STRING` |
| `synthesis.py:446` | `_combinations` unions over the signatures |

`_combinations` splits in two. Its current body becomes
`_combinations_for_signature(primitive, candidates, signature)` unchanged; the
caller loops the signatures, concatenates, and deduplicates with
`dict.fromkeys`. Deduplication is required because families overlap: a
`Datetime` column matches `HAS_DATE` and `TEMPORAL` both, so a primitive
declaring signatures over each would otherwise emit one feature twice.

The `stack_on_self` filter, the commutative deduplication and the
matched/unmatched bookkeeping stay where they are and read the unioned result,
so "matched nothing anywhere" keeps its meaning.

`_warn_categorical` gains one consequence worth naming: `equal` declares a
`STRING` signature, making it the first shipped primitive to fire
`CategoricalDtypeWarning`. Nothing in tusk declares `F.STRING` today. The
message currently reports only that the column is skipped; it will also name
`equal_categorical` as the primitive that handles it.

### Primitives

Comparisons, all `output_dtype = nw.Boolean`, all declaring
`((NUMERIC, NUMERIC), (HAS_DATE, HAS_DATE))` unless noted:

| Name | Expression | Commutative |
| --- | --- | --- |
| `greater_than` | `left > right` | no |
| `greater_than_equal_to` | `left >= right` | no |
| `less_than` | `left < right` | no |
| `less_than_equal_to` | `left <= right` | no |
| `equal` | `left == right` | yes |
| `not_equal` | `left != right` | yes |

`equal` and `not_equal` additionally declare `(BOOLEAN, BOOLEAN)` and
`(STRING, STRING)`: equality is meaningful for every type that compares safely
on both backends.

featuretools' third comparison signature is `Ordinal`. tusk has no logical
types by design (`src/tusk/dtypes.py:1-6`), so there is nothing to map it to.

`modulo_numeric`, `(NUMERIC, NUMERIC)`:

```py
return ((left % right) + right) % right
```

The doubled operation is what makes the result floored on a backend that
truncates. It carries a comment saying so, since a plain `%` is what a reader
expects to find.

`multiply_numeric_boolean`, `(NUMERIC, BOOLEAN)`, output dtype falling through
to the numeric input. featuretools declares both argument orders as separate
signatures; tusk needs one, because synthesis generates each (numeric,
boolean) pair once regardless of order.

`equal_categorical` and `not_equal_categorical`, `(CATEGORICAL, CATEGORICAL)`,
`output_dtype = nw.Boolean`, tusk-only:

```py
return left.cast(nw.String) == right.cast(nw.String)
```

The cast is not cosmetic. Two `Enum` columns with different member lists raise
on polars —

```
SchemaError: Enum mismatch.
Operations mixing different Enums are often not supported, you may have to cast.
```

— while duckdb compares them by value. Casting both sides to text makes the
answer the same on both, and covers a `Categorical` compared against an `Enum`.

`DtypeFamily.CATEGORICAL` matches `Categorical` and `Enum` both
(`src/tusk/dtypes.py:70`), which is why the cast is what makes these
primitives safe to ship at all rather than a convenience.

## Testing

The mechanism is tested apart from the primitives, because it is the part that
can break primitives that already work.

- `tests/test_primitives_base.py` — `signatures` normalizes a flat tuple, a
  tuple of tuples, and empty; mismatched arity raises `PrimitiveError`.
- `tests/test_synthesis.py` — a multi-signature primitive generates features
  for each signature, never crosses slots, and emits one feature where two
  signatures both match a column.
- `tests/test_primitives_transform.py` — resolution, declared signatures,
  output dtype and values for each new primitive, including `-7 % 2 == 1` and
  `Enum` against `Enum`.
- `tests/test_backend_duckdb.py` — the comparisons and floored modulo survive
  translation to SQL. This is the test that catches the `-1` divergence, and
  so is the point of decision 3.
- `tests/differential/test_binary_transforms.py` — all twelve primitives
  against featuretools on a fixture carrying nulls, a zero denominator and
  negative operands. This is what earns ✅ in the coverage table.

Divergences found there are asserted on both sides, as
`tests/differential/test_boolean_transforms.py` does for three-valued logic,
rather than excluded from the comparison. `EQUAL` over null rows is the likely
candidate: SQL answers `null = null → null`.

## Documentation

- `docs/api/primitives.md` — ten new entries.
- `docs/guide/primitives.md` — the comparison group, floored modulo, the null
  rule, and when to reach for `equal_categorical` over `equal`.
- `docs/guide/primitive-coverage.md` — twelve rows to ✅, `multiply_boolean` to
  ⚠️ pointing at `and`, two ➕ rows for the categorical equality primitives, and
  ⚠️ notes where tusk's domain is narrower than featuretools'.

⚠️ ("Diverges on purpose") covers "narrower domain, identical where both
apply". No new legend icon.

## Out of scope

- Scalar-argument primitives, already ⛔ across the table.
- `Ordinal` as a dtype family.
- Casting categorical inputs in the compiler (decision 5).
- Comparison over `Duration`, which is orderable but has no featuretools
  counterpart.

## Outcome

Five rows landed ✅: the four arithmetic primitives and
`multiply_numeric_boolean`, agreeing with featuretools everywhere, nulls
included. The rest — the six comparisons, `equal_categorical`/
`not_equal_categorical`, `modulo_numeric` and `and` — landed ⚠️ instead of
the predicted ✅, each for a measured null-handling divergence: Numeric,
Boolean and String operands agree on a null (null gives null both sides),
but a Datetime operand does not (featuretools compares `NaT` with plain
`datetime64[ns]` semantics), and featuretools' generic `equal`/`not_equal`
treat a null label the same ordinarily-unequal way. `EQUAL` over numeric
operands, the divergence this plan predicted, does not exist.
`multiply_boolean` is not implemented as its own primitive: in tusk it
would be the exact expression `and` already is, so the ⚠️ records identical
values under two names, not a missing primitive.
