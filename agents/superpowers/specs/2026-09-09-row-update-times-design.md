# Row update times: per-column visibility under a cutoff

Design for `add_table(row_update_times=...)`, the featuretools equivalent of a
secondary time index. Closes issue #15.

## Problem

`row_creation_time` decides whether a *row* is visible at a cutoff. It cannot
say anything about a *column*. A table that records when a row was last edited

```
id | placed_at  | status    | shipper_id | updated_at
 1 | 2024-01-05 | delivered |         77 | 2024-09-01
```

is fully visible at `cutoff_time=2024-06-01` because `placed_at <= cutoff`, and
every column comes back holding a value written three months after the cutoff.
`MODE(orders.status)` learns the outcome. `MAX(orders.updated_at)` learns when
the row will next be touched. Both leak the target.

featuretools answers this with
[`EntitySet.set_secondary_time_index`](https://docs.featuretools.com/en/stable/generated/featuretools.EntitySet.set_secondary_time_index.html),
which nulls the affected columns for rows whose secondary index is after the
cutoff.

The complication featuretools does not address: a table like the one above
stores only the *current* value. When `status` was overwritten, its previous
value was not kept, so tusk cannot reconstruct it. Nulling is one guess among
several — a status column's pre-update value is far more likely `"pending"`
than unknown. Forcing the user to name that value is what makes the feature
honest rather than merely non-leaking.

## Decisions

These were settled with the maintainer.

1. **One nested mapping on `add_table`.** The update-time column, the columns
   it masks, and each masked column's pre-update value are one decision and
   travel in one argument. A separate `fallbacks=` argument would let a user
   declare a masked column and forget its fallback.
2. **A null update time means the row was never updated.** Its stored value is
   the original and is visible at every cutoff. This is the shape a nullable
   audit column normally has: set on first edit, null until then.
3. **An update-time column masks itself.** If the user does not list it,
   `add_table` inserts it with fallback `None` and warns, so the user can
   override the fallback. It is not silently correct to leave it unmasked:
   `MAX(orders.updated_at)` reads the future.
4. **The masked update-time column stays in the feature matrix.** Once masked
   it holds no post-cutoff value, so there is nothing to protect it from. This
   deliberately differs from `row_creation_time`, which
   `output_excluded_columns` drops.
5. **Foreign keys are maskable; the primary key and the row creation time are
   not.** A foreign key genuinely changes — a reassigned shipper, a merged
   account — and masking it to `None` correctly stops that child row
   aggregating into any parent at the earlier cutoff. Masking a primary key
   would rewrite the matrix index; masking a row creation time would
   contradict the filter that already guarantees it is at or before the
   cutoff.
6. **No "fallback defaulted to null" warning.** Issue #15 asks for one, but
   under decision 1 every fallback the user writes is an explicit value and
   `None` is a typed-out choice. The only implicit fallback left is the
   self-mask of decision 3, which does warn.

## Architecture

### Public API

```python
db.add_table(
    "orders",
    orders,
    primary_key="id",
    row_creation_time="placed_at",
    row_update_times={
        "updated_at": {"status": "pending", "shipper_id": None},
    },
)
```

`row_update_times: Mapping[str, Mapping[str, Any]] | None = None`. The outer
key names a Datetime column recording when the row was last edited. The inner
mapping names the columns that edit rewrote, each mapped to the value it held
beforehand.

A table may declare several update-time columns, each masking a different set
of columns. No column may be masked by two of them.

### TableSchema

`TableSchema` gains a fifth field:

```python
row_update_times: Mapping[str, Mapping[str, Any]]
```

Normalized at construction to a plain `dict` of `dict`, `{}` when the argument
is omitted, with the self-mask entry of decision 3 already inserted. Every
consumer therefore reads a complete, uniform mapping and never re-derives it.

### Masking

`compiler.base_frame` keeps its row filter and appends one `with_columns`:

```python
nw.when(nw.col(update_time).is_null() | (nw.col(update_time) <= cutoff_time))
  .then(nw.col(column))
  .otherwise(nw.lit(fallback, dtype=schema.dtypes[column]))
  .alias(column)
```

one expression per `(update_time, column, fallback)` triple, all in a single
`with_columns` call.

The single call is load-bearing. Every expression in one `with_columns` reads
the frame as it was before the call, so no mask can consume another mask's
output and the self-mask of decision 3 is well defined — its condition reads
the original `updated_at` while its `then` branch rewrites it.

`nw.lit` is given the column's declared dtype explicitly. A bare
`nw.lit(None)` has no type for a backend to infer.

`base_frame` is the only masking site. It runs before every join and every
aggregation, so depth-2 stacks, aggregations onto a parent, and masked foreign
keys all inherit the mask without further work. With `cutoff_time=None` nothing
is masked, exactly as nothing is filtered today.

### Excluded columns

`input_excluded_columns` and `output_excluded_columns` are unchanged, per
decision 4.

## Validation

Seven new entries in `TABLE_CHECKS`. Six read no rows and run by default from
`add_table`; the seventh scans and does not.

| Name | Raises when | Default |
| --- | --- | --- |
| `datetime_row_update_times` | an update-time column is not `Datetime` | on |
| `unchained_row_update_times` | an update time is updated by another update time | on |
| `singly_updated_columns` | one column is updated by two update times | on |
| `never_updated_primary_key` | the primary key is updated | on |
| `never_updated_row_creation_time` | the row creation time is updated | on |
| `matching_earlier_value_dtypes` | an earlier value does not fit its column's dtype | on |
| `ordered_row_times` | some `update_time < row_creation_time` | off |

`unchained_row_update_times` refuses a declaration where one update time is
listed under another. Every column is rewound in one pass against the stored
timestamps, so a chained update time would decide the columns below it using a
value already known to be from after the cutoff — leaking it.

It runs before `singly_updated_columns`, which a chain also trips: the chained
update time does not list itself, so `add_table` inserts its self-entry, and it
ends up listed twice — once under the update time above it, once under itself.
Running first means the error names the chain instead of reporting a duplicate
the user never wrote.

`singly_updated_columns` counts the self-entry of decision 3 like any other,
so declaring `{"a": {"b_time": 0}, "b_time": {...}}` fails: `a` updates
`b_time` and `b_time` updates itself.

`datetime_row_update_times` mirrors the existing
`datetime_row_creation_time`: a `Date` has no time of day and compares against
a cutoff differently across backends.

`matching_earlier_value_dtypes` compares each earlier value against
`schema.dtypes[column]` without reading a row, the same shape of check as
`matching_key_dtypes`. Its purpose is the error site: `{"score": "pending"}`
against an `Int64` otherwise fails inside a lazy query plan at `collect()`,
far from the `add_table` call that caused it. A `None` always passes.

`ordered_row_times` is the only check here that scans, so it stays off and is
reachable through `db.validate(tables="ordered_row_times")`. An update
recorded before the row existed means the two columns do not mean what they
were declared to mean.

### The default selector

`add_table`'s `validate=` default is the bare string
`"datetime_row_creation_time"` today. It becomes a module-level tuple in
`validation.py`:

```python
DEFAULT_TABLE_CHECKS = (
    "datetime_row_creation_time",
    "datetime_row_update_times",
    "singly_updated_columns",
    "never_updated_primary_key",
    "never_updated_row_creation_time",
    "matching_earlier_value_dtypes",
)
```

`_select_checks` already accepts an iterable of names, so its grammar is
unchanged and no notion of a check *group* is introduced.

### Structural errors

Raised by `add_table` itself, next to the existing missing-column checks,
because they describe a malformed declaration rather than a defect in data:

- an update-time column that is not a column of the table
- a masked column that is not a column of the table
- `row_update_times` given as anything but a mapping of mappings

### The self-mask warning

New in `exceptions.py`:

```python
class ImplicitEarlierValueWarning(UserWarning):
    """Warns that an update-time column was masked with a null fallback."""
```

Its own class, so it can be filtered independently, matching
`UnmatchedPrimitiveWarning` and `MissingPrimaryKeyWarning`.

## Plotting

`describe_comments` in `plotting.py` gains two phrases:

- `"row update time"` on an update-time column
- `"updated by <update_time>"` on a masked column

`columns="structural"` needs no change: it shows exactly the columns whose
`describe_role` is non-empty, so both new phrases pull their columns in
automatically. `from_database`'s docstring and `Database.plot`'s are amended to
say so.

## Documentation

- `docs/guide/databases.md`: a "Row update times" section — the declaration,
  what a null update time means, what a fallback is for, and why the primary
  key and row creation time cannot be masked.
- `docs/guide/deep-feature-synthesis.md`: the cutoff section gains the
  column-level half of the story.
- Cutoff docstrings in `api.py`, `feature_list.py` and `compiler.py`.

Every one of these states plainly that a column updated in place but never
declared in `row_update_times` is invisible to tusk and will leak. tusk cannot
detect such a column, and a user who believes the feature protects them
generally is worse off than one who knows it protects only what they declared.

## Testing

- Masking arithmetic across every backend: value kept before the cutoff,
  fallback after it.
- A null update time keeps its value.
- A self-masking update time returns its fallback after the cutoff, and the
  masks of the columns it governs are unaffected by its own rewrite.
- A masked foreign key changes an aggregation: the child stops contributing to
  its parent at the earlier cutoff.
- A mask survives a depth-2 stack.
- `cutoff_time=None` masks nothing.
- Each of the six checks, raising and passing.
- The self-mask warning fires when the column is absent and stays silent when
  the user lists it.
- Both new plotting phrases, under `columns=True` and `"structural"`.
- Differential against featuretools' `set_secondary_time_index` for the
  all-null-fallback case, which is the only case featuretools can express.

## Out of scope

- Per-row cutoff frames. `cutoff_time` remains one global value.
- Reconstructing a column's history from an audit or slowly-changing-dimension
  table. tusk masks the value it was given; it does not recover the old one.
- Detecting undeclared in-place updates. Nothing in the schema distinguishes a
  column that is overwritten from one that is not.
