# Running DFS

[`deep_feature_synthesis()`][tusk.deep_feature_synthesis] walks the database
from a target table, stacking primitives up to `max_depth`, and returns a
`(feature_matrix, features)` tuple.

```python
feature_matrix, features = tusk.deep_feature_synthesis(
    database=db,
    target_table="customers",
    agg_primitives=["mean", "count"],
    trans_primitives=["month", "weekday"],
    max_depth=2,
)
```

The matrix has one row per *visible* row of the target table, keyed by its
primary key. The target must have a `primary_key`; without one, compilation
raises [`SchemaError`][tusk.exceptions.SchemaError].

## Lazy out, always

tusk builds one query plan and **never collects**. Whatever you feed it — a
`pl.DataFrame`, a `pl.LazyFrame`, a pandas frame, a duckdb relation — the
matrix comes back uncomputed, in your backend's native type, and you decide
when to compute:

```python
matrix = feature_matrix.collect()
```

On a backend with no separate lazy type, such as pandas or pyarrow, there is
nothing to collect and you already have your frame. The only `collect()` calls
in tusk are the ones [validation](databases.md#validation) makes when you ask
for a check by name.

## Definitions without computation

`features_only=True` returns just the [`FeatureList`][tusk.FeatureList] of
[`Feature`][tusk.features.Feature] definitions, skipping compilation entirely.
Synthesis raises [`SchemaError`][tusk.exceptions.SchemaError] instead if the
walk generates no features at all. Feed the definitions back later:

```python
features = tusk.deep_feature_synthesis(db, "customers", features_only=True)
matrix = features.apply(db_new)
```

This is how you apply a feature set fitted on training data to new data. All
features in a `FeatureList` — and so all features passed to
[`apply_features()`][tusk.apply_features] — must share one target table.

A `FeatureList` indexes, iterates and slices like a list, and a slice is
itself a `FeatureList`, so `features[:10].apply(db)` works. It cannot be
empty: `features[:0]`, or any slice past the end, raises
[`SchemaError`][tusk.exceptions.SchemaError] rather than returning nothing.

## Cutoff times

`cutoff_time` is **one global value**, not a per-row frame. Only rows whose
`row_creation_time` is at or before it are visible.

It must be a `datetime`, not just a `date`. A date has no time of day, and
therefore risks differing behaviors across dataframe backends.

Its time zone awareness must match the database's row creation times. A
comparison between a tz-aware timestamp and a naive one has no defined
ordering, so a mismatch raises
[`ValidationError`][tusk.exceptions.ValidationError].

It filters the target table too, so the matrix can have fewer rows than the
target — a row that did not exist yet at the cutoff has no features to compute.

Tables with no `row_creation_time` are timeless and pass through unfiltered, so
a cutoff on a database that declares none is silently a no-op.

With `features_only=True` the cutoff is ignored entirely, since nothing is
computed and feature definitions do not record it.

## Feature names are SQL identifiers

Featuretools writes `MEAN(transactions.amount)`; tusk writes
`MEAN__transactions__amount`. On a backend that generates SQL, dots and
parentheses parse as table qualifiers and function calls rather than as part of
a name, so the conventional form is unusable there. Every construct joins with
`__`:

| meaning | column name |
| --- | --- |
| a parent's column | `customers__age` |
| an aggregation | `MEAN__transactions__amount` |
| a zero-arity aggregation | `COUNT__transactions` |
| stacked to depth 2 | `MEAN__sessions__MEAN__transactions__amount` |
| a grouped transform | `CUM_SUM__amount__by__session_id` |
| one output of a multi-output primitive | `QUANTILES__amount__0` |

The conventional form is kept on
[`Feature.display_name`][tusk.features.Feature.display_name] for logs, docs and
error messages. If a source column name collides with a generated one,
`apply_features` raises rather than silently dropping a column.

## Warnings

DFS reports what it quietly skipped rather than failing:

- [`UnmatchedPrimitiveWarning`][tusk.exceptions.UnmatchedPrimitiveWarning] — a
  requested primitive matched no column of its input dtypes anywhere in the
  walk. Skipping is correct, since raising would break a zero-configuration
  `deep_feature_synthesis()` on any schema that happens to lack a dtype family; skipping
  *silently* is not.
- [`CategoricalDtypeWarning`][tusk.exceptions.CategoricalDtypeWarning] — a
  Categorical or Enum column was skipped because the primitive requires a
  string input.

Both have their own class so you can filter them independently.
