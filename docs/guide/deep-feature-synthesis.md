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

tusk builds one query plan. The matrix comes back on the backend you put in, as
a lazy frame where that backend has one, so you decide when to compute it, the
way that backend computes anything.

On a backend with no separate lazy type, such as pandas or pyarrow, you already
have your frame. [Validation](databases.md#validation) is the one part of tusk
that computes on its own, and only for the checks you ask for.

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
features in a `FeatureList`, and so all features passed to
[`apply_features()`][tusk.apply_features], must share one target table.

## Cutoff times

`cutoff_time` is **one global value**, not a per-row frame. Only rows whose
`row_creation_time` is at or before it are visible.

It must be a `datetime`, not just a `date`. A date has no time of day, and
therefore risks differing behaviors across dataframe backends.

Its time zone awareness must match every `Datetime` column in the database,
including the ones beyond the declared row creation times. A comparison between
a tz-aware timestamp and a naive one has no defined ordering, so a mismatch
raises [`ValidationError`][tusk.exceptions.ValidationError] up front, for the
whole database. The cutoff reaches columns beyond the ones it filters on:
`TimeSince` subtracts it from whatever column its feature was built on.
`validate(database="consistent_time_zones")` reports the same mixing without a
cutoff.

It filters the target table too, so the matrix can have fewer rows than the
target. A row that did not exist yet at the cutoff has no features to
compute.

Tables with no `row_creation_time` are timeless and pass through unfiltered, so
a cutoff on a database that declares none is silently a no-op.

A cutoff also reaches inside a row. A table that declares
[`row_update_times`](databases.md#row-update-times) gives back the earlier value
of every column filled in after the cutoff, so a taxi fare settled at the end of
a ride is not visible from a cutoff taken while the ride was still running. This
happens before any join or aggregation, so aggregated and stacked features see
those earlier values too.

With `features_only=True` the cutoff is ignored entirely, since nothing is
computed and feature definitions do not record it.

## Warnings

DFS skips what it cannot build and warns instead of failing:

- [`UnmatchedPrimitiveWarning`][tusk.exceptions.UnmatchedPrimitiveWarning]: a
  requested primitive matched no column of its input dtypes anywhere in the
  walk, so it contributed no features.
- [`CategoricalDtypeWarning`][tusk.exceptions.CategoricalDtypeWarning]: a
  Categorical or Enum column was skipped because the primitive requires a
  string input.

Both have their own class so you can filter them independently.
