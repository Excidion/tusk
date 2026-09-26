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

The feature matrix has one row per *visible* row of the target table, keyed
by its primary key. The target must have a `primary_key`. Without one,
compilation raises [`SchemaError`][tusk.exceptions.SchemaError].

## Lazy out, always

tusk builds one query plan. The feature matrix comes back on the backend you
give it, as that backend's native lazy table where it has one. You decide
when to compute it, the same way you compute anything else on that backend.

On a backend with no separate lazy type, such as pandas or pyarrow, you
already have your table. [Validation](databases.md#validation) is the only
part of tusk that computes on its own, and only for the checks you ask for.

## Definitions without computation

`features_only=True` returns only the [`FeatureList`][tusk.FeatureList] of
[`Feature`][tusk.features.Feature] definitions. It skips compilation
entirely. Synthesis raises [`SchemaError`][tusk.exceptions.SchemaError]
instead if the walk builds no features at all. Feed the definitions back
later:

```python
features = tusk.deep_feature_synthesis(db, "customers", features_only=True)
matrix = features.apply(db_new)
```

This is how you apply a feature set fitted on training data to new data. All
features in a `FeatureList`, and so all features passed to
[`apply_features()`][tusk.apply_features], must share one target table.

## Cutoff times

`cutoff_time` is **one global value**, not one value per row. Only rows
whose `row_creation_time` is at or before it are visible.

It must be a `datetime`, not just a `date`. A date has no time of day, so
its behavior can differ across backends.

Its time zone awareness must match every `Datetime` column in the database,
including the ones beyond the declared row creation times. A comparison
between a tz-aware timestamp and a naive one has no defined ordering. A
mismatch therefore raises
[`ValidationError`][tusk.exceptions.ValidationError] up front, for the whole
database. The cutoff reaches columns beyond the ones it filters on.
`TimeSince` subtracts it from its feature's input column.
`validate(database="consistent_time_zones")` reports the same mixing
without a cutoff.

It filters the target table too. The feature matrix can therefore have
fewer rows than the target. A row that did not exist yet at the cutoff has
no features to compute.

Tables with no `row_creation_time` are timeless and pass through unfiltered.
A cutoff on a database that declares none is therefore silently a no-op.

A cutoff also reaches inside a row. A table that declares
[`row_update_times`](databases.md#row-update-times) gives back the earlier
value of every column filled in after the cutoff. A taxi fare settled at the
end of a ride is therefore not visible from a cutoff taken while the ride
was still running. This happens before any join or aggregation. Aggregated
and stacked features therefore see those earlier values too.

With `features_only=True`, synthesis ignores the cutoff entirely. The
compiler never runs, and feature definitions do not record it.

## Warnings

DFS skips what it cannot build and warns instead of failing:

- [`UnmatchedPrimitiveWarning`][tusk.exceptions.UnmatchedPrimitiveWarning]: a
  requested primitive matched no column of its input dtypes anywhere in the
  walk. It therefore contributed no features.
- [`CategoricalDtypeWarning`][tusk.exceptions.CategoricalDtypeWarning]: DFS
  skipped a Categorical or Enum column because the primitive requires a
  string input.

Both have their own class so you can filter them independently.
