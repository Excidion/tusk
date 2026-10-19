# DFSTransformer feature matrix cache

Issue: Excidion/tusk#37, "Caching of sklearn DFS transformers to speedup
gridsearch".

`DFSTransformer` computes the feature matrix in every `fit_transform` and every
`transform`. A grid search does that once per candidate and fold, although the
matrix only changes with the synthesized features, the database and the cutoff
time. This spec adds an opt-in cache that computes the matrix once for every
visible row of the target table, and serves each call by selecting its keys.

## Why not `Pipeline(memory=...)`

joblib keys that cache on a hash of the routed `database`. For polars the hash
follows the data. For a duckdb database it does not: two databases with
different values hash the same, so a cache directory can serve a stale matrix.
`Pipeline(memory=)` also caches `fit_transform` only. Each validation fold still
runs `transform` against the database, so on a live database (for example
ibis) the train folds would be frozen and the validation folds would not.

The cache in this spec freezes one matrix and serves both.

## Parameter

```python
DFSTransformer(target_table, ..., feature_matrix_cache=None)
DFSSelectorTransformer(target_table, ..., feature_matrix_cache=None)
```

`feature_matrix_cache` is `None` (no caching, the current behavior) or a
`FeatureMatrixCache`, exported from `tusk.sklearn`:

```python
cache = FeatureMatrixCache()
search = GridSearchCV(Pipeline([("dfs", DFSTransformer("customers",
    feature_matrix_cache=cache)), ("clf", clf)]), grid)
```

### Why an object, not a boolean

`GridSearchCV` clones the estimator for every fit, so state on the transformer
is lost between folds. `clone` copies a parameter with `copy.deepcopy`.
`FeatureMatrixCache.__deepcopy__` returns the same object, so every clone
shares one cache.

## `FeatureMatrixCache`

A mapping from a key to an eager feature matrix.

- The key is `(id(database), cutoff_time, output_names, output_backend)`. The
  cache holds a reference to each database it has seen, so the `id` stays
  valid.
- `output_names` is `tuple(features.output_names)`. Two feature lists with the
  same output names build the same matrix.
- The matrix holds every row of the target table visible at the cutoff time,
  with the primary key column, as an eager narwhals table.
- Each process has its own copy. Under `n_jobs > 1` with a process-based joblib
  backend the object is pickled into each worker, so workers do not share it.
  The threading backend does. The guide says so.
- `clear()` drops every matrix.
- The cache does not expire. It never detects that the database changed. The
  user builds a new cache to get fresh values.

## Behavior

`DFSTransformer.transform` with a cache:

1. `check_keys_are_visible` runs as today, so a stale key fails before any
   computation.
2. The cache returns the matrix for the key, or computes it: `features_.apply`
   for the database and cutoff time, collected to `output_backend`, with no key
   filter.
3. The rows for `X` are selected and put in key order with the code that
   `collect_feature_matrix` uses today. That code takes a matrix and keys and
   does no computation, so it is split from the collection.

`fit` is unchanged. It reads no rows.

`DFSSelectorTransformer.fit` calls `transform` through its parent, so its fit
reads the cache too. Its own encode and select steps run per fold and are not
cached: they depend on `y`.

With `feature_matrix_cache=None` nothing changes.

## Memory

The cached matrix has one row per visible row of the target table, not one per
key in `X`. The guide says so, and says to leave the cache off when the
training keys are a small share of the table.

## Errors

- `SchemaError` for duplicate keys and for a key with no visible row keeps its
  current message.
- A `feature_matrix_cache` that is neither `None` nor a `FeatureMatrixCache`
  raises `TypeError` at `fit`: "feature_matrix_cache must be None or a
  FeatureMatrixCache".

## Testing

Tests assert behavior, not source text.

- Two `transform` calls with the same cache compute the matrix once. The test
  counts calls with a table whose column computation records each run.
- The result with a cache equals the result without one, for the same keys, in
  key order, including after another call with different keys.
- A grid search over a non-DFS parameter computes the matrix once. One over
  `dfs__max_depth` computes it once per depth.
- A different cutoff time and a different database each get their own matrix.
- A key that is not visible at the cutoff time raises before the matrix is
  computed.
- Validation folds see the frozen matrix: change the underlying table after the
  first call and the second call returns the old values. This is the contract
  the cache exists for.
- `DFSSelectorTransformer` with a cache equals one without.
- `clone` of a transformer keeps the same cache object.
- A non-cache value raises `TypeError`.

## Docs

`docs/guide/sklearn.md` gets a "Speeding up a grid search" section: the cache
example, what freezes, the memory cost, and the process-backend limit.
`docs/api/sklearn.md` lists `FeatureMatrixCache`.

## Out of scope

- Disk persistence.
- Detecting a changed database.
- Caching the selector's encoder output.
