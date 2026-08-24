# scikit-learn transformers: DFS as a pipeline step

Status: draft — pending review
Date: 2026-08-24

## Problem

tusk produces a feature matrix. Everything downstream of that — encoding,
selection, model fitting, cross-validation — is scikit-learn's job, and there
is currently no way to hand the one to the other without a manual step in
between. A user who wants DFS inside a `Pipeline` has to compute the matrix
themselves, collect it, align it with their targets, and give up any hope of
`GridSearchCV` touching a DFS parameter.

featuretools solved this with
[`featuretools-sklearn-transformer`](https://github.com/alteryx/featuretools-sklearn-transformer),
and [PR #25](https://github.com/alteryx/featuretools-sklearn-transformer/pull/25)
extends it with feature *selection*: fit a selector on the computed matrix,
then prune the stored feature definitions so that later `transform` calls
compute only the features that survived. That pruning is the point. A DFS run
that generates 800 features and keeps 40 should compute 40 at inference time,
not 800.

Porting that idea to tusk is not a translation exercise, because upstream's
pruning rests on a property tusk does not have and does not want. featuretools'
`encode_features` returns *feature definitions* for the one-hot columns it
creates, so the selector's support mask and the feature list stay 1:1 and the
mask maps straight back. tusk has no encoding primitives and will not grow
any — encoding is the user's choice of sklearn transformer, so that they keep
full control of the strategy. The moment encoding is an sklearn object, the
link between "column the selector kept" and "feature tusk should compute"
is severed, and re-establishing it is the central design problem here.

## Decision

Add `src/tusk/sklearn.py` with two estimators:

- **`DFSTransformer`** — deep feature synthesis as a pipeline step, nothing more.
- **`DFSSelectorTransformer`** — the same, plus a user-supplied encode-and-select
  pipeline whose decision is mapped back onto tusk features so that later
  `transform` calls compute only the survivors.

Upstream's middle class, `DFSSelectionTransformer`, is **not** ported. It wraps
`featuretools.selection.remove_highly_null_features` and
`remove_single_value_features`; tusk has no such helpers and is not growing
them. Null-ratio and single-value filtering are `VarianceThreshold` and
friends, and belong in the selector the user supplies.

### Public surface

| Addition | Meaning |
| --- | --- |
| `src/tusk/sklearn.py` | New module. Not imported by `tusk/__init__.py`. |
| `sklearn.DFSTransformer` | DFS as a `TransformerMixin`. |
| `sklearn.DFSSelectorTransformer` | DFS plus lineage-aware feature pruning. |
| `exceptions.LineageWarning(UserWarning)` | Lineage could not be recovered; nothing was pruned. |
| `exceptions.LineageError(TuskError)` | The post-refit lineage invariant was violated. |
| `[project.optional-dependencies] sklearn` | `tusk[sklearn]`, pulling `scikit-learn`. |

Nothing is added to `tusk.__all__`, and `tusk/__init__.py` does not import the
module. Importing sklearn eagerly would break every user who installed tusk
without the extra, which is most of them. The estimators are reachable only as
`from tusk.sklearn import DFSTransformer`, matching `xgboost.sklearn` and
`lightgbm.sklearn`.

## `X` is the key column, the database is metadata

The binding decision. sklearn requires `X` to be row-aligned with `y` and
sliceable; a `Database` is neither. Upstream passes the `EntitySet` as `X` and
accepts the consequences. tusk inverts it:

```python
sklearn.set_config(enable_metadata_routing=True)

pipe.fit(train_ids, y, database=db)
pipe.predict(test_ids, database=db_test)
```

`X` is the target table's primary-key column — a 1-D sequence, an `(n, 1)`
array, or a one-column frame. The database travels as routed metadata,
declared at class level so users never call `set_fit_request` themselves:

```python
__metadata_request__fit = {"database": True}
__metadata_request__transform = {"database": True}
```

Three properties follow, and together they are why this shape was chosen over
the upstream one:

- **`y` misalignment becomes impossible.** The keys *are* `X`, so `transform`
  reorders the matrix onto them and returns rows in exactly that order. `X` and
  `y` are the same length by construction, and sklearn's own CV splitters slice
  them together.
- **`cross_val_score` and `GridSearchCV` work**, including searching
  `dfs__max_depth` and primitive sets. With a `Database` as `X` sklearn refuses
  outright: `InvalidParameterError: 'X' must be array-like`.
- **Inference compute is bounded.** The key list pushes into the query as a
  filter, so predicting on three rows never materializes the whole target table.

### Why the database cannot be a constructor parameter

`clone` deep-copies non-estimator constructor parameters, and duckdb relations
are not picklable:

```
TypeError: cannot pickle '_duckdb.DuckDBPyRelation' object
```

`GridSearchCV` clones before every fit, so a `database=` constructor argument
would hard-crash on a backend tusk supports and benchmarks on. It has to be
metadata. `cutoff_time`, being a scalar datetime, is clone-safe and stays a
constructor parameter — and is therefore searchable.

### The scorer fallback

Metadata reaches `fit` and `transform`, but sklearn's scorers call
`estimator.predict(X_test)` with no metadata at all, so CV scoring would
otherwise transform against `database=None` and return `nan` for every fold.
`fit` therefore records `self.database_`, and `transform` falls back to it when
no database is routed. Passing `database=` explicitly still overrides, which is
what makes predicting against a *different* database work.

### The `fit_transform` override

`TransformerMixin.fit_transform` does not forward metadata to `transform`;
sklearn warns about exactly this. Both classes override it:

```python
def fit_transform(self, X, y=None, database=None, **kwargs):
    return self.fit(X, y, database=database).transform(X, database=database)
```

### No `**kwargs` in `__init__`

Upstream's `DFSTransformer` takes `**dfs_kwargs` and hand-writes `get_params`
to compensate, which is what breaks `clone`. Every parameter here is declared
explicitly, so `BaseEstimator` supplies `get_params`, `set_params`, `clone` and
`GridSearchCV` support for free. `DFSSelectorTransformer` repeats its parent's
parameters rather than inheriting them through `**kwargs`. The duplication is
the price of the sklearn contract and is paid deliberately.

## `DFSTransformer`

```python
DFSTransformer(
    target_table,
    agg_primitives=None,
    trans_primitives=None,
    groupby_trans_primitives=None,
    max_depth=2,
    cutoff_time=None,
    output_backend=None,
)
```

`fit` calls `synthesize` and nothing else. Synthesis reads the schema and never
touches a row, so fitting feature *definitions* is free; the expensive pass
happens in `transform`. It sets `self.features_` and `self.database_`.

`transform`:

1. resolve the database — routed, else `self.database_`;
2. `apply_features(self.features_, database, self.cutoff_time)`;
3. filter to the keys in `X`, collect (see `output_backend`);
4. reorder onto `X`;
5. drop the primary key and return the feature columns.

Step 4 **raises** on a key that produced no row, whether because it is absent
from the table or because `cutoff_time` filtered it out. Returning fewer rows
than `y` has is the precise failure this design exists to prevent, so it is
never silent. Duplicate keys raise for the same reason.

The primary key is dropped from the output: it is a join key, not a feature,
consistent with `Database.output_excluded_columns`.

`get_feature_names_out` returns the flattened `output_names` of `self.features_`
in column order. Multi-output primitives contribute more than one name, so the
matrix is wider than `len(features_)`, and the two must not be conflated.

## `DFSSelectorTransformer`

Takes everything `DFSTransformer` takes, plus `inner`: a user-supplied
transformer whose last step is a `SelectorMixin`.

```python
Pipeline([
    ("dfs", DFSSelectorTransformer(
        target_table="customers",
        inner=Pipeline([
            ("enc", OneHotEncoder(handle_unknown="ignore")),
            ("sel", SelectKBest(k=50)),
        ]),
    )),
    ("clf", ExtraTreesClassifier()),
])
```

`inner` is validated in `fit`, not `__init__`, per sklearn convention. A bare
`SelectorMixin` with no encoder is supported; the encoder prefix is then the
identity.

### Two column spaces

The design turns on keeping these separate, and conflating them is the bug this
section exists to prevent:

| Space | Produced by | Indexed by |
| --- | --- | --- |
| **tusk space** | `apply_features`, `M` columns | feature `output_names` |
| **encoded space** | the encoder prefix, `E` columns | `get_feature_names_out()` |

The selector's support mask lives in **encoded** space. Pruning happens in
**tusk** space. `support` can never index the tusk matrix.

### `fit`

```python
matrix  = self._compute(X, database)              # tusk space, M columns
probe   = matrix.rename(sentinels)                # _t<rand>_0000, _t<rand>_0001, ...
inner   = clone(self.inner)
inner.fit(probe, y)

encoded = inner[:-1].get_feature_names_out()      # E names, sentinel-bearing
support = inner[-1].get_support()                 # E bools
kept    = encoded[support]

self.features_    = prune(self.features_, lineage(kept))
surviving_columns = flattened output_names of self.features_   # tusk space
self.encoder_     = encoder_prefix(clone(self.inner))
self.encoder_.fit(matrix[surviving_columns].rename(same_sentinels))
refit   = self.encoder_.get_feature_names_out()

if not set(kept) <= set(refit):
    raise LineageError(...)
self.kept_names_ = [name for name in refit if name in kept]
```

`surviving_columns` is named rather than written `matrix[surviving]` on
purpose: it indexes **tusk** space. `support` indexes encoded space and can
never index `matrix`.

Note `inner[:-1]`: on a pipeline ending in a selector,
`inner.get_feature_names_out()` returns the names *after* selection. The
pre-selection encoded names come from the prefix. `encoder_prefix` is
`inner[:-1]` when `inner` is a `Pipeline`, and a passthrough identity when
`inner` is a bare `SelectorMixin` — the latter does not support slicing.

The refit reuses the **identical** sentinel mapping on the surviving subset.
That is what keeps the recorded `kept` names matchable afterwards.

### The selector is fitted once and frozen

The selector is never refitted. Its decision is recorded as names and replayed
as a static mask. Refitting it on the pruned matrix would let it make a
*different* decision — `SelectKBest(k=50)` choosing 50 from 60 pruned columns
need not choose the same 50 it chose from 500, and a randomized selector like
`RFE` over a forest would diverge further. Freezing removes the question of
which round won.

This costs nothing at inference: a fitted `SelectorMixin.transform` is a column
mask either way. Only the encoder prefix is refitted, and only at fit time, on
a matrix already collected and narrower than the one the first fit saw.

### `transform`

```python
matrix  = self._compute(X, database)              # pruned features only
encoded = self.encoder_.transform(matrix.rename(sentinels))
return select(encoded, self.kept_names_)
```

`self.kept_names_` is stored as names, not as a positional boolean mask,
because `output_backend=None` means `encoded` may be a named frame (polars,
pandas) or a bare ndarray depending on the backend and on the user's
`set_output`. `select` resolves names against `self.encoder_`'s
`get_feature_names_out()` — the authoritative order for a fitted encoder — and
indexes positionally from that. Storing positions directly would silently
mis-slice if the encoder's output container changed between fit and transform.

Nothing is fitted here. Both fits happen inside `fit`; sklearn calls
`fit_transform` on intermediate steps during `Pipeline.fit` and `transform`
during `Pipeline.predict`, never `fit`.

`get_feature_names_out` substitutes sentinels back to real names, so the user
reads `oh__MODE__transactions__category_a`, never `oh___t9f3a_0001_a`.

## Lineage by sentinel names

The mechanism that reconnects encoded space to tusk space. Before handing the
matrix to `inner`, tusk renames its columns to opaque tokens with a random
per-fit prefix, then reads them back out of the encoder's output names:

```python
matrix.columns = ["_t9f3a_0000", "_t9f3a_0001", "_t9f3a_0002"]
for name in encoder.get_feature_names_out():
    sources = pattern.findall(name)          # 0, 1, or many
```

sklearn's convention is that output names derive from input names, so this
recovers lineage through `ColumnTransformer`, nested `Pipeline`s,
`OneHotEncoder`, `OrdinalEncoder`, scalers, `passthrough`, and multi-input
transformers. `PolynomialFeatures` reports `_t9f3a_0000 _t9f3a_0002` and is
correctly attributed to both sources — hence *all* matches are collected, not
just the first.

A tusk feature survives if any of its matrix columns is a source of any kept
encoded column. Feature order is preserved.

### Why not read `ColumnTransformer.output_indices_`

The structural alternative — require a `ColumnTransformer` and read
`output_indices_` against each sub-transformer's declared columns — is
*coarser* on exactly the pipelines it was meant to serve. A
`("num", StandardScaler(), [200 numeric columns])` entry is one block: keeping
one output would keep all 200 features. Sentinels give per-column precision on
the same pipeline, because `ColumnTransformer` embeds the input column name in
each output name (`num___t9f3a_0000`). It also imposes no structure on what the
user may supply.

### Failure modes, and why they are safe

Two things sentinels cannot do. Both are detected.

**Opaque output names.** `PCA` produces `pca0`, mentioning no sentinel. Lineage
is unknown, and known to be unknown: tusk keeps *all* features and emits
`LineageWarning`. Pruning is an optimization and correctness never depends on
it, so the degenerate case is simply the un-pruned one.

**Sentinel collision.** A categorical *value* that happens to look like a
sentinel is attributed as an extra source. This is verified to occur: a value
of `_t9f3a_0000` in a one-hot column yields `oh___t9f3a_0001__t9f3a_0000`,
matching two sentinels where one is real. The result is a **superset** — a
feature is spuriously kept, costing compute and never correctness. A random
per-fit prefix makes it vanishingly unlikely regardless.

The dangerous direction — a feature wrongly *pruned* — is closed off by the
invariant after the refit: every name the frozen mask needs must exist in the
refitted encoder's output. If lineage missed a source, a required name goes
missing and `fit` raises rather than serving wrong columns.

## `output_backend`

narwhals collects a duckdb-backed database to a pyarrow `Table` by default, and
sklearn's `ColumnTransformer` rejects pyarrow outright:

```
TypeError: Index must either be string or integer
```

polars and pandas both work. So the collect target is a choice, and the default
is `None`, meaning **collect natively**. A growing set of sklearn-compatible
transformers — scikit-lego among them — are narwhals-native and are best served
the frame the database already uses, with no conversion.

`tusk[sklearn]` therefore depends on `scikit-learn` alone. It does not pull
pandas.

Setting `output_backend="pandas"` without pandas installed raises a tusk error
naming the fix. And any exception raised out of the inner pipeline is re-raised
`from e`, original traceback intact, with a note giving the backend the matrix
was collected as and suggesting `output_backend="pandas"`.

The hint attaches **at the point of failure**, not as a warning on every fit. A
proactive warning would fire on every pyarrow run, including all the ones that
work.

## Errors and warnings

| Condition | Behaviour |
| --- | --- |
| Key in `X` produced no row | raise |
| Duplicate key in `X` | raise |
| `inner` does not end in a `SelectorMixin` | raise, in `fit` |
| Kept column mentions no sentinel | keep all features, `LineageWarning` |
| `kept ⊄ refit` after encoder refit | raise `LineageError` |
| Selection eliminated every feature | raise |
| Inner pipeline raised | re-raise `from e` with backend hint |
| `output_backend` package missing | raise, naming the install |

## Testing

Behaviour and contracts, not source text.

- **Lineage recovery**, parametrized over `ColumnTransformer`, `OneHotEncoder`,
  nested `Pipeline`, `PolynomialFeatures` (multi-source), and `PCA` (opaque →
  warns, keeps all).
- **Pruning is real**: assert a pruned feature's column is absent from the
  compiled plan, not merely dropped from the output.
- **The frozen mask**: fit, then confirm `transform`'s columns equal the round-1
  selection, and that no selector is refitted.
- **Train/serve**: fit on one database, predict on another via routed metadata.
- **sklearn integration**: `cross_val_score` and `GridSearchCV` over
  `dfs__max_depth` both run and produce finite scores.
- **Backends**: polars and duckdb; the pyarrow failure carries the hint.
- **Row alignment**: a missing key and a duplicate key each raise.

Gated on `pytest.importorskip("sklearn")`, with `scikit-learn` added to the dev
group.

## Open items

- **sklearn floor.** The dependency on metadata routing through
  `Pipeline.predict` → `transform` is verified on 1.6.1. Pin the floor at the
  oldest version that passes a routing test rather than guessing; start at
  `>=1.4` and raise it if that fails.
