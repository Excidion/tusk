# TableEncoder

Replace `dtype_selector` with `TableEncoder`, a transformer that encodes each
column of a feature matrix by its narwhals dtype. It resembles skrub's
`TableVectorizer`, but the column's dtype alone chooses the encoder. There is
no cardinality switch: the user picks the encoding of a text column by casting
it to `String`, `Categorical` or `Enum`.

`NarwhalsMixin` makes any scikit-learn estimator accept a table narwhals can
read. The tusk encoders and `TableEncoder` use it too.

## Why

`dtype_selector` picks columns by `DtypeFamily`. The families were built to
decide which primitives apply to a column, so they overlap (`Datetime` is in
`has_date` and `has_time`). An encoder must give each column one owner. Users
also still build a `ColumnTransformer` by hand, which fails on pyarrow, the
backend a duckdb database collects to.

## Components

All components live in `tusk.sklearn` and are public. They need the `sklearn`
extra.

### `NarwhalsMixin`

`NarwhalsMixin` makes an estimator accept any table narwhals can read, eager
or lazy, on any backend. It is mixed in before the estimator's class:

```python
class PandasTargetEncoder(NarwhalsMixin, TargetEncoder):
    convert_to = "pandas"
```

- The class attribute `convert_to: Literal["narwhals", "numpy", "pandas",
  "polars"] = "numpy"` sets what the estimator underneath receives. It is a
  class attribute, not an `__init__` parameter: scikit-learn reads parameters
  from the `__init__` signature, which a mixin cannot extend without hiding
  the estimator's own. A class attribute also survives `clone`.
- `fit`, `transform` and `fit_transform` convert `X` to `convert_to`, then
  call the estimator's own method. `fit_transform` is converted too, because
  some estimators define their own, `TargetEncoder` among them.
- `"numpy"`, `"pandas"` and `"polars"` collect a lazy table first.
  `"narwhals"` hands over the narwhals table unchanged, lazy or eager.
- At fit the mixin records the column names as `narwhals_columns_`.
  `get_feature_names_out(input_features=None)` falls back to them, so an
  estimator fitted on numpy still reports the real names. The mixin does not
  set `feature_names_in_`: scikit-learn would then warn at every `transform`
  that numpy input has no feature names.
- Input narwhals cannot read, such as a numpy array from an earlier step,
  reaches the estimator unchanged.
- If `convert_to` names pandas or polars and the package is not installed, it
  raises (see [Errors](#errors)). Neither package is a dependency of the
  `sklearn` extra.

The output is scikit-learn's own `set_output`. The mixin does not change it.

### `NarwhalsEncoder`

The base class of the tusk encoders and `TableEncoder`, and of any
narwhals-native encoder a user writes:
`NarwhalsEncoder(NarwhalsMixin, TransformerMixin, BaseEstimator)` with
`convert_to = "narwhals"`.

- `fit(X, y=None)` reads the schema with `collect_schema()` and does not
  collect. It sets `feature_names_in_`, `n_features_in_` and `schema_in_`
  (column name to narwhals dtype). Here `feature_names_in_` is safe to set,
  because `transform` receives a table with names.
- Subclasses override `_fit(table, y)`, `_expressions()`, `_output_names()`
  and, if they need values, `_transform(table)`. `_fit` and `_transform`
  receive the table as given, lazy or eager; `transform` collects the result
  of `_transform`. The default `_transform` selects `_expressions()`, and the
  default `_expressions()` selects every input column, so the base class
  returns the table unchanged.
- `set_output(transform=...)` accepts `"default"` (numpy), `"pandas"` and
  `"polars"`, the values scikit-learn accepts. The base class converts the
  narwhals result itself instead of through scikit-learn's output wrapper. A
  polars `Categorical` column then becomes a pandas `category` column, not an
  object array. `Pipeline.set_output` reaches it like any other step.
- `get_feature_names_out()` returns the output column names.

Used alone, `NarwhalsEncoder()` is a convert step. It passes every column
through unchanged, and `set_output` sets what it returns:

```python
Pipeline([
    ("dfs", DFSTransformer(target_table="customers")),
    ("convert", NarwhalsEncoder().set_output(transform="pandas")),
    ("model", HistGradientBoostingClassifier()),
])
```

It fits on the schema alone, so `TableEncoder` accepts it as a group's
estimator, for example to pass a group through in a chosen form.

### Column encoders

Each column encoder is a `NarwhalsEncoder` subclass that encodes every
column it is given. Each rejects a column of a dtype it cannot encode (see
[Errors](#errors)). Output names are `{column}_{suffix}`, so the input column
name is always part of the output name.

| Encoder | Input dtype | Output per column | Fits on |
| --- | --- | --- | --- |
| `EnumEncoder` | `Enum` | position in the category order, as `Float64`; null → NaN | schema |
| `DateEncoder` | `Date` | one column per component | schema |
| `TimeEncoder` | `Time` | one column per component | schema |
| `DatetimeEncoder` | `Datetime` | one column per component | schema |
| `DurationEncoder` | `Duration` | one column per component | schema |
| `StringEncoder` | `String` | 30 `Float64` columns | values |

`EnumEncoder` reads the categories from the dtype at fit. pandas
`Categorical(ordered=True)` reaches narwhals as `Enum`, so it is encoded the
same way. Output name: `{column}_code`.

The four temporal encoders subclass the public base `TemporalEncoder`, which
holds the shared logic. A subclass sets the class attributes
`accepted_dtype` and `allowed_components`. They take `components:
Sequence[str]`. The allowed
components are the numeric results of narwhals' `dt` namespace:

| Encoder | Allowed | Default |
| --- | --- | --- |
| `DateEncoder` | `year`, `month`, `day`, `weekday`, `ordinal_day`, `timestamp` | `month`, `day` |
| `TimeEncoder` | `hour`, `minute`, `second`, `millisecond`, `microsecond`, `nanosecond` | `hour`, `minute` |
| `DatetimeEncoder` | the `DateEncoder` and `TimeEncoder` components | `month`, `day`, `hour`, `minute` |
| `DurationEncoder` | `total_minutes`, `total_seconds`, `total_milliseconds`, `total_microseconds`, `total_nanoseconds` | `total_seconds` |

They compute the components as narwhals expressions on the native backend.
Output name: `{column}_{component}`.

`StringEncoder(n_components=30)` encodes each column separately:

1. Fill null with `""`.
2. `TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))`: each value
   becomes a TF-IDF vector over the character 3- and 4-grams seen at fit.
3. `TruncatedSVD(n_components)`: each vector becomes its coordinates along
   the 30 directions of highest variance.

If the vocabulary has fewer n-grams than `n_components`, the SVD keeps as
many components as it can, and zero columns fill the rest. If the vocabulary
is empty (every value null or `""`), every output column
is zero. The output width is always `n_components`. Output name:
`{column}_svd_{i}`.

Schema-only encoders carry the class attribute `_fits_on_schema = True`.

Backend limits, found by the prototype:

- pandas has no `Date` or `Time` dtype. A pandas date column reads as
  `Datetime`, a time column as `Object` (group `other`).
- duckdb's `TIME` reads as narwhals `Unknown` (group `other`).
- duckdb raises `NotImplementedError` for the `timestamp` and
  `total_nanoseconds` components. Neither is a default.
- A pyarrow dictionary column reads as `Categorical`, not `Enum`.

### `TableEncoder`

`TableEncoder(NarwhalsEncoder)` has one parameter per dtype group. Each
takes an estimator, `"passthrough"` or `"drop"`. The groups are disjoint, so
each column has exactly one owner.

| Parameter | narwhals dtypes | Default |
| --- | --- | --- |
| `numeric` | `Int*`, `UInt*`, `Float*`, `Decimal` | `"passthrough"` |
| `boolean` | `Boolean` | `"passthrough"` |
| `string` | `String` | `StringEncoder()` |
| `categorical` | `Categorical` | `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` |
| `enum` | `Enum` | `EnumEncoder()` |
| `date` | `Date` | `DateEncoder()` |
| `time` | `Time` | `TimeEncoder()` |
| `datetime` | `Datetime` | `DatetimeEncoder()` |
| `duration` | `Duration` | `DurationEncoder()` |
| `other` | `Binary`, `List`, `Array`, `Struct`, `Object`, `Unknown` | `"drop"` |

The defaults are estimator instances in the signature, as in skrub, so
nested parameters such as `date__components` work with `set_params`. Every
`TableEncoder` shares these instances. `fit` clones them, and `set_params`
clones a shared default before it sets a nested parameter on it, so no
`TableEncoder` changes another's defaults.

An estimator with `NarwhalsMixin` receives the narwhals table and converts
it to its own `convert_to`. Any other estimator receives numpy, and
`TableEncoder` passes the column names to
`get_feature_names_out(input_features=columns)`. To give one group pandas,
the user mixes `NarwhalsMixin` into that group's estimator.

`fit`:

1. Read the schema. Cast `Decimal` columns to `Float64`.
2. Split the columns into the groups. A group with no columns is skipped.
3. Clone each group's estimator.
4. Fit each estimator on its group's columns:
   - An estimator with `_fits_on_schema = True` receives the table as it
     came, lazy or eager. Nothing is collected.
   - Every other estimator receives its columns collected: as a narwhals
     table if it has `NarwhalsMixin`, else as numpy. This covers
     `StringEncoder`, `OneHotEncoder` and any estimator the user supplies.
5. Set `groups_`: group name to (fitted estimator, input columns).

`transform` builds one `select` on the native table. It holds:

- the expressions of the schema-only encoders, which expose them as
  `_expressions() -> list[nw.Expr]` after fit,
- the passthrough columns,
- the raw columns of the value-based groups.

It collects that `select` once. On duckdb, date parts and codes are then
computed in the database. Each value-based estimator transforms its columns,
converted as at fit. A sparse output is made dense. The blocks are
concatenated horizontally, by position. The result goes through
`NarwhalsEncoder`'s output conversion.

Order:

- Rows: every block comes from the one collected table, and every encoder
  maps row i to row i. Before the concatenation, `transform` checks that
  every block has the same row count.
- Columns: groups in the order of the parameter table above; within a group,
  the estimator's `get_feature_names_out()` order. Both are fixed at fit.

Output names are `{group}__{name}`, for example `date__signup_month`, the
same scheme as `ColumnTransformer(verbose_feature_names_out=True)`.

`TableEncoder` does its own dispatch instead of building a
`ColumnTransformer`. A `ColumnTransformer` needs the whole table eager and
cannot read pyarrow, which defeats a schema-only fit.

## Integration with `DFSSelectorTransformer`

- Lineage works unchanged: every output name contains the input column name,
  which `DFSSelectorTransformer` has replaced with a sentinel.
- `_require_feature_names` passes: every component implements
  `get_feature_names_out`.
- `_reject_explicit_columns` does not apply: `TableEncoder` is not a
  `ColumnTransformer`. Its message now names `TableEncoder` as the fix.
- Refitting on a narrowed feature matrix recomputes the groups.

## Removed

- `dtype_selector`, its tests and its section in `docs/guide/sklearn.md`.
- The `DtypeFamily` docstring sentence that mentions `dtype_selector`.
  `DtypeFamily` stays, because primitives use it.

## Errors

| Condition | When | Raises |
| --- | --- | --- |
| A group parameter is not an estimator, `"passthrough"` or `"drop"` | fit | `ValueError` naming the parameter and the value |
| A component is not allowed for the encoder | fit | `ValueError` listing the allowed components |
| A column encoder gets a column of a dtype it cannot encode | fit | `EncoderError` naming the column, its dtype and the accepted dtype |
| `transform` gets columns that differ from `feature_names_in_` | transform | `EncoderError` naming the extra and absent columns |
| A column's dtype differs from the dtype seen at fit, including an `Enum` with other categories | transform | `EncoderError` naming the column and both dtypes |
| `set_output` gets a value other than `"default"`, `"pandas"`, `"polars"` | set_output | `ValueError` from scikit-learn's own check |
| `convert_to` or `set_output` names pandas or polars and the package is not installed | fit, set_output | `TuskError`: "`PandasTargetEncoder` converts to pandas, which is not installed; `uv add pandas`" |
| A block has a different row count from the others | transform | `EncoderError` naming the group and both counts |

## Testing

Behaviour tests, on polars, pandas, pyarrow and duckdb where the backend
matters:

- `NarwhalsMixin`: mixed into `StandardScaler`, `OneHotEncoder` and
  `TargetEncoder`, it accepts polars, pandas, pyarrow and a lazy duckdb
  table; each `convert_to` reaches the estimator as that type;
  `get_feature_names_out` reports the real names after a numpy fit; a
  `transform` after a numpy fit issues no feature-name warning; `clone`
  keeps `convert_to`; an absent pandas raises `TuskError` (patch the import).
- `NarwhalsEncoder`: fit on a lazy duckdb table collects nothing (patch
  `collect` to raise); `set_output` returns numpy, pandas and polars; a
  polars `Categorical` becomes a pandas `category`; as a pipeline step
  between `DFSTransformer` on duckdb and a model, it hands the model the
  `set_output` type with the values unchanged.
- Each column encoder: the values for a small fixed input; the output names;
  null handling; an unrecognized component and a wrong dtype raise.
- `EnumEncoder`: codes follow the category order, not the order the values
  appear in; a pandas ordered categorical gets the same codes.
- `StringEncoder`: output width is `n_components` for a large vocabulary, a
  small one and an empty one; strings with shared n-grams are closer than
  strings without; a value first seen at transform is encoded.
- `TableEncoder`: each dtype reaches its group; defaults encode a table with
  every dtype; `"drop"` and `"passthrough"` work; a schema-only group does
  not collect at fit; an empty group is skipped; `get_feature_names_out` matches the output columns;
  a bare user estimator receives numpy, and one with `NarwhalsMixin`
  receives its `convert_to`;
  `transform` on a lazy duckdb table collects once;
  `clone` and `get_params`/`set_params` work, including nested parameters
  such as `date__components`.
- `DFSSelectorTransformer` with a `TableEncoder` followed by a selector, on
  duckdb: lineage keeps the right features, and the refit on the narrowed
  matrix succeeds.
- scikit-learn's `check_estimator` is not used: it feeds numpy arrays, which
  the column encoders reject by design.

## Docs

- `docs/guide/sklearn.md`: replace "Choosing columns with `dtype_selector`"
  with a `TableEncoder` section. It states the rule: cast a text column to
  `Categorical` for one-hot, to `Enum` for ordered codes, and leave it
  `String` for the n-gram encoding. Update the examples that use
  `dtype_selector`.
- `docs/api/sklearn.md` picks the new classes up from `tusk.sklearn`.
