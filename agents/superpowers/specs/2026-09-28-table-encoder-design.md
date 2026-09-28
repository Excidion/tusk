# TableEncoder

Replace `dtype_selector` with `TableEncoder`, a transformer that encodes each
column of a feature matrix by its narwhals dtype. It resembles skrub's
`TableVectorizer`, but the column's dtype alone chooses the encoder. There is
no cardinality switch: the user picks the encoding of a text column by casting
it to `String`, `Categorical` or `Enum`.

`TableEncoder` and its column encoders subclass `NarwhalsConverter`, a
scikit-learn compatibility layer that accepts any table narwhals can read.

## Why

`dtype_selector` picks columns by `DtypeFamily`. The families were built to
decide which primitives apply to a column, so they overlap (`Datetime` is in
`has_date` and `has_time`). An encoder must give each column one owner. Users
also still build a `ColumnTransformer` by hand, which fails on pyarrow, the
backend a duckdb database collects to.

## Components

All components live in `tusk.sklearn` and are public. They need the `sklearn`
extra.

### `NarwhalsConverter`

`NarwhalsConverter(TransformerMixin, BaseEstimator)` converts a narwhals
table into the output scikit-learn asks for.

- `fit(X, y=None)` accepts an eager or lazy native table. It reads the schema
  with `collect_schema()` and does not collect. It sets `feature_names_in_`,
  `n_features_in_` and `schema_in_` (column name to narwhals dtype).
- `transform(X)` collects the table and returns it in the configured output.
- `set_output(transform=...)` accepts `"default"` (numpy), `"pandas"` and
  `"polars"`, the values scikit-learn accepts. The converter converts through
  narwhals itself instead of through scikit-learn's output wrapper. A polars
  table with a `Categorical` column then becomes a pandas table with a
  `category` column, not an object array. `Pipeline.set_output` reaches it
  like any other step.
- `get_feature_names_out()` returns the output column names.
- Subclasses override two hooks and receive a narwhals table:
  `_fit(table: nw.LazyFrame | nw.DataFrame, y)` and
  `_transform(table: nw.DataFrame) -> nw.DataFrame`. The base class does the
  conversion on both sides.

Used alone, it is the step that hands a duckdb or pyarrow feature matrix to
scikit-learn.

### Column encoders

Each column encoder is a `NarwhalsConverter` subclass that encodes every
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

The four temporal encoders take `components: Sequence[str]`. The allowed
components are the numeric results of narwhals' `dt` namespace:

| Encoder | Allowed | Default |
| --- | --- | --- |
| `DateEncoder` | `year`, `month`, `day`, `weekday`, `ordinal_day`, `timestamp` | `month`, `day`, `weekday` |
| `TimeEncoder` | `hour`, `minute`, `second`, `millisecond`, `microsecond`, `nanosecond` | `hour`, `minute` |
| `DatetimeEncoder` | the `DateEncoder` and `TimeEncoder` components | `month`, `day`, `weekday`, `hour`, `minute` |
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

### `TableEncoder`

`TableEncoder(NarwhalsConverter)` has one parameter per dtype group. Each
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

The defaults are estimator instances in the signature, as in skrub. `fit`
clones them, so instances are never shared or mutated, and nested parameters
such as `date__components` work with `set_params`.

`fit`:

1. Read the schema. Cast `Decimal` columns to `Float64`.
2. Split the columns into the groups. A group with no columns is skipped.
3. Clone each group's estimator.
4. Fit each estimator on its group's columns:
   - An estimator with `_fits_on_schema = True` receives the table as it
     came, lazy or eager. Nothing is collected.
   - Every other estimator receives its columns collected as pandas. This
     covers `StringEncoder`, `OneHotEncoder` and any estimator the user
     supplies.
5. Set `groups_`: group name to (fitted estimator, input columns).

`transform` collects the table once. Each group's output is converted to a
narwhals table named by the estimator's `get_feature_names_out()`, and the
tables are joined horizontally in the order of the parameter table above.
A sparse output is made dense. The result goes through `NarwhalsConverter`'s
output conversion.

Output names are `{group}__{name}`, for example `date__signup_month`, the
same scheme as `ColumnTransformer(verbose_feature_names_out=True)`.

`TableEncoder` does its own dispatch instead of building a
`ColumnTransformer`. A `ColumnTransformer` needs the whole table eager and
cannot read pyarrow, which defeats a schema-only fit.

Non-tusk estimators receive pandas, so the `sklearn` extra gains `pandas`.

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

## Testing

Behaviour tests, on polars, pandas, pyarrow and duckdb where the backend
matters:

- `NarwhalsConverter`: fit on a lazy duckdb table collects nothing (patch
  `collect` to raise); `set_output` returns numpy, pandas and polars; a
  polars `Categorical` becomes a pandas `category`.
- Each column encoder: the values for a small fixed input; the output names;
  null handling; an unrecognized component and a wrong dtype raise.
- `EnumEncoder`: codes follow the category order, not the order the values
  appear in; a pandas ordered categorical gets the same codes.
- `StringEncoder`: output width is `n_components` for a large vocabulary, a
  small one and an empty one; strings with shared n-grams are closer than
  strings without; a value first seen at transform is encoded.
- `TableEncoder`: each dtype reaches its group; defaults encode a table with
  every dtype; `"drop"` and `"passthrough"` work; a user estimator receives
  pandas; a schema-only group does not collect at fit; an empty group is
  skipped; `get_feature_names_out` matches the output columns;
  `clone` and `get_params`/`set_params` work, including nested parameters
  such as `date__components`.
- `DFSSelectorTransformer` with a `TableEncoder` followed by a selector, on
  duckdb: lineage keeps the right features, and the refit on the narrowed
  matrix succeeds.
- scikit-learn's `check_estimator` on the column encoders, as far as its
  numeric-only test data allows.

## Docs

- `docs/guide/sklearn.md`: replace "Choosing columns with `dtype_selector`"
  with a `TableEncoder` section. It states the rule: cast a text column to
  `Categorical` for one-hot, to `Enum` for ordered codes, and leave it
  `String` for the n-gram encoding. Update the examples that use
  `dtype_selector`.
- `docs/api/sklearn.md` picks the new classes up from `tusk.sklearn`.
