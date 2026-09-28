# scikit-learn pipelines

`tusk.sklearn` runs deep feature synthesis as a step in a scikit-learn
pipeline. Install the extra:

```bash
uv add "tusk-ml[sklearn]"
```

`X` is the target table's primary key: one value per row, in the order you
want the rows back. You pass the database separately, as scikit-learn
metadata:

```python
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from tusk.sklearn import DFSTransformer, TableEncoder

sklearn.set_config(enable_metadata_routing=True)

pipeline = Pipeline(
    steps=[
        ("dfs", DFSTransformer(target_table="customers", max_depth=2)),
        ("encode", TableEncoder()),
        ("model", HistGradientBoostingClassifier()),
    ],
)

pipeline.fit([1, 2, 3], y_train, database=db)
pipeline.predict([4, 5], database=db)
```

`fit` reads the schema and synthesizes feature definitions. It touches no
rows. `transform` computes them and returns one row per key, in key order.

`sklearn.set_config(enable_metadata_routing=True)` is what lets `database=`
reach the transformer through the pipeline. Set it once per process.

The feature matrix holds whatever dtypes synthesis produced, so it can carry
strings and nulls, which most estimators do not take.
[`TableEncoder`](#encoding-by-dtype-with-tableencoder) encodes each column by
its dtype. A null stays a null (NaN). Choose a model that accepts NaN, such as
`HistGradientBoostingClassifier`, or give the `numeric` group an imputer.

Pass `database=` to `predict` to score a different set of keys, from either
the same database or another one built to the same schema.

Give `cutoff_time=` in the same way. See [Cutoff time](#cutoff-time).

## What `X` may be

Any iterable of key values, such as a list, a 1-D array or a Series:

```python
pipeline.fit([1, 2, 3], y_train, database=db)
pipeline.fit(np.array([1, 2, 3]), y_train, database=db)
```

Each element is one key. A key with no matching row raises, as does a
repeated key. Both would misalign the feature matrix against `y`.

## Cutoff time

Give `cutoff_time` to `fit` and to `predict`, together with `database`:

```python
X_train = db.get_keys("customers", train_cutoff_time)
y_train = compute_labels(X_train)

pipeline.fit(
    X_train,
    y_train,
    database=db,
    cutoff_time=train_cutoff_time,
)
pipeline.predict(new_keys, database=db, cutoff_time=now)
```

[`Database.get_keys`][tusk.Database.get_keys] returns the primary key column
of a table at a cutoff time, as the backend's native (lazy) table. Compute
the labels based on this table. `X` and `y` have to have the same row order.

If `transform` gets no cutoff time, it uses the cutoff time from `fit`.
scikit-learn's scorers give no metadata to `predict`. Thus cross-validation
uses the training cutoff time.

Always give `cutoff_time` to `predict`. If you do not, `predict` uses the
training cutoff time.


### Keys that did not exist at the cutoff time

`cutoff_time` also filters the target table. A key that has no row at the
cutoff time raises `SchemaError`:
```
no row for 3 of 100 keys, e.g. [17, 41, 88]; they are absent from
'customer_id' or were excluded by cutoff_time
```
`transform` finds these keys before it computes the features. It reads only
the primary key column to do this. To prevent this error, get `X` from
`db.get_keys` at the same cutoff time.

## Cross-validation and search

Because `X` is an ordinary column of values, the usual tools work:

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

cross_val_score(pipeline, keys, y_train, cv=3, params={"database": db})

search = GridSearchCV(pipeline, {"dfs__max_depth": [1, 2, 3]})
search.fit(keys, y_train, database=db)
```

## Automatic feature selection

Synthesis builds many more features than a model needs. Recomputing them all
at inference time wastes work.

`DFSSelectorTransformer` takes a pipeline that encodes and selects. It fits
that pipeline once, determines which tusk features the selector kept, and
drops the others from its feature list. Every later call computes only what
remains:

```python
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer

from tusk.sklearn import DFSSelectorTransformer

selector = DFSSelectorTransformer(
    target_table="customers",
    selection_pipeline=Pipeline(
        steps=[
            ("encode", TableEncoder(numeric=SimpleImputer())),
            ("select", SelectKBest(k=50)),
        ],
    ),
)

selector.fit(keys, y_train, database=db)
```

`selection_pipeline` must end in a scikit-learn selector, one with a
`get_support()` mask. Everything before it encodes.

After fitting, `features_` holds the kept feature definitions as a
[`FeatureList`][tusk.FeatureList]. `get_feature_names_out()` gives the encoded
column names with tusk's names substituted back, such as
`string__MODE__orders__products__category_svd_0`.

Two cases change what you get:

- If an encoder names its outputs without reference to its inputs, tusk
  cannot tell which feature produced an output. For example, `PCA` produces
  `pca0`, `pca1`.
  tusk keeps every feature and warns with `LineageWarning`.
  Selection still applies to the model. You lose only the saving at inference
  time.
  Consider placing the `PCA` further downstream in the main `Pipeline` and
  not in the `selection_pipeline`.
- If a feature reaches no encoder, tusk drops it and warns with
  `UnencodedFeatureWarning` naming how many. A `TableEncoder` group set to
  `"drop"` is the usual cause. The `other` group, for dtypes such as `List`
  and `Struct`, drops by default.

## Encoding by dtype with `TableEncoder`

`TableEncoder` puts each column into one group by its narwhals dtype and
encodes each group with its own estimator:

| Parameter | Dtypes | Default |
| --- | --- | --- |
| `numeric` | integers, floats, `Decimal` | `"passthrough"` |
| `boolean` | `Boolean` | `"passthrough"` |
| `string` | `String` | `StringEncoder()` |
| `categorical` | `Categorical` | `OneHotEncoder(handle_unknown="ignore")` |
| `enum` | `Enum` | `EnumEncoder()` |
| `date` | `Date` | `DateEncoder()`: month, day |
| `time` | `Time` | `TimeEncoder()`: hour, minute |
| `datetime` | `Datetime` | `DatetimeEncoder()`: month, day, hour, minute |
| `duration` | `Duration` | `DurationEncoder()`: total seconds |
| `other` | every other dtype | `"drop"` |

Each parameter takes an estimator, `"passthrough"` or `"drop"`. Output names
are `{group}__{name}`, such as `date__signed_up_at_month`.

The dtype of a text column chooses its encoding:

- `Categorical`: one column per value (one-hot). Use it for a few unordered
  values.
- `Enum`: the value's position in the category order. Use it for ordered
  values.
- `String`: `StringEncoder` computes a TF-IDF vector over the value's
  character 3- and 4-grams and reduces it to 30 coordinates. Values that
  share spelling get close coordinates. Use it for free text and for values
  with many distinct entries.

Cast a column in the database to choose its encoding.

The temporal encoders take `components`:

```python
from tusk.sklearn import DateEncoder

TableEncoder(date=DateEncoder(components=["year", "month", "weekday"]))
```

Synthesis builds the feature matrix's column names, so you cannot know them
all in advance. tusk therefore rejects a `ColumnTransformer` given an
explicit list of names, raising `EncoderError`. `TableEncoder` needs no
names.

### Any estimator, any backend

A group gets numpy unless its estimator has `NarwhalsMixin`. Mix it in to
give an estimator a table narwhals can read, from any backend, converted to
`convert_to`:

```python
from sklearn.preprocessing import TargetEncoder

from tusk.sklearn import NarwhalsMixin


class PandasTargetEncoder(NarwhalsMixin, TargetEncoder):
    convert_to = "pandas"


TableEncoder(string=PandasTargetEncoder())
```

`convert_to` is `"numpy"`, `"pandas"`, `"polars"` or `"narwhals"`.

`NarwhalsEncoder` on its own passes every column through unchanged.
`set_output` sets what it returns. Use it to hand a feature matrix from any
backend to scikit-learn:

```python
from tusk.sklearn import NarwhalsEncoder

NarwhalsEncoder().set_output(transform="pandas")
```

## Table backends

tusk collects the feature matrix to whatever backend the database already
uses, so narwhals-native transformers get the native table type they want.
Set `output_backend` to change it:

```python
DFSTransformer(target_table="customers", output_backend="pandas")
```

Install pandas yourself for `output_backend="pandas"`. `tusk[sklearn]` does
not include it. Two cases need it:

- `ColumnTransformer` cannot read pyarrow tables, which is what a duckdb
  database collects to. Use `"pandas"` or `"polars"` in that case.
  `TableEncoder` and `NarwhalsEncoder` read every backend.
- scikit-learn reads polars DataFrames through a dataframe interchange
  protocol that polars deprecated, so fitting one emits harmless
  `DeprecationWarning`s. `"pandas"` avoids them.
