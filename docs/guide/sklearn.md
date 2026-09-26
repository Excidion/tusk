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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from tusk.sklearn import DFSTransformer, dtype_selector

sklearn.set_config(enable_metadata_routing=True)

pipeline = Pipeline(
    steps=[
        ("dfs", DFSTransformer(target_table="customers", max_depth=2)),
        (
            "encode",
            ColumnTransformer(
                [("numbers", SimpleImputer(), dtype_selector("numeric"))],
            ),
        ),
        ("model", ExtraTreesClassifier()),
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
strings and nulls, which most estimators do not take. How to encode them, and
what value to substitute for a null, is yours to choose.

Pass `database=` to `predict` to score a different set of keys, from either
the same database or another one built to the same schema.

## What `X` may be

Any iterable of key values, such as a list, a 1-D array or a Series:

```python
pipeline.fit([1, 2, 3], y_train, database=db)
pipeline.fit(np.array([1, 2, 3]), y_train, database=db)
```

Each element is one key. A key with no matching row raises, as does a
repeated key. Both would misalign the feature matrix against `y`.

## Cutoff time and the target table

`cutoff_time` filters every table, the target included. A target row that
did not exist yet at the cutoff time will have no row in the feature matrix.
Therefore, any key in `X` identifying such a row raises `SchemaError`:
```
no row for 3 of 100 keys, e.g. [17, 41, 88]; they are absent from
'customer_id' or were excluded by cutoff_time
```
`transform` raises this error before it computes the full feature matrix. It
first reads the target table's primary key alone, under the same cutoff
time. A stale key therefore costs a one-column scan, rather than a full pass
over the features.

To avoid this error, filter `X` and `y` to the keys that existed at the
cutoff time. Then hand them to the pipeline.

`transform` checks the keys before it computes the feature matrix.

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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tusk.sklearn import DFSSelectorTransformer

selector = DFSSelectorTransformer(
    target_table="customers",
    selection_pipeline=Pipeline(
        steps=[
            (
                "encode",
                ColumnTransformer(
                    [
                        (
                            "categories",
                            OneHotEncoder(handle_unknown="ignore"),
                            dtype_selector("string"),
                        ),
                        ("numbers", StandardScaler(), dtype_selector("numeric")),
                    ],
                ),
            ),
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
`categories__MODE__orders__products__category_a`.

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
  `UnencodedFeatureWarning` naming how many. Cover every dtype in your
  feature matrix, or set `remainder="passthrough"`.

## Choosing columns with `dtype_selector`

Synthesis builds the feature matrix's column names, so you cannot know them
all in advance. Which features exist depends on your schema, your primitives
and `max_depth`. tusk therefore rejects a `ColumnTransformer` given an
explicit list of names, raising `EncoderError`.

`dtype_selector` picks columns by dtype instead. It takes a
[`DtypeFamily`](../api/dtypes.md), the same families that decide which
primitives apply to which columns:

| Family | Matches |
| --- | --- |
| `"numeric"` | integers and floats |
| `"temporal"` | `Date`, `Datetime`, `Duration`, `Time` (every temporal dtype) |
| `"has_date"` | `Date`, `Datetime` (columns you can read a calendar position from) |
| `"has_time"` | `Datetime`, `Time` (columns you can read an hour or minute from) |
| `"duration"` | `Duration` |
| `"string"` | `String` |
| `"categorical"` | `Categorical`, `Enum` |
| `"boolean"` | `Boolean` |

`"string"` and `"categorical"` are separate: a column you declared
`Categorical` is a label, a `String` column carries no such declaration.

scikit-learn's own `make_column_selector` does the same job but accepts only
pandas. `dtype_selector` reads the schema through narwhals, so it works on
every backend a tusk database can use.

## Frame backends

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
- scikit-learn reads polars DataFrames through a dataframe interchange
  protocol that polars deprecated, so fitting one emits harmless
  `DeprecationWarning`s. `"pandas"` avoids them.
