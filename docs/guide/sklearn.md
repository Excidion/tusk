# scikit-learn pipelines

`tusk.sklearn` runs deep feature synthesis as a step in a scikit-learn
pipeline. Install the extra:

```bash
uv add "tusk[sklearn]"
```

`tusk` itself does not import scikit-learn, so reach the estimators by name:

```python
from tusk.sklearn import DFSTransformer
```

## Setting up

`X` is the target table's primary key — one value per row, in the order you
want the rows back. The database is passed separately, as scikit-learn
metadata, which you enable once per process:

```python
import sklearn

sklearn.set_config(enable_metadata_routing=True)
```

```python
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from tusk.sklearn import dtype_selector

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

pipeline.fit([1, 2, 3], y, database=db)
pipeline.predict([4, 5], database=db_test)
```

`fit` reads the schema and synthesizes feature definitions; it touches no
rows. `transform` computes them and returns one row per key, in key order.

The matrix holds whatever dtypes synthesis produced, and aggregates over an
empty group are null, so a step between tusk and the model is the normal
shape: most estimators take neither strings nor nulls. The encoding is yours
to choose — tusk never encodes anything itself.

Pass `database=` to `predict` to score against a different database. Omit it
and the one seen at `fit` is used.

### What `X` may be

A list, any iterable, a 1-D array, or a one-column dataframe:

```python
pipeline.fit([1, 2, 3], y, database=db)
pipeline.fit(pl.LazyFrame({"id": [1, 2, 3]}), y, database=db)
```

A column vector of shape `(n, 1)` — scikit-learn's usual `X` shape — is
rejected. Each element of `X` is one key, not one row of features.

A key with no matching row raises, as does a repeated key. Both would
misalign the matrix against `y`.

### Cross-validation and search

Because `X` is an ordinary column of values, the usual tools work:

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

cross_val_score(pipeline, keys, y, cv=3, params={"database": db})

search = GridSearchCV(pipeline, {"dfs__max_depth": [1, 2, 3]})
search.fit(keys, y, database=db)
```

## Computing only the features you keep

Deep feature synthesis generates far more features than a model needs.
`DFSSelectorTransformer` fits a pipeline that encodes and selects, works out
which tusk features the selector kept, and drops the rest. Later calls compute
only what is left:

```python
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tusk.sklearn import DFSSelectorTransformer, dtype_selector

selector = DFSSelectorTransformer(
    target_table="customers",
    selection_pipeline=Pipeline(
        steps=[
            (
                "encode",
                ColumnTransformer(
                    [
                        ("categories", OneHotEncoder(handle_unknown="ignore"),
                         dtype_selector("string")),
                        ("numbers", StandardScaler(), dtype_selector("numeric")),
                    ],
                ),
            ),
            ("select", SelectKBest(k=50)),
        ],
    ),
)
```

`selection_pipeline` must end in a scikit-learn selector — one with a
`get_support()` mask. Everything before it encodes.

After fitting, `features_` holds the definitions that were kept, and
`get_feature_names_out()` gives the encoded column names with tusk's names
substituted back, such as `categories__MODE__transactions__category_a`.

### Choosing columns with `dtype_selector`

Give a `ColumnTransformer` an explicit list of column names and tusk raises.
Synthesis generates those names, so they do not exist until it has run, and
`ColumnTransformer` drops any column you did not name.

`dtype_selector` picks columns by dtype instead. It takes a
[`DtypeFamily`](../api/dtypes.md), the same families that decide which
primitives apply to which columns:

| Family | Matches |
| --- | --- |
| `"numeric"` | integers and floats |
| `"temporal"` | `Date`, `Datetime`, `Duration` |
| `"string"` | `String` |
| `"categorical"` | `Categorical`, `Enum` |
| `"boolean"` | `Boolean` |

`"string"` and `"categorical"` are separate: a column you declared
`Categorical` is a label, a `String` column carries no such declaration.

scikit-learn's own `make_column_selector` does the same job but accepts only
pandas. `dtype_selector` reads the schema through narwhals, so it works on
every backend a tusk database can use.

A feature no encoder looks at is dropped, and tusk warns with
`UnencodedFeatureWarning` naming how many. Cover every dtype in your matrix,
or set `remainder="passthrough"`.

### When the selector's names carry no lineage

Some transformers name their outputs without reference to their inputs —
`PCA` gives `pca0`, `pca1`. tusk cannot then tell which feature an output came
from, so it keeps every feature and warns with `LineageWarning`. Selection
still applies to the model; only the saving at inference time is lost.

## Frame backends

The feature matrix is collected to whatever backend the database already uses,
so narwhals-native transformers get the frame type they want. Set
`output_backend` to change it:

```python
DFSTransformer(target_table="customers", output_backend="pandas")
```

`tusk[sklearn]` does not depend on pandas, so install it yourself to use that
option.

Two backend notes:

- `ColumnTransformer` cannot read pyarrow tables, which is what a duckdb
  database collects to by default. Set `output_backend` to `"pandas"` or
  `"polars"` for those.
- scikit-learn reads polars frames through a dataframe interchange protocol
  that polars has deprecated, so fitting one emits `DeprecationWarning`s from
  scikit-learn. They are harmless. `output_backend="pandas"` avoids them.
