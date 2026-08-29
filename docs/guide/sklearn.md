# scikit-learn pipelines

`tusk.sklearn` runs deep feature synthesis as a step in a scikit-learn
pipeline. Install the extra:

```bash
uv add "tusk[sklearn]"
```

`X` is the target table's primary key — one value per row, in the order you
want the rows back — and the database is passed separately, as scikit-learn
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

`fit` reads the schema and synthesizes feature definitions; it touches no
rows. `transform` computes them and returns one row per key, in key order.

`sklearn.set_config(enable_metadata_routing=True)` is what lets `database=`
reach the transformer through the pipeline. Set it once per process.

The matrix holds whatever dtypes synthesis produced, and aggregates over an
empty group are null, so a step between tusk and the model is the normal
shape: most estimators take neither strings nor nulls. The encoding is yours
to choose — tusk never encodes anything itself.

Pass `database=` to `predict` to score a different set of keys, from either
the same database or another one built to the same schema.

## What `X` may be

Any iterable of key values — a list, a 1-D array, a Series:

```python
pipeline.fit([1, 2, 3], y_train, database=db)
pipeline.fit(np.array([1, 2, 3]), y_train, database=db)
```

Each element is one key. A key with no matching row raises, as does a
repeated key: both would misalign the matrix against `y`.

## Cross-validation and search

Because `X` is an ordinary column of values, the usual tools work:

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

cross_val_score(pipeline, keys, y_train, cv=3, params={"database": db})

search = GridSearchCV(pipeline, {"dfs__max_depth": [1, 2, 3]})
search.fit(keys, y_train, database=db)
```

## Automatic feature selection

Synthesis generates many more features than a model needs, and computing all
of them again at inference time is wasted work.

`DFSSelectorTransformer` takes a pipeline that encodes and selects. It fits
that pipeline once, works out which tusk features the selector kept, and drops
the others from its feature list. Every later call computes only what is
left:

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

`selection_pipeline` must end in a scikit-learn selector — one with a
`get_support()` mask. Everything before it encodes.

After fitting, `features_` holds the feature definitions that were kept, and
`get_feature_names_out()` gives the encoded column names with tusk's names
substituted back, such as `categories__MODE__transactions__category_a`.

Two cases change what you get:

- If an encoder names its outputs without reference to its inputs tusk cannot tell which feature an output came from - e. g. `PCA` produces `pca0`, `pca1`.
  It keeps every feature and warns with `LineageWarning`.
  Selection still applies to the model; only the saving at inference time is lost.
  Consider placing the `PCA` further downstream in the main `Pipeline` and not in the `selection_pipeline`.
- If a feature reaches no encoder, it is dropped and tusk warns with
  `UnencodedFeatureWarning` naming how many. Cover every dtype in your
  matrix, or set `remainder="passthrough"`.

## Choosing columns with `dtype_selector`

Synthesis generates the matrix's column names, so you cannot know them all in advance.
Which features exist depends on your schema, your primitives and
`max_depth`. A `ColumnTransformer` given an explicit list of names is
therefore rejected with `EncoderError`.

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

## Frame backends

The feature matrix is collected to whatever backend the database already
uses, so narwhals-native transformers get the frame type they want. Set
`output_backend` to change it:

```python
DFSTransformer(target_table="customers", output_backend="pandas")
```

`tusk[sklearn]` does not depend on pandas, so install it yourself to use that
option. Two cases call for it:

- `ColumnTransformer` cannot read pyarrow tables, which is what a duckdb
  database collects to. Use `"pandas"` or `"polars"` there.
- scikit-learn reads polars frames through a dataframe interchange protocol
  that polars has deprecated, so fitting one emits harmless
  `DeprecationWarning`s. `"pandas"` avoids them.
