# tusk

In nature, narwhals use their tusk to find mates.

In data science, you can use tusk to connect [narwhals](https://narwhals-dev.github.io/narwhals/) dataframes.

This package implements deep feature synthesis to automate feature engineering with the power of your favorite dataframe library.
Powered by [narwhals](https://narwhals-dev.github.io/narwhals/), inspired by [featuretools](https://featuretools.alteryx.com/).

## Install

```bash
uv add tusk
```

## Usage

```python
from datetime import datetime

import tusk
from tusk.primitives import Quantiles

db = tusk.Database("retail")
db.add_table("customers", customers_lf, primary_key="id", row_creation_time="signed_up_at")
db.add_table("sessions", sessions_lf, primary_key="id", row_creation_time="started_at")
db.add_table("transactions", tx_lf, primary_key="id", row_creation_time="occurred_at")

db.add_relationship(parent="customers", child="sessions", foreign_key="customer_id")
db.add_relationship(parent="sessions", child="transactions", foreign_key="session_id")

db.validate()  # optional: confirm the keys really are keys, before you trust the numbers

feature_matrix, features = tusk.deep_feature_synthesis(
    database=db,
    target_table="customers",
    agg_primitives=["mean", "count", Quantiles(qs=(0.25, 0.5, 0.75))],
    trans_primitives=["month", "weekday"],
    max_depth=2,
    cutoff_time=datetime(2026, 1, 1),
)
```

`feature_matrix` comes back as an uncomputed query plan — tusk never collects —
so on a backend with a lazy frame type you get one back and decide when to
compute:

```python
matrix = feature_matrix.collect()
```

`features` is a list of inspectable definitions you can re-apply to new data:

```python
matrix = tusk.apply_features(features, db_new)
```

## Documentation

Full documentation lives in [`docs/`](docs/index.md):

- [Databases](docs/guide/databases.md) — tables, keys, relationships, and
  [validating](docs/guide/databases.md#validation) them against the data.
- [Running DFS](docs/guide/deep-feature-synthesis.md) — depth, cutoff times, and the column naming scheme.
- [Primitives](docs/guide/primitives.md) — what ships with tusk and how it behaves.
- [Custom primitives](docs/guide/custom-primitives.md) — the extension point.
- [scikit-learn pipelines](docs/guide/sklearn.md) — DFS as a pipeline step, and
  computing only the features you keep.
- [Differences from featuretools](docs/guide/featuretools.md) — if you are porting.
- [API reference](docs/api/index.md) — every public symbol.

Build the site locally with:

```bash
uv run --group docs zensical serve
```
