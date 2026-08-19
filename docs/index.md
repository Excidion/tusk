# tusk

In nature, narwhals use their tusk to find mates.

In data science, you can use tusk to connect
[narwhals](https://narwhals-dev.github.io/narwhals/) dataframes.

This package implements deep feature synthesis to automate feature engineering
with the power of your favorite dataframe library. Powered by
[narwhals](https://narwhals-dev.github.io/narwhals/), inspired by
[featuretools](https://featuretools.alteryx.com/).

## Install

```bash
uv add tusk
```

## Quickstart

```python
from datetime import datetime

import tusk
from tusk.primitives import Quantiles

es = tusk.EntitySet("retail")
es.add_dataframe(
    "customers",
    customers_lf,
    primary_key="id",
    row_creation_time="signed_up_at",
)
es.add_dataframe(
    "sessions",
    sessions_lf,
    primary_key="id",
    row_creation_time="started_at",
)
es.add_dataframe(
    "transactions",
    tx_lf,
    primary_key="id",
    row_creation_time="occurred_at",
)

es.add_relationship(parent="customers", child="sessions", foreign_key="customer_id")
es.add_relationship(parent="sessions", child="transactions", foreign_key="session_id")

feature_matrix, features = tusk.dfs(
    entityset=es,
    target_dataframe_name="customers",
    agg_primitives=["mean", "count", Quantiles(qs=(0.25, 0.5, 0.75))],
    trans_primitives=["month", "weekday"],
    max_depth=2,
    cutoff_time=datetime(2026, 1, 1),
)
```

`feature_matrix` comes back in the frame type you put in — lazy in, lazy out —
so if you passed lazy frames, nothing is computed until you collect it.
`features` is a list of inspectable definitions you can re-apply to new data:

```python
matrix = tusk.calculate_feature_matrix(features, es_new)
```

## Where to go next

- [Entity sets](guide/entity-sets.md) — declaring tables, keys and relationships.
- [Running DFS](guide/dfs.md) — depth, cutoff times, and the column naming scheme.
- [Primitives](guide/primitives.md) — what ships with tusk and how it behaves.
- [Custom primitives](guide/custom-primitives.md) — the extension point.
- [Differences from featuretools](guide/featuretools.md) — if you are porting.
- [API reference](api/index.md) — every public symbol.
