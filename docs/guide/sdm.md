# Relational deep learning with sdm

`tusk.sdm` exports a database to
[structured-data-models](https://github.com/NVIDIA/structured-data-models)
(`sdm`), the library behind `KumoRelational`. The model reads the related
tables directly, so it needs no feature matrix. Install the extra:

```bash
uv add "tusk-ml[sdm]"
```

`sdm` requires torch. Its sampler also needs `pyg-lib`, which has wheels only
for stable torch releases. Install it as `sdm` explains, for example with
`pip install pyg-lib -f https://data.pyg.org/whl/torch-2.14.0+cu130.html`.
Without it, `to_relational_sampler` raises `sdm`'s `ImportError`.
[`to_relational_data`][tusk.sdm.to_relational_data] returns the same tables and
relationships without a sampler, and needs no `pyg-lib`.

## Export a database

```python
from tusk.sdm import to_relational_data, to_relational_sampler

data = to_relational_data(db.as_of(cutoff_time))  # sdm's RelationalData
sampler = to_relational_sampler(db.as_of(cutoff_time))  # needs pyg-lib
```

[`to_relational_sampler`][tusk.sdm.to_relational_sampler] returns the `sdm`
`RelationalSampler` of the database. It holds the `RelationalData`, and limits the rows it samples by each table's
`row_creation_time`. The sampler turns the data into the `RelatedTables` that
`KumoRelational` takes. You build the task table and the task link, because
they belong to your model task, not to the database.

```python
task_table = TableTensor.from_columns(
    {"customer": [1, 2], "at": [t1, t2]},
    {"customer": "id", "at": "datetime"},
)
output = sampler(
    task_table,
    {"task_column": "customer", "table": "customers", "table_column": "id"},
    num_neighbors=[8, 8],
    task_time_column="at",
)
output.related_tables  # RelatedTables
```

## Cutoff time

The export reads the database as it stands at its cutoff time. Cut with
[`as_of`][tusk.database.Database.as_of] first, and the tables hold only the
rows created up to the cutoff time, with each column in `row_update_times`
restored to the value it held then. The sampler limits the rows again for each
task row, by the task row's own time.

`row_update_times` has no counterpart in `sdm`, which cannot restore an old
value of a column. Cut with `as_of`, and pass the sampler a task time at or
after that cutoff time.

## Columns

The dtype of a column chooses its `sdm` semantic type. To change it, cast the
column before you add it to the database.

| narwhals dtype | sdm semantic type | Column handed to `sdm` |
| --- | --- | --- |
| `Int*`, `UInt*`, `Float*` | `numerical` | as is |
| `Decimal` | `numerical` | cast to `Float64` |
| `Boolean` | `numerical` | 0 and 1; a null stays null |
| `Enum` | `numerical` | position in the category order; a null stays null |
| `Duration` | `numerical` | total seconds as `Float64` |
| `String`, `Categorical` | `categorical` | as is |
| `Date`, `Datetime` | `datetime` | as is |
| `Time`, `Binary`, `List`, `Array`, `Struct`, `Object`, `Unknown` | none | the column is left out |

A primary key or a foreign key is `id`, whatever its dtype. Columns left out
raise one [`UnsupportedDtypeWarning`][tusk.exceptions.UnsupportedDtypeWarning]
per table.

tusk casts only what is a number by meaning. It does not encode a column, and
it ships no `sdm` recipe. The recipe of the model you use decides how `sdm`
treats each semantic type. The default recipe of `KumoRelational` turns a
`categorical` column into ordinal codes in alphabetical order, an order with
no meaning. Pass your own recipe to the model for something else, for example
one with `AddCategoryCounts`. Cast a `Boolean` or an `Enum` to `String` first
if you want it to be `categorical`.

## Backends

Any backend narwhals can convert to pyarrow works. A cudf table goes to the
GPU through `TableTensor.from_cudf` instead. tusk's own test suite does not run
the cudf path.

duckdb reads an `INTERVAL` column as `Unknown`, so a `Duration` column on
duckdb is left out.

## What is not exported

- The `where` and `when` conditions of a table.
- Composite keys, which a database does not accept.
- Task tables and task links.
