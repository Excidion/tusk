# Relational data export: a database as sdm relational data

Design for `tusk.sdm.to_relational_sampler(database)`, a function that
turns a database into the sampler that
[structured-data-models](https://github.com/NVIDIA/structured-data-models)
(`sdm`) uses to feed `KumoRelational`. Issue 41.

## Problem

tusk describes a database: tables, primary keys, relationships, row creation
times, row update times. Deep feature synthesis is one consumer of that
description. Relational deep learning is another. A user who wants to try
`sdm.models.KumoRelational` on a tusk database rebuilds the keys and
relationships by hand for `sdm`, and has no guard against rows from after a
cutoff time.

tusk positions the database as the one description of the data. Each consumer
is an adapter on top of it: `deep_feature_synthesis`, the scikit-learn
transformers, and now this export. The mission is to make the database usable
by other frameworks. The adapter does not duplicate what the framework
already does.

## What sdm provides

- `RelationalData(tables, relationships)`: named `TableTensor`s and join
  relationships. It is the whole database. Every column of a relationship must
  have the semantic type `id`.
- `RelationalSampler(data, time_columns)`: samples a subgraph around task
  rows. A row of a table listed in `time_columns` is sampled only if its time
  does not exceed the task row's query time. It exposes `data`.
- `RelatedTables`: the sampled, per-task result. `KumoRelational` consumes it.
  The user gets it from the sampler, so tusk does not build it.
- `TableTensor.from_arrow(table, stypes)`. A column of the table that
  `stypes` does not name is left out of the tensor.
- `KumoRelational`'s default recipe already expands datetime columns, aligns
  categories, imputes and standardizes. It takes numerical and datetime inputs.

## Decisions

1. **The target is `RelationalData`, wrapped in a `RelationalSampler`.** Not
   `RelatedTables`, which is a per-task slice. Not a Kumo `Graph`, which is a
   different product that needs pandas and a separate client.
2. **The export is a function in an optional submodule, not a `Database`
   method.** `tusk.sdm` needs the `sdm` extra, like `tusk.sklearn` needs the
   `sklearn` extra. `tusk` does not import it.
3. **The function returns the `RelationalSampler`.** The user reads
   `sampler.data` for the `RelationalData`. One call returns one object, and
   no caller has to keep a `RelationalData` and its `time_columns` in step.
4. **tusk settles what a column is. The `sdm` recipe settles how a model
   treats it.** tusk gives each column the semantic type its dtype states,
   and casts the three dtypes whose meaning is a number (below). Categories,
   datetimes, nulls and scale stay with `sdm`'s recipes. This avoids encoding
   twice and avoids rebuilding what `sdm.processing` has.
5. **Time is passed through and cut at export.** The function reads the
   database as it stands, so `to_relational_sampler(db.as_of(t))` filters rows
   and restores updated columns first, with the code `as_of` already has. It
   also returns `time_columns` built from each `row_creation_time`, so the
   sampler cuts again per task row. A database with no cutoff time exports
   every row, and the sampler alone limits them.
6. **tusk maps narwhals dtypes to semantic types, and the function takes no
   semantic type argument.** The dtype alone chooses the semantic type, as it
   does in `TableEncoder` and the primitives. To change a column's semantic
   type, the caller casts the column before it is added to the database.
   `Database` gains no semantic type concept, and `infer_stypes` is not
   used: its statistics depend on the values and would differ between runs.
   The cast is also how a caller keeps a dtype out of its conversion: a
   `Boolean` cast to `String` is `categorical`, and an `Enum` cast to
   `Categorical` is unordered.
7. **Key columns get the semantic type `id`, whatever their dtype.** These are
   every primary key and every foreign key of a relationship.
8. **tusk ships no `sdm` recipe.** A recipe is model preprocessing, it belongs
   to one model, and `sdm` is Alpha. The user passes their own recipe to the
   model. Because tusk's casts only state what a column is, a recipe written
   for `numerical`, `categorical` and `datetime` columns fits the export.
   Revisit if the default recipe proves poor for exported databases.
9. **Two conversion paths, chosen by the table's backend.** A table on cudf
   goes to `TableTensor.from_cudf`, and stays on the GPU. A table on any other
   backend narwhals can convert goes to pyarrow with `to_arrow()` and then
   `TableTensor.from_arrow`. The caller does not choose. A database may mix
   backends, and each table takes its own path.

### Dtype to semantic type

| narwhals dtype | sdm semantic type | Column handed to `sdm` |
| --- | --- | --- |
| `Int*`, `UInt*`, `Float*`, `Decimal` | `numerical` | as is |
| `Boolean` | `numerical` | 0 and 1; a null stays null |
| `Enum` | `numerical` | position in the dtype's category order; a null stays null |
| `Duration` | `numerical` | total seconds as `Float64`, from `total_microseconds() / 1e6` |
| `String`, `Categorical` | `categorical` | as is |
| `Date`, `Datetime` | `datetime` | as is |
| `Time`, `Binary`, `List`, `Array`, `Struct`, `Object`, `Unknown` | none | the column is left out |

A key column is `id` before this table applies. A column left out raises
`UnsupportedDtypeWarning` once per table, naming the columns.

`Boolean`, `Enum` and `Duration` are numbers by meaning, so tusk casts them.
`Enum` keeps its declared order, which the default recipe would otherwise
discard: it sorts categories by value. The recipe turns `String` and
`Categorical` columns into ordinal codes in alphabetical order, an order with
no meaning. That is `sdm`'s choice, and the guide says so. A caller who wants
something else passes a recipe, for example one with `AddCategoryCounts`.

`String` is `categorical`, not `text`. `KumoRelational` takes numerical and
datetime feature columns and warns that it does not consume other types. Its
default recipe turns categorical columns into ordinal codes (`AlignCategories`
and `ToNumerical`), and has no step for `text`: `TFIDF` and
`SentenceTransformer` exist, but the caller must pass them in a custom recipe.
Most `String` columns in a database hold categories, such as a status, so
mapping them to `text` would drop them from the default model. A `text` column
is out of reach until a consumer needs it. Whether the model's warning also
fires for `categorical` columns that the recipe goes on to use is a test below.

## Architecture

```python
from tusk.sdm import to_relational_sampler

sampler = to_relational_sampler(db.as_of(cutoff_time))
data = sampler.data
```

```python
def to_relational_sampler(database: Database) -> RelationalSampler: ...
```

For each table of the database:

1. Read it with `database.get_table(name)` and collect it. Lazy backends
   collect here, and duckdb collects to pyarrow.
2. Cast the columns the dtype table casts, and build the semantic types from
   the table's schema: `id` for its key columns, the dtype table above for the
   rest. A column left out is not given a type, so `from_arrow` ignores it.
3. Build the tensor. A cudf table gives
   `TableTensor.from_cudf(native, types)`. Any other table gives
   `TableTensor.from_arrow(table.to_arrow(), types)`. Both ignore a column
   that `types` does not name.

Then:

- Each tusk `Relationship(parent, child, foreign_key)` becomes
  `sdm.relational.Relationship(left_table=child, left_columns=(foreign_key,),
  right_table=parent, right_columns=(parent's primary key,))`, as in the `sdm`
  documentation's example.
- `time_columns` maps each table with a `row_creation_time` to that column.
- The function returns `RelationalSampler(RelationalData(tables,
  relationships), time_columns)`.

The casts of step 2 are narwhals expressions, so they run on the table's own
backend before either path. A cudf table is never moved to the host. A sampler
over tensors on a CUDA device is `sdm`'s GPU sampler, which `sdm` documents as
needing its own GPU dependencies (cudf, cupy, pylibcugraph). tusk adds none of
them to the `sdm` extra and passes no `device`, so the tensors land where
`sdm` puts them by default.

The function uses only `Database`'s public methods: `table_names`,
`relationships`, `get_schema`, `get_table`, `require_primary_key`.

### Package

- The `Enum` code expression, `code_expression` in
  `tusk/sklearn/_column_encoders.py`, moves to a private module that needs no
  `sklearn`. `EnumOrdinalEncoder` and the export import it from there.

- `pyproject.toml` gains the extra `sdm = ["structured-data-models>=0.2.1"]`.
  The package is Apache-2.0 and Alpha, and it requires torch, so it stays
  optional. The module docstring says so, as `tusk.sklearn`'s does.
- `src/tusk/sdm/__init__.py` exports `to_relational_sampler`. The code is in
  one private module.
- Inside the package, `import sdm` is the absolute import of the library.

## Errors

- `sdm` not installed: the import raises the library's own `ImportError`.
- A column of a dtype with no semantic type: the column is left out and
  `UnsupportedDtypeWarning` names it. A key column of such a dtype is not
  left out, and `sdm` raises its own error.
- A `row_creation_time` column that `sdm` does not type as `datetime`: the
  sampler raises its `ValueError`. The test below shows whether a date column
  needs a cast.
- A table without a primary key but used as a parent: `require_primary_key`
  raises as it does elsewhere.

## Limits

- `row_update_times` cannot follow the sampler's per task row time. `sdm`
  cannot restore an old value of a column. The columns hold the value at the
  database's cutoff time, so callers cut with `as_of`. The guide says so.
- `where` and `when` conditions have no `sdm` counterpart and are ignored.
- The cudf path is written from `sdm`'s and narwhals' source and was not run:
  the build environment has no GPU. Whether the casts work on cudf is open:
  cudf reads an `Enum` as `Categorical` in narwhals, so the `Enum` cast does
  not apply there, and `Duration` seconds on cudf is unchecked. The cudf tests
  below skip without a GPU and run where one exists.
- Composite keys stay unsupported, because `Database` rejects them. `sdm`
  would take them.
- Task tables and task links stay with the caller. They belong to the model
  task, not to the database.

## Tests

`pytest.importorskip("sdm")` guards them.

- A three table database exports tables, relationships with the child on the
  left, and `id` types on the keys.
- The dtype table: one column of each dtype gets its semantic type, or is
  left out with the warning. A `Boolean` becomes 0 and 1, an `Enum` its
  category position, a `Duration` its seconds. A cast to `String` keeps a
  `Boolean` or an `Enum` as `categorical`. A key
  column is `id` whatever its dtype.
- `as_of` drops later rows and restores updated columns in the exported
  tables.
- `time_columns` holds each `row_creation_time` and omits tables without one.
- A polars, a pandas and a duckdb database export the same tables.
- A cudf database exports the same tables as a pandas one, on a CUDA device.
  It skips without `cudf` and a GPU.
- A database with a cudf table and a polars table exports both.
- A `Date` `row_creation_time` builds a sampler or fails with a clear message.
- The sampler samples one task row end to end and returns `RelatedTables`.
- `KumoRelational` runs on the sampled tables. The test records which warnings
  it issues for `categorical` columns.

## Docs

A section in `docs/guide/` on the export, with the example above and the
limits. `docs/guide/index.md` and the API page link to it. Run the docs build.

## Out of scope

- A Kumo `Graph` export.
- Encoding categoricals, datetimes or text for `sdm`.
- A semantic type concept on `Database`, and a `text` semantic type.
- Building task tables or `RelatedTables`.
