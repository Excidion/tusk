# Relational data export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tusk.sdm.to_relational_sampler(database)`, which turns a database into the `sdm` `RelationalSampler` that feeds `KumoRelational`.

**Architecture:** An optional submodule `tusk.sdm`, like `tusk.sklearn`. Three private modules: `_columns` casts a table's columns and names their semantic types, `_graph` builds the relationships and `time_columns`, `_export` converts each table to a `TableTensor` (cudf tables through `from_cudf`, every other table through pyarrow) and builds the sampler. All three read the database through its public methods.

**Tech Stack:** Python 3.11+, narwhals, `structured-data-models` (`sdm`, optional), polars and pandas and duckdb (tests), pytest.

**Spec:** `agents/superpowers/specs/2026-10-17-relational-data-export-design.md`

## Global Constraints

- Work on the branch `claude/adoring-gauss-3xp78z`. Do not open a pull request.
- The export is a function in `tusk.sdm`, not a `Database` method. `tusk` does not import `tusk.sdm`.
- `import tusk.sdm` without `sdm` installed raises the library's own `ImportError`. Import `sdm` at the top of the module that uses it.
- tusk ships no `sdm` recipe, passes no `device`, and adds no cudf or GPU packages to any extra.
- Semantic types are the plain strings `"numerical"`, `"categorical"`, `"datetime"` and `"id"`, which `sdm` accepts as `StypeLike`. `_columns` and `_graph` import nothing from `sdm`.
- Docs, docstrings and identifiers follow `agents/GLOSSARY.md` (table, null, build, compute, backend, cutoff time). Docstrings say what a thing does, not why, except on exception and warning classes.
- Every public and private function has a Google-style docstring (`ruff` `D` and `interrogate` at 100%). Tests are exempt.
- Run tests with `uv run pytest`, lint with `just lint`.
- The build environment has no GPU. Write the cudf path and its tests, but do not claim they pass.

---

### Task 1: Move `code_expression` out of `tusk.sklearn`

**Files:**
- Create: `src/tusk/_codes.py`
- Modify: `src/tusk/sklearn/_column_encoders.py` (delete `code_expression`, import it)
- Test: `tests/test_codes.py`

**Interfaces:**
- Produces: `tusk._codes.code_expression(name: str, categories: list[str]) -> nw.Expr`, unchanged behaviour.

- [ ] **Step 1: Write the failing test** in `tests/test_codes.py`: build a polars `Enum` column `["b", "a", None]` with categories `["a", "b"]`, select `code_expression("x", ["a", "b"])`, assert the values `[1.0, 0.0, None]` and the dtype `Float64`. Run `uv run pytest tests/test_codes.py`; expect `ModuleNotFoundError: tusk._codes`.
- [ ] **Step 2: Move the function.** Cut `code_expression` and its comment from `src/tusk/sklearn/_column_encoders.py` into `src/tusk/_codes.py`, with a module docstring. `_column_encoders.py` imports `from tusk._codes import code_expression`. The only caller is `EnumOrdinalEncoder._expressions`.
- [ ] **Step 3: Run** `uv run pytest tests/test_codes.py tests/test_sklearn_column_encoders.py tests/test_sklearn_table_encoder.py`. Expect all to pass.
- [ ] **Step 4: Commit** `Move code_expression out of tusk.sklearn`.

---

### Task 2: `UnsupportedDtypeWarning`

**Files:**
- Modify: `src/tusk/exceptions.py`
- Test: `tests/test_packaging.py`

**Interfaces:**
- Produces: `tusk.exceptions.UnsupportedDtypeWarning(UserWarning)`.

- [ ] **Step 1: Write the failing test** in `tests/test_packaging.py`, next to `test_missing_primary_key_warning_is_filterable`: `assert issubclass(UnsupportedDtypeWarning, UserWarning)`. Run it; expect `ImportError`.
- [ ] **Step 2: Add the class** after `UnencodedFeatureWarning`. Its docstring says it is issued when the export leaves a column out because no `sdm` semantic type fits its dtype, why a table's columns are not all exported, and that it has its own class so a caller can filter it. Follow the wording of the neighbouring warnings.
- [ ] **Step 3: Run** `uv run pytest tests/test_packaging.py`; expect a pass. Check that the docs build lists it: `docs/api` renders `tusk.exceptions` whole, so no edit is needed.
- [ ] **Step 4: Commit** `Add UnsupportedDtypeWarning`.

---

### Task 3: The `sdm` extra, group, marker and recipe

**Files:**
- Modify: `pyproject.toml`, `justfile`

- [ ] **Step 1: Add the extra** `sdm = ["structured-data-models>=0.2.1"]` under `[project.optional-dependencies]`. `requires-python` is already `>=3.11`, which `sdm` needs.
- [ ] **Step 2: Add a dependency group** `sdm = ["structured-data-models>=0.2.1"]`, and the marker `"sdm: tests that need structured-data-models, which requires torch (opt-in)"`. Extend `addopts` to `-m 'not differential and not benchmark and not sdm'`. torch is large, so the default run and the default `uv sync --all-groups` job must still work: check that `just install` (`--all-groups --all-extras`) resolves. If the resolve fails, say so and stop.
- [ ] **Step 3: Add the recipe** `test-sdm` to the `justfile`, modelled on `test-differential`: `uv run --group sdm pytest -m sdm`. Add it to `check`.
- [ ] **Step 4: Run** `uv lock` and `uv run pytest -q`. Expect the existing suite to pass unchanged.
- [ ] **Step 5: Commit** `Add the sdm extra and test recipe`.

---

### Task 4: Cast the columns and name their semantic types

**Files:**
- Create: `src/tusk/sdm/__init__.py` (a docstring only until Task 7), `src/tusk/sdm/_columns.py`
- Test: `tests/test_sdm_columns.py`

**Interfaces:**
- Produces: `prepare_table(table: nw.DataFrame, schema: TableSchema, key_columns: frozenset[str]) -> tuple[nw.DataFrame, dict[str, str]]`. The table holds the cast columns and the columns left out are removed. The mapping is column name to semantic type string.
- Produces: `semantic_type(dtype) -> str | None` for the dtype table in the spec.

Task 7 makes `tusk/sdm/__init__.py` import `sdm`, so every `tusk.sdm` module needs `sdm` installed. The tests of Tasks 4, 5 and 7 therefore carry the marker `sdm` (`pytestmark = pytest.mark.sdm`) and `pytest.importorskip("sdm")` at the top. They run under `just test-sdm`.

- [ ] **Step 1: Write failing tests.** Build a polars table with one column per dtype in the spec's table (`Int64`, `UInt8`, `Float32`, `Decimal`, `Boolean`, `Enum`, `Duration`, `String`, `Categorical`, `Date`, `Datetime`, `Time`, `Binary`, `List`, `Struct`) and run `prepare_table`. Assert:
  - the semantic types: numeric dtypes, `Boolean`, `Enum`, `Duration` are `numerical`; `String` and `Categorical` are `categorical`; `Date` and `Datetime` are `datetime`.
  - the casts: `Boolean` gives `[1.0, 0.0, None]` as `Float64`; `Enum` gives the category position; `Duration` of 1.5 seconds gives `1.5`; a `Float` is unchanged.
  - the columns of `Time`, `Binary`, `List` and `Struct` are not in the result, and one `UnsupportedDtypeWarning` names all of them (`pytest.warns`, check the message holds each name once).
  - a key column of dtype `Time` stays in the table with the type `id`, with no warning for it.
  - a key column of dtype `String` is `id`, not `categorical`.
  - a cast to `String` before the call keeps a `Boolean` as `categorical`.
- [ ] **Step 2: Run** `uv run pytest tests/test_sdm_columns.py`; expect `ModuleNotFoundError`.
- [ ] **Step 3: Implement `_columns.py`.** Read `table.schema` and `schema.dtypes`. For each non-key column choose by dtype: numeric (`dtype.is_numeric()`) as is; `nw.Boolean` `nw.col(c).cast(nw.Float64)`; `nw.Enum` `code_expression(c, list(dtype.categories))`; `nw.Duration` `nw.col(c).dt.total_microseconds().cast(nw.Float64) / 1_000_000`; `nw.String` and `nw.Categorical` as is; `nw.Date` and `nw.Datetime` as is. Compare with `dtype == nw.Enum`, the way `tusk/dtypes.py` does. Collect the left-out names, warn once per table, name the table and the columns in the message. `semantic_type` holds the mapping so that the code and the spec's table are one place.
- [ ] **Step 4: Run** the new tests; expect a pass. Run them on pandas and duckdb tables too (parametrize the construction), and note any dtype a backend reads differently in a test comment, as the table encoder spec does (pandas has no `Date`, duckdb reads `TIME` as `Unknown`).
- [ ] **Step 5: Commit** `Add column preparation for the sdm export`.

---

### Task 5: Relationships and time columns

**Files:**
- Create: `src/tusk/sdm/_graph.py`
- Test: `tests/test_sdm_graph.py`

**Interfaces:**
- Produces: `relationship_mappings(database: Database) -> list[dict[str, str]]`, each with the keys `left_table`, `left_column`, `right_table`, `right_column`.
- Produces: `time_columns(database: Database) -> dict[str, str]`.
- Produces: `key_columns(database: Database, name: str) -> frozenset[str]`, the table's primary key and its foreign keys.

Mark the tests as Task 4 does.

- [ ] **Step 1: Write failing tests** with the `conftest.py` three-table database: `customers -> sessions -> transactions`. Assert `relationship_mappings` gives the child as `left` and the parent's primary key as `right`, in insertion order; `time_columns` holds only tables that declare a `row_creation_time` (give one table none); `key_columns("sessions")` is its primary key and `customer_id`; a table without a primary key but used only as a child has only its foreign keys; a parent without a primary key raises `SchemaError` through `require_primary_key`.
- [ ] **Step 2: Run**; expect `ModuleNotFoundError`.
- [ ] **Step 3: Implement** with `database.relationships`, `database.table_names`, `database.get_schema(name)`, `database.require_primary_key(parent)`, `database.parents_of(name)`.
- [ ] **Step 4: Run** the tests; expect a pass.
- [ ] **Step 5: Commit** `Add relationships and time columns for the sdm export`.

---

### Task 6: Verify sdm's behaviour on the open points

This task decides open points of the spec. It writes no production code, only tests and possibly a spec edit. Run it with the `sdm` group: `uv run --group sdm pytest -m sdm`.

**Files:**
- Test: `tests/test_sdm_assumptions.py`, marked `sdm`, `importorskip("sdm")`.

- [ ] **Step 1: Write and run tests that record `sdm`'s behaviour** on a pyarrow table, with `TableTensor.from_arrow`:
  - `Decimal` as `numerical`: accepted, or rejected? If rejected, the plan casts `Decimal` to `Float64` in `_columns` and the spec's table changes.
  - `Date` as `datetime`: does `from_arrow` accept a `date32` column, and does `RelationalSampler(data, {"t": "d"})` accept it? If not, `_columns` casts `Date` to `Datetime`.
  - A `Datetime` with a time zone as `datetime`.
  - A `Float64` `numerical` column with a null.
  - A `UInt64` or `Int64` key as `id`, and a `String` key as `id`.
  - A column in `types` that is absent from the table.
- [ ] **Step 2: Apply the findings.** For each rejected case, add the cast to Task 4's `prepare_table` with a test, edit the spec's dtype table to match, and commit the spec change separately. For each accepted case, keep the test as a regression test.
- [ ] **Step 3: Commit** `Record sdm's accepted dtypes`.

---

### Task 7: The export function

**Files:**
- Create: `src/tusk/sdm/_export.py`
- Modify: `src/tusk/sdm/__init__.py`
- Test: `tests/test_sdm_export.py`

**Interfaces:**
- Produces: `tusk.sdm.to_relational_sampler(database: Database) -> sdm.relational.RelationalSampler`.

`__init__.py` gets the docstring of `tusk/sklearn/__init__.py` as a model: the function, the extra (`pip install "tusk-ml[sdm]"`) and the sentence that `tusk` does not import this package. It exports `to_relational_sampler` in `__all__`.

- [ ] **Step 1: Write failing tests** (module-level `pytest.importorskip("sdm")`, marker `sdm`, using the `conftest.py` database):
  - the sampler's `data.tables` has the three tables, each with the columns of its types, keys as `id`.
  - `data.relationships` has the child on the left.
  - `sampler.time_columns` holds each declared `row_creation_time`.
  - `to_relational_sampler(db.as_of(t))` drops the rows created after `t` and restores a `row_update_times` column (reuse the setup of the `as_of` tests in `tests/test_database.py`).
  - a polars, a pandas and a duckdb database give the same tensors (`tensor.to_pandas()` equal after sorting columns).
  - the sampler runs end to end: build a task table with `TableTensor.from_columns({"id": [...]}, {"id": "id"})`, call `sampler(task_table, {"task_column": "id", "table": "customers", "table_column": "id"}, [2, 2])` and assert the result holds a `RelatedTables`.
  - `KumoRelational` runs on the sampled tables with `pretrained=False`, and the test records the warnings it issues for `categorical` columns (the spec's open point). Mark this one `skip` if the model cannot run without a checkpoint download, and say why in the skip reason.
  - a cudf test, skipped without `cudf` and `torch.cuda.is_available()`: a cudf database gives the same tables as the pandas one, on a CUDA device.
- [ ] **Step 2: Run**; expect `ModuleNotFoundError`.
- [ ] **Step 3: Implement.** For each table name: `frame = nw.from_native(database.get_table(name))`, collect it if lazy, `prepare_table(frame, schema, key_columns(database, name))`. If `frame.implementation.is_cudf()` build `TableTensor.from_cudf(frame.to_native(), types)`, otherwise `TableTensor.from_arrow(frame.to_arrow(), types)`. Then `RelationalData(tables, relationship_mappings(database))` and `RelationalSampler(data, time_columns(database))`. Keep the three steps in small functions.
- [ ] **Step 4: Run** `uv run --group sdm pytest -m sdm`; expect a pass apart from the skipped cudf test.
- [ ] **Step 5: Commit** `Add to_relational_sampler`.

---

### Task 8: Docs

**Files:**
- Create: `docs/guide/sdm.md`
- Modify: `docs/guide/index.md`, `docs/api/index.md`, `zensical.toml`, `README.md`

- [ ] **Step 1: Write the guide page** after the pattern of `docs/guide/sklearn.md`: install (`uv add "tusk-ml[sdm]"`), the two-line example from the spec, the dtype table, the sentence that `String` and `Categorical` become ordinal codes in alphabetical order under the default recipe and how to pass a recipe (name `AddCategoryCounts`), the limits (`row_update_times`, `where` and `when`, composite keys, task tables), and the sentence that the cudf path was not run in tusk's own CI.
- [ ] **Step 2: Link it** from `docs/guide/index.md`, add `guide/sdm.md` to the nav in `zensical.toml` after `guide/sklearn.md`, add `- [sdm][tusk.sdm]: the export to relational deep learning.` to `docs/api/index.md`, and one line about the export in `README.md` if it lists the optional parts.
- [ ] **Step 3: Build** with `just docs`; expect no warnings about the new page or the `tusk.sdm` reference. If mkdocstrings cannot import `tusk.sdm` because `sdm` is missing from the docs group, add `structured-data-models` to the `docs` group and say so in the commit.
- [ ] **Step 4: Commit** `Document the sdm export`.

---

### Task 9: Check everything

- [ ] **Step 1: Run** `just lint`; fix what it reports.
- [ ] **Step 2: Run** `uv run pytest` and `uv run --group sdm pytest -m sdm`; expect both to pass.
- [ ] **Step 3: Re-read the diff adversarially:** terms against `agents/GLOSSARY.md` (no "frame", "dataframe", "generate", "create", "cheap"), docstrings on every function, no import of `sdm` in `_columns` or `_graph`, `tusk` does not import `tusk.sdm`.
- [ ] **Step 4: Push** the branch. Report which tests ran, which skipped and why (cudf, a checkpoint), and that the cudf path is unverified on a GPU.
