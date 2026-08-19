# Vocabulary Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move tusk's public surface onto one vocabulary — `EntitySet` becomes `Database`, `add_dataframe` becomes `add_table`, `dfs` becomes `deep_feature_synthesis`, `calculate_feature_matrix` becomes `apply_features` — with no behaviour change and no compatibility aliases.

**Architecture:** Five rename tasks, each ending with the full suite green, then a verification sweep. Order matters: the container first (everything imports it), then the public entry points, then internal parameter names, then test file and fixture names, then documentation. Each task is a self-contained commit that leaves the repo importable and the suite passing.

**Tech Stack:** Python 3.10+, narwhals, polars/pyarrow/duckdb backends, pytest, uv, ruff + ty + interrogate + pydoclint via pre-commit, zensical for docs.

**Spec:** `docs/superpowers/specs/2026-08-19-vocabulary-rename-design.md`

## Global Constraints

- **Clean break.** No aliases, no deprecation shims, no re-exports of the old names. After Task 2, `tusk.EntitySet`, `tusk.dfs` and `tusk.calculate_feature_matrix` must not exist.
- **Behaviour is frozen.** This is a rename. No signature reordering, no new arguments, no logic edits. Every assertion in the suite must keep passing unchanged except where it names a renamed symbol.
- **`tests/differential/test_vs_featuretools.py` calls two libraries.** `featuretools.EntitySet`, `.add_dataframe(dataframe_name=…, index=…)`, `featuretools.dfs(entityset=…, target_dataframe_name=…)` are *featuretools'* API and MUST NOT be renamed. Only the `tusk.` half of each comparison changes. Never run an unreviewed global substitution over this file.
- **Prose keeps "DFS" and "deep feature synthesis".** Only code identifiers lose the abbreviation. Do not rewrite prose uses of "DFS".
- **`docs/superpowers/specs/` and `docs/superpowers/plans/`** are historical records. They contain many old names. Leave every file in them untouched (this plan and its spec included).
- **Out of scope, do not touch:** `agg_primitives`, `trans_primitives`, `groupby_trans_primitives`, `AGG_DEFAULTS`, `TRANS_DEFAULTS`, `max_depth`, `cutoff_time`, `row_creation_time`, `primary_key`, `foreign_key`, `parent`, `child`, `features_only`, "feature matrix" as the name of the output, and the internal function names `synthesis.synthesize()` and `compiler.compile_features()`.
- **Standard test command:** `uv run --group validation pytest -q`. The validation group must be present or `tests/differential/` fails to *collect* (it imports pandas/featuretools) even though the marker deselects it. Baseline before any work: `149 passed, 1 skipped, 41 deselected`.
- **Differential test command:** `uv run --group validation pytest -q -m differential`. Baseline: `36 passed, 6 skipped`.
- **Pre-commit gates every commit** with ruff, ty, interrogate and pydoclint. pydoclint checks that docstring `Args:` names match the signature, so a renamed parameter whose docstring was not updated fails at commit time, not at test time. Update docstrings in the same edit as the signature.

---

### Task 1: Container — `EntitySet` becomes `Database`

**Files:**
- Rename: `src/tusk/entityset.py` → `src/tusk/database.py`
- Modify: `src/tusk/database.py`, `src/tusk/__init__.py`, `src/tusk/api.py`, `src/tusk/compiler.py`, `src/tusk/synthesis.py`, `src/tusk/features.py`
- Modify: `tests/conftest.py`, `tests/test_entityset.py`, `tests/test_compiler_single_table.py`, `tests/test_compiler_aggregation.py`, `tests/differential/test_vs_featuretools.py`, `benchmarks/test_relbench.py`
- Test: `tests/test_packaging.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `tusk.Database`, constructed as `Database(name: str)` with attribute `.name`; method `add_table(name: str, table: Any, primary_key: str | None = None, row_creation_time: str | None = None) -> Database`. All other members keep their current names: `add_relationship`, `table_names`, `schema`, `frame`, `is_eager`, `children_of`, `parents_of`, `input_excluded_columns`, `output_excluded_columns`. Module path is `tusk.database`; `Relationship` and `TableSchema` still live there and are still re-exported from `tusk`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_packaging.py`:

```python
def test_container_is_named_database():
    db = tusk.Database("retail")
    assert db.name == "retail"
    assert not hasattr(tusk, "EntitySet")
    assert not hasattr(db, "add_dataframe")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group validation pytest tests/test_packaging.py::test_container_is_named_database -q`
Expected: FAIL with `AttributeError: module 'tusk' has no attribute 'Database'`

- [ ] **Step 3: Move the module**

```bash
git mv src/tusk/entityset.py src/tusk/database.py
```

- [ ] **Step 4: Rename the class and its members in `src/tusk/database.py`**

Edit by hand, not by blanket substitution:

- Module docstring: `"""The schema model: tables, relationships, and the database that holds them."""`
- `class EntitySet:` → `class Database:`, docstring `"""A collection of related tables that DFS can synthesize features over."""` (unchanged text — "DFS" is prose).
- Constructor:

```python
    def __init__(self, name: str) -> None:
        """Create an empty database.

        Args:
            name: Human-readable identifier for this database.
        """
        self.name = name
```

- `add_dataframe` → `add_table`, with the `dataframe` parameter renamed to `table`:

```python
    def add_table(
        self,
        name: str,
        table: Any,
        primary_key: str | None = None,
        row_creation_time: str | None = None,
    ) -> Database:
        """Add a table to the database.

        Args:
            name: Name to register the table under.
            table: A native frame or a narwhals frame.
            primary_key: Column uniquely identifying a row. Required for a
                table used as a relationship parent or as the DFS target.
            row_creation_time: Column recording when a row became knowable.
                Required for order-dependent primitives on this table.

        Returns:
            This database, to allow chaining.

        Raises:
            SchemaError: If the name is taken, a declared column is missing,
                a key is composite, or the backend differs from earlier tables.

        Warns:
            MissingPrimaryKeyWarning: If ``primary_key`` is omitted.
        """
```

  In its body, rename the local uses of `dataframe` to `table`:

```python
        frame = (
            table
            if isinstance(table, (nw.DataFrame, nw.LazyFrame))
            else nw.from_native(table)
        )
```

- Both `-> EntitySet` return annotations (`add_table`, `add_relationship`) → `-> Database`, and `add_relationship`'s `Returns:` line → `This database, to allow chaining.`
- Error message strings that say "entity set":
  - `f"table {name!r} is already in this entity set"` → `f"table {name!r} is already in this database"`
  - the backend-mismatch message `"... but this "` / `f"entity set uses {self._backend}; ..."` → `f"database uses {self._backend}; ..."`
- Leave the `MissingPrimaryKeyWarning` message alone — `tests/test_entityset.py` matches `"cannot be used as a relationship parent"` and the message's "DFS target" is prose.

- [ ] **Step 5: Update the importers in `src/tusk/`**

```bash
sed -i 's/from tusk\.entityset import/from tusk.database import/' src/tusk/api.py src/tusk/compiler.py src/tusk/synthesis.py src/tusk/features.py
sed -i 's/\bEntitySet\b/Database/g' src/tusk/api.py src/tusk/compiler.py src/tusk/synthesis.py
```

Then update `src/tusk/__init__.py` by hand:

```python
from tusk import exceptions
from tusk.api import calculate_feature_matrix, dfs
from tusk.database import Database, Relationship, TableSchema

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Database",
    "Relationship",
    "TableSchema",
    "calculate_feature_matrix",
    "dfs",
    "exceptions",
]
```

Leave the parameter *names* (`entityset`) in `api.py`, `compiler.py` and `synthesis.py` alone — Task 3 handles those. Only the type annotations and imports change here.

- [ ] **Step 6: Update the test and benchmark call sites**

```bash
sed -i 's/from tusk\.entityset import/from tusk.database import/; s/tusk\.EntitySet(/tusk.Database(/g; s/\.add_dataframe(/.add_table(/g' \
  tests/test_entityset.py tests/test_compiler_single_table.py tests/test_compiler_aggregation.py tests/conftest.py benchmarks/test_relbench.py
```

`benchmarks/test_relbench.py` also has prose mentions of `EntitySet` in docstrings (around lines 102, 105, 117, 180) and a local variable `entityset` (lines 143-160, 183-200). Rename the local to `database` and the docstring mentions to "Database". Do not touch `tusk.dfs`/`tusk.calculate_feature_matrix` there yet — Task 2 owns those.

- [ ] **Step 7: Update the tusk half of the differential test by hand**

In `tests/differential/test_vs_featuretools.py`, change **only** the tusk constructions (around lines 127-129, 286-288 and any others):

```python
        tusk.Database("s")
        .add_table("customers", pl.from_pandas(customers).lazy(), primary_key="id")
        .add_table("sessions", pl.from_pandas(sessions).lazy(), primary_key="id")
```

Leave every `featuretools.EntitySet(...)`, `es.add_dataframe(dataframe_name=..., index=...)` and the module docstring's reference to featuretools 1.31.0's `EntitySet.add_dataframe` exactly as they are.

- [ ] **Step 8: Run the tests**

Run: `uv run --group validation pytest -q`
Expected: `150 passed, 1 skipped, 41 deselected` (the new packaging test brings the count from 149 to 150).

Run: `uv run --group validation pytest -q -m differential`
Expected: `36 passed, 6 skipped`

- [ ] **Step 9: Verify no stragglers**

Run:

```bash
grep -rIn -E "EntitySet|entityset|add_dataframe" src tests benchmarks | grep -v "^tests/differential/"
```

Expected: only `entityset` parameter names in `src/tusk/api.py`, `src/tusk/compiler.py` and `src/tusk/synthesis.py` (Task 3 removes them). No `EntitySet`, no `add_dataframe`, no `tusk.entityset` module path.

```bash
grep -rIn "featuretools.EntitySet\|dataframe_name" tests/differential/test_vs_featuretools.py
```

Expected: still present — those are featuretools' names and must survive.

- [ ] **Step 10: Commit**

```bash
git add -A src tests benchmarks
git commit -m "refactor: rename EntitySet to Database and add_dataframe to add_table"
```

---

### Task 2: Entry points — `deep_feature_synthesis` and `apply_features`

**Files:**
- Modify: `src/tusk/api.py`, `src/tusk/__init__.py`
- Modify: `tests/test_dfs.py`, `tests/test_backend_duckdb.py`, `tests/test_compiler_ordering.py`, `tests/differential/test_vs_featuretools.py`, `benchmarks/test_relbench.py`
- Test: `tests/test_packaging.py`

**Interfaces:**
- Consumes: `tusk.Database` from Task 1.
- Produces:
  - `tusk.deep_feature_synthesis(database: Database, target_table: str, agg_primitives=None, trans_primitives=None, groupby_trans_primitives=None, max_depth: int = 2, cutoff_time: Any = None, features_only: bool = False) -> Any` — parameter order unchanged from `dfs`, so existing positional calls such as `tusk.dfs(es, "customers", features_only=True)` keep working as `tusk.deep_feature_synthesis(db, "customers", features_only=True)`.
  - `tusk.apply_features(features: Sequence[Feature], database: Database, cutoff_time: Any = None) -> Any`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_packaging.py`:

```python
def test_entry_points_are_spelled_out():
    assert callable(tusk.deep_feature_synthesis)
    assert callable(tusk.apply_features)
    assert not hasattr(tusk, "dfs")
    assert not hasattr(tusk, "calculate_feature_matrix")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group validation pytest tests/test_packaging.py::test_entry_points_are_spelled_out -q`
Expected: FAIL with `AttributeError: module 'tusk' has no attribute 'deep_feature_synthesis'`

- [ ] **Step 3: Rename the functions in `src/tusk/api.py`**

- Module docstring: `"""The public entry points: :func:`deep_feature_synthesis` and :func:`apply_features`."""`
- `def dfs(` → `def deep_feature_synthesis(`, with `entityset: Database` → `database: Database` and `target_dataframe_name: str` → `target_table: str`.
- Docstring: summary becomes `Run deep feature synthesis over a database.`; the `Args:` entries are renamed to `database:` and `target_table:` (pydoclint fails the commit otherwise); every prose occurrence of "entity set" in the docstring becomes "database".
- Body:

```python
    features = synthesize(
        entityset=database,
        target_dataframe_name=target_table,
        agg_primitives=AGG_DEFAULTS if agg_primitives is None else agg_primitives,
        trans_primitives=TRANS_DEFAULTS
        if trans_primitives is None
        else trans_primitives,
        groupby_trans_primitives=groupby_trans_primitives or (),
        max_depth=max_depth,
    )
    if features_only:
        return features
    return apply_features(features, database, cutoff_time), features
```

  (`synthesize`'s own parameter names still read `entityset`/`target_dataframe_name` here — Task 3 renames them.)

- `def calculate_feature_matrix(` → `def apply_features(`, with `entityset: Database` → `database: Database`. Docstring summary: `Apply existing feature definitions to a database.`; `Args:` entry `database: The database to compute over.`; prose "entity set" → "database". Body:

```python
    frame = compile_features(features, database, cutoff_time)
    if database.is_eager:
        return frame.collect().to_native()
    return frame.to_native()
```

- [ ] **Step 4: Update `src/tusk/__init__.py`**

```python
from tusk import exceptions
from tusk.api import apply_features, deep_feature_synthesis
from tusk.database import Database, Relationship, TableSchema

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Database",
    "Relationship",
    "TableSchema",
    "apply_features",
    "deep_feature_synthesis",
    "exceptions",
]
```

- [ ] **Step 5: Update the call sites**

```bash
sed -i 's/tusk\.dfs(/tusk.deep_feature_synthesis(/g; s/tusk\.calculate_feature_matrix(/tusk.apply_features(/g; s/\bentityset=/database=/g; s/\btarget_dataframe_name=/target_table=/g' \
  tests/test_dfs.py tests/test_backend_duckdb.py tests/test_compiler_ordering.py benchmarks/test_relbench.py
```

Then check for keyword-free calls the sed missed:

```bash
grep -rIn "dfs\|calculate_feature_matrix" tests benchmarks | grep -v "^tests/differential/"
```

Fix any remaining tusk call sites by hand. Function *names* containing "dfs" in test names (e.g. `def test_dfs_end_to_end`) are cosmetic; rename them to `test_deep_feature_synthesis_end_to_end` style for consistency, but they are not load-bearing.

- [ ] **Step 6: Update the tusk half of the differential test by hand**

In `tests/differential/test_vs_featuretools.py`, the tusk calls (around lines 135-136 and 275-276 — check which block belongs to which library before editing):

```python
    got = tusk.deep_feature_synthesis(
        database=db,
        target_table="customers",
        ...
    )
```

The `featuretools.dfs(entityset=es, target_dataframe_name="customers", ...)` calls stay exactly as they are.

- [ ] **Step 7: Run the tests**

Run: `uv run --group validation pytest -q`
Expected: `151 passed, 1 skipped, 41 deselected`

Run: `uv run --group validation pytest -q -m differential`
Expected: `36 passed, 6 skipped`

- [ ] **Step 8: Commit**

```bash
git add -A src tests benchmarks
git commit -m "refactor: rename dfs to deep_feature_synthesis and calculate_feature_matrix to apply_features"
```

---

### Task 3: Internals — parameter names, attributes and docstring prose

**Files:**
- Modify: `src/tusk/synthesis.py`, `src/tusk/compiler.py`, `src/tusk/api.py`, `src/tusk/exceptions.py`
- Modify: `tests/test_synthesis.py`

**Interfaces:**
- Consumes: `tusk.Database`, `tusk.deep_feature_synthesis`, `tusk.apply_features` from Tasks 1-2.
- Produces: `synthesis.synthesize(database: Database, target_table: str, agg_primitives, trans_primitives, groupby_trans_primitives, max_depth) -> list[Feature]` and `compiler.compile_features(features, database: Database, cutoff_time) -> nw.LazyFrame`. Both function names are unchanged; only their parameters are renamed. `_SynthesisContext` exposes `self.database`.

- [ ] **Step 1: Rename the parameters and attributes**

```bash
sed -i 's/\bentityset\b/database/g; s/\btarget_dataframe_name\b/target_table/g' src/tusk/synthesis.py src/tusk/compiler.py src/tusk/api.py
```

This covers signatures, bodies, `self.entityset` → `self.database`, and the docstring `Args:` entries, keeping them in sync for pydoclint.

- [ ] **Step 2: Fix the prose and cross-references left behind**

```bash
grep -rIn "entity set\|tusk\.entityset\|dfs()" src
```

Rewrite each hit:
- `src/tusk/exceptions.py` — `"""Raised when a database's schema is invalid or inconsistent."""`
- `src/tusk/synthesis.py` — the cross-reference `:meth:`~tusk.entityset.EntitySet.output_excluded_columns`` becomes `:meth:`~tusk.database.Database.output_excluded_columns``; the class docstring `"""Carries the entity set and resolved primitives through the recursion."""` becomes `"""Carries the database and resolved primitives through the recursion."""`
- `src/tusk/compiler.py` — every `The entity set holding the frames.` / `The entity set, used to find ordering columns.` / `The entity set holding the schemas.` becomes "The database …".
- `src/tusk/api.py` — any remaining "entity set" in docstring prose becomes "database".
- `src/tusk/exceptions.py:30` and `src/tusk/synthesis.py:269` — the docstring prose
  ``zero-configuration ``dfs()`` on any schema …`` becomes
  ``zero-configuration ``deep_feature_synthesis()`` on any schema …``. These are code
  references in prose, not the permitted uppercase "DFS", and step 5's grep fails
  without this edit.

- [ ] **Step 3: Update `tests/test_synthesis.py`**

```bash
sed -i 's/\bentityset=/database=/g; s/\btarget_dataframe_name=/target_table=/g' tests/test_synthesis.py
```

- [ ] **Step 4: Run the tests**

Run: `uv run --group validation pytest -q`
Expected: `151 passed, 1 skipped, 41 deselected`

- [ ] **Step 5: Verify no old vocabulary survives in `src/`**

Run: `grep -rIn -E "EntitySet|entityset|entity set|add_dataframe|target_dataframe_name|calculate_feature_matrix|\bdfs\b" src`
Expected: no output at all. (Prose "DFS" in capitals is allowed and this pattern does not match it.)

- [ ] **Step 6: Commit**

```bash
git add -A src tests
git commit -m "refactor: rename internal entityset parameters to database"
```

---

### Task 4: Test module and fixture names

**Files:**
- Rename: `tests/test_entityset.py` → `tests/test_database.py`, `tests/test_dfs.py` → `tests/test_deep_feature_synthesis.py`
- Modify: `tests/conftest.py` and every test module using the `es` fixture

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: pytest fixture `db` (was `es`), returning a three-table retail `Database`. Every test that took `es` as an argument now takes `db`.

- [ ] **Step 1: Rename the test modules**

```bash
git mv tests/test_entityset.py tests/test_database.py
git mv tests/test_dfs.py tests/test_deep_feature_synthesis.py
```

- [ ] **Step 2: Rename the fixture in `tests/conftest.py`**

```python
@pytest.fixture
def db():
    """A three-table retail database."""
    customers, sessions, transactions = _frames()
    return (
        tusk.Database("retail")
        .add_table(
            "customers", customers, primary_key="id", row_creation_time="signed_up_at"
        )
        ...
    )
```

Keep the body otherwise identical — only the fixture name, its docstring and the already-renamed constructor calls change.

- [ ] **Step 3: Run the tests to watch them fail**

Run: `uv run --group validation pytest -q`
Expected: FAIL — many errors reading `fixture 'es' not found`. This is the proof that the next step is actually needed.

- [ ] **Step 4: Rename the fixture usages**

```bash
grep -rIln "\bes\b" tests | grep -v "^tests/differential/"
```

For each file listed, apply:

```bash
sed -i 's/\bes\b/db/g' <file>
```

Review each diff before committing: `\bes\b` is a short token and could in principle appear inside a docstring or a string literal. Check with `git diff` that only identifiers changed.

- [ ] **Step 5: Handle the differential test by hand**

`tests/differential/test_vs_featuretools.py` uses a local `es` for **both** libraries' objects. Rename only the tusk-side local to `db`, leaving `es = featuretools.EntitySet(...)` and its `.add_dataframe(...)` chain named `es`. The result should read as two clearly distinct objects per test.

- [ ] **Step 6: Run the tests**

Run: `uv run --group validation pytest -q`
Expected: `151 passed, 1 skipped, 41 deselected`

Run: `uv run --group validation pytest -q -m differential`
Expected: `36 passed, 6 skipped`

- [ ] **Step 7: Commit**

```bash
git add -A tests
git commit -m "test: rename entity-set fixture and test modules to database vocabulary"
```

---

### Task 5: Documentation and README

**Files:**
- Rename: `docs/guide/entity-sets.md` → `docs/guide/databases.md`, `docs/guide/dfs.md` → `docs/guide/deep-feature-synthesis.md`, `docs/api/entityset.md` → `docs/api/database.md`
- Modify: `docs/index.md`, `docs/guide/index.md`, `docs/guide/databases.md`, `docs/guide/deep-feature-synthesis.md`, `docs/guide/primitives.md`, `docs/guide/custom-primitives.md`, `docs/guide/featuretools.md`, `docs/api/index.md`, `docs/api/tusk.md`, `docs/api/database.md`, `zensical.toml`, `README.md`

**Interfaces:**
- Consumes: the full renamed public API from Tasks 1-4.
- Produces: no code. mkdocstrings identifiers must resolve to `tusk.Database`, `tusk.database`, `tusk.deep_feature_synthesis`, `tusk.apply_features`.

- [ ] **Step 1: Rename the documentation files**

```bash
git mv docs/guide/entity-sets.md docs/guide/databases.md
git mv docs/guide/dfs.md docs/guide/deep-feature-synthesis.md
git mv docs/api/entityset.md docs/api/database.md
```

- [ ] **Step 2: Update the nav in `zensical.toml`**

```toml
nav = [
    "index.md",
    { "Guide" = [
        "guide/index.md",
        "guide/databases.md",
        "guide/deep-feature-synthesis.md",
        "guide/primitives.md",
        "guide/custom-primitives.md",
        "guide/featuretools.md",
    ] },
    { "API reference" = [
        "api/index.md",
        "api/tusk.md",
        "api/database.md",
        "api/primitives.md",
        "api/features.md",
        "api/dtypes.md",
        "api/exceptions.md",
    ] },
]
```

- [ ] **Step 3: Rewrite the API reference pages**

`docs/api/tusk.md`:

```markdown
# Top level

::: tusk
    options:
      members: false

## deep_feature_synthesis

::: tusk.deep_feature_synthesis

## apply_features

::: tusk.apply_features
```

`docs/api/database.md`:

```markdown
# Databases

::: tusk.database
    options:
      members: false

## Database

::: tusk.Database

## Relationship

::: tusk.Relationship

## TableSchema

::: tusk.TableSchema
```

`docs/api/index.md` — update the two bullets:

```markdown
- [Top level](tusk.md) — `deep_feature_synthesis()` and `apply_features()`.
- [Databases](database.md) — `Database`, `Relationship`, `TableSchema`.
```

- [ ] **Step 4: Update the guide and index pages**

Apply to `docs/index.md`, `docs/guide/index.md`, `docs/guide/databases.md`, `docs/guide/deep-feature-synthesis.md`, `docs/guide/primitives.md`, `docs/guide/custom-primitives.md`, `README.md`:

```bash
sed -i \
 -e 's/tusk\.EntitySet(/tusk.Database(/g' \
 -e 's/\.add_dataframe(/.add_table(/g' \
 -e 's/tusk\.dfs(/tusk.deep_feature_synthesis(/g' \
 -e 's/tusk\.calculate_feature_matrix(/tusk.apply_features(/g' \
 -e 's/\bentityset=/database=/g' \
 -e 's/\btarget_dataframe_name=/target_table=/g' \
 -e 's/\[tusk\.EntitySet\]/[tusk.Database]/g' \
 -e 's/\[tusk\.EntitySet\./[tusk.Database./g' \
 -e 's/\[tusk\.dfs\]/[tusk.deep_feature_synthesis]/g' \
 -e 's/\[tusk\.calculate_feature_matrix\]/[tusk.apply_features]/g' \
 -e 's/`EntitySet`/`Database`/g' \
 -e 's/`add_dataframe`/`add_table`/g' \
 -e 's/`dfs()`/`deep_feature_synthesis()`/g' \
 -e 's/`calculate_feature_matrix`/`apply_features`/g' \
 -e 's/`calculate_feature_matrix()`/`apply_features()`/g' \
 -e 's/entity-sets\.md/databases.md/g' \
 -e 's/\bdfs\.md/deep-feature-synthesis.md/g' \
 -e 's/entityset\.md/database.md/g' \
 docs/index.md docs/guide/index.md docs/guide/databases.md docs/guide/deep-feature-synthesis.md docs/guide/primitives.md docs/guide/custom-primitives.md docs/guide/featuretools.md docs/api/index.md README.md

The link patterns match the *basename* on purpose. The links in the tree are
`(dfs.md#cutoff-times)`, `(dfs.md#feature-names-are-sql-identifiers)`,
`(guide/dfs.md)` and `(docs/guide/dfs.md)` — anchors and path prefixes vary, so a
pattern anchored on `(` misses most of them.
```

Then fix the remaining prose by hand:
- `docs/guide/databases.md` — the `# Entity sets` heading becomes `# Databases`; the opening sentence becomes "A [`Database`][tusk.Database] holds the tables you want features over and the relationships between them."; "adding a dataframe reads its schema" becomes "adding a table reads its schema"; "return the entity set" becomes "return the database"; "One entity set uses one backend" becomes "One database uses one backend".
- `docs/guide/deep-feature-synthesis.md` — the `# Running DFS` heading may stay (prose), but the opening `[`dfs()`][tusk.dfs] walks the entity set` becomes `[`deep_feature_synthesis()`][tusk.deep_feature_synthesis] walks the database`; the positional example `tusk.dfs(es, "customers", features_only=True)` becomes `tusk.deep_feature_synthesis(db, "customers", features_only=True)`; remaining "entity set" occurrences become "database".
- `docs/guide/index.md` — the numbered list: "Build an [entity set](entity-sets.md)" becomes "Build a [database](databases.md)"; the link labels "Entity sets" → "Databases" and "Running DFS" → its new path.
- `docs/index.md` and `README.md` — rename the local variable in the usage example from `es` to `db` (and `es_new` to `db_new`) so the snippets match the tests, and update the documentation link list.

- [ ] **Step 5: Extend `docs/guide/featuretools.md`**

Update the opening line — tusk no longer keeps featuretools' vocabulary wholesale — and add a mapping section so porters can translate:

```markdown
- **Renamed container and entry points.** The concepts are featuretools', the
  names are not:

  | featuretools | tusk |
  | --- | --- |
  | `EntitySet` | `Database` |
  | `EntitySet(id=…)` | `Database(name=…)` |
  | `es.add_dataframe(dataframe_name=…, index=…, time_index=…)` | `db.add_table(name, table, primary_key=…, row_creation_time=…)` |
  | `dfs(entityset=…, target_dataframe_name=…)` | `deep_feature_synthesis(database=…, target_table=…)` |
  | `calculate_feature_matrix(features, entityset)` | `apply_features(features, database)` |

  A collection of tables joined by primary and foreign keys is a database, and
  a function name should be a verb or a spelled-out term of art rather than an
  acronym. `apply_features` is neutral about laziness: it returns a query plan
  for lazy input and a materialized frame for eager input.
```

Also update the file's own prose uses of "entity set" (lines about backends and cutoff times) to "database".

- [ ] **Step 6: Verify the docs build**

Run: `uv run --group docs zensical build`
Expected: succeeds with no unresolved mkdocstrings identifiers and no broken internal links. `site/` is gitignored — do not commit it.

- [ ] **Step 7: Check the README example actually runs**

The README snippet is exercised by `tests/test_deep_feature_synthesis.py::test_readme_example_compiles_and_collects`. Confirm the snippet in `README.md` and the test still match, argument for argument.

Run: `uv run --group validation pytest tests/test_deep_feature_synthesis.py -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add -A docs README.md zensical.toml
git commit -m "docs: rename entity sets to databases and spell out the entry points"
```

---

### Task 6: Verification sweep

**Files:** none modified unless a check fails.

**Interfaces:**
- Consumes: everything.
- Produces: evidence that the rename is complete and behaviour-preserving.

- [ ] **Step 1: Full suite**

Run: `uv run --group validation pytest -q`
Expected: `151 passed, 1 skipped, 41 deselected`

- [ ] **Step 2: Differential suite**

Run: `uv run --group validation pytest -q -m differential`
Expected: `36 passed, 6 skipped` — unchanged from baseline, proving featuretools' own API is still called correctly.

- [ ] **Step 3: Import surface**

Run:

```bash
uv run python -c "
import tusk
assert tusk.Database and tusk.deep_feature_synthesis and tusk.apply_features
for gone in ('EntitySet', 'dfs', 'calculate_feature_matrix'):
    assert not hasattr(tusk, gone), gone
import tusk.database
try:
    import tusk.entityset
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('tusk.entityset still importable')
print('surface ok')
"
```

Expected: `surface ok`

- [ ] **Step 4: Grep for survivors**

Run:

```bash
grep -rIn -E "EntitySet|entityset|entity set|entity-set|add_dataframe|target_dataframe_name|calculate_feature_matrix" \
  src tests benchmarks docs/guide docs/api docs/index.md README.md zensical.toml
```

Expected: hits **only** in `tests/differential/test_vs_featuretools.py`, and every one of them on a `featuretools.` line or its module docstring. Anything else is a miss to fix.

Run:

```bash
grep -rIn "\bdfs\b" src tests benchmarks docs/guide docs/api docs/index.md README.md
```

Expected: no tusk call sites. Prose "DFS" (capitals) does not match and is fine.

- [ ] **Step 5: Lint gates**

Run: `uv run pre-commit run --all-files`
Expected: all hooks pass — notably interrogate (100% docstring coverage) and pydoclint (docstring `Args:` names match the renamed signatures).

- [ ] **Step 6: Review the whole diff**

Run: `git diff f92ab86..HEAD --stat` then read the source diff in full.
Confirm: no behaviour change, no signature reordering, no stray edits under `docs/superpowers/`.

- [ ] **Step 7: Commit any fixes**

If Steps 1-6 surfaced misses, fix and commit:

```bash
git add -A
git commit -m "refactor: finish the database vocabulary rename"
```
