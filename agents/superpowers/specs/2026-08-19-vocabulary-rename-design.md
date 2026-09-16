# Vocabulary rename: entity sets and dataframes become databases and tables

Status: approved for planning
Date: 2026-08-19

## Problem

tusk speaks two vocabularies at once.

The schema layer already broke from featuretools deliberately, and
`docs/guide/featuretools.md` documents why: `primary_key` and
`row_creation_time` instead of `index` and `time_index`, three-argument
relationships with an explicit `foreign_key`. Internally the unit of data is a
*table* throughout — `TableSchema`, `table_names`, `EntitySet.schema()`,
`Feature.table`.

The public container and entry points never followed. They are still
featuretools' words: `EntitySet`, `add_dataframe`, `target_dataframe_name`,
`dfs`, `calculate_feature_matrix`. So a user adds a *dataframe* to an *entity
set* and gets back an object whose own API calls the result a *table*.

Two further defects in the entry points:

- `dfs` is an abbreviation of a noun phrase. A function is an action; its name
  should be a verb or a spelled-out term of art, not an acronym.
- `calculate_feature_matrix` implies computation, but with lazy frames in,
  nothing is computed — a query plan comes back. The name is wrong on the
  library's headline path.

## Decision

Adopt one register — the physical one — across the public surface. A
collection of tables joined by primary and foreign keys is a database; that is
the word every user already knows for it, it matches every internal name, and
it is what relbench (which the benchmarks compare against) calls this exact
object.

### Public name map

| Now | Becomes |
| --- | --- |
| `tusk.EntitySet` | `tusk.Database` |
| `EntitySet(id="retail")` | `Database(name="retail")` |
| `EntitySet.id` | `Database.name` |
| `.add_dataframe(name, dataframe, …)` | `.add_table(name, table, …)` |
| `tusk.dfs(entityset=…, target_dataframe_name=…)` | `tusk.deep_feature_synthesis(database=…, target_table=…)` |
| `tusk.calculate_feature_matrix(features, entityset, cutoff_time)` | `tusk.apply_features(features, database, cutoff_time)` |
| `src/tusk/entityset.py` | `src/tusk/database.py` |
| "entity set" in prose, docstrings and error messages | "database" |

Unchanged, because they are already right: `add_relationship(parent, child,
foreign_key)`, `table_names`, `schema()`, `frame()`, `is_eager`,
`input_excluded_columns()`, `output_excluded_columns()`, `TableSchema`,
`Relationship`, `Feature.table`, and every primitive, dtype and exception name.

### Why these names

- **`Database`** over `Ontology`: an ontology names entity *types* and
  relations at the conceptual level, and the honest version of that choice
  would drag `primary_key`, `foreign_key` and `parent`/`child` along with it.
  Those are settled, documented decisions. `Database` costs nothing internally
  and makes the library speak one register. It carries a faint suggestion of a
  live connection, which is false here — the object is built from frames — but
  that is the only cost.
- **`name` over `id`**: `id` shadows a builtin and overstates what the value
  is. It is a human-readable label, never used for lookup.
- **`deep_feature_synthesis`**: the term of art, spelled out. Discoverable by
  anyone arriving from featuretools, and no abbreviation.
- **`apply_features`**: laziness-neutral. `materialize` and `compute` overclaim
  when frames are lazy (a plan comes back); `compile` and `plan` underclaim
  when frames are eager (real data comes back, after the library's single
  `collect()`). "Apply these definitions to this database" is true in both
  directions, and is what the existing docstring already says the function is
  for.

### Alternatives rejected

- **Deprecated aliases for the old names.** tusk is unreleased: version 0.1.0,
  no git tags, no dependents. A compat layer would ship a permanent second
  vocabulary to serve nobody. Clean break.
- **Splitting the phases into three entry points** (`featurize` /
  `deep_feature_synthesis` / `apply_features`, dropping `features_only`). A
  genuine improvement — `features_only=True` changes the return type and makes
  `cutoff_time` silently inert — but it is an API shape change, not a rename.
  Out of scope; worth revisiting separately.
- **`Ontology` with `add_table`.** Mixes registers within a single line of
  code.
- **`Schema` as the container name.** Collides with `TableSchema` and the
  existing `.schema(name)` method.

## Scope

**In:** the container class and its `add_dataframe` method; the two public
entry points and their parameter names; the internal `entityset` parameter and
attribute names in `compiler.py`, `synthesis.py` and `features.py`, which name
the renamed class; the module file; prose, docstrings and error messages that
say "entity set"; tests and their fixtures; the documentation site.

**Out:** primitive vocabulary (`agg_primitives`, `trans_primitives`,
`groupby_trans_primitives`, `AGG_DEFAULTS`, `TRANS_DEFAULTS`, `max_depth`);
time vocabulary (`cutoff_time`, `row_creation_time`, "timeless" tables);
key vocabulary (`primary_key`, `foreign_key`, `parent`, `child`);
`features_only`; "feature matrix" as the name of the output; all behaviour.

## Change inventory

Source (`src/tusk/`):

- `entityset.py` → `database.py`. `EntitySet` → `Database`, `id` → `name`,
  `add_dataframe` → `add_table` (its `dataframe` parameter → `table`), module
  and class docstrings, and the `SchemaError` messages that say "entity set".
- `api.py`: both functions renamed, parameters `entityset` → `database` and
  `target_dataframe_name` → `target_table`, module docstring, and the
  docstring prose that says "entity set".
- `synthesis.py`, `compiler.py`, `features.py`: imports, type annotations,
  parameter and attribute names (`self.entityset` → `self.database`), and the
  `~tusk.entityset.EntitySet` docstring cross-references.
- `exceptions.py`: `SchemaError` docstring.
- `__init__.py`: imports and `__all__`.

Internal function names `synthesis.synthesize()` and
`compiler.compile_features()` stay. They do not collide with the new public
names and are not part of the public surface.

Tests:

- `test_entityset.py` → `test_database.py`; `test_dfs.py` →
  `test_deep_feature_synthesis.py`.
- `conftest.py`: the `es` fixture → `db`, and its docstring.
- Every test module that constructs an entity set or calls the entry points,
  plus the `es` local variable convention.
- `benchmarks/test_relbench.py`.

Docs:

- `docs/guide/entity-sets.md` → `docs/guide/databases.md`; `docs/guide/dfs.md`
  → `docs/guide/deep-feature-synthesis.md`; `docs/api/entityset.md` →
  `docs/api/database.md`. Update `zensical.toml` nav and every cross-link,
  including `docs/index.md`, `docs/guide/index.md`, `docs/api/index.md` and the
  README's documentation list.
- `docs/api/tusk.md` and `docs/api/database.md`: the `::: tusk.…` mkdocstrings
  identifiers.
- `docs/guide/featuretools.md`: add the vocabulary mapping (`EntitySet` →
  `Database`, `add_dataframe` → `add_table`, `dfs` →
  `deep_feature_synthesis`, `calculate_feature_matrix` → `apply_features`) so
  the porting guide covers the container and entry points as it already covers
  keys and relationships.
- `README.md`: the usage example and the doc links.

## Hazards

1. **`tests/differential/test_vs_featuretools.py` calls featuretools' own
   API.** `featuretools.EntitySet`, `add_dataframe(dataframe_name=…, index=…)`,
   `featuretools.dfs(entityset=…, target_dataframe_name=…)` are that library's
   names and must survive untouched. Only the `tusk.…` half of each comparison
   changes. A blind global substitution breaks this file; it must be edited
   deliberately. The same applies to the module docstring's reference to
   featuretools 1.31.0's `EntitySet.add_dataframe`.
2. **"DFS" in prose is not a code identifier.** Docstrings and guides use "DFS"
   and "deep feature synthesis" to name the algorithm. That prose is fine and
   stays; only identifiers lose the abbreviation. Renaming prose occurrences of
   "DFS" is out of scope.
3. **`superpowers/specs/` and `superpowers/plans/`** hold dated
   records of past decisions and contain the largest concentration of the old
   names. They are history and stay verbatim. This spec is the record of the
   change.
4. **`site/` is generated output.** Regenerate rather than edit; do not
   hand-patch its HTML.

## Verification

The rename preserves behaviour, so the existing suite is the oracle.

1. `uv run pytest` fully green, including
   `tests/differential/test_vs_featuretools.py` (which proves featuretools'
   own API still works alongside the renamed tusk API) and
   `tests/test_backend_duckdb.py`.
2. `uv run --group docs zensical build` succeeds with no broken mkdocstrings
   identifiers and no dead internal links.
3. `grep -rIn -E "EntitySet|entityset|entity set|add_dataframe|target_dataframe_name|calculate_feature_matrix"`
   over `src`, `tests`, `benchmarks`, `docs/guide`, `docs/api`, `docs/index.md`,
   `README.md` and `zensical.toml` returns only the deliberate featuretools
   references in `tests/differential/test_vs_featuretools.py`.
4. `grep -rIn "\bdfs\b"` over the same paths returns no tusk call sites — only
   prose and the featuretools differential test.
5. `python -c "import tusk; tusk.Database; tusk.deep_feature_synthesis; tusk.apply_features"`
   and confirmation that `tusk.EntitySet`, `tusk.dfs` and
   `tusk.calculate_feature_matrix` no longer exist.
