# STE rewrite of the guide and the docstrings

Rewrite tusk's guide pages and docstrings with the Simplified Technical English
skill (`.claude/skills/simplified-technical-english`), on one fixed set of
terms.

## Scope

| Text | Mode | Change |
| --- | --- | --- |
| `docs/guide/*.md` | STE-flavored | Sentences only. Pages, headings and their order stay. |
| Every docstring in `src/`, public and private | Strict | Sentences, and the terms below. |
| Code identifiers in `src/`, `tests/`, `docs/` | — | Renamed where a name uses a banned term. |

Not changed: `README.md`, `docs/index.md`, code comments, and the existing
specs and plans in `agents/superpowers/`. A spec or plan changes only if a
rename breaks a statement in it.

## Glossary

One term per concept, in the guide, the docstrings and the identifiers.

| Concept | Term | Not |
| --- | --- | --- |
| A missing value | null | missing, NaN, None, unknown. "NaN" only when quoting featuretools output. |
| A value that is not null | non-null value | known value |
| A table in a database, and the data object that holds it | table | frame, dataframe, data frame, entity |
| The result of synthesis | feature matrix | matrix |
| The time that limits which rows are used | cutoff time | cutoff |
| Work out a value | compute | calculate |
| Make an expression, a feature, a name | build | generate, create |
| Parts of a table | row, column | record, field |
| polars, duckdb, pandas, ... | backend | engine, dataframe backend |
| The child rows that share one foreign key value | group | — |
| A group without rows | empty group | — |
| A set of tables and relationships | database | entity set |
| An operation that makes features | primitive | — |

Exceptions: names from other libraries stay as they are (`nw.LazyFrame`,
featuretools' `EntitySet`, `dataframe_name`, `calculate_feature_matrix`), and
so does "entity-relationship diagram".

The glossary lives in `agents/GLOSSARY.md`. `CLAUDE.md` imports it with
`@agents/GLOSSARY.md`. `agents/AGENTS.md` gets one line that points to it.

## Renames

Every identifier that uses a banned term is renamed to the glossary term.
Known cases:

- `Primitive.generate_name` → `build_name`
- `Primitive.generate_display_name` → `build_display_name`
- `_generate_…_column_name` → `_build_…_column_name`
- `base_frame`, `_table_frame`, and parameters and variables named `frame` →
  names with `table`

The full list comes from a search at the start of the work. Behavior does not
change. Renaming the two public `Primitive` methods breaks custom primitives
that override them. The pull request says so.

## Order of work

1. `agents/GLOSSARY.md`, the import in `CLAUDE.md`, the line in `AGENTS.md`.
2. Renames, in one commit.
3. Docstrings, one module at a time.
4. Guide pages, one page at a time.

## Checks

For each step:

- `uv run pytest -q`, `uv run --group validation pytest -q -m differential`
- `uv run ruff check`, `ruff format --check`, `ty check`, `interrogate`,
  `pydoclint src`
- the docs build (`uv run --group docs zensical build --clean`)
- a search of the changed files for the banned terms in the glossary
- for guide pages: `ste-lint.py` on the page. Its findings are reported, not
  enforced.

No new pre-commit hook and no new check script.
