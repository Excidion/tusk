# STE rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the guide pages (loose) and every docstring in `src/` (strict) with Simplified Technical English, on one glossary, and rename `src/` identifiers that use banned terms.

**Architecture:** A glossary file comes first. Then renames in `src/`, which do not change behavior. Then docstrings, one group of modules per task. Then guide pages, one group of pages per task. Each task ends with the full checks and one commit.

**Tech Stack:** Python 3.10, narwhals, uv, pytest, ruff, ty, interrogate, pydoclint, zensical.

**Spec:** `agents/superpowers/specs/2026-09-25-ste-docs-rewrite-design.md`

## Global Constraints

- Work only in `/home/excidion/dev/tusk/.claude/worktrees/docs-ste` on branch `docs-ste`. Never commit on `main`. Never push.
- Read `.claude/skills/simplified-technical-english/SKILL.md` before the first rewrite. Docstrings use its **Strict** mode. Guide pages use its **STE-flavored** mode.
- Use only the glossary terms in `agents/GLOSSARY.md` (Task 1). Names from other libraries stay (`nw.LazyFrame`, `DataFrame`, featuretools' `EntitySet`, `dataframe_name`, `calculate_feature_matrix`, `dataclasses.field`, sklearn's `handle_unknown`), and so does "entity-relationship diagram".
- "unknown" meaning "not recognized" (e.g. "unknown table") is allowed. "unknown" meaning null is not.
- Docstrings say **what** the code does, never why. Comments are not changed.
- Keep Google-style sections (`Args:`, `Returns:`, `Raises:`, `Attributes:`), every documented name, and every fact: conditions, defaults, units, exceptions. Never add a fact.
- Do not change behavior. Do not change `README.md`, `docs/index.md`, code comments, or files in `agents/superpowers/` other than this plan's checkboxes.
- Guide pages: sentences only. Pages, headings and their order stay. Tables keep their columns.
- Line length 88 (ruff).
- End every commit message with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0141PC66J9LXb1n8ZFhUGkq6
  ```
- **Full checks**, run before each commit:
  ```bash
  uv run pytest -q
  uv run --group validation pytest -q -m differential
  uv run ruff check src tests && uv run ruff format --check src tests
  uv run ty check && uv run interrogate && uv run pydoclint src
  uv run --group docs zensical build --clean
  ```
  `git commit` runs the pre-commit hooks. `pre-commit run --all-files` does not work with the installed git 2.25.
- **Banned-term search**, run on the files a task changed. Every hit must be an allowed exception from the list above:
  ```bash
  grep -nwiP "frames?|dataframes?|(?<!feature )matrix|calculat\w*|creat\w*|generat\w*|cutoff(?![ _]time)|known|missing|NaN|records?|fields?|engines?|entity" <files>
  ```
- **STE linter**, for guide pages. Report its findings in the task report, but they do not block the task:
  ```bash
  python3 .claude/skills/simplified-technical-english/scripts/ste-lint.py <files>
  ```

## Strict mode for docstrings, in short

- One idea per sentence. At most 20 words for an instruction, 25 for a description.
- No semicolons. Split into two sentences.
- Active voice, simple present tense.
- No phrasal verbs ("set up", "look up"). Use one plain verb.
- A verb, not a noun, for an action ("computes", not "performs the computation of").
- Keep articles and subjects ("The column that holds…", not "Column holding…").
- Summary line: a verb for functions and methods ("Build …", "Return …"), a noun phrase for classes, modules and properties.

Examples from this code base:

| Before | After |
| --- | --- |
| `Number of known values of a numeric column above the group's mean.` | `Number of non-null values of a numeric column above the group's mean.` |
| `Most frequent known value of a label column; a tie gives the smallest value.` | `Most frequent non-null value of a label column. A tie gives the smallest value.` |
| `Return a table's lazy frame.` | `Return a table as a narwhals LazyFrame.` |
| `The child frame, already filtered to the condition.` | `The child table, after the condition filter.` |
| `The named count expression; null where the value is null.` | `The named count expression. It is null where the value is null.` |

---

### Task 1: Glossary

**Files:**
- Create: `agents/GLOSSARY.md`
- Modify: `CLAUDE.md`, `agents/AGENTS.md`

- [ ] **Step 1: Create `agents/GLOSSARY.md`** with exactly:

```markdown
# Glossary

Use one term per concept, in docs, docstrings and identifiers.

| Concept | Term | Do not use |
| --- | --- | --- |
| A missing value | null | missing, NaN, None, unknown. Use "NaN" only to quote featuretools output. |
| A value that is not null | non-null value | known value |
| A table in a database, and the data object that holds it | table | frame, dataframe, data frame, entity |
| The result of synthesis | feature matrix | matrix |
| The time that limits which rows are used | cutoff time | cutoff |
| Work out a value | compute | calculate |
| Make an expression, a feature or a name | build | generate, create |
| Parts of a table | row, column | record, field |
| polars, duckdb, pandas, ... | backend | engine, dataframe backend |
| The child rows that share one foreign key value | group | |
| A group without rows | empty group | |
| A set of tables and relationships | database | entity set |
| An operation that makes features | primitive | |

Names from other libraries stay as they are, for example `nw.LazyFrame`,
featuretools' `EntitySet` and `calculate_feature_matrix`. "Unknown" in the
sense "not recognized", as in "unknown table", is allowed.
```

- [ ] **Step 2: Import it in `CLAUDE.md`.** Add the line `@agents/GLOSSARY.md` after `@agents/AGENTS.md`.

- [ ] **Step 3: Point to it from `agents/AGENTS.md`.** Append:

```markdown

## Terms
Use the terms in `agents/GLOSSARY.md` in docs, docstrings and identifiers.
```

- [ ] **Step 4: Check and commit.** Run `git diff --stat`. It must show 3 files. Then:

```bash
git add agents/GLOSSARY.md CLAUDE.md agents/AGENTS.md
git commit -F - <<'EOF'
glossary of terms for docs, docstrings and identifiers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0141PC66J9LXb1n8ZFhUGkq6
EOF
```

---

### Task 2: Public renames

**Files:** `src/tusk/primitives/base.py`, `src/tusk/database.py`, `src/tusk/compiler.py`, and every file in `src/`, `tests/`, `docs/` that uses the renamed names.

**Interfaces:**
- Produces: `Primitive.build_name`, `Primitive.build_display_name`, `Database.table(name)`, `tusk.compiler.base_table`. The old names no longer exist.

- [ ] **Step 1: Find every use.**

```bash
grep -rn "generate_name\|generate_display_name\|\.frame(\|def frame\|base_frame" src tests docs
```

- [ ] **Step 2: Rename.**
  - `Primitive.generate_name` → `build_name`, and every override and call.
  - `Primitive.generate_display_name` → `build_display_name`, and every override and call.
  - `Database.frame` → `Database.table`, and every call.
  - `compiler.base_frame` → `base_table`, and every call.
  Update the docstrings of the renamed methods to the new verb ("Build the column name…").

- [ ] **Step 3: Check that no old name is left.** The Step 1 command must print nothing.

- [ ] **Step 4: Full checks and commit.** Message: `rename public names to the glossary terms`. The commit body lists the four renames and says that custom primitives that override `generate_name` or `generate_display_name` must rename them.

---

### Task 3: Private renames in `src/`

**Files:** all of `src/tusk/`, and tests that import a renamed private name.

**Interfaces:**
- Consumes: the Task 2 names.

- [ ] **Step 1: List the names to change.**

```bash
grep -rnoE "\b[A-Za-z_]*(generate|frame|matrix|calculate|create|cutoff|known)[A-Za-z_]*\b" src --include=*.py | grep -v "LazyFrame\|DataFrame\|cutoff_time\|unknown\|dataframe_name\|add_dataframe\|entityset\|feature_matrix"
```

- [ ] **Step 2: Rename** each function, method, parameter, variable, attribute and module:
  - `frame`/`frames` → `table`/`tables`. Where a function already has a `table` that holds a table's **name** (a `str`), rename that one to `table_name` first, then rename the frame to `table`. Examples: `_table_frame` → `_read_table`, `self._frames` → `self._tables`, `times_frame` → `times_table`.
  - Module `src/tusk/sklearn/_frames.py` → `_tables.py`, and its imports.
  - `matrix` alone → `feature_matrix`, e.g. `collect_matrix` → `collect_feature_matrix`.
  - `cutoff` alone → `cutoff_time`, e.g. `cutoff_aware` → `cutoff_time_aware`, `_names_measuring_against_cutoff` → `_names_measuring_against_cutoff_time`. `check_cutoff_time_zone` stays.
  - `known` → `non_null`, e.g. `known_flags` → `non_null_flags`.
  - `generate…`/`created` → `build…`/`built`, e.g. `_generate_comparison_column_name` → `_build_comparison_column_name`.
  - Names that are public API (in `__all__` or documented in `docs/api/`) other than Task 2's are **not** renamed here. List any you find in the report instead.

- [ ] **Step 3: Check.** The Step 1 command prints only allowed names. Full checks pass.

- [ ] **Step 4: Commit.** Message: `rename private names to the glossary terms`.

---

### Tasks 4–10: Docstrings, strict mode

Each task rewrites every docstring (module, class, function, method, property) in its files. The procedure is the same for each:

- [ ] **Step 1: Rewrite** each docstring with the strict rules and the glossary. Keep every section, name and fact.
- [ ] **Step 2: Banned-term search** on the task's files. Every hit is an allowed exception.
- [ ] **Step 3: Full checks.**
- [ ] **Step 4: Commit.** Message: `STE docstrings: <module names>`.

| Task | Files | Docstrings |
| --- | --- | --- |
| 4 | `src/tusk/primitives/transform.py` | 113 |
| 5 | `src/tusk/primitives/aggregation.py` | 91 |
| 6 | `src/tusk/primitives/base.py`, `registry.py`, `primitives/__init__.py`, `src/tusk/dtypes.py`, `src/tusk/exceptions.py` | 54 |
| 7 | `src/tusk/features.py`, `feature_list.py`, `api.py`, `src/tusk/__init__.py` | 74 |
| 8 | `src/tusk/compiler.py`, `synthesis.py` | 45 |
| 9 | `src/tusk/database.py`, `validation.py`, `plotting.py` | 74 |
| 10 | `src/tusk/sklearn/*.py` | 40 |

---

### Tasks 11–14: Guide pages, STE-flavored mode

Each task rewrites the sentences of its pages. The procedure is the same for each:

- [ ] **Step 1: Rewrite** the prose with the STE-flavored rules and the glossary. Pages, headings, their order, code blocks, links and table columns stay. Code blocks change only where they use a name renamed in Tasks 2–3.
- [ ] **Step 2: Banned-term search** and **STE linter** on the task's pages. Report the linter's findings.
- [ ] **Step 3: Full checks** (the docs build shows broken links).
- [ ] **Step 4: Commit.** Message: `STE guide: <page names>`.

| Task | Pages |
| --- | --- |
| 11 | `docs/guide/databases.md` |
| 12 | `docs/guide/index.md`, `deep-feature-synthesis.md`, `primitives.md` |
| 13 | `docs/guide/custom-primitives.md`, `sklearn.md` |
| 14 | `docs/guide/featuretools.md`, `primitive-coverage.md` |

---

### Task 15: Pull request

Run by the controller after the final review, not by a subagent.

- [ ] **Step 1:** `git push -u origin docs-ste`
- [ ] **Step 2:** `gh pr create --base main --title "STE rewrite of the guide and the docstrings"` with a body that lists the glossary, the public renames and their break for custom primitives, and what is not changed.
