# Value-count aggregations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute `mode` from per-group value counts so it runs on duckdb and the other SQL backends, through a reusable `ValueCountAggregationPrimitive`, and add `mode` to `AGG_DEFAULTS`.

**Architecture:** Before the child's `group_by`, the compiler adds a column holding how often each row's value occurs in its foreign-key group (`nw.len().over(foreign_key, value)`). A value-count primitive is a group-relative aggregation whose `build_per_row(values, counts)` reads that column; the existing per-row machinery wraps it in `.over(foreign_key)` and `build` reduces it inside the `group_by().agg()`.

**Tech Stack:** Python 3.10, narwhals ≥ 2.24, polars, duckdb, pytest, uv.

**Spec:** `agents/superpowers/specs/2026-09-24-value-count-aggregations-design.md`

## Global Constraints

- Work only in the worktree `/home/excidion/dev/tusk/.claude/worktrees/mode-count-then-pick` on branch `mode-count-then-pick`. Never commit on `main`.
- Follow `agents/style/CODESTYLE.md`. Function and method names are verbs or verb phrases; class names are nouns; one word per concept: `build` makes an expression, `add` puts columns on a frame, `generate` makes a name. No abbreviations.
- Docstrings say **what** a function does, never why or how. A comment explains **why**, only where the code would otherwise look wrong.
- Google-style docstrings with `Args:` and `Returns:` on every function in `src/` (pydoclint and interrogate enforce this; tests follow the file's existing style).
- Line length 88 (ruff).
- End every commit message with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0141PC66J9LXb1n8ZFhUGkq6
  ```
- Checks, run from the worktree: `uv run pytest -q`, `uv run --group validation pytest -q -m differential`, `uv run ruff check src tests`, `uv run ruff format --check src tests`, `uv run ty check`, `uv run interrogate`, `uv run pydoclint src`. `git commit` runs the same hooks; `pre-commit run --all-files` does not work with the installed git 2.25.

---

### Task 1: `ValueCountAggregationPrimitive` and its compiler support

**Files:**
- Modify: `src/tusk/primitives/base.py` (after `GroupRelativeAggregationPrimitive`, around line 257)
- Modify: `src/tusk/primitives/__init__.py` (import from `base`, `__all__`)
- Modify: `src/tusk/compiler.py` (imports; `_join_condition_aggregations` around line 440; `_build_per_row_column` around line 508; new functions)
- Modify: `docs/api/primitives.md` (base classes list)
- Modify: `docs/guide/custom-primitives.md` (base-class paragraph)
- Test: `tests/test_compiler_aggregation.py`

**Interfaces:**
- Produces: `tusk.primitives.ValueCountAggregationPrimitive`, subclass of `GroupRelativeAggregationPrimitive`, with abstract `build_per_row(self, values: nw.Expr, counts: nw.Expr) -> nw.Expr` and the inherited abstract `build(self, per_row: nw.Expr) -> nw.Expr`. `counts` is null wherever `values` is null.

- [ ] **Step 1: Write the failing test**

Add to the imports of `tests/test_compiler_aggregation.py`:

```python
from tusk.primitives.base import (
    AggregationPrimitive,
    NeedsCutoffTime,
    ValueCountAggregationPrimitive,
)
```

(replacing the existing `from tusk.primitives.base import AggregationPrimitive, NeedsCutoffTime`), and add below the `CutoffAggregation` class:

```python
@dataclass(frozen=True)
class SummedValueCount(ValueCountAggregationPrimitive):
    """Sum over a group's rows of how often each row's value occurs in it."""

    name = "summed_value_count"
    input_dtypes = (F.STRING,)
    output_dtype = nw.Int64

    def build_per_row(self, values, counts):
        return counts

    def build(self, per_row):
        return per_row.sum()
```

and at the end of the file:

```python
def test_a_value_count_aggregation_reads_each_rows_value_count():
    # parent 1: a, a, b, null -> counts 2, 2, 1, null
    # parent 2: c, null, null -> counts 1, null, null; a null value has no count
    # parent 3: no children
    labels = (
        tusk.Database("labels")
        .add_table("parents", pl.LazyFrame({"id": [1, 2, 3]}), primary_key="id")
        .add_table(
            "children",
            pl.LazyFrame(
                {
                    "id": [1, 2, 3, 4, 5, 6, 7],
                    "parent_id": [1, 1, 1, 1, 2, 2, 2],
                    "label": ["a", "a", "b", None, "c", None, None],
                },
            ),
            primary_key="id",
        )
        .add_relationship(parent="parents", child="children", foreign_key="parent_id")
    )
    feature = AggregationFeature(
        SummedValueCount(),
        (IdentityFeature("children", "label", nw.String()),),
        Relationship("parents", "children", "parent_id"),
    )
    got = collect([feature], labels)
    assert got["SUMMED_VALUE_COUNT__children__label"].to_list() == [5, 1, None]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_compiler_aggregation.py`
Expected: collection error, `ImportError: cannot import name 'ValueCountAggregationPrimitive'`.

- [ ] **Step 3: Add the base class**

In `src/tusk/primitives/base.py`, directly after `GroupRelativeAggregationPrimitive`:

```python
class ValueCountAggregationPrimitive(GroupRelativeAggregationPrimitive):
    """A group-relative aggregation reading each row's value and its count.

    It takes one input column. :meth:`build_per_row` receives that column and,
    per row, how often the row's value occurs in its group.
    """

    @abstractmethod
    def build_per_row(self, values: nw.Expr, counts: nw.Expr) -> nw.Expr:
        """Build the per-row expression from each row's value and its count.

        Args:
            values: The input column.
            counts: How often each row's value occurs in its group; null
                where the value is null.

        Returns:
            A narwhals expression with one value per row.
        """
```

In `src/tusk/primitives/__init__.py`, add `ValueCountAggregationPrimitive` to the `from tusk.primitives.base import (...)` block and `"ValueCountAggregationPrimitive"` to `__all__`, then run `uv run ruff check --fix src/tusk/primitives/__init__.py` to sort both.

- [ ] **Step 4: Teach the compiler**

In `src/tusk/compiler.py`, add `ValueCountAggregationPrimitive` to the `from tusk.primitives.base import (...)` block.

In `_join_condition_aggregations`, replace

```python
    child = _add_per_row_columns(child, relationship, batch)
```

with

```python
    child = _add_value_count_columns(child, relationship, batch)
    child = _add_per_row_columns(child, relationship, batch)
```

Insert directly above `_add_per_row_columns`:

```python
def _add_value_count_columns(
    child: nw.LazyFrame,
    relationship: Relationship,
    batch: Sequence[AggregationFeature],
) -> nw.LazyFrame:
    """Add the count column every value-count aggregation reads.

    Args:
        child: The child frame, already filtered to the condition.
        relationship: The relationship being aggregated across.
        batch: The aggregation features sharing this condition.

    Returns:
        The child frame with one column per value-count feature.
    """
    counts = [
        _build_value_count_column(feature, relationship)
        for feature in batch
        if isinstance(feature.primitive, ValueCountAggregationPrimitive)
    ]
    return child.with_columns(*counts) if counts else child


def _build_value_count_column(
    feature: AggregationFeature,
    relationship: Relationship,
) -> nw.Expr:
    """Build how often each row's value occurs in its group.

    Args:
        feature: The value-count aggregation feature.
        relationship: The relationship whose foreign key forms the groups.

    Returns:
        The named count expression; null where the value is null.
    """
    value = feature.base_features[0].name
    count = nw.len().over(relationship.foreign_key, value)
    return (
        nw.when(~nw.col(value).is_null())
        .then(count)
        .alias(_generate_value_count_column_name(feature))
    )
```

Replace the body of `_build_per_row_column` (keep its signature and docstring):

```python
    inputs = _select_build_per_row_inputs(feature)
    per_row = primitive.build_per_row(*inputs).over(relationship.foreign_key)
    return per_row.alias(_generate_per_row_column_name(feature))
```

Insert directly below `_build_per_row_column`:

```python
def _select_build_per_row_inputs(feature: AggregationFeature) -> list[nw.Expr]:
    """Build the columns a group-relative aggregation's per-row expression reads.

    Args:
        feature: The group-relative aggregation feature.

    Returns:
        The feature's input columns, then its count column if it is a
        value-count aggregation.
    """
    inputs = [nw.col(b.name) for b in feature.base_features]
    if isinstance(feature.primitive, ValueCountAggregationPrimitive):
        inputs.append(nw.col(_generate_value_count_column_name(feature)))
    return inputs
```

Insert directly below `_generate_per_row_column_name`:

```python
def _generate_value_count_column_name(feature: AggregationFeature) -> str:
    """Name the child column holding a value-count aggregation's counts.

    Args:
        feature: The value-count aggregation feature.

    Returns:
        The column name.
    """
    return f"{feature.name}__value_count"
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest -q tests/test_compiler_aggregation.py`
Expected: all pass.

- [ ] **Step 6: Document the base class**

In `docs/api/primitives.md`, after `::: tusk.primitives.GroupRelativeAggregationPrimitive` add a blank line and:

```markdown
::: tusk.primitives.ValueCountAggregationPrimitive
```

In `docs/guide/custom-primitives.md`, replace

```markdown
[`GroupRelativeAggregationPrimitive`][tusk.primitives.GroupRelativeAggregationPrimitive],
since SQL backends reject an aggregate nested in an aggregate. Subclass
```

with

```markdown
[`GroupRelativeAggregationPrimitive`][tusk.primitives.GroupRelativeAggregationPrimitive],
since SQL backends reject an aggregate nested in an aggregate. One that needs
how often each value occurs in its group, as `mode` does, subclasses
[`ValueCountAggregationPrimitive`][tusk.primitives.ValueCountAggregationPrimitive]:
its `build_per_row(values, counts)` receives the column and each row's count.
Subclass
```

- [ ] **Step 7: Run the full checks**

Run: `uv run pytest -q && uv run ruff check src tests && uv run ruff format --check src tests && uv run ty check && uv run interrogate && uv run pydoclint src`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/tusk/primitives/base.py src/tusk/primitives/__init__.py src/tusk/compiler.py tests/test_compiler_aggregation.py docs/api/primitives.md docs/guide/custom-primitives.md
git commit -F - <<'EOF'
value-count aggregations

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0141PC66J9LXb1n8ZFhUGkq6
EOF
```

---

### Task 2: `mode` as a value-count aggregation

**Files:**
- Modify: `src/tusk/primitives/aggregation.py` (module docstring, imports, `Mode`)
- Modify: `tests/test_backend_duckdb.py` (restore the pre-`mode` version)
- Modify: `docs/guide/primitive-coverage.md` (`mode` row)

**Interfaces:**
- Consumes: `ValueCountAggregationPrimitive` from Task 1.
- Produces: `Mode` with `build_per_row(values, counts)` and `build(per_row)`; the registry name `"mode"` is unchanged.

- [ ] **Step 1: Write the failing test**

Restore the duckdb suite to its state before `mode` existed, which parametrizes over every entry of `EXPECTED`, `mode` included:

```bash
git checkout 26e1ded -- tests/test_backend_duckdb.py
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_backend_duckdb.py -k "standalone_aggregations and mode"`
Expected: FAIL with `NotImplementedError: 'drop_nulls' is not implemented for ... duckdb`.

- [ ] **Step 3: Rebuild `Mode` on value counts**

In `src/tusk/primitives/aggregation.py`, restore the module docstring's second paragraph to:

```python
"""Built-in aggregation primitives.

Every expression here is legal inside a lazy ``group_by().agg()``. Length-changing
expressions such as ``mode()`` are not -- narwhals rejects them on lazy frames --
which is why ``quantiles`` rather than ``n_most_common`` is the multi-output
primitive.
"""
```

Add `ValueCountAggregationPrimitive` to the `from tusk.primitives.base import (...)` block, and replace the whole `Mode` class with:

```python
@register
@dataclass(frozen=True)
class Mode(ValueCountAggregationPrimitive):
    """Most frequent known value of a label column; a tie gives the smallest value."""

    name = "mode"
    input_dtypes = ((F.STRING,), (F.CATEGORICAL,))

    def build_per_row(self, values: nw.Expr, counts: nw.Expr) -> nw.Expr:
        """Build each row's value where it is among its group's most frequent.

        Args:
            values: The label column.
            counts: How often each row's value occurs in its group.

        Returns:
            A narwhals expression; null on every other row.
        """
        return nw.when(counts == counts.max()).then(values)

    def build(self, per_row: nw.Expr) -> nw.Expr:
        """Build the smallest of the most frequent values.

        Args:
            per_row: The most frequent values, null on every other row.

        Returns:
            A narwhals expression.
        """
        return per_row.min()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/test_backend_duckdb.py tests/test_primitives_aggregation.py && uv run --group validation pytest -q -m differential tests/differential/test_aggregations.py`
Expected: all pass, including `test_standalone_aggregations_give_the_polars_values_on_duckdb[mode]`, `test_standalone_aggregations_on_every_kind_of_group[mode]` and `test_standalone_aggregations_match_featuretools_on_every_parent_row[mode-mode-MODE(children.label)]`.

- [ ] **Step 5: Update the coverage row**

In `docs/guide/primitive-coverage.md`, in the row starting `| Aggregation | [\`mode\`][tusk.primitives.Mode]`, replace the final cell ` Not available on duckdb or other SQL backends. |` with `  |`, leaving the comment empty like the other ✅ rows.

- [ ] **Step 6: Run the full checks**

Run: `uv run pytest -q && uv run ruff check src tests && uv run ruff format --check src tests && uv run ty check && uv run interrogate && uv run pydoclint src`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/tusk/primitives/aggregation.py tests/test_backend_duckdb.py docs/guide/primitive-coverage.md
git commit -F - <<'EOF'
compute mode from value counts, so it runs on SQL backends

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0141PC66J9LXb1n8ZFhUGkq6
EOF
```

---

### Task 3: `mode` joins the default aggregations

**Files:**
- Modify: `src/tusk/primitives/aggregation.py` (`AGG_DEFAULTS`)
- Modify: `tests/test_primitives_aggregation.py` (`test_defaults_are_the_documented_set`)
- Modify: `tests/conftest.py` (`_frames`, `transactions`)
- Modify: `tests/test_sklearn_dfs_transformer.py` (`test_it_routes_the_database_through_a_pipeline`)

**Interfaces:**
- Consumes: `Mode` from Task 2.
- Produces: `AGG_DEFAULTS` ending in `"mode"`.

- [ ] **Step 1: Write the failing test**

In `tests/test_primitives_aggregation.py`, add `"mode",` after `"percent_true",` in `test_defaults_are_the_documented_set`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_primitives_aggregation.py -k test_defaults_are_the_documented_set`
Expected: FAIL, the tuples differ by `'mode'`.

- [ ] **Step 3: Add `mode` to the defaults**

In `src/tusk/primitives/aggregation.py`, add `"mode",` after `"percent_true",` in `AGG_DEFAULTS`.

- [ ] **Step 4: Run the suite to see the zero-config test fail**

Run: `uv run pytest -q tests/test_primitives_aggregation.py tests/test_synthesis.py`
Expected: `test_defaults_are_the_documented_set` passes; `test_zero_config_run_warns_about_nothing` FAILS, since no child table of the `db` fixture has a string column for `mode`.

- [ ] **Step 5: Give the standard schema a label column**

In `tests/conftest.py`, in `_frames`, add to the `transactions` frame after `"verified": [True, False, True, False],`:

```python
            "channel": ["web", "web", "shop", "web"],
```

- [ ] **Step 6: Run the suite to see the pipeline test fail**

Run: `uv run pytest -q`
Expected: only `tests/test_sklearn_dfs_transformer.py::test_it_routes_the_database_through_a_pipeline` FAILS, because default DFS now outputs the string column `MODE__...channel`, which `SimpleImputer` rejects.

- [ ] **Step 7: Impute only the numeric columns**

In `tests/test_sklearn_dfs_transformer.py`, add the imports

```python
from sklearn.compose import ColumnTransformer
```

(next to the other `sklearn` imports) and change `from tusk.sklearn import DFSTransformer` to `from tusk.sklearn import DFSTransformer, dtype_selector`. In `test_it_routes_the_database_through_a_pipeline`, replace the comment and the `("impute", ...)` step so the test reads:

```python
@pytest.mark.filterwarnings(_INTERCHANGE_DEPRECATION)
def test_it_routes_the_database_through_a_pipeline(db):
    # An imputer sits between dfs and clf: customer 3 has no sessions and
    # session 30 has no transactions, so max_depth=2 legitimately produces
    # null MEAN/MIN/MAX/STD features for those nested empty groups (confirmed
    # against tusk.deep_feature_synthesis directly, independent of this
    # wrapper) -- real DFS output, not something LogisticRegression accepts
    # unpreprocessed. It takes the numeric columns only, since MODE is a
    # string. What is under test here is metadata routing, not
    # feature-matrix completeness, so the fix belongs in the fixture pipeline
    # rather than in DFSTransformer.
    with sklearn.config_context(enable_metadata_routing=True):
        pipe = Pipeline(
            [
                ("dfs", _transformer()),
                (
                    "impute",
                    ColumnTransformer(
                        [
                            (
                                "numbers",
                                SimpleImputer(keep_empty_features=True),
                                dtype_selector("numeric"),
                            ),
                        ],
                    ),
                ),
                ("clf", LogisticRegression()),
            ],
        )
        pipe.fit(KEYS, Y, database=db)
        assert len(pipe.predict([1, 2], database=db)) == 2
```

- [ ] **Step 8: Run the full checks**

Run: `uv run pytest -q && uv run --group validation pytest -q -m differential && uv run ruff check src tests && uv run ruff format --check src tests && uv run ty check && uv run interrogate && uv run pydoclint src`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/tusk/primitives/aggregation.py tests/test_primitives_aggregation.py tests/conftest.py tests/test_sklearn_dfs_transformer.py
git commit -F - <<'EOF'
mode joins the default aggregations

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0141PC66J9LXb1n8ZFhUGkq6
EOF
```

---

### Task 4: Pull request

**Files:** none.

- [ ] **Step 1: Review the branch diff**

Run: `git diff main...HEAD --stat` and `git diff main...HEAD -- src`
Check: no helper column leaks into the feature matrix, every new function name is a verb phrase, and no docstring explains why.

- [ ] **Step 2: Push and open the pull request**

```bash
git push -u origin mode-count-then-pick
gh pr create --base main --title "Compute mode from value counts" --body-file - <<'EOF'
`mode` now runs on duckdb and the other SQL backends, and joins the default aggregations.

- `ValueCountAggregationPrimitive`: a group-relative aggregation whose `build_per_row(values, counts)` also receives how often each row's value occurs in its group. The compiler adds that count as a window column (`len().over(foreign_key, value)`) before the existing per-row column.
- `mode` is built on it: the smallest of the values holding the group's highest count. Values are unchanged, and still match featuretools.
- `mode` joins `AGG_DEFAULTS`, as featuretools' DFS uses it by default. Default DFS output now holds string columns, so the sklearn pipeline test imputes numeric columns only.

Backends, probed with narwhals 2.24: polars, pandas, pyarrow, duckdb, sqlframe and ibis give featuretools' answer; dask rejects `when/then` inside `over`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_0141PC66J9LXb1n8ZFhUGkq6
EOF
```

Expected: the PR URL is printed. Do not merge.
