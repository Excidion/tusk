# Aggregation primitive parity

Spec: none. The authority is `docs/guide/primitive-coverage.md` (the parity
table built on this branch) plus the decisions recorded under Global
Constraints below.

## Context

`docs/guide/primitive-coverage.md` lists every featuretools 1.31.0 primitive
beside its tusk counterpart. In the Aggregation group, three rows are not
backed by a per-primitive test:

| tusk | featuretools | Status now |
| --- | --- | --- |
| `median` | `median` | ❓ both implement it, no test |
| `percent_true` | `percent_true` | ❌ diverges on nulls, no test |
| `quantiles` | — | ➕ tusk only, no test link |

`n_unique` is ⚠️ (intentional divergence on empty groups) and already pinned by
`test_n_unique_of_an_empty_group_diverges_from_featuretools`. No work.

Missing featuretools aggregations (`all`, `any`, `skew`, ...) are out of scope
for this plan.

## Global Constraints

- **One commit per primitive.** Each task ends in exactly one commit covering
  its code, test, and coverage-table row.
- **One test script per primitive group.** All new aggregation differential
  tests go in `tests/differential/test_aggregations.py`, a new file. Do not
  modify `tests/differential/test_vs_featuretools.py`; that file tests the DFS
  pipeline, not individual primitives.
- **Differential tests are opt-in.** Mark the module
  `pytestmark = pytest.mark.differential` and gate the imports with
  `pytest.importorskip`, as `test_vs_featuretools.py` does. Run them with
  `uv run --group validation pytest -m differential`.
- **Test one primitive at a time.** Each test drives
  `tusk.deep_feature_synthesis` with a single entry in `agg_primitives` and the
  featuretools counterpart with a single entry in `agg_primitives`, then
  compares the one column. "Simple" is the requirement — no matrix-wide
  parametrization.
- **`percent_true` null decision (user ruling, 2026-08-30):** tusk changes to
  match featuretools. A null counts as `False` in the denominator, i.e.
  `[True, False, True, True, None]` → `0.6`, not `0.75`.
- **Code style:** `CODESTYLE.md` applies. Docstrings say what, comments say why.
- **Verification:** every task runs
  `uv run --group validation pytest -m differential` and `uv run pytest` before
  committing, and reports the output.

## Task 1: A per-primitive differential harness, proven on `median`

Create `tests/differential/test_aggregations.py`.

It needs a fixture giving one parent table and one child table where the child
carries a numeric column with nulls and at least one parent row with no
children, plus a helper pair that runs a single named aggregation through tusk
and through featuretools and returns the two columns aligned on the parent key.
Model the entity-set construction and the tusk↔featuretools name translation on
`tests/differential/test_vs_featuretools.py` (`_as_tusk`, `_featuretools_matrix`,
`_tusk_matrix`), but keep this file self-contained rather than importing from
it.

Then add the `median` test: assert tusk's `MEDIAN(child.value)` equals
featuretools' `MEDIAN(child.value)` on every parent row, empty groups included.
If they disagree, do not change tusk — stop and report the disagreement with the
values, because that turns a ❓ into a divergence question for the user.

Update the `median` row of `docs/guide/primitive-coverage.md`: Status ✅, Test
column linking to the new test function in the new file, following the exact
link format already used in that table.

Commit.

## Task 2: `percent_true` matches featuretools on nulls

Change `PercentTrue.build` in `src/tusk/primitives/aggregation.py` so a null
counts as `False`: fill nulls with `False` before averaging. The expression must
stay legal inside a lazy `group_by().agg()` — a reduction only, no
length-changing (filtration) call such as `drop_nulls`. Update the class
docstring to state the null rule.

`tests/test_primitives_aggregation.py` has a `percent_true` case in
`test_aggregations_over_group_one` whose fixture column `b` has no nulls, so it
should still pass unchanged. Add a unit test there pinning the null rule on a
column that does contain nulls.

Add the differential test to `tests/differential/test_aggregations.py`, using
the Task 1 harness over a boolean child column containing nulls and an empty
parent group.

Update the `percent_true` row of `docs/guide/primitive-coverage.md`: Status ✅,
Test column linking to the new differential test, and clear the comment (the
divergence is gone).

Commit.

## Task 3: `quantiles` gets its test link

`quantiles` is tusk-only, so it has no differential counterpart. Its behaviour
is already covered by `test_quantiles_is_multi_output` in
`tests/test_primitives_aggregation.py`, which asserts the multi-output shape but
not the values.

Extend that file with a test asserting the actual quantile values for a group
with hand-checkable data, and asserting the default `qs` are `(0.25, 0.5, 0.75)`.

Update the `quantiles` row of `docs/guide/primitive-coverage.md` to link the
Test column at that test, using the same link format as the other rows but
pointing into `tests/test_primitives_aggregation.py`. Add a one-sentence comment
explaining the test is a unit test because featuretools has no counterpart.

Commit.
