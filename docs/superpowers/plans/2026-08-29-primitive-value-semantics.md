# Primitive Value Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject primitives that are not frozen dataclasses at the point they enter tusk, so features deduplicate correctly and a pickled feature set equals the one it was saved from.

**Architecture:** One guard in `resolve()`, the single function every primitive passes through on its way into synthesis. It runs on the resolved instance, so it covers both a user-constructed instance and a registered name. No public signature changes; the failure moves from a misleading compile-time error to an immediate one naming the class.

**Tech Stack:** Python 3, `dataclasses`, pytest, ruff, ty, pre-commit.

**Spec:** `docs/superpowers/specs/2026-08-29-primitive-value-semantics-design.md`

## Global Constraints

- The error type is `PrimitiveError` (`src/tusk/exceptions.py:14`). Do not add a new exception type.
- The predicate is `dataclasses.is_dataclass(cls)` **and** `cls.__dataclass_params__.frozen`, read off `type(primitive)`.
- No signature changes to `resolve`, `resolve_all`, `register`, or any public entry point.
- All 28 built-in primitives already comply. If a change makes one fail, the change is wrong — do not edit a built-in to suit it.
- Every commit must pass `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check src tests`, and `uv run ty check`. The pre-commit hooks run these plus `pydoclint` and `interrogate`, which require Google-style docstrings with `Args:`/`Returns:`/`Raises:` sections on every public function. A `Raises:` section must list only exceptions raised in that function's own body.
- Do not use `git commit --no-verify`.

## Already covered — do not re-implement

Spec test 5 ("every registered built-in passes the check") already exists as
`test_every_registered_primitive_is_a_frozen_dataclass` in
`tests/test_primitives_base.py:96`, alongside
`test_every_registered_primitive_round_trips_through_pickle` at line 104. Both
iterate `_REGISTRY`. Leave them alone; Task 1 must not break them.

**Consequence for test design:** those tests assert that *everything in
`_REGISTRY`* is a frozen dataclass. Registering a deliberately-bad primitive at
module scope in any test file would make them fail. Task 1 therefore registers
its bad primitive inside a fixture that restores the registry afterwards.

---

### Task 1: Reject non-frozen-dataclass primitives in `resolve()`

**Files:**
- Modify: `src/tusk/primitives/registry.py:35-54` (the `resolve` function)
- Modify: `docs/guide/custom-primitives.md:26-32`
- Test: `tests/test_primitives_base.py`

**Interfaces:**
- Consumes: `PrimitiveError` from `tusk.exceptions`; `_REGISTRY`, `register`, `resolve` from `tusk.primitives.registry`.
- Produces: `resolve(spec)` keeps its signature `(str | Primitive) -> Primitive` and still returns the *same object* when given an instance (`resolve(inst) is inst`), which `test_registry_resolves_names_and_passes_instances_through` at line 78 depends on. Adds a private `_require_value_semantics(primitive: Primitive) -> None` in the same module.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_primitives_base.py`. Note the existing imports at the top
of that file already cover `dataclass`, `is_dataclass`, `pytest`,
`PrimitiveError`, `TransformPrimitive`, `F`, `_REGISTRY`, `register` and
`resolve` — add nothing to the import block.

```python
@pytest.fixture
def restore_registry():
    """Undo any registration a test performs.

    test_every_registered_primitive_is_a_frozen_dataclass asserts that every
    entry in _REGISTRY complies, so a deliberately-bad primitive must not
    outlive the test that needs it.
    """
    before = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(before)


class PlainClassPrimitive(TransformPrimitive):
    """A primitive written the ordinary way: no dataclass, identity equality."""

    name = "plain_class_primitive"
    input_dtypes = (F.NUMERIC,)

    def __init__(self, k=2.0):
        self.k = k

    def build(self, expr):
        return expr * self.k


@dataclass
class UnfrozenPrimitive(TransformPrimitive):
    """A dataclass that forgot frozen=True, so it is unhashable."""

    name = "unfrozen_primitive"
    input_dtypes = (F.NUMERIC,)
    k: float = 2.0

    def build(self, expr):
        return expr * self.k


def test_a_plain_class_primitive_instance_is_rejected():
    with pytest.raises(PrimitiveError, match="PlainClassPrimitive"):
        resolve(PlainClassPrimitive(3.0))


def test_a_plain_class_primitive_is_rejected_by_registered_name(restore_registry):
    register(PlainClassPrimitive)
    with pytest.raises(PrimitiveError, match="PlainClassPrimitive"):
        resolve("plain_class_primitive")


def test_an_unfrozen_dataclass_primitive_is_rejected():
    # Without the check this reaches synthesis and dies with an unhelpful
    # "unhashable type" much further downstream.
    with pytest.raises(PrimitiveError, match="UnfrozenPrimitive"):
        resolve(UnfrozenPrimitive(3.0))


def test_the_rejection_names_the_fix():
    with pytest.raises(PrimitiveError, match="frozen dataclass"):
        resolve(PlainClassPrimitive(3.0))


def test_a_compliant_primitive_still_resolves_unchanged():
    instance = Pair(scale=3.0)
    assert resolve(instance) is instance
    assert resolve("doubled") == Doubled()


def test_the_rejection_reaches_a_user_through_the_public_entry_point(db):
    with pytest.raises(PrimitiveError, match="PlainClassPrimitive"):
        tusk.deep_feature_synthesis(
            database=db,
            target_table="customers",
            agg_primitives=[],
            trans_primitives=[PlainClassPrimitive(3.0)],
            features_only=True,
        )
```

The last test needs the `db` fixture from `tests/conftest.py`, which pytest
supplies automatically. It needs no new import: the existing
`import tusk.primitives` at the top of the file already binds the name `tusk`,
so `tusk.deep_feature_synthesis` resolves. Do not add `import tusk` — it is
redundant.

The unannotated `build(self, expr)` and `__init__(self, k=2.0)` above match
the primitives already defined in this file (`Doubled`, `Spread`, `Pair` at
lines 14-47). `pyproject.toml:79-88` already scopes a ty override for
`invalid-method-override` over `tests/**`, so the fixed-arity `build` is
accepted.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_primitives_base.py -q`

Expected: the four rejection tests FAIL. `test_a_plain_class_primitive_instance_is_rejected` fails with `DID NOT RAISE <class 'tusk.exceptions.PrimitiveError'>`. `test_the_rejection_reaches_a_user_through_the_public_entry_point` fails the same way — it currently returns a `FeatureList` instead of raising. `test_a_compliant_primitive_still_resolves_unchanged` PASSES already; it is a regression guard, not a new behavior.

- [ ] **Step 3: Write the implementation**

In `src/tusk/primitives/registry.py`, add `import dataclasses` to the import
block (above `from collections.abc import Iterable`, so ruff's isort rule `I`
keeps stdlib grouped and alphabetical), then replace the body of `resolve` and
add the helper below it:

```python
def resolve(spec: str | Primitive) -> Primitive:
    """Turn a name or instance into a primitive instance.

    Args:
        spec: A registered primitive name, or an already-built instance.

    Returns:
        A primitive instance.

    Raises:
        PrimitiveError: If the name is not registered, or the primitive is not
            a frozen dataclass.
    """
    if isinstance(spec, Primitive):
        primitive = spec
    else:
        try:
            primitive = _REGISTRY[spec]()
        except KeyError:
            known = ", ".join(sorted(_REGISTRY))
            msg = f"unknown primitive {spec!r}; available: {known}"
            raise PrimitiveError(msg) from None
    _require_value_semantics(primitive)
    return primitive


def _require_value_semantics(primitive: Primitive) -> None:
    """Reject a primitive that compares by identity rather than by value.

    Features are frozen dataclasses holding a primitive, so a feature's
    equality and hash are only as good as its primitive's. Deduplication
    during synthesis and matching a saved feature set against a fresh one both
    depend on that.

    Args:
        primitive: The resolved primitive to check.

    Raises:
        PrimitiveError: If it is not a frozen dataclass.
    """
    cls = type(primitive)
    params = getattr(cls, "__dataclass_params__", None)
    if dataclasses.is_dataclass(cls) and params is not None and params.frozen:
        return
    raise PrimitiveError(
        f"primitive {cls.__name__!r} must be a frozen dataclass. tusk compares "
        "primitives by value to deduplicate features and to match a saved "
        "feature set against a fresh one; a primitive that compares by "
        "identity silently produces duplicate features. Decorate it with "
        "@dataclass(frozen=True) -- see docs/guide/custom-primitives.md.",
    )
```

Note `resolve` now assigns and returns at the end rather than returning early,
so the check covers both paths. Returning `primitive` (not a copy) preserves
`resolve(inst) is inst`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_primitives_base.py -q`
Expected: PASS.

Then the whole suite, which must be unchanged apart from the new tests:

Run: `uv run pytest -q`
Expected: `358 passed, 42 deselected` (352 before, plus the 6 new tests).

- [ ] **Step 5: Update the custom-primitives guide**

`docs/guide/custom-primitives.md` currently shows the frozen dataclass and says
parameters are ordinary dataclass fields. Make the requirement explicit.
Replace the paragraph at lines 26-32:

```markdown
Then pass `"range"` or `Range()` to `deep_feature_synthesis()`. Parameters are
ordinary dataclass fields.

`@dataclass(frozen=True)` is required, not a style preference. tusk compares
primitives by value — to drop duplicate features during synthesis, and to
match a feature set loaded from disk against a freshly synthesized one. A
plain class compares by object identity, so two identically configured
instances would look different and both would survive. A primitive that is
not a frozen dataclass is rejected with `PrimitiveError` as soon as it reaches
`deep_feature_synthesis()`.

There is no second, shorter way to declare one. Every built-in primitive is a
frozen dataclass written out like this, so `Year` and `Count` are the same kind
of object as `Range` — nothing in tusk can reach a definition path your own
code cannot.
```

- [ ] **Step 6: Verify the docs build**

Run: `uv run --group docs zensical build`
Expected: `No issues found`.

- [ ] **Step 7: Run every check, then commit**

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run ty check
uv run pytest -q
git add src/tusk/primitives/registry.py tests/test_primitives_base.py docs/guide/custom-primitives.md
git commit -m "Require primitives to be frozen dataclasses

A primitive that is a plain class compares by object identity, so two
identically configured instances are unequal. Features hold primitives and
are compared by value, so that broke deduplication during synthesis and
made a pickled feature set unequal to the one it was saved from.

resolve() now rejects one, on both the instance and the registered-name
path, naming the class and the fix. All 28 built-ins already complied."
```

---

### Task 2: Pin the feature-set pickle guarantee

**Files:**
- Test: `tests/test_feature_list.py`

**Interfaces:**
- Consumes: `resolve`'s new guarantee from Task 1; `tusk.deep_feature_synthesis`; the `db` fixture from `tests/conftest.py`; the `features` fixture already defined at the top of `tests/test_feature_list.py`.
- Produces: nothing other tasks consume. This is the end of the plan.

**Why this is its own task:** Task 1 protects a property of *features*, but every existing test of that property is about *primitives* (`tests/test_primitives_base.py:104` pickles primitives, not feature sets). Nothing currently pins the thing the change exists for.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_feature_list.py`. Add `import pickle` to its import
block, at the top of the stdlib group above `from collections.abc import
Sequence`.

```python
def test_a_feature_list_survives_pickle_equal_and_hash_stable(features):
    """The property the frozen-dataclass rule on primitives exists to protect.

    A loaded feature set has to equal the one it was saved from, or it will
    not deduplicate against a freshly synthesized set and the combination
    produces duplicate columns.
    """
    restored = pickle.loads(pickle.dumps(features))
    assert restored == features
    assert hash(restored) == hash(features)
    assert restored.target_table == features.target_table
    assert restored.output_names == features.output_names


def test_a_restored_feature_list_still_computes(features, db):
    restored = pickle.loads(pickle.dumps(features))
    assert restored.apply(db).collect().equals(features.apply(db).collect())


def test_restored_features_deduplicate_against_a_fresh_run(features, db):
    """A loaded set combined with a fresh one must collapse, not double."""
    restored = pickle.loads(pickle.dumps(features))
    combined = tusk.FeatureList(dict.fromkeys([*restored, *features]))
    assert len(combined) == len(features)
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_feature_list.py -q`

Expected: PASS on all three. These are regression guards — the behavior is
already correct for compliant primitives, and Task 1 is what keeps a
non-compliant one from reaching this code path at all. If any fails, stop:
either Task 1 broke something or the guarantee was never real.

- [ ] **Step 3: Run every check, then commit**

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run ty check
uv run pytest -q
git add tests/test_feature_list.py
git commit -m "Pin the feature-set pickle guarantee

Nothing tested that a FeatureList round-trips equal and hash-stable, which
is the property the frozen-dataclass rule on primitives protects."
```

---

## Verification

After both tasks:

```bash
uv run pytest -q                       # 361 passed, 42 deselected
uv run ruff check .                    # All checks passed!
uv run ruff format --check src tests   # 45 files already formatted
uv run ty check                        # All checks passed!
uv run --group docs zensical build     # No issues found
```

Counts: 352 before this plan, +6 in Task 1, +3 in Task 2.

## Out of scope

- Guarding hand-built features (`TransformFeature(MyPrim(), ...)`). The spec accepts this gap; `Feature` classes are public and bypass `resolve()`.
- Any change to `register()`. It still accepts a non-compliant class; `resolve()` is what rejects it. Moving the check into `register()` would miss unregistered instances.
- A JSON serialization format. Pickle is the chosen route.
