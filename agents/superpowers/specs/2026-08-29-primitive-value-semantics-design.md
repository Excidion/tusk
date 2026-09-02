# Primitives must be frozen dataclasses

Status: draft — pending review
Date: 2026-08-29

## Problem

A primitive is the extension point: users subclass `TransformPrimitive` or
`AggregationPrimitive` to add operations tusk does not ship.
`docs/guide/custom-primitives.md` writes every example as a frozen dataclass,
and all 28 built-ins are written that way. Nothing checks it.

The distinction matters because tusk compares primitives by value. A frozen
dataclass compares by its fields, so two separately constructed
`Quantiles(qs=(0.5,))` are equal and hash alike. A plain class — the ordinary
thing a Python programmer writes — compares by object identity, so two
identically configured instances are not equal.

Features are frozen dataclasses holding a primitive, so a feature's equality
and hash are only as good as its primitive's. Two places depend on that:

- **Deduplication during synthesis.** The same feature is often reached by
  more than one route through the graph; `synthesize` collapses the repeats
  with `dict.fromkeys`, which is hash-and-equality based.
- **Identity across a save and load.** A pickled feature set is only useful if
  it equals the set it was saved from.

### Observed failures

Three shapes are writable today. Only the third is a problem, and it is the
default thing to write.

| How the primitive is written | What happens |
| --- | --- |
| `@dataclass(frozen=True)` | Correct. |
| `@dataclass` | `TypeError: unhashable type` during synthesis. Loud, immediate, no wrong answer. |
| plain class | Two failures, below. |

**Duplicate features, misdiagnosed.** Deduplication misses, so two identical
features survive. Compilation then rejects them with `features 'SCALED(x)' and
'SCALED(x)' both compile to column 'SCALED__x'; rename a source column to
break the tie`. The advice is wrong for this cause and sends the user looking
at their data instead of their primitive.

**Broken identity across pickle.** Verified on a round-trip: a feature set
built from a plain-class primitive comes back unequal to the original, with a
different hash. A loaded set does not recognize itself, and does not
deduplicate against a fresh run. Combining a loaded set with a newly
synthesized one yields duplicate columns, with nothing to indicate why. This
is the failure that motivates the work now that pickle is the intended
persistence route.

## Decision

Require every primitive entering tusk to be a frozen dataclass, and reject the
others where they enter, with an error naming the class and the fix.

### Where the check goes

`resolve()` in `src/tusk/primitives/registry.py:35`. Every primitive reaching
synthesis passes through it — `resolve_all` is the only caller, from
`synthesis.py:85-89` — whether it arrives as a registry name or as an instance
the user constructed.

The check applies to the resolved instance on **both** paths, not only to
user-constructed instances. `register()` accepts a plain class today, so a
registered name can also yield a non-compliant primitive; checking after
resolution covers both with one test.

The predicate, on the resolved instance:

```python
dataclasses.is_dataclass(prim) and type(prim).__dataclass_params__.frozen
```

`is_dataclass` accepts an instance, and `__dataclass_params__` lives on the
class, so both parts read the class behind the instance.

### Alternatives rejected

**`Primitive.__init_subclass__`.** It runs when the class body ends, before
the `@dataclass` decorator is applied, so it observes `is_dataclass=False` for
every primitive including the compliant built-ins. It cannot work.

**Make `Primitive` itself a frozen dataclass.** A subclass that adds a field
without repeating the decorator inherits a comparison over zero fields, making
a primitive with `k=3` equal to one with `k=5`. That replaces a loud failure
with a silent wrong answer and is strictly worse than today.

**`register()` only.** It runs after the decorator, so it works, but it covers
only registered classes. An instance passed directly to
`deep_feature_synthesis` never registers, and that is a supported path.

**Documentation alone.** The failure surfaces far from the mistake and names
the wrong cause. This is what the repo does today.

## Public surface

No signature changes. One new error, raised from `resolve()`:

```
PrimitiveError: primitive 'Scaled' must be a frozen dataclass: tusk compares
primitives by value to deduplicate features and to match a saved feature set
against a fresh one, and a plain class compares by identity. Decorate it with
@dataclass(frozen=True). See docs/guide/custom-primitives.md.
```

`PrimitiveError` is the existing type for primitive problems and is already
raised by `resolve()` for unknown names.

## Scope

**Covered.** Every primitive passed to `deep_feature_synthesis`,
`synthesize`, or the sklearn transformers, as a name or an instance.

**Not covered.** Features constructed by hand, e.g.
`TransformFeature(MyPrim(), (base,))`. The `Feature` classes are public
(`docs/api/features.md`) and bypass `resolve()`. This is an advanced path and
is left unguarded rather than adding a second check inside the feature
constructors.

The gap is acceptable because the normal route is covered and the uncovered
route is one a user reaches only by importing `tusk.features` deliberately.

## Breaking change

Anyone who wrote a primitive as a plain class gets an error where they
previously got duplicate features and a confusing compile failure. All 28
built-ins already comply, verified, so nothing internal changes.

tusk is at 0.1.0 and unreleased, so no deprecation period is offered.

## Testing

1. A frozen-dataclass primitive resolves unchanged — instance and registry
   name.
2. A plain-class primitive raises `PrimitiveError` naming the class, whether
   passed as an instance or resolved from a registered name.
3. A non-frozen `@dataclass` primitive raises the same error, rather than the
   `TypeError: unhashable type` it raises today. Reaching the check earlier is
   an improvement, not a regression.
4. The error reaches a user through the public entry point, not only through
   `resolve()` directly.
5. Every registered built-in passes the check. This pins the guarantee the
   breaking-change section claims.
6. A feature set built from a compliant primitive survives a pickle round-trip
   equal and hash-stable. This is the property the whole change exists to
   protect, and nothing currently tests it.

## Non-goals

- Enforcing anything about `build()`, dtypes, or naming.
- Guarding hand-built features.
- Any change to how primitives are registered or resolved by name.
- A serialization format. Pickle is the chosen route; JSON is out of scope.
