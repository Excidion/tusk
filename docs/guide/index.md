# Guide

tusk turns a set of related tables into a single wide feature matrix. The
workflow is always the same three steps:

1. **Describe your data.** Build a [database](databases.md): register each
   table, say which column is its primary key and which column records when a
   row became knowable, then link the tables with relationships.
2. **Synthesize.** Call [`deep_feature_synthesis()`](deep-feature-synthesis.md). It walks the relationship graph,
   stacking [primitives](primitives.md) up to `max_depth`, and returns both the
   feature matrix and the feature definitions that produced it.
3. **Re-apply.** Feed those definitions back to `apply_features()` to
   compute the same columns on new data.

Everything tusk builds is a narwhals expression, so the whole pipeline is one
query plan on the backend you already use. Nothing is materialized until you
ask for it.

- [Databases](databases.md)
- [Running DFS](deep-feature-synthesis.md)
- [Primitives](primitives.md)
- [Custom primitives](custom-primitives.md)
- [Differences from featuretools](featuretools.md)
