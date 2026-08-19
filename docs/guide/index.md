# Guide

tusk turns a set of related tables into a single wide feature matrix. The
workflow is always the same three steps:

1. **Describe your data.** Build an [entity set](entity-sets.md): register each
   table, say which column is its primary key and which column records when a
   row became knowable, then link the tables with relationships.
2. **Synthesize.** Call [`dfs()`](dfs.md). It walks the relationship graph,
   stacking [primitives](primitives.md) up to `max_depth`, and returns both the
   feature matrix and the feature definitions that produced it.
3. **Re-apply.** Feed those definitions back to `calculate_feature_matrix()` to
   compute the same columns on new data.

Everything tusk builds is a narwhals expression, so the whole pipeline is one
query plan on the backend you already use. Nothing is materialized until you
ask for it.

- [Entity sets](entity-sets.md)
- [Running DFS](dfs.md)
- [Primitives](primitives.md)
- [Custom primitives](custom-primitives.md)
- [Differences from featuretools](featuretools.md)
