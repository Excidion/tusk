# Guide

tusk turns a set of related tables into one wide feature matrix. The workflow
always follows the same three steps:

1. **Describe your data.** Build a [database](databases.md). Register each
   table. Name its primary key column and the column that records when a row
   becomes visible. Then link the tables with relationships. tusk takes these
   declarations on trust. Call [`validate()`](databases.md#validation) to
   check them against the data.
2. **Synthesize.** Call [`deep_feature_synthesis()`](deep-feature-synthesis.md).
   It walks the relationship graph, stacks [primitives](primitives.md) up to
   `max_depth`, and returns the feature matrix and the feature definitions
   used to compute it.
3. **Re-apply.** Feed those definitions back to `apply_features()` to
   compute the same columns on new data.

Everything tusk builds is a narwhals expression. The whole pipeline is one
query plan on the backend you already use.

- [Databases](databases.md)
- [Running DFS](deep-feature-synthesis.md)
- [Primitives](primitives.md)
- [Custom primitives](custom-primitives.md)
- [scikit-learn pipelines](sklearn.md)
- [Differences from featuretools](featuretools.md)
