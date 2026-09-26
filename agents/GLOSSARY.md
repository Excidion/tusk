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
