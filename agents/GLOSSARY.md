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
| Make an expression, a feature, a name, a database or a table | build | generate, create |
| Parts of a table | row, column | record, field |
| polars, duckdb, pandas, ... | backend | engine, dataframe backend |
| The child rows that share one foreign key value | group | |
| A group without rows | empty group | |
| A set of tables and relationships | database | entity set |
| An operation that makes features | primitive | |

Names from other libraries stay as they are, for example `nw.LazyFrame`,
featuretools' `EntitySet` and `calculate_feature_matrix`. "Unknown" in the
sense "not recognized", as in "unknown table", is allowed.

## Allowed exceptions

- A backend's native object may be named by its library type, for example
  "polars DataFrame" or `nw.LazyFrame`. A generic "frame" in prose is not
  allowed; use "table".
- "Record(s)" is allowed as a verb for storing a time or a state, for
  example "the column records when a row becomes knowable". It stays
  banned as a noun for a row.
- "NaN" is allowed for the float not-a-number value, and when quoting
  featuretools output.
- "Row creation time" is allowed for the `row_creation_time` concept,
  alongside `row_creation_time` itself.
- "Entity" is allowed where it names Mermaid's diagram syntax or an
  entity-relationship diagram, not where it means a table or a database.
- "Unknown" is allowed in the sense "not recognized" (see above).
- featuretools' own API names are allowed in text that compares tusk to
  featuretools.

"Missing" is not used in prose; use "null" for a missing value. Existing
identifiers keep "missing", for example `MissingPrimaryKeyWarning` and
`_reject_missing_keys`.

## Actors

Name synthesis as the actor that builds feature definitions, and the
compiler as the actor that builds the feature matrix's query. The backend
computes the values, when the caller collects. Do not name an actor that
the text being edited did not already name.

## Error messages

Error messages follow this glossary.

## Docstrings

A docstring on an exception or a warning class may say why it is raised or
issued rather than only what it is, because that is what the reader needs
to decide how to react. Every other docstring says what a thing does, not
why.
