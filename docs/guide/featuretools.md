# Differences from featuretools

tusk is inspired by [featuretools](https://featuretools.alteryx.com/) and keeps
its vocabulary — entity sets, relationships, primitives, deep feature synthesis
— but differs where narwhals or SQL backends force a different answer.

- **Lazy in, lazy out.** Feed tusk lazy frames and it returns a lazy frame: it
  builds one query plan and never collects, so you decide when to compute. Feed
  it eager frames and it returns an eager frame, collecting once at the end —
  that single `collect()` is the only one in the library.

- **Any narwhals backend**, not just pandas. One entity set uses one backend.

- **Feature names are SQL identifiers.** Featuretools writes
  `MEAN(transactions.amount)`; tusk writes `MEAN__transactions__amount`. On a
  backend that generates SQL, dots and parentheses parse as table qualifiers
  and function calls rather than as part of a name, so the conventional form is
  unusable there. See [the full naming
  scheme](dfs.md#feature-names-are-sql-identifiers). The conventional form is
  kept on [`Feature.display_name`][tusk.features.Feature.display_name].

- **`primary_key` and `row_creation_time`** rather than `index` and
  `time_index`. Narwhals has no index concept, and `row_creation_time` names
  what the column means: when the row became knowable.

- **Three-argument relationships.** `add_relationship(parent=, child=,
  foreign_key=)` — the parent side is always the parent's primary key.

- **One global `cutoff_time`**, not per-row cutoff times. It filters the target
  table too, so the matrix can have fewer rows than the target — a row that did
  not exist yet at the cutoff has no features to compute. Tables with no
  `row_creation_time` are timeless and pass through unfiltered, so a cutoff on
  an entity set that declares none is silently a no-op. With
  `features_only=True` the cutoff is ignored entirely, since nothing is
  computed and feature definitions do not record it.

- **`primary_key` is optional**, but a table without one cannot be a
  relationship parent or a DFS target, and order-dependent primitives on it
  have non-deterministic tiebreaks. You get a
  [`MissingPrimaryKeyWarning`][tusk.exceptions.MissingPrimaryKeyWarning].

- **`N_UNIQUE` on an empty group is `0`**, not `NaN`. See [empty
  groups](primitives.md#empty-groups) for the reasoning and the full table.
