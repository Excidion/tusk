# Primitive parity roadmap

The order in which the remaining ❌ rows of
`docs/guide/primitive-coverage.md` are worked off, and the rules every phase
shares. Each phase gets its own spec, plan, worktree and pull request.

## Shared rules

These were settled with the maintainer and hold for every phase.

1. **Both backends or ⛔.** A primitive ships only if it runs on lazy polars and
   duckdb with the same meaning. Otherwise its row becomes ⛔ with the reason.
2. **Only data-independent parameters.** A dataclass field such as `n` or a
   user-supplied holiday list is allowed. A threshold that only means something
   for one dataset, such as `count_greater_than(threshold)`, is not.
3. **tusk semantics, divergences marked ⚠️.** Nulls follow tusk's existing rules
   and empty groups get an identity value where one exists, otherwise null,
   decided primitive by primitive. Every divergence from featuretools is
   measured by a differential test asserting both sides and marked ⚠️.
4. **Names follow tusk's convention.**
   - Counts start with `n_`: `num_true` → `n_true`, `count_above_mean` →
     `n_above_mean`, `number_of_unique_words` → `n_unique_words`.
   - One word per concept: `greater`/`less` become `above`/`below`, as in
     `num_consecutive_greater_mean` → `n_consecutive_above_mean`.
   - No abbreviations: `avg_time_between` → `mean_time_between`.
   - A name that would make the class shadow a Python builtin or a `typing`
     name gets a suffix: featuretools' `all` and `any` ship as `all_true` and
     `any_true`.
   - Every other primitive keeps its featuretools name.
5. **No synonyms.** A featuretools primitive that computes what an existing tusk
   primitive computes is not implemented; its row becomes ⚠️ pointing at the
   existing one, as `multiply_boolean` points at `and`.
6. **Defaults follow featuretools' defaults.** A primitive joins
   `AGG_DEFAULTS` or `TRANS_DEFAULTS` exactly when featuretools' DFS uses it by
   default.
7. **❓ rows get tests** in the phase covering their group. The divergent
   `cum_count` and `is_weekend` get a test asserting the divergence and move
   from ❌ to ⚠️.
8. **No cross-row statistics in transforms.** A transform whose value for a
   row depends on the other rows of the dataset, such as a rank or a scaled
   value, is a data leakage risk; scaling is the job of downstream sklearn
   transformers. Such rows become ⛔.

## Phases

Easy wins first, foundations after.

| # | Phase | Primitives | Also |
| --- | --- | --- | --- |
| 1 | Standalone aggregations | `all_true`, `any_true`, `n_true`, `skew`, `kurtosis`, `variance`, `max_min_delta`, `first_last_time_delta`, `is_unique`, `percent_unique`, `n_unique_days`, `n_unique_days_of_calendar_year`, `n_unique_days_of_month`, `n_unique_months` | `skew`, `percent_true` join `AGG_DEFAULTS`; `day` joins `TRANS_DEFAULTS`; ⚠️ `has_no_duplicates` → `is_unique`; `n_unique` counts null as a value; the ⛔ reclassifications below |
| 2 | Standalone and ordered transforms | `is_null`, `negate`, `square_root`, `sine`, `cosine`, `minute`, `second`, `day_of_year`, `is_leap_year`, `cum_mean`, `same_as_previous`, `absolute_diff`, `percent_change`, `cumulative_time_since_last_true`, `cumulative_time_since_last_false` | ⚠️ `cum_count`; ⚠️ `diff_datetime` → `time_since_previous`; ❓ tests for `cum_sum`, `cum_min`, `cum_max`, `diff`, `year`, `month`, `day`, `hour`, `absolute`, `natural_log` |
| 3 | Strings, email and URL | `n_characters`, `n_words`, `n_unique_words`, `email_address_to_domain`, `url_to_domain`, `url_to_protocol`, `url_to_tld` | `n_words`, `n_characters` join `TRANS_DEFAULTS` |
| 4 | Staged and ordered aggregations | `first`, `last`, `trend`, `time_since_last_max`, `time_since_last_min`, `mean_time_between`, `n_above_mean`, `n_below_mean`, `max_consecutive_true`, `max_consecutive_false`, `max_consecutive_positives`, `max_consecutive_negatives`, `max_consecutive_zeros`, `n_consecutive_above_mean`, `n_consecutive_below_mean`, `n_false_since_last_true`, `n_true_since_last_false`, `is_monotonically_increasing`, `is_monotonically_decreasing`, `n_peaks`, `n_zero_crossings`, `entropy`, `max_count`, `min_count`, `median_count`, `average_count_per_unique` | ⚠️ `date_first_event` → `time_since_first` |
| 5 | Calendar helpers | `quarter`, `is_month_start`, `is_month_end`, `is_quarter_start`, `is_quarter_end`, `is_year_start`, `is_year_end`, `days_in_month`, `week`, `season`, `n_unique_weeks` | ⚠️ `is_weekend` |
| 6 | Data-independent parameters | `n_inside_nth_std`, `n_outside_nth_std`, `date_to_holiday`, `distance_to_holiday`, `is_federal_holiday` | |

### Foundations introduced by phase 4

Two probes against narwhals 2.24 shape phase 4; its own spec designs them.

- **Ordered aggregations.** `expr.first(order_by=...)` and `last(order_by=...)`
  work inside a lazy `group_by().agg()` on both backends. Order is the child's
  `(row_creation_time, primary_key)`. A child without `row_creation_time` is
  skipped by DFS with a warning. Ties on an extreme resolve to the latest row.
- **Staged aggregations.** duckdb rejects a window inside an aggregate
  (`aggregate function calls cannot contain window function calls`), so a
  comparison against the group mean or a run length is computed as a column on
  the child, `.over(foreign_key, order_by=...)`, before the `group_by`. Both
  backends accept the two-stage form.

## ⛔ reclassifications

These rows change from ❌ to ⛔ in phase 1, since no phase will implement them.

| Row | Reason |
| --- | --- |
| `count_greater_than`, `count_less_than`, `count_inside_range`, `count_outside_range` | Data-dependent threshold argument. |
| `mode`, `n_most_common`, `n_most_common_frequency` | Length-changing inside a lazy `group_by().agg()`. |
| `exponential_weighted_average`, `exponential_weighted_std`, `exponential_weighted_variance` | duckdb has no exponentially weighted window; lazy polars rejects `ewm_mean`. |
| `date_to_time_zone` | duckdb keeps the time zone on the connection, not the dtype. |
| `percentile` | Data leakage risk, scaling is the job of downstream sklearn transformers. |
