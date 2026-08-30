# Primitive coverage

Every primitive in featuretools 1.31.0 beside its tusk counterpart. Time-series
primitives (rolling, expanding, lag) are out of scope and not listed.

| | Meaning |
| --- | --- |
| ✅ | Same values, covered by a test in the [differential suite][suite] |
| ❓ | Both implement it, but no test yet |
| ⚠️ | Diverges on purpose |
| ❌ | Diverges or missing |
| ➕ | tusk only |

[suite]: https://github.com/Excidion/tusk/tree/main/tests/differential

| Group | tusk | featuretools | Status | Test | Comment |
| --- | --- | --- | --- | --- | --- |
| Aggregation | [`count`][tusk.primitives.Count] | [`count`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Count.html) | ✅ | [`test_values_match_featuretools`](https://github.com/Excidion/tusk/blob/main/tests/differential/test_vs_featuretools.py) |  |
| Aggregation | [`max`][tusk.primitives.Max] | [`max`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Max.html) | ✅ | [`test_values_match_featuretools`](https://github.com/Excidion/tusk/blob/main/tests/differential/test_vs_featuretools.py) |  |
| Aggregation | [`mean`][tusk.primitives.Mean] | [`mean`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Mean.html) | ✅ | [`test_values_match_featuretools`](https://github.com/Excidion/tusk/blob/main/tests/differential/test_vs_featuretools.py) |  |
| Aggregation | [`median`][tusk.primitives.Median] | [`median`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Median.html) | ❓ |  |  |
| Aggregation | [`min`][tusk.primitives.Min] | [`min`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Min.html) | ✅ | [`test_values_match_featuretools`](https://github.com/Excidion/tusk/blob/main/tests/differential/test_vs_featuretools.py) |  |
| Aggregation | [`n_unique`][tusk.primitives.NUnique] | [`num_unique`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.NumUnique.html) | ⚠️ | [`test_n_unique_of_an_empty_group_diverges_from_featuretools`](https://github.com/Excidion/tusk/blob/main/tests/differential/test_vs_featuretools.py) | An empty group is `0` in tusk and `NaN` in featuretools; null handling agrees. |
| Aggregation | [`percent_true`][tusk.primitives.PercentTrue] | [`percent_true`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.PercentTrue.html) | ❌ |  | featuretools counts nulls as `False` in the denominator; tusk ignores them. |
| Aggregation | [`quantiles`][tusk.primitives.Quantiles] | — | ➕ |  |  |
| Aggregation | [`std`][tusk.primitives.Std] | [`std`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Std.html) | ✅ | [`test_stacked_values_match_featuretools`](https://github.com/Excidion/tusk/blob/main/tests/differential/test_vs_featuretools.py) |  |
| Aggregation | [`sum`][tusk.primitives.Sum] | [`sum`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Sum.html) | ✅ | [`test_values_match_featuretools`](https://github.com/Excidion/tusk/blob/main/tests/differential/test_vs_featuretools.py) |  |
| Aggregation | — | [`all`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.All.html) | ❌ |  |  |
| Aggregation | — | [`any`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Any.html) | ❌ |  |  |
| Aggregation | — | `average_count_per_unique` | ❌ |  |  |
| Aggregation | — | [`avg_time_between`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.AvgTimeBetween.html) | ❌ |  |  |
| Aggregation | — | [`count_above_mean`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CountAboveMean.html) | ❌ |  |  |
| Aggregation | — | [`count_below_mean`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CountBelowMean.html) | ❌ |  |  |
| Aggregation | — | [`count_greater_than`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CountGreaterThan.html) | ❌ |  |  |
| Aggregation | — | [`count_inside_nth_std`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CountInsideNthSTD.html) | ❌ |  |  |
| Aggregation | — | [`count_inside_range`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CountInsideRange.html) | ❌ |  |  |
| Aggregation | — | [`count_less_than`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CountLessThan.html) | ❌ |  |  |
| Aggregation | — | [`count_outside_nth_std`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CountOutsideNthSTD.html) | ❌ |  |  |
| Aggregation | — | [`count_outside_range`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CountOutsideRange.html) | ❌ |  |  |
| Aggregation | — | `date_first_event` | ❌ |  |  |
| Aggregation | — | [`entropy`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Entropy.html) | ❌ |  | Needs per-group value counts. |
| Aggregation | — | [`first`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.First.html) | ❌ |  |  |
| Aggregation | — | `first_last_time_delta` | ❌ |  |  |
| Aggregation | — | `has_no_duplicates` | ❌ |  |  |
| Aggregation | — | `is_monotonically_decreasing` | ❌ |  | Needs an ordered pairwise scan. |
| Aggregation | — | `is_monotonically_increasing` | ❌ |  | Needs an ordered pairwise scan. |
| Aggregation | — | `is_unique` | ❌ |  |  |
| Aggregation | — | `kurtosis` | ❌ |  |  |
| Aggregation | — | [`last`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Last.html) | ❌ |  |  |
| Aggregation | — | [`max_consecutive_false`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.MaxConsecutiveFalse.html) | ❌ |  | Needs a run-length scan. |
| Aggregation | — | [`max_consecutive_negatives`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.MaxConsecutiveNegatives.html) | ❌ |  | Needs a run-length scan. |
| Aggregation | — | [`max_consecutive_positives`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.MaxConsecutivePositives.html) | ❌ |  | Needs a run-length scan. |
| Aggregation | — | [`max_consecutive_true`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.MaxConsecutiveTrue.html) | ❌ |  | Needs a run-length scan. |
| Aggregation | — | [`max_consecutive_zeros`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.MaxConsecutiveZeros.html) | ❌ |  | Needs a run-length scan. |
| Aggregation | — | `max_count` | ❌ |  | Needs per-group value counts. |
| Aggregation | — | `max_min_delta` | ❌ |  |  |
| Aggregation | — | `median_count` | ❌ |  | Needs per-group value counts. |
| Aggregation | — | `min_count` | ❌ |  | Needs per-group value counts. |
| Aggregation | — | [`mode`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Mode.html) | ❌ |  | Length-changing inside a lazy `group_by().agg()`. |
| Aggregation | — | [`n_most_common`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.NMostCommon.html) | ❌ |  | Length-changing inside a lazy `group_by().agg()`. |
| Aggregation | — | `n_most_common_frequency` | ❌ |  | Length-changing inside a lazy `group_by().agg()`. |
| Aggregation | — | `n_unique_days` | ❌ |  |  |
| Aggregation | — | `n_unique_days_of_calendar_year` | ❌ |  |  |
| Aggregation | — | `n_unique_days_of_month` | ❌ |  |  |
| Aggregation | — | `n_unique_months` | ❌ |  |  |
| Aggregation | — | `n_unique_weeks` | ❌ |  |  |
| Aggregation | — | [`num_consecutive_greater_mean`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.NumConsecutiveGreaterMean.html) | ❌ |  | Needs a run-length scan. |
| Aggregation | — | [`num_consecutive_less_mean`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.NumConsecutiveLessMean.html) | ❌ |  | Needs a run-length scan. |
| Aggregation | — | `num_false_since_last_true` | ❌ |  | Needs a run-length scan. |
| Aggregation | — | `num_peaks` | ❌ |  | Needs a windowed scan. |
| Aggregation | — | [`num_true`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.NumTrue.html) | ❌ |  |  |
| Aggregation | — | `num_true_since_last_false` | ❌ |  | Needs a run-length scan. |
| Aggregation | — | `num_zero_crossings` | ❌ |  | Needs a windowed scan. |
| Aggregation | — | `percent_unique` | ❌ |  |  |
| Aggregation | — | [`skew`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Skew.html) | ❌ |  |  |
| Aggregation | — | [`time_since_first`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.TimeSinceFirst.html) | ❌ |  | Needs a reference clock. |
| Aggregation | — | [`time_since_last`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.TimeSinceLast.html) | ❌ |  | Needs a reference clock. |
| Aggregation | — | [`time_since_last_false`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.TimeSinceLastFalse.html) | ❌ |  | Needs a reference clock. |
| Aggregation | — | [`time_since_last_max`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.TimeSinceLastMax.html) | ❌ |  | Needs a reference clock. |
| Aggregation | — | [`time_since_last_min`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.TimeSinceLastMin.html) | ❌ |  | Needs a reference clock. |
| Aggregation | — | [`time_since_last_true`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.TimeSinceLastTrue.html) | ❌ |  | Needs a reference clock. |
| Aggregation | — | [`trend`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Trend.html) | ❌ |  | Needs a least-squares fit. |
| Binary transform | [`add_numeric`][tusk.primitives.AddNumeric] | [`add_numeric`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.AddNumeric.html) | ❓ |  |  |
| Binary transform | [`divide_numeric`][tusk.primitives.DivideNumeric] | `divide_numeric` | ❓ |  |  |
| Binary transform | [`multiply_numeric`][tusk.primitives.MultiplyNumeric] | `multiply_numeric` | ❓ |  |  |
| Binary transform | [`subtract_numeric`][tusk.primitives.SubtractNumeric] | [`subtract_numeric`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.SubtractNumeric.html) | ❓ |  |  |
| Binary transform | — | [`add_numeric_scalar`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.AddNumericScalar.html) | ❌ |  | Scalar argument; tusk primitives take columns only. |
| Binary transform | — | [`divide_by_feature`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.DivideByFeature.html) | ❌ |  | Scalar argument; tusk primitives take columns only. |
| Binary transform | — | [`divide_numeric_scalar`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.DivideNumericScalar.html) | ❌ |  | Scalar argument; tusk primitives take columns only. |
| Binary transform | — | [`equal`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Equal.html) | ❌ |  |  |
| Binary transform | — | [`equal_scalar`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.EqualScalar.html) | ❌ |  | Scalar argument; tusk primitives take columns only. |
| Binary transform | — | [`greater_than`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.GreaterThan.html) | ❌ |  |  |
| Binary transform | — | [`greater_than_equal_to`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.GreaterThanEqualTo.html) | ❌ |  |  |
| Binary transform | — | [`greater_than_equal_to_scalar`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.GreaterThanEqualToScalar.html) | ❌ |  | Scalar argument; tusk primitives take columns only. |
| Binary transform | — | [`greater_than_scalar`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.GreaterThanScalar.html) | ❌ |  | Scalar argument; tusk primitives take columns only. |
| Binary transform | — | [`less_than`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.LessThan.html) | ❌ |  |  |
| Binary transform | — | [`less_than_equal_to`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.LessThanEqualTo.html) | ❌ |  |  |
| Binary transform | — | [`less_than_equal_to_scalar`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.LessThanEqualToScalar.html) | ❌ |  | Scalar argument; tusk primitives take columns only. |
| Binary transform | — | [`less_than_scalar`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.LessThanScalar.html) | ❌ |  | Scalar argument; tusk primitives take columns only. |
| Binary transform | — | [`modulo_by_feature`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.ModuloByFeature.html) | ❌ |  | Scalar argument; tusk primitives take columns only. |
| Binary transform | — | [`modulo_numeric`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.ModuloNumeric.html) | ❌ |  |  |
| Binary transform | — | [`modulo_numeric_scalar`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.ModuloNumericScalar.html) | ❌ |  | Scalar argument; tusk primitives take columns only. |
| Binary transform | — | [`multiply_boolean`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.MultiplyBoolean.html) | ❌ |  |  |
| Binary transform | — | [`multiply_numeric_boolean`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.MultiplyNumericBoolean.html) | ❌ |  |  |
| Binary transform | — | [`multiply_numeric_scalar`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.MultiplyNumericScalar.html) | ❌ |  | Scalar argument; tusk primitives take columns only. |
| Binary transform | — | [`not_equal`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.NotEqual.html) | ❌ |  |  |
| Binary transform | — | [`not_equal_scalar`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.NotEqualScalar.html) | ❌ |  | Scalar argument; tusk primitives take columns only. |
| Binary transform | — | [`scalar_subtract_numeric_feature`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.ScalarSubtractNumericFeature.html) | ❌ |  | Scalar argument; tusk primitives take columns only. |
| Binary transform | — | [`subtract_numeric_scalar`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.SubtractNumericScalar.html) | ❌ |  | Scalar argument; tusk primitives take columns only. |
| Boolean transform | — | [`and`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.And.html) | ❌ |  |  |
| Boolean transform | — | [`isin`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.IsIn.html) | ❌ |  |  |
| Boolean transform | — | [`not`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Not.html) | ❌ |  |  |
| Boolean transform | — | [`or`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Or.html) | ❌ |  |  |
| Cumulative transform | [`cum_count`][tusk.primitives.CumCount] | [`cum_count`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CumCount.html) | ❌ |  | featuretools counts every row; tusk counts only non-null values. |
| Cumulative transform | [`cum_max`][tusk.primitives.CumMax] | [`cum_max`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CumMax.html) | ❓ |  |  |
| Cumulative transform | [`cum_min`][tusk.primitives.CumMin] | [`cum_min`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CumMin.html) | ❓ |  |  |
| Cumulative transform | [`cum_sum`][tusk.primitives.CumSum] | [`cum_sum`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CumSum.html) | ❓ |  |  |
| Cumulative transform | [`diff`][tusk.primitives.Diff] | [`diff`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Diff.html) | ❓ |  |  |
| Cumulative transform | [`time_since_previous`][tusk.primitives.TimeSincePrevious] | [`time_since_previous`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.TimeSincePrevious.html) | ❓ |  |  |
| Cumulative transform | — | [`cum_mean`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CumMean.html) | ❌ |  |  |
| Cumulative transform | — | `cumulative_time_since_last_false` | ❌ |  | Needs a conditional forward-fill; narwhals has none. |
| Cumulative transform | — | `cumulative_time_since_last_true` | ❌ |  | Needs a conditional forward-fill; narwhals has none. |
| Cumulative transform | — | [`diff_datetime`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.DiffDatetime.html) | ❌ |  |  |
| Datetime transform | [`day`][tusk.primitives.Day] | [`day`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Day.html) | ❓ |  |  |
| Datetime transform | [`hour`][tusk.primitives.Hour] | [`hour`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Hour.html) | ❓ |  |  |
| Datetime transform | [`is_weekend`][tusk.primitives.IsWeekend] | [`is_weekend`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.IsWeekend.html) | ❌ |  | Agrees on real dates; a null datetime is null in tusk and `False` in featuretools. |
| Datetime transform | [`month`][tusk.primitives.Month] | [`month`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Month.html) | ❓ |  |  |
| Datetime transform | [`weekday`][tusk.primitives.Weekday] | [`weekday`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Weekday.html) | ⚠️ |  | tusk is ISO 1-7 (Monday = 1); featuretools is 0-6 (Monday = 0). |
| Datetime transform | [`year`][tusk.primitives.Year] | [`year`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Year.html) | ❓ |  |  |
| Datetime transform | — | [`age`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Age.html) | ❌ |  | Needs a reference clock. |
| Datetime transform | — | [`date_to_holiday`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.DateToHoliday.html) | ❌ |  | Needs the `holidays` package. |
| Datetime transform | — | [`date_to_time_zone`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.DateToTimeZone.html) | ❌ |  |  |
| Datetime transform | — | [`day_of_year`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.DayOfYear.html) | ❌ |  |  |
| Datetime transform | — | [`days_in_month`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.DaysInMonth.html) | ❌ |  |  |
| Datetime transform | — | [`distance_to_holiday`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.DistanceToHoliday.html) | ❌ |  | Needs the `holidays` package. |
| Datetime transform | — | [`is_federal_holiday`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.IsFederalHoliday.html) | ❌ |  | Needs the `holidays` package. |
| Datetime transform | — | `is_first_week_of_month` | ❌ |  |  |
| Datetime transform | — | [`is_leap_year`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.IsLeapYear.html) | ❌ |  |  |
| Datetime transform | — | [`is_lunch_time`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.IsLunchTime.html) | ❌ |  |  |
| Datetime transform | — | [`is_month_end`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.IsMonthEnd.html) | ❌ |  |  |
| Datetime transform | — | [`is_month_start`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.IsMonthStart.html) | ❌ |  |  |
| Datetime transform | — | [`is_quarter_end`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.IsQuarterEnd.html) | ❌ |  |  |
| Datetime transform | — | [`is_quarter_start`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.IsQuarterStart.html) | ❌ |  |  |
| Datetime transform | — | [`is_working_hours`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.IsWorkingHours.html) | ❌ |  |  |
| Datetime transform | — | [`is_year_end`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.IsYearEnd.html) | ❌ |  |  |
| Datetime transform | — | [`is_year_start`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.IsYearStart.html) | ❌ |  |  |
| Datetime transform | — | [`minute`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Minute.html) | ❌ |  |  |
| Datetime transform | — | `nth_week_of_month` | ❌ |  | narwhals exposes no ISO week number. |
| Datetime transform | — | [`part_of_day`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.PartOfDay.html) | ❌ |  |  |
| Datetime transform | — | [`quarter`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Quarter.html) | ❌ |  |  |
| Datetime transform | — | [`season`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Season.html) | ❌ |  |  |
| Datetime transform | — | [`second`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Second.html) | ❌ |  |  |
| Datetime transform | — | `time_since` | ❌ |  | Needs a reference clock. |
| Datetime transform | — | [`week`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Week.html) | ❌ |  | narwhals exposes no ISO week number. |
| Email and URL transform | — | [`email_address_to_domain`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.EmailAddressToDomain.html) | ❌ |  |  |
| Email and URL transform | — | [`is_free_email_domain`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.IsFreeEmailDomain.html) | ❌ |  | Needs a bundled domain list. |
| Email and URL transform | — | [`url_to_domain`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.URLToDomain.html) | ❌ |  | Needs URL parsing. |
| Email and URL transform | — | [`url_to_protocol`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.URLToProtocol.html) | ❌ |  | Needs URL parsing. |
| Email and URL transform | — | [`url_to_tld`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.URLToTLD.html) | ❌ |  | Needs URL parsing. |
| Exponential transform | — | [`exponential_weighted_average`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.ExponentialWeightedAverage.html) | ❌ |  |  |
| Exponential transform | — | [`exponential_weighted_std`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.ExponentialWeightedSTD.html) | ❌ |  | narwhals exposes only `ewm_mean`. |
| Exponential transform | — | [`exponential_weighted_variance`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.ExponentialWeightedVariance.html) | ❌ |  | narwhals exposes only `ewm_mean`. |
| General transform | [`absolute`][tusk.primitives.Absolute] | [`absolute`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Absolute.html) | ❓ |  |  |
| General transform | [`natural_log`][tusk.primitives.NaturalLog] | [`natural_logarithm`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.NaturalLogarithm.html) | ❓ |  |  |
| General transform | — | [`absolute_diff`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.AbsoluteDiff.html) | ❌ |  |  |
| General transform | — | [`cosine`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Cosine.html) | ❌ |  |  |
| General transform | — | [`is_null`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.IsNull.html) | ❌ |  |  |
| General transform | — | [`negate`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Negate.html) | ❌ |  |  |
| General transform | — | `percent_change` | ❌ |  |  |
| General transform | — | [`percentile`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Percentile.html) | ❌ |  |  |
| General transform | — | [`rate_of_change`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.RateOfChange.html) | ❌ |  |  |
| General transform | — | [`same_as_previous`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.SameAsPrevious.html) | ❌ |  |  |
| General transform | — | `savgol_filter` | ❌ |  | Needs SciPy. |
| General transform | — | [`sine`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Sine.html) | ❌ |  |  |
| General transform | — | [`square_root`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.SquareRoot.html) | ❌ |  |  |
| General transform | — | [`tangent`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Tangent.html) | ❌ |  | narwhals exposes no `tan`. |
| General transform | — | [`variance`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Variance.html) | ❌ |  |  |
| Location transform | — | [`cityblock_distance`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CityblockDistance.html) | ❌ |  | featuretools' `LatLong` column has no narwhals equivalent. |
| Location transform | — | [`geomidpoint`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.GeoMidpoint.html) | ❌ |  | featuretools' `LatLong` column has no narwhals equivalent. |
| Location transform | — | [`haversine`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Haversine.html) | ❌ |  | featuretools' `LatLong` column has no narwhals equivalent. |
| Location transform | — | [`is_in_geobox`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.IsInGeoBox.html) | ❌ |  | featuretools' `LatLong` column has no narwhals equivalent. |
| Location transform | — | [`latitude`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Latitude.html) | ❌ |  | featuretools' `LatLong` column has no narwhals equivalent. |
| Location transform | — | [`longitude`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.Longitude.html) | ❌ |  | featuretools' `LatLong` column has no narwhals equivalent. |
| Natural-language transform | — | [`count_string`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.CountString.html) | ❌ |  |  |
| Natural-language transform | — | `file_extension` | ❌ |  |  |
| Natural-language transform | — | `full_name_to_first_name` | ❌ |  | Needs name parsing. |
| Natural-language transform | — | `full_name_to_last_name` | ❌ |  | Needs name parsing. |
| Natural-language transform | — | `full_name_to_title` | ❌ |  | Needs name parsing. |
| Natural-language transform | — | [`mean_characters_per_word`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.MeanCharactersPerWord.html) | ❌ |  |  |
| Natural-language transform | — | [`median_word_length`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.MedianWordLength.html) | ❌ |  | Needs a per-row list reduction. |
| Natural-language transform | — | [`num_characters`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.NumCharacters.html) | ❌ |  |  |
| Natural-language transform | — | [`num_unique_separators`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.NumUniqueSeparators.html) | ❌ |  | Needs a regex count; narwhals has none. |
| Natural-language transform | — | [`num_words`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.NumWords.html) | ❌ |  |  |
| Natural-language transform | — | [`number_of_common_words`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.NumberOfCommonWords.html) | ❌ |  | Needs tokenisation against a word list. |
| Natural-language transform | — | [`number_of_hashtags`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.NumberOfHashtags.html) | ❌ |  | Needs a regex count; narwhals has none. |
| Natural-language transform | — | [`number_of_mentions`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.NumberOfMentions.html) | ❌ |  | Needs a regex count; narwhals has none. |
| Natural-language transform | — | [`number_of_unique_words`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.NumberOfUniqueWords.html) | ❌ |  | Needs a per-row list reduction. |
| Natural-language transform | — | [`number_of_words_in_quotes`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.NumberOfWordsInQuotes.html) | ❌ |  | Needs a regex count; narwhals has none. |
| Natural-language transform | — | [`punctuation_count`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.PunctuationCount.html) | ❌ |  | Needs a regex count; narwhals has none. |
| Natural-language transform | — | [`title_word_count`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.TitleWordCount.html) | ❌ |  | Needs a regex count; narwhals has none. |
| Natural-language transform | — | [`total_word_length`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.TotalWordLength.html) | ❌ |  |  |
| Natural-language transform | — | [`upper_case_count`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.UpperCaseCount.html) | ❌ |  | Needs a regex count; narwhals has none. |
| Natural-language transform | — | [`upper_case_word_count`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.UpperCaseWordCount.html) | ❌ |  | Needs a regex count; narwhals has none. |
| Natural-language transform | — | [`whitespace_count`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.WhitespaceCount.html) | ❌ |  | Needs a regex count; narwhals has none. |
| Postal-code transform | — | [`one_digit_postal_code`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.OneDigitPostalCode.html) | ❌ |  |  |
| Postal-code transform | — | [`two_digit_postal_code`](https://featuretools.alteryx.com/en/stable/generated/featuretools.primitives.TwoDigitPostalCode.html) | ❌ |  |  |
