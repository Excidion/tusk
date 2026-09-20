# Primitives

::: tusk.primitives
    options:
      members: false

## Base classes

::: tusk.primitives.Primitive

::: tusk.primitives.AggregationPrimitive

::: tusk.primitives.TransformPrimitive

::: tusk.primitives.GroupTransformPrimitive

::: tusk.primitives.OrderedTransformPrimitive

::: tusk.primitives.NeedsCutoffTime

## Registry

::: tusk.primitives.register

::: tusk.primitives.resolve

::: tusk.primitives.resolve_all

## Aggregation primitives

::: tusk.primitives.aggregation.AGG_DEFAULTS

::: tusk.primitives.Count

::: tusk.primitives.Sum

::: tusk.primitives.Mean

::: tusk.primitives.Min

::: tusk.primitives.Max

::: tusk.primitives.Std

::: tusk.primitives.Median

::: tusk.primitives.NUnique

::: tusk.primitives.PercentTrue

::: tusk.primitives.Quantiles

::: tusk.primitives.TimeSinceFirst

::: tusk.primitives.TimeSinceLast

::: tusk.primitives.TimeSinceLastTrue

::: tusk.primitives.TimeSinceLastFalse

::: tusk.primitives.AllTrue

::: tusk.primitives.AnyTrue

::: tusk.primitives.NTrue

::: tusk.primitives.Skew

::: tusk.primitives.Kurtosis

::: tusk.primitives.Variance

::: tusk.primitives.MaxMinDelta

::: tusk.primitives.FirstLastTimeDelta

::: tusk.primitives.IsUnique

::: tusk.primitives.PercentUnique

::: tusk.primitives.NUniqueDays

::: tusk.primitives.NUniqueDaysOfCalendarYear

::: tusk.primitives.NUniqueDaysOfMonth

::: tusk.primitives.NUniqueMonths

## Transform primitives

::: tusk.primitives.transform.TRANS_DEFAULTS

::: tusk.primitives.Year

::: tusk.primitives.Month

::: tusk.primitives.Day

::: tusk.primitives.Hour

::: tusk.primitives.Weekday

::: tusk.primitives.IsWeekend

::: tusk.primitives.Minute

::: tusk.primitives.Second

::: tusk.primitives.DayOfYear

::: tusk.primitives.IsLeapYear

::: tusk.primitives.TimeSince

::: tusk.primitives.Absolute

::: tusk.primitives.NaturalLog

::: tusk.primitives.IsNull

::: tusk.primitives.Negate

::: tusk.primitives.SquareRoot

::: tusk.primitives.Sine

::: tusk.primitives.Cosine

::: tusk.primitives.NWords

::: tusk.primitives.NUniqueWords

::: tusk.primitives.AddNumeric

::: tusk.primitives.SubtractNumeric

::: tusk.primitives.MultiplyNumeric

::: tusk.primitives.DivideNumeric

::: tusk.primitives.ModuloNumeric

::: tusk.primitives.MultiplyNumericBoolean

::: tusk.primitives.GreaterThan

::: tusk.primitives.GreaterThanEqualTo

::: tusk.primitives.LessThan

::: tusk.primitives.LessThanEqualTo

::: tusk.primitives.Equal

::: tusk.primitives.NotEqual

::: tusk.primitives.EqualCategorical

::: tusk.primitives.NotEqualCategorical

::: tusk.primitives.Not

::: tusk.primitives.And

::: tusk.primitives.Or

## Group transform primitives

::: tusk.primitives.Percentile

## Ordered transform primitives

::: tusk.primitives.CumSum

::: tusk.primitives.CumCount

::: tusk.primitives.CumMin

::: tusk.primitives.CumMax

::: tusk.primitives.Diff

::: tusk.primitives.TimeSincePrevious

::: tusk.primitives.CumMean

::: tusk.primitives.SameAsPrevious

::: tusk.primitives.AbsoluteDiff

::: tusk.primitives.PercentChange

::: tusk.primitives.CumulativeTimeSinceLastTrue

::: tusk.primitives.CumulativeTimeSinceLastFalse
