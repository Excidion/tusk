# Where-Clause Aggregations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user declare named row conditions on a table and have DFS generate a masked variant of each selected aggregation, so conditional features like `SUM(contracts.subscription_types.price WHEN current)` are reachable through a join path.

**Architecture:** Conditions are declared on `add_table` in two dicts — `where` for static `nw.Expr`, `when` for callables taking the cutoff. `AggregationFeature` carries the condition's kind and key (never its expression), so a `FeatureList` stays re-applicable. The compiler masks the child frame's rows before grouping with `nw.col(x).filter(mask)`, which is what lets a mask on a contract row reach a price that arrived as a direct feature.

**Tech Stack:** Python, narwhals, polars (core suite), duckdb (portability suite), pytest, uv, just.

**Spec:** `agents/superpowers/specs/2026-09-20-where-clause-aggregations-design.md`

## Global Constraints

- **Run tests with** `uv run pytest`, never bare `pytest`. Full gate is `just check`.
- **Docstrings are enforced** by `interrogate` and `pydoclint` pre-commit hooks. Every new function, method and class needs a docstring with `Args:`, `Returns:` and `Raises:` sections matching the real signature. A missing or mismatched section fails the commit.
- **Style** follows `agents/style/CODESTYLE.md`: descriptive names, no abbreviations, functions do one thing, comments explain *why* only. Never write a comment restating the code.
- **Feature names are plain SQL identifiers.** Parts join with `__`. No dots, parens or spaces — duckdb parses those as table qualifiers and function calls. `display_name` carries the readable form.
- **Clause key separator token:** `WHERE` for `where` keys, `WHEN` for `when` keys, uppercase, as a name part.
- **`where_primitives` default is `("count", "sum")`.**
- **A clause never consumes depth.**
- **Unclaused features must compile byte-identically to today.** Every task keeps `clause=None` on the existing path.
- **Commit after every task.** Conventional commits (`feat:`, `test:`, `docs:`, `refactor:`).

---

### Task 1: Schema accepts `where` and `when`

**Files:**
- Modify: `src/tusk/database.py` (`TableSchema`, `Database.add_table`)
- Modify: `src/tusk/validation.py` (three new checks, `TABLE_CHECKS`, `DEFAULT_TABLE_CHECKS`)
- Test: `tests/test_database.py`, `tests/test_validation.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `TableSchema.where: Mapping[str, nw.Expr]`, `TableSchema.when: Mapping[str, Callable[[datetime], nw.Expr]]`, both defaulting to `{}`. `Database.add_table(..., where=None, when=None)`. Validation check names `"where_clauses_are_expressions"`, `"when_clauses_are_callable"`, `"clause_keys"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_database.py`:

```python
def test_add_table_stores_where_and_when_clauses(db):
    """Declared clauses reach the schema unchanged."""
    current = lambda cutoff: nw.col("occurred_at") <= cutoff
    verified = nw.col("verified")
    database = tusk.Database("shop").add_table(
        "transactions",
        _transactions_frame(),
        primary_key="id",
        row_creation_time="occurred_at",
        where={"verified": verified},
        when={"current": current},
    )

    schema = database.schema("transactions")
    assert schema.where["verified"] is verified
    assert schema.when["current"] is current


def test_add_table_defaults_clauses_to_empty_mappings(db):
    """A table declaring no clause has two empty mappings, never None."""
    schema = db.schema("transactions")
    assert schema.where == {}
    assert schema.when == {}
```

Add a module-level helper to `tests/test_database.py` if one does not already exist:

```python
def _transactions_frame():
    """A minimal transactions frame for schema-only tests."""
    return pl.LazyFrame(
        {
            "id": [100, 101],
            "amount": [1.0, 3.0],
            "verified": [True, False],
            "occurred_at": [dt.datetime(2024, 3, 4), dt.datetime(2024, 3, 5)],
        },
    )
```

Add to `tests/test_validation.py`:

```python
def test_where_clause_must_be_an_expression():
    """A callable in where names the parameter it belongs in."""
    with pytest.raises(ValidationError, match="when"):
        tusk.Database("shop").add_table(
            "transactions",
            _clause_frame(),
            primary_key="id",
            where={"current": lambda cutoff: nw.col("occurred_at") <= cutoff},
        )


def test_when_clause_must_be_callable():
    """An expression in when names the parameter it belongs in."""
    with pytest.raises(ValidationError, match="where"):
        tusk.Database("shop").add_table(
            "transactions",
            _clause_frame(),
            primary_key="id",
            when={"verified": nw.col("verified")},
        )


def test_clause_key_rejects_the_name_separator():
    """A key holding __ would break feature name parsing."""
    with pytest.raises(ValidationError, match="__"):
        tusk.Database("shop").add_table(
            "transactions",
            _clause_frame(),
            primary_key="id",
            where={"is__verified": nw.col("verified")},
        )


def _clause_frame():
    """A minimal frame for clause validation tests."""
    return pl.LazyFrame(
        {
            "id": [100, 101],
            "verified": [True, False],
            "occurred_at": [dt.datetime(2024, 3, 4), dt.datetime(2024, 3, 5)],
        },
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_database.py -k clause -v && uv run pytest tests/test_validation.py -k clause -v`
Expected: FAIL — `add_table() got an unexpected keyword argument 'where'`.

- [ ] **Step 3: Add the schema fields**

In `src/tusk/database.py`, add to the `TableSchema` dataclass, after `row_update_times`:

```python
    where: Mapping[str, Any] = field(default_factory=dict)
    when: Mapping[str, Any] = field(default_factory=dict)
```

Extend the `TableSchema` class docstring's `Attributes:` block with:

```
        where: Named row conditions as narwhals expressions.
        when: Named row conditions as callables taking the cutoff time and
            returning a narwhals expression.
```

Add the property that the compiler and synthesis both need:

```python
    @property
    def clauses(self) -> tuple[tuple[str, str], ...]:
        """Every declared condition as a (kind, key) pair, where first."""
        return tuple(
            [("where", key) for key in self.where]
            + [("when", key) for key in self.when],
        )
```

- [ ] **Step 4: Thread the parameters through `add_table`**

In `Database.add_table`, add two parameters after `row_update_times`:

```python
        where: Mapping[str, Any] | None = None,
        when: Mapping[str, Any] | None = None,
```

Extend the docstring's `Args:` block:

```
            where: Named row conditions as narwhals expressions, used to mask
                a child table's rows before an aggregation groups them. Each
                key becomes a feature name part.
            when: Named row conditions as callables receiving the cutoff time
                and returning a narwhals expression. Use this for a condition
                measured against the cutoff, such as a validity interval.
```

Immediately before the `TableSchema(...)` construction, normalize:

```python
        where = dict(where or {})
        when = dict(when or {})
```

and pass `where=where, when=when` to `TableSchema(...)`.

- [ ] **Step 5: Add the three validation checks**

In `src/tusk/validation.py`, add:

```python
def check_where_clauses_are_expressions(
    frame: nw.LazyFrame,
    schema: TableSchema,
) -> None:
    """Confirm every ``where`` clause is a narwhals expression.

    Reads the schema only.

    Args:
        frame: The table's lazy frame. Unused.
        schema: The table's schema, holding the clauses to check.

    Raises:
        ValidationError: If a ``where`` clause is not an ``nw.Expr``.
    """
    for key, clause in schema.where.items():
        if isinstance(clause, nw.Expr):
            continue
        raise ValidationError(
            f"where clause {key!r} of {schema.name!r} is not a narwhals "
            f"expression; a clause that needs the cutoff time belongs in when",
        )


def check_when_clauses_are_callable(
    frame: nw.LazyFrame,
    schema: TableSchema,
) -> None:
    """Confirm every ``when`` clause is callable.

    Reads the schema only.

    Args:
        frame: The table's lazy frame. Unused.
        schema: The table's schema, holding the clauses to check.

    Raises:
        ValidationError: If a ``when`` clause is not callable.
    """
    for key, clause in schema.when.items():
        if callable(clause):
            continue
        raise ValidationError(
            f"when clause {key!r} of {schema.name!r} is not callable; a clause "
            f"that does not need the cutoff time belongs in where",
        )


def check_clause_keys(frame: nw.LazyFrame, schema: TableSchema) -> None:
    """Confirm no clause key holds the feature name separator.

    Reads the schema only.

    Args:
        frame: The table's lazy frame. Unused.
        schema: The table's schema, holding the keys to check.

    Raises:
        ValidationError: If a clause key contains ``__``.
    """
    for _, key in schema.clauses:
        if "__" not in key:
            continue
        raise ValidationError(
            f"clause key {key!r} of {schema.name!r} contains '__', which "
            f"separates the parts of a feature name; rename it",
        )
```

Register all three in `TABLE_CHECKS` and add all three to `DEFAULT_TABLE_CHECKS` — every one answers from the schema without reading a row, which is the documented bar for the default set.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_database.py tests/test_validation.py -v`
Expected: PASS, including every pre-existing test.

- [ ] **Step 7: Commit**

```bash
git add src/tusk/database.py src/tusk/validation.py tests/test_database.py tests/test_validation.py
git commit -m "feat(database): accept named where and when clauses on add_table"
```

---

### Task 2: `AggregationFeature` carries a clause

**Files:**
- Modify: `src/tusk/features.py` (`AggregationFeature`)
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `TableSchema.clauses` from Task 1.
- Produces: `AggregationFeature.clause: tuple[str, str] | None = None`, where the first element is `"where"` or `"when"`. `name` and `display_name` render the clause.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_features.py`:

```python
def test_clause_appears_in_aggregation_name():
    """A where clause becomes a trailing name part."""
    feature = AggregationFeature(
        resolve("count"),
        (),
        Relationship("customers", "orders", "customer_id"),
        clause=("where", "enterprise"),
    )
    assert feature.name == "COUNT__orders__WHERE__enterprise"
    assert feature.display_name == "COUNT(orders WHERE enterprise)"


def test_when_clause_uses_the_when_token():
    """A when clause is distinguishable from a where clause of the same key."""
    relationship = Relationship("customers", "orders", "customer_id")
    where = AggregationFeature(
        resolve("count"), (), relationship, clause=("where", "current"),
    )
    when = AggregationFeature(
        resolve("count"), (), relationship, clause=("when", "current"),
    )
    assert where.name == "COUNT__orders__WHERE__current"
    assert when.name == "COUNT__orders__WHEN__current"
    assert where != when


def test_clause_renders_after_the_last_argument():
    """A clause on a one-input aggregation sits inside the parentheses."""
    feature = AggregationFeature(
        resolve("sum"),
        (IdentityFeature("orders", "amount", nw.Float64()),),
        Relationship("customers", "orders", "customer_id"),
        clause=("when", "current"),
    )
    assert feature.name == "SUM__orders__amount__WHEN__current"
    assert feature.display_name == "SUM(orders.amount WHEN current)"


def test_unclaused_aggregation_name_is_unchanged():
    """The existing path keeps its exact name."""
    feature = AggregationFeature(
        resolve("count"), (), Relationship("customers", "orders", "customer_id"),
    )
    assert feature.name == "COUNT__orders"
    assert feature.display_name == "COUNT(orders)"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_features.py -k clause -v`
Expected: FAIL — `AggregationFeature.__init__() got an unexpected keyword argument 'clause'`.

- [ ] **Step 3: Add the field and render it**

In `src/tusk/features.py`, add the field to `AggregationFeature` after `relationship`:

```python
    clause: tuple[str, str] | None = None
```

Extend the class docstring's `Attributes:` block:

```
        clause: The condition masking the child's rows, as a (kind, key)
            pair where kind is ``"where"`` or ``"when"``. None aggregates
            every row.
```

Replace the `name` and `display_name` properties with:

```python
    @property
    def name(self) -> str:
        """Generated name, e.g. ``MEAN__transactions__amount``.

        Zero-arity primitives name the child table instead of a column, giving
        ``COUNT__transactions``. A clause adds two trailing parts, giving
        ``COUNT__transactions__WHEN__current``.
        """
        child = self.relationship.child
        if not self.bases:
            return self.primitive.generate_name([child, *self._clause_parts])
        names = [f"{child}__{b.name}" for b in self.bases]
        return self.primitive.generate_name([*names, *self._clause_parts])

    @property
    def display_name(self) -> str:
        """Readable name, e.g. ``MEAN(transactions.amount)``.

        A clause is appended to the final argument, giving
        ``COUNT(transactions WHEN current)``.
        """
        child = self.relationship.child
        if not self.bases:
            arguments = [child]
        else:
            arguments = [f"{child}.{b.display_name}" for b in self.bases]
        arguments[-1] += self._clause_display
        return self.primitive.generate_display_name(arguments)

    @property
    def _clause_parts(self) -> tuple[str, ...]:
        """The clause's name parts, empty when the feature has no clause."""
        if self.clause is None:
            return ()
        kind, key = self.clause
        return (kind.upper(), key)

    @property
    def _clause_display(self) -> str:
        """The clause's readable suffix, empty when the feature has no clause."""
        if self.clause is None:
            return ""
        kind, key = self.clause
        return f" {kind.upper()} {key}"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_features.py -v`
Expected: PASS, including every pre-existing naming test.

- [ ] **Step 5: Commit**

```bash
git add src/tusk/features.py tests/test_features.py
git commit -m "feat(features): give aggregation features an optional clause"
```

---

### Task 3: The compiler masks the aggregation

**Files:**
- Modify: `src/tusk/compiler.py` (`_add_aggregations`, new `_clause_expr`)
- Test: `tests/test_compiler_aggregation.py`

**Interfaces:**
- Consumes: `TableSchema.where`/`when` (Task 1), `AggregationFeature.clause` (Task 2).
- Produces: `_clause_expr(schema: TableSchema, clause: tuple[str, str] | None, cutoff_time: datetime | None) -> nw.Expr | None`, raising `SchemaError` for an unknown key.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_compiler_aggregation.py`:

```python
def test_where_clause_masks_the_aggregated_rows():
    """Only rows passing the clause reach the aggregation."""
    database = _clause_database()
    features = FeatureList(
        "customers",
        [
            AggregationFeature(
                resolve("sum"),
                (IdentityFeature("orders", "amount", nw.Float64()),),
                Relationship("customers", "orders", "customer_id"),
                clause=("where", "large"),
            ),
        ],
    )
    matrix = nw.from_native(features.apply(database)).lazy().collect()
    values = dict(zip(matrix["id"], matrix["SUM__orders__amount__WHERE__large"]))
    assert values == {1: 30.0, 2: 0.0}


def test_clause_reaches_a_direct_feature_from_a_third_table():
    """A mask on the child's rows masks values joined onto that child."""
    database = _clause_database()
    price = DirectFeature(
        IdentityFeature("products", "price", nw.Float64()),
        Relationship("products", "orders", "product_id"),
    )
    features = FeatureList(
        "customers",
        [
            AggregationFeature(
                resolve("sum"),
                (price,),
                Relationship("customers", "orders", "customer_id"),
                clause=("where", "large"),
            ),
        ],
    )
    matrix = nw.from_native(features.apply(database)).lazy().collect()
    column = [c for c in matrix.columns if c.startswith("SUM__orders")][0]
    assert dict(zip(matrix["id"], matrix[column])) == {1: 7.0, 2: 0.0}


def test_when_clause_receives_the_cutoff_time():
    """The same feature gives different values at two cutoffs."""
    database = _clause_database()
    features = FeatureList(
        "customers",
        [
            AggregationFeature(
                resolve("count"),
                (),
                Relationship("customers", "orders", "customer_id"),
                clause=("when", "open"),
            ),
        ],
    )
    early = nw.from_native(
        features.apply(database, cutoff_time=dt.datetime(2024, 5, 1)),
    ).lazy().collect()
    late = nw.from_native(
        features.apply(database, cutoff_time=dt.datetime(2024, 8, 1)),
    ).lazy().collect()
    name = "COUNT__orders__WHEN__open"
    assert dict(zip(early["id"], early[name])) == {1: 2, 2: 1}
    assert dict(zip(late["id"], late[name])) == {1: 1, 2: 1}


def _clause_database():
    """Customers, orders with a closing date, and products with a price.

    Customer 1 has orders of 10.0 and 20.0 (both large) and one of 1.0;
    customer 2 has a single order of 2.0. Order 10 closes 2024-06-01, so it
    is open at a 2024-05-01 cutoff and closed at 2024-08-01.
    """
    customers = pl.LazyFrame({"id": [1, 2], "signed_up_at": [dt.datetime(2024, 1, 1)] * 2})
    orders = pl.LazyFrame(
        {
            "id": [10, 11, 12, 13],
            "customer_id": [1, 1, 1, 2],
            "product_id": [1, 2, 1, 2],
            "amount": [10.0, 20.0, 1.0, 2.0],
            "placed_at": [dt.datetime(2024, 3, 1)] * 4,
            "closed_at": [
                dt.datetime(2024, 6, 1),
                None,
                dt.datetime(2024, 4, 1),
                None,
            ],
        },
    )
    products = pl.LazyFrame({"id": [1, 2], "price": [3.0, 4.0]})
    return (
        tusk.Database("shop")
        .add_table(
            "customers",
            customers,
            primary_key="id",
            row_creation_time="signed_up_at",
        )
        .add_table(
            "orders",
            orders,
            primary_key="id",
            row_creation_time="placed_at",
            where={"large": nw.col("amount") >= 10.0},
            when={
                "open": lambda cutoff: nw.col("closed_at").is_null()
                | (nw.col("closed_at") > cutoff),
            },
        )
        .add_table("products", products, primary_key="id")
        .add_relationship(parent="customers", child="orders", foreign_key="customer_id")
        .add_relationship(parent="products", child="orders", foreign_key="product_id")
    )
```

The `test_clause_reaches_a_direct_feature_from_a_third_table` expectation of `7.0` is the sum of `products.price` over customer 1's *large* orders: order 10 on product 1 (3.0) and order 11 on product 2 (4.0). Order 12 is small and excluded. Customer 2 has no large order, so `Sum.default_value` of `0` applies.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_compiler_aggregation.py -k clause -v`
Expected: FAIL — the clause is ignored, so `SUM__orders__amount__WHERE__large` is `31.0` for customer 1 rather than `30.0`.

- [ ] **Step 3: Add `_clause_expr`**

In `src/tusk/compiler.py`, add below `_add_aggregations`:

```python
def _clause_expr(
    schema: TableSchema,
    clause: tuple[str, str] | None,
    cutoff_time: datetime | None,
) -> nw.Expr | None:
    """Build the mask a clause selects on the child's rows.

    Args:
        schema: The child table's schema, holding the declared clauses.
        clause: The (kind, key) pair, or None for an unclaused feature.
        cutoff_time: The cutoff, passed to a ``when`` clause's callable.

    Returns:
        The mask expression, or None when the feature has no clause.

    Raises:
        SchemaError: If the key is not declared on the child table.
    """
    if clause is None:
        return None

    kind, key = clause
    declared = schema.where if kind == "where" else schema.when
    if key not in declared:
        raise SchemaError(
            f"{kind} clause {key!r} is not declared on table {schema.name!r}; "
            f"declare it in add_table({kind}=...) or drop the feature",
        )
    return declared[key] if kind == "where" else declared[key](cutoff_time)
```

Import `TableSchema` from `tusk.database` at the top of the module.

- [ ] **Step 4: Batch by clause and filter the child frame**

**Do not mask the expressions.** `nw.col(x).filter(mask).sum()` looks like the
natural approach and it is wrong: `Count` builds `nw.len()`, and polars rejects
`nw.len().filter(...)` with `InvalidOperationError: Can't apply filtration
(e.g. drop_nulls) to scalar-like expression`. Filtering the frame instead needs
no per-primitive special case, and makes an empty mask take the same left-join
path an empty group already takes.

Replace the body of `_add_aggregations` after the `child = _table_frame(...)`
line with:

```python
    child_schema = database.schema(relationship.child)
    for clause, features in _by_clause(batch):
        mask = _clause_expr(child_schema, clause, cutoff_time)
        frame = _join_one_clause(
            frame,
            child if mask is None else child.filter(mask),
            database,
            table,
            relationship,
            features,
            cutoff_time,
        )
    return frame
```

Add the two helpers below it:

```python
def _by_clause(
    batch: Sequence[AggregationFeature],
) -> list[tuple[tuple[str, str] | None, list[AggregationFeature]]]:
    """Group a relationship's aggregations by the clause masking them.

    Args:
        batch: Every aggregation feature using one relationship.

    Returns:
        One (clause, features) pair per distinct clause, unclaused first.
    """
    grouped: dict[tuple[str, str] | None, list[AggregationFeature]] = {}
    for feature in batch:
        grouped.setdefault(feature.clause, []).append(feature)
    return sorted(grouped.items(), key=lambda item: item[0] is not None)


def _join_one_clause(
    frame: nw.LazyFrame,
    child: nw.LazyFrame,
    database: Database,
    table: str,
    relationship: Relationship,
    batch: Sequence[AggregationFeature],
    cutoff_time: datetime | None,
) -> nw.LazyFrame:
    """Fold one clause's aggregations into the parent with a single join.

    Args:
        frame: The parent frame being built.
        child: The child frame, already filtered to the clause.
        database: The database holding the schemas.
        table: The parent table's name.
        relationship: The relationship being aggregated across.
        batch: The aggregation features sharing this clause.
        cutoff_time: The cutoff, or None.

    Returns:
        The parent frame with this clause's columns joined on.
    """
    exprs = []
    for feature in batch:
        inputs = [nw.col(b.name) for b in feature.base_features]
        built = _build_expressions(feature.primitive, inputs, cutoff_time)
        exprs.extend(
            e.alias(n) for e, n in zip(built, feature.output_names, strict=True)
        )

    grouped = child.group_by(relationship.foreign_key).agg(*exprs)
    frame = frame.join(
        grouped,
        left_on=database.schema(table).primary_key,
        right_on=relationship.foreign_key,
        how="left",
    )

    defaults = [
        nw.col(name).fill_null(feature.primitive.default_value).alias(name)
        for feature in batch
        if feature.primitive.default_value is not None
        for name in feature.output_names
    ]
    return frame.with_columns(*defaults) if defaults else frame
```

`_join_one_clause` is the old `_add_aggregations` body from the
`exprs = []` line onward, moved verbatim and given the already-filtered child
frame. `_add_aggregations` keeps only the frame-building and the clause loop.

Unclaused features sort first and take `mask=None`, so they compile through the
identical expressions and the identical single join as today.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_compiler_aggregation.py -v`
Expected: PASS, including every pre-existing aggregation test — unclaused features take `mask=None` and compile unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/tusk/compiler.py tests/test_compiler_aggregation.py
git commit -m "feat(compiler): mask a child's rows with an aggregation's clause"
```

---

### Task 4: A `when` clause requires a cutoff time

**Files:**
- Modify: `src/tusk/compiler.py` (`_require_cutoff_time`)
- Test: `tests/test_compiler_aggregation.py`

**Interfaces:**
- Consumes: `AggregationFeature.clause` (Task 2), `_clause_database()` (Task 3).
- Produces: no new names. `_require_cutoff_time` now also rejects `when`-claused features.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_compiler_aggregation.py`:

```python
def test_when_clause_without_a_cutoff_time_is_rejected():
    """A cutoff-measuring clause names itself in the error."""
    database = _clause_database()
    features = FeatureList(
        "customers",
        [
            AggregationFeature(
                resolve("count"),
                (),
                Relationship("customers", "orders", "customer_id"),
                clause=("when", "open"),
            ),
        ],
    )
    with pytest.raises(ValidationError, match="open"):
        features.apply(database)


def test_where_clause_without_a_cutoff_time_is_allowed():
    """A static clause measures nothing, so it needs no cutoff."""
    database = _clause_database()
    features = FeatureList(
        "customers",
        [
            AggregationFeature(
                resolve("count"),
                (),
                Relationship("customers", "orders", "customer_id"),
                clause=("where", "large"),
            ),
        ],
    )
    matrix = nw.from_native(features.apply(database)).lazy().collect()
    assert dict(
        zip(matrix["id"], matrix["COUNT__orders__WHERE__large"]),
    ) == {1: 2, 2: 0}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_compiler_aggregation.py -k cutoff_time -v`
Expected: the first test FAILS — `_clause_expr` calls the `when` callable with `None`, raising a `TypeError` from the comparison rather than a `ValidationError`.

- [ ] **Step 3: Extend `_require_cutoff_time`**

Replace the body of `_require_cutoff_time` in `src/tusk/compiler.py`:

```python
    if cutoff_time is not None:
        return
    measuring = {
        primitive.name
        for feature in features
        if isinstance(primitive := getattr(feature, "primitive", None), NeedsCutoffTime)
    } | {name for feature in features if (name := _measuring_clause(feature))}
    if measuring:
        raise ValidationError(
            f"{', '.join(sorted(measuring))} needs a cutoff_time; pass one "
            "when applying the features",
        )
```

and add the predicate beside it, rather than inlining it into the
comprehension:

```python
def _measuring_clause(feature: Feature) -> str | None:
    """Name a feature's clause if that clause measures against the cutoff.

    Args:
        feature: The feature to inspect.

    Returns:
        A readable name for the clause, or None when the feature has no
        clause or its clause is static.
    """
    clause = getattr(feature, "clause", None)
    if clause is None or clause[0] != "when":
        return None
    return f"when clause {clause[1]!r}"
```

Update the docstring's summary line and `Raises:` block to say that a
primitive *or a when clause* may require the cutoff.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_compiler_aggregation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tusk/compiler.py tests/test_compiler_aggregation.py
git commit -m "feat(compiler): require a cutoff time for when-claused features"
```

---

### Task 5: Synthesis generates clause variants

**Files:**
- Modify: `src/tusk/synthesis.py` (`synthesize`, `_Context.__init__`, `_Context._aggregations`)
- Modify: `src/tusk/api.py` (`deep_feature_synthesis`)
- Test: `tests/test_synthesis.py`, `tests/test_deep_feature_synthesis.py`

**Interfaces:**
- Consumes: `TableSchema.clauses` (Task 1), `AggregationFeature.clause` (Task 2).
- Produces: `synthesize(..., where_primitives: Iterable[str | Primitive] | None = None)` and the same parameter on `deep_feature_synthesis`. `None` selects `WHERE_DEFAULTS = ("count", "sum")`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_synthesis.py`:

```python
def test_clause_variants_are_generated_for_where_primitives():
    """Each selected primitive gains one variant per declared clause."""
    database = _clause_database()
    features = synthesize(
        database,
        "customers",
        agg_primitives=["count"],
        trans_primitives=[],
        where_primitives=["count"],
        max_depth=1,
    )
    names = {f.name for f in features}
    assert "COUNT__orders" in names
    assert "COUNT__orders__WHERE__large" in names
    assert "COUNT__orders__WHEN__open" in names


def test_primitives_outside_where_primitives_get_no_variant():
    """A primitive not selected for clauses stays unclaused."""
    database = _clause_database()
    features = synthesize(
        database,
        "customers",
        agg_primitives=["count", "sum"],
        trans_primitives=[],
        where_primitives=["count"],
        max_depth=1,
    )
    names = {f.name for f in features}
    assert "COUNT__orders__WHERE__large" in names
    assert not any(n.startswith("SUM__") and "WHERE" in n for n in names)


def test_empty_where_primitives_generates_no_clause_features():
    """Passing () turns the whole mechanism off."""
    database = _clause_database()
    features = synthesize(
        database,
        "customers",
        agg_primitives=["count"],
        trans_primitives=[],
        where_primitives=(),
        max_depth=1,
    )
    assert not any("WHERE" in f.name or "WHEN" in f.name for f in features)


def test_a_clause_does_not_consume_depth():
    """A clause variant has the same depth as its unclaused twin."""
    database = _clause_database()
    features = synthesize(
        database,
        "customers",
        agg_primitives=["count"],
        trans_primitives=[],
        where_primitives=["count"],
        max_depth=1,
    )
    by_name = {f.name: f for f in features}
    assert by_name["COUNT__orders__WHERE__large"].depth == by_name["COUNT__orders"].depth
```

Copy `_clause_database()` from Task 3 into `tests/test_synthesis.py` as a
module-level helper, or move it to `tests/conftest.py` as a fixture named
`clause_db` and use that in both files. Prefer the fixture — two copies of a
six-table builder will drift.

Add to `tests/test_deep_feature_synthesis.py`:

```python
def test_where_primitives_defaults_to_count_and_sum():
    """The documented default reaches the motivating feature."""
    database = _clause_database()
    features = tusk.deep_feature_synthesis(
        database,
        "customers",
        agg_primitives=["count", "sum", "mean"],
        trans_primitives=[],
        max_depth=1,
        cutoff_time=dt.datetime(2024, 5, 1),
        features_only=True,
    )
    names = {f.name for f in features}
    assert "COUNT__orders__WHEN__open" in names
    assert "SUM__orders__amount__WHEN__open" in names
    assert not any(n.startswith("MEAN__") and "WHEN" in n for n in names)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_synthesis.py tests/test_deep_feature_synthesis.py -k where -v`
Expected: FAIL — `synthesize() got an unexpected keyword argument 'where_primitives'`.

- [ ] **Step 3: Add the default and thread the parameter**

In `src/tusk/primitives/aggregation.py`, below `AGG_DEFAULTS`:

```python
WHERE_DEFAULTS: tuple[str, ...] = ("count", "sum")
```

Export it from `src/tusk/primitives/__init__.py` alongside `AGG_DEFAULTS`, in
both the import block and `__all__`.

In `src/tusk/synthesis.py`, add the parameter to `synthesize` after
`trans_primitives`:

```python
    where_primitives: Iterable[str | Primitive] | None = None,
```

and in the body, beside the existing resolution:

```python
    where_agg = resolve_all(
        WHERE_DEFAULTS if where_primitives is None else where_primitives,
    )
    for primitive in where_agg:
        _require_kind(primitive, AggregationPrimitive, "where_primitives")
```

Pass `where_agg=where_agg` into `_Context(...)`, and add the matching
parameter and `self.where_agg = where_agg` assignment to `_Context.__init__`,
documenting it in that method's `Args:` block.

Document the parameter in `synthesize`'s `Args:` block:

```
        where_primitives: Aggregation primitives that additionally get one
            masked variant per clause declared on the child table, as names
            or instances. None selects ``WHERE_DEFAULTS``; ``()`` generates
            no clause features.
```

- [ ] **Step 4: Generate the variants**

In `_Context._aggregations`, replace the primitive loop with:

```python
            for primitive in self.agg:
                out.extend(self._aggregations_for(primitive, rel, usable, None))
            clauses = self.database.schema(rel.child).clauses
            for primitive in self.where_agg:
                for clause in clauses:
                    out.extend(
                        self._aggregations_for(primitive, rel, usable, clause),
                    )
        return out

    def _aggregations_for(
        self,
        primitive: Primitive,
        relationship: Relationship,
        usable: Sequence[Feature],
        clause: tuple[str, str] | None,
    ) -> list[Feature]:
        """Build every aggregation of one primitive across one relationship.

        Args:
            primitive: The aggregation primitive to apply.
            relationship: The parent-child link being aggregated across.
            usable: Features on the child that may serve as inputs.
            clause: The (kind, key) pair masking the child's rows, or None.

        Returns:
            One feature per usable input combination, or a single zero-arity
            feature when the primitive declares no signatures.
        """
        if not primitive.signatures:
            return [AggregationFeature(primitive, (), relationship, clause)]
        return [
            AggregationFeature(primitive, combo, relationship, clause)
            for combo in self._combinations(primitive, usable, relationship.child)
        ]
```

- [ ] **Step 5: Thread it through the public entry point**

In `src/tusk/api.py`, add `where_primitives: Iterable[str | Primitive] | None = None`
to `deep_feature_synthesis` after `trans_primitives`, pass it to `synthesize`,
and document it in the `Args:` block with the same wording used in
`synthesize`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_synthesis.py tests/test_deep_feature_synthesis.py -v`
Expected: PASS. Pre-existing tests that pass no `where_primitives` now also
generate clause variants — but only for tables declaring clauses, and no
existing fixture declares any, so their feature sets are unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/tusk/synthesis.py src/tusk/api.py src/tusk/primitives/ tests/
git commit -m "feat(synthesis): generate a masked variant per declared clause"
```

---

### Task 6: `FeatureList.apply` resolves clauses against the new database

**Files:**
- Test: `tests/test_feature_list.py`
- Modify: `src/tusk/feature_list.py` (docstring only)

**Interfaces:**
- Consumes: `_clause_expr`'s `SchemaError` (Task 3).
- Produces: no new names. This task proves the re-application contract and documents it.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_feature_list.py`:

```python
def test_clause_is_resolved_against_the_database_being_applied_to():
    """Features carry the key; the new database supplies the expression."""
    features = tusk.deep_feature_synthesis(
        _clause_database(),
        "customers",
        agg_primitives=["count"],
        trans_primitives=[],
        where_primitives=["count"],
        max_depth=1,
        cutoff_time=dt.datetime(2024, 5, 1),
        features_only=True,
    )
    relaxed = _clause_database(large_threshold=1.0)
    matrix = nw.from_native(
        features.apply(relaxed, cutoff_time=dt.datetime(2024, 5, 1)),
    ).lazy().collect()
    assert dict(
        zip(matrix["id"], matrix["COUNT__orders__WHERE__large"]),
    ) == {1: 3, 2: 1}


def test_missing_clause_on_the_new_database_names_the_key():
    """Applying to a database that never declared the clause fails loudly."""
    features = tusk.deep_feature_synthesis(
        _clause_database(),
        "customers",
        agg_primitives=["count"],
        trans_primitives=[],
        where_primitives=["count"],
        max_depth=1,
        cutoff_time=dt.datetime(2024, 5, 1),
        features_only=True,
    )
    without = _clause_database(declare_clauses=False)
    with pytest.raises(SchemaError, match="large"):
        features.apply(without, cutoff_time=dt.datetime(2024, 5, 1))
```

Extend the shared `_clause_database()` helper with two keyword arguments,
`large_threshold: float = 10.0` and `declare_clauses: bool = True`, so the
`where`/`when` mappings are built from them and can be omitted entirely.
Update its docstring to document both.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_feature_list.py -k clause -v`
Expected: the second test FAILS with `KeyError` rather than `SchemaError` if
Task 3's guard was skipped; if Task 3 is complete, both should pass — in which
case they are regression tests and you move straight to Step 4.

- [ ] **Step 3: Fix any gap the tests expose**

If the `KeyError` surfaces, the `key not in declared` guard in `_clause_expr`
is missing or unreachable. Restore it exactly as written in Task 3, Step 3.

- [ ] **Step 4: Document the contract**

In `src/tusk/feature_list.py`, extend `apply`'s docstring `Raises:` line for
`SchemaError` to include: *or a feature's clause key is not declared on the
table it masks in ``database``*.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_feature_list.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/tusk/feature_list.py tests/test_feature_list.py
git commit -m "test(feature-list): prove clauses resolve against the applied database"
```

---

### Task 7: Cross-backend and leakage guarantees

**Files:**
- Test: `tests/test_backend_duckdb.py`, `tests/test_compiler_aggregation.py`

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: no new names.

This is the task that earns the spec's central claims. It is separate because a
reviewer could reasonably accept the mechanism and reject its guarantees.

- [ ] **Step 1: Write the empty-mask matrix test**

Add to `tests/test_backend_duckdb.py`:

```python
def test_empty_mask_falls_back_to_the_primitive_default(duckdb_database):
    """No matching rows gives the same value as no rows at all."""
    features = FeatureList(
        "customers",
        [
            AggregationFeature(
                resolve(name),
                (IdentityFeature("orders", "amount", nw.Float64()),),
                Relationship("customers", "orders", "customer_id"),
                clause=("where", "impossible"),
            )
            for name in ("sum", "mean")
        ],
    )
    matrix = nw.from_native(features.apply(duckdb_database)).lazy().collect()
    assert matrix["SUM__orders__amount__WHERE__impossible"].to_list() == [0.0, 0.0]
    assert matrix["MEAN__orders__amount__WHERE__impossible"].to_list() == [None, None]
```

Build `duckdb_database` as a fixture mirroring the existing duckdb fixtures in
that file, declaring `where={"impossible": nw.col("amount") < 0.0}` on
`orders`. Add the polars twin of this test to
`tests/test_compiler_aggregation.py`, using `_clause_database()` with the same
`impossible` clause, so the two backends are asserted against identical
expected values.

`Sum.default_value` is `0` and `Mean` declares none, so the expectations above
are the primitives' own declared fallbacks — the point of the test is that both
backends reach them.

- [ ] **Step 2: Write the leakage test**

Add to `tests/test_compiler_aggregation.py`:

```python
def test_clause_sees_pre_update_values(updating_db):
    """A clause on an updated column reads the value restored by the cutoff."""
    database = updating_db
    schema = database.schema("orders")
    database._schemas["orders"] = replace(
        schema,
        where={"delivered": nw.col("status") == "delivered"},
    )
    features = FeatureList(
        "customers",
        [
            AggregationFeature(
                resolve("count"),
                (),
                Relationship("customers", "orders", "customer_id"),
                clause=("where", "delivered"),
            ),
        ],
    )
    matrix = nw.from_native(
        features.apply(database, cutoff_time=dt.datetime(2024, 6, 1)),
    ).lazy().collect()
    assert dict(
        zip(matrix["id"], matrix["COUNT__orders__WHERE__delivered"]),
    ) == {1: 0, 2: 1}
```

Order 10 is `delivered` in the raw frame but was updated on 2024-09-01, after
the cutoff, so `row_update_times` restores it to `pending` and the clause must
not count it. Order 11 is genuinely `pending`. Order 12 is `delivered` and was
updated before the cutoff, so it counts. A result of `{1: 1, 2: 1}` means the
mask read post-cutoff data and the feature leaks.

Reaching into `database._schemas` is deliberate: it keeps the shared
`updating_db` fixture untouched for every other test. If the reviewer prefers,
add a `clauses` keyword to the `updating_db` fixture instead and pass the
clause in — either is acceptable, but do not change the fixture's default
shape.

Import `replace` from `dataclasses` at the top of the test module.

- [ ] **Step 3: Write the join-path scope test**

This pins the semantics Task 8 documents. It asserts the behaviour that
surprises people, so that a future change cannot quietly "fix" it into
something else.

Add to `tests/test_compiler_aggregation.py`:

```python
def test_a_clause_does_not_scope_the_tables_below_it():
    """A clause masks its own table's rows, never its children's.

    Car 1 was owned by customer 1 and is now owned by customer 2, and was
    repaired twice under each owner. A current clause on cars selects the
    car for customer 2 only, and that car brings its whole repair history
    with it -- all four repairs, not the two from customer 2's era.
    """
    customers = pl.LazyFrame(
        {"id": [1, 2], "signed_up_at": [dt.datetime(2024, 1, 1)] * 2},
    )
    cars = pl.LazyFrame(
        {
            "id": [1, 2],
            "customer_id": [1, 2],
            "bought_at": [dt.datetime(2024, 1, 1)] * 2,
            "sold_at": [dt.datetime(2024, 6, 1), None],
        },
    )
    repairs = pl.LazyFrame(
        {
            "id": [100, 101, 102, 103],
            "car_id": [1, 1, 1, 1],
            "cost": [1.0, 2.0, 4.0, 8.0],
            "repaired_at": [
                dt.datetime(2024, 3, 1),
                dt.datetime(2024, 4, 1),
                dt.datetime(2024, 8, 1),
                dt.datetime(2024, 9, 1),
            ],
        },
    )
    database = (
        tusk.Database("garage")
        .add_table(
            "customers",
            customers,
            primary_key="id",
            row_creation_time="signed_up_at",
        )
        .add_table(
            "cars",
            cars,
            primary_key="id",
            row_creation_time="bought_at",
            when={
                "current": lambda cutoff: nw.col("sold_at").is_null()
                | (nw.col("sold_at") > cutoff),
            },
        )
        .add_table(
            "repairs",
            repairs,
            primary_key="id",
            row_creation_time="repaired_at",
        )
        .add_relationship(parent="customers", child="cars", foreign_key="customer_id")
        .add_relationship(parent="cars", child="repairs", foreign_key="car_id")
    )

    repair_count = AggregationFeature(
        resolve("count"), (), Relationship("cars", "repairs", "car_id"),
    )
    feature = AggregationFeature(
        resolve("sum"),
        (repair_count,),
        Relationship("customers", "cars", "customer_id"),
        clause=("when", "current"),
    )
    features = FeatureList("customers", [feature])
    matrix = nw.from_native(
        features.apply(database, cutoff_time=dt.datetime(2024, 12, 1)),
    ).lazy().collect()

    values = dict(zip(matrix["id"], matrix[feature.name]))
    assert values == {1: 0, 2: 4}
```

Customer 1 currently owns no car, so `Sum.default_value` of `0` applies.
Customer 2 owns car 1, whose lifetime repair count is `4`. A result of
`{2: 2}` would mean the clause had somehow scoped the repairs, which is issue
#29 and explicitly not this feature.

- [ ] **Step 4: Run the three tests to verify they fail or pass honestly**

Run: `uv run pytest tests/test_compiler_aggregation.py tests/test_backend_duckdb.py -k "empty_mask or pre_update" -v`
Expected: PASS if Task 3 built the mask from the frame `base_frame` returned.
If the leakage test FAILS, the mask is being built against the raw frame —
fix `_add_aggregations` to take `child` (the already-restored frame) rather
than re-reading `database.frame(...)`.

- [ ] **Step 5: Run the whole suite**

Run: `just test && just test-differential`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/
git commit -m "test: pin clause behaviour across backends, leakage and join paths"
```

---

### Task 8: Documentation

**Files:**
- Modify: `README.md`
- Modify or create: the docs page covering `add_table` under `docs/`
- Test: `uv run pytest tests/test_packaging.py`

**Interfaces:**
- Consumes: the finished feature.
- Produces: user-facing documentation, including the join-path warning.

- [ ] **Step 1: Add the clause example to the README**

Extend the `db.add_table("orders", ...)` call in the README's Usage section
with a `where` and a `when` clause, and add `where_primitives` to the
`deep_feature_synthesis` call, so the headline example shows the feature:

```python
db.add_table(
    "orders",
    orders_lf,
    primary_key="id",
    row_creation_time="placed_at",
    row_update_times={"payed_at": {"payed_at": None, "payment_method": None}},
    where={"large": nw.col("amount") >= 100.0},
    when={"open": lambda cutoff: nw.col("closed_at").is_null()
                               | (nw.col("closed_at") > cutoff)},
)
```

- [ ] **Step 2: Document what a clause scopes**

Add a subsection to the docs page covering `add_table`, titled
"What a clause scopes". It must state, in prose a user reads before trusting a
feature:

- A clause masks the rows of the table it is declared on, at the moment that
  table is grouped. It does not reach the tables below it.
- Worked example: with `customers <- cars <- repairs` and a `current` clause on
  `cars`, `SUM(cars.COUNT(cars.repairs) WHEN current)` is "over the cars this
  customer currently owns, each car's **lifetime** repair count". Repairs from a
  previous owner's era are attributed to the current owner.
- Normalizing ownership into its own table does not change this. It decides
  which customer a car belongs to in a window, not which repairs.
- The split falls out for free only when the descendant already carries the
  bridge key, so each repair hangs off exactly one ownership row.
- Link to issue #29 for the general case.

Copy the reasoning from the spec's "Clause scope along a join path" section
rather than re-deriving it.

- [ ] **Step 3: Verify the docs build**

Run: `just docs`
Expected: the site builds into `site/` with no error.

- [ ] **Step 4: Run the full gate**

Run: `just check`
Expected: lint, tests and differential tests all PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/
git commit -m "docs: document where and when clauses and what they scope"
```

---

## Notes for the executor

- **Task 3 is the load-bearing one.** Mask by filtering the child frame, never
  by filtering each expression. `nw.len().filter(mask)` raises
  `InvalidOperationError` on polars, so expression masking silently excludes
  `Count` — the single most useful clause feature. If frame filtering fails on
  a backend, stop and report rather than working around it.
- **Tasks 1-2 are independent of each other** and can be done in either order.
  Tasks 3-7 are strictly sequential.
- **Do not implement `valid_from`/`valid_until`.** Deferred by spec decision 11.
- **Do not implement interval-scoped join paths.** That is issue #29.
