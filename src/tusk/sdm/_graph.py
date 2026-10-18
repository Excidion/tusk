"""Builds the relationships and the time columns of the `sdm` export."""

from __future__ import annotations

from tusk.database import Database


def relationship_mappings(database: Database) -> list[dict[str, str]]:
    """Return the relationships of a database as `sdm` relationship mappings.

    The child table is the left table and its foreign key the left column. The
    parent table is the right table and its primary key the right column.

    Args:
        database: The database.

    Returns:
        One mapping per relationship, in insertion order.
    """
    return [
        {
            "left_table": relationship.child,
            "left_column": relationship.foreign_key,
            "right_table": relationship.parent,
            "right_column": database.require_primary_key(relationship.parent),
        }
        for relationship in database.relationships
    ]


def time_columns(database: Database) -> dict[str, str]:
    """Return the row creation time column of each table that declares one.

    Args:
        database: The database.

    Returns:
        A mapping of table name to its row creation time column.
    """
    schemas = (database.get_schema(name) for name in database.table_names)
    return {
        schema.name: schema.row_creation_time
        for schema in schemas
        if schema.row_creation_time is not None
    }


def key_columns(database: Database, name: str) -> frozenset[str]:
    """Return the key columns of a table.

    Args:
        database: The database.
        name: The table's name.

    Returns:
        The table's primary key, if declared, and its foreign keys.
    """
    primary_key = database.get_schema(name).primary_key
    foreign_keys = {r.foreign_key for r in database.parents_of(name)}
    keys = foreign_keys if primary_key is None else foreign_keys | {primary_key}
    return frozenset(keys)
