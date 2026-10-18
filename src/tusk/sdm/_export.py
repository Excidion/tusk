"""Exports a database as the `sdm` sampler that feeds `KumoRelational`."""

from __future__ import annotations

import narwhals as nw
from sdm import TableTensor
from sdm.relational import RelationalData, RelationalSampler

from tusk.database import Database
from tusk.sdm._columns import prepare_table
from tusk.sdm._graph import key_columns, relationship_mappings, time_columns


def _read_table(database: Database, name: str) -> nw.DataFrame:
    """Return a table of a database as a collected narwhals table.

    Args:
        database: The database.
        name: The table's name.

    Returns:
        The table as the database shows it at its cutoff time.
    """
    table = nw.from_native(database.get_table(name))
    if isinstance(table, nw.LazyFrame):
        return table.collect()
    return table


def _build_tensor(database: Database, name: str) -> TableTensor:
    """Build the `sdm` tensor of one table of a database.

    A cudf table goes to the GPU through `TableTensor.from_cudf`. Every other
    table goes through pyarrow.

    Args:
        database: The database.
        name: The table's name.

    Returns:
        The table as a `TableTensor`.
    """
    table, types = prepare_table(
        name,
        _read_table(database, name),
        key_columns(database, name),
    )
    if table.implementation.is_cudf():
        return TableTensor.from_cudf(table.to_native(), types)
    return TableTensor.from_arrow(table.to_arrow(), types)


def to_relational_data(database: Database) -> RelationalData:
    """Build the `sdm` relational data of a database.

    The data holds the database's tables and relationships. The database is
    read as it stands at its cutoff time, so
    `to_relational_data(database.as_of(cutoff_time))` exports the rows created
    up to the cutoff time. Unlike the sampler, the data needs no `pyg-lib`.

    A key column gets the semantic type `id`. A `Boolean`, `Enum`,
    `Duration` or `Decimal` column becomes `Float64`. A column whose dtype has
    no semantic type is left out.

    Args:
        database: The database.

    Returns:
        The relational data of the database.

    Warns:
        UnsupportedDtypeWarning: If a column is left out of its table.
    """
    tables = {name: _build_tensor(database, name) for name in database.table_names}
    return RelationalData(tables, relationship_mappings(database))


def to_relational_sampler(database: Database) -> RelationalSampler:
    """Build the `sdm` sampler of a database.

    The sampler holds the relational data of
    [`to_relational_data`][tusk.sdm.to_relational_data], and limits the rows it
    samples by each table's row creation time. It reads the relational data
    from its `data` attribute. Building it needs `pyg-lib`.

    Args:
        database: The database.

    Returns:
        The sampler over the database's relational data.

    Warns:
        UnsupportedDtypeWarning: If a column is left out of its table.
    """
    return RelationalSampler(to_relational_data(database), time_columns(database))
