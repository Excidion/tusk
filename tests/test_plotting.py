import datetime as dt
import sys

import narwhals as nw
import polars as pl
import pytest

import tusk
from tusk.plotting import (
    SchemaDiagram,
    build_schema_source,
    render_column_name,
    render_dtype,
    render_table_name,
)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (nw.Int64(), "Int64"),
        (nw.String(), "String"),
        (nw.Float64(), "Float64"),
        (nw.Date(), "Date"),
        (nw.Datetime(time_unit="us"), "Datetime[us]"),
        (nw.Datetime(time_unit="ns", time_zone="UTC"), "Datetime[ns-UTC]"),
        (
            nw.Datetime(time_unit="ns", time_zone="America/New_York"),
            "Datetime[ns-America_New_York]",
        ),
        (nw.Duration(time_unit="ms"), "Duration[ms]"),
        (nw.List(nw.Int64()), "List[Int64]"),
        (nw.List(nw.List(nw.Int64())), "List[List[Int64]]"),
        (nw.Enum(["a", "b", "c"]), "Enum[3]"),
        (nw.Struct({"x": nw.Int64(), "y": nw.Int64()}), "Struct[2]"),
    ],
)
def test_dtype_renders_as_a_mermaid_safe_token(dtype, expected):
    assert render_dtype(dtype) == expected


def test_timezone_punctuation_is_replaced():
    # Slashes, plus signs and colons are all parse errors in Mermaid's type
    # slot, so every character outside the safe set collapses to underscore.
    rendered = render_dtype(nw.Datetime(time_unit="ns", time_zone="UTC+02:00"))
    assert rendered == "Datetime[ns-UTC_02_00]"


FIGURE_SPACE = "\u2007"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("amount", "amount"),
        ("total_2024", "total_2024"),
        ("n-items", "n-items"),
        ("a.b", "a.b"),
        ("straße", "straße"),
        ("unit price", f"unit{FIGURE_SPACE}price"),
        ("2024_total", "_2024_total"),
        ("2024 total", f"_2024{FIGURE_SPACE}total"),
    ],
)
def test_column_name_is_made_parseable(name, expected):
    assert render_column_name(name) == expected


def test_table_name_is_quoted():
    # Quoting is what lets a table name contain a space, which an attribute
    # name cannot.
    assert render_table_name("order items") == '"order items"'


def test_a_quote_in_a_table_name_is_dropped():
    # An embedded quote would close the entity name early and break the whole
    # diagram, not just this one label.
    assert render_table_name('a"b') == '"ab"'


@pytest.fixture
def two_table_db():
    """A parent and a child, plus one table related to neither."""
    return (
        tusk.Database("shop")
        .add_table(
            "customers",
            pl.LazyFrame(
                {
                    "id": [1],
                    "signed_up_at": [dt.datetime(2024, 1, 1)],
                    "region": ["eu"],
                },
            ),
            primary_key="id",
            row_creation_time="signed_up_at",
        )
        .add_table(
            "orders",
            pl.LazyFrame({"id": [1], "customer_id": [1], "amount": [1.0]}),
            primary_key="id",
        )
        .add_table(
            "regions",
            pl.LazyFrame({"code": ["eu"]}),
            primary_key="code",
        )
        .add_relationship(parent="customers", child="orders", foreign_key="customer_id")
    )


def test_every_table_and_relationship_appears(two_table_db):
    source = build_schema_source(two_table_db, columns=True)
    assert source.startswith("erDiagram\n")
    assert '"customers" ||--o{ "orders" : customer_id' in source
    # A table with no relationships still has to be drawn.
    assert '"regions"' in source


def test_columns_true_lists_every_column_with_markers(two_table_db):
    source = build_schema_source(two_table_db, columns=True)
    assert "Int64 id PK" in source
    assert "Int64 customer_id FK" in source
    assert 'Datetime[us] signed_up_at "row creation time"' in source
    assert "Float64 amount" in source


def test_columns_false_omits_every_attribute(two_table_db):
    source = build_schema_source(two_table_db, columns=False)
    assert "Float64" not in source
    assert "amount" not in source
    assert '"customers" ||--o{ "orders" : customer_id' in source


def test_columns_structural_keeps_only_keys_and_the_time_index(two_table_db):
    source = build_schema_source(two_table_db, columns="structural")
    assert "Int64 customer_id FK" in source
    assert 'Datetime[us] signed_up_at "row creation time"' in source
    assert "String region" not in source
    assert "amount" not in source


def test_a_column_that_is_both_keys_gets_both_markers():
    # orders.id is the primary key and also the foreign key to customers.
    db = (
        tusk.Database("d")
        .add_table("customers", pl.LazyFrame({"id": [1]}), primary_key="id")
        .add_table("orders", pl.LazyFrame({"id": [1]}), primary_key="id")
        .add_relationship(parent="customers", child="orders", foreign_key="id")
    )
    assert "Int64 id PK, FK" in build_schema_source(db, columns=True)


def test_a_table_without_a_primary_key_has_no_marker():
    with pytest.warns(tusk.exceptions.MissingPrimaryKeyWarning):
        db = tusk.Database("d").add_table("t", pl.LazyFrame({"a": [1]}))
    source = build_schema_source(db, columns=True)
    assert "Int64 a" in source
    assert "PK" not in source


def test_an_unknown_columns_value_is_rejected(two_table_db):
    with pytest.raises(ValueError, match="structural"):
        build_schema_source(two_table_db, columns="all")


def test_generated_source_parses(two_table_db):
    # The golden-string tests above compare text and would happily accept
    # source Mermaid cannot parse. This is the test that catches an escaping
    # regression, so it renders for real.
    mermaidx = pytest.importorskip("mermaidx")
    mermaidx.render(build_schema_source(two_table_db, columns=True)).svg()


def test_hostile_names_still_parse():
    db = tusk.Database("d").add_table(
        'order "items"',
        pl.LazyFrame({"id": [1], "unit price": [1.0], "2024 total": [2.0]}),
        primary_key="id",
    )
    mermaidx = pytest.importorskip("mermaidx")
    mermaidx.render(build_schema_source(db, columns=True)).svg()


SOURCE = 'erDiagram\n  "t" {\n    Int64 id PK\n  }\n'


def test_str_is_the_source():
    assert str(SchemaDiagram(SOURCE)) == SOURCE


def test_markdown_repr_is_a_mermaid_block():
    # Jupyter, GitHub and the docs site all render a fenced mermaid block.
    rendered = SchemaDiagram(SOURCE)._repr_markdown_()
    assert rendered.startswith("```mermaid\n")
    assert rendered.endswith("```")
    assert SOURCE in rendered


@pytest.mark.parametrize("suffix", [".mmd", ".md"])
def test_saving_source_needs_no_renderer(tmp_path, suffix, monkeypatch):
    # Hiding mermaidx proves the text formats never reach for it.
    monkeypatch.setitem(sys.modules, "mermaidx", None)
    path = tmp_path / f"schema{suffix}"
    SchemaDiagram(SOURCE).save(path)
    assert path.read_text(encoding="utf-8") == SOURCE


@pytest.mark.parametrize("suffix", [".svg", ".png"])
def test_saving_an_image_writes_a_file(tmp_path, suffix):
    pytest.importorskip("mermaidx")
    path = tmp_path / f"schema{suffix}"
    SchemaDiagram(SOURCE).save(path)
    assert path.stat().st_size > 0


def test_an_unsupported_suffix_is_rejected(tmp_path):
    with pytest.raises(ValueError, match=".mmd"):
        SchemaDiagram(SOURCE).save(tmp_path / "schema.gif")


def test_a_missing_renderer_names_the_extra(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mermaidx", None)
    with pytest.raises(ImportError, match=r"tusk-ml\[plot\]"):
        SchemaDiagram(SOURCE).save(tmp_path / "schema.svg")


def test_plot_returns_a_diagram_of_the_database(db):
    diagram = db.plot()
    assert isinstance(diagram, SchemaDiagram)
    assert '"customers" ||--o{ "sessions" : customer_id' in diagram.source
    assert '"sessions" ||--o{ "transactions" : session_id' in diagram.source


def test_plot_passes_the_column_mode_through(db):
    assert "age" in db.plot(columns=True).source
    assert "age" not in db.plot(columns="structural").source
    assert "age" not in db.plot(columns=False).source


def test_plot_reads_no_rows(db, monkeypatch):
    # The whole diagram comes from declared schema. Collecting would make
    # plotting cost as much as computing, on a method that looks free.
    def fail(*args, **kwargs):
        raise AssertionError("plot() must not collect")

    monkeypatch.setattr(nw.LazyFrame, "collect", fail)
    db.plot()
