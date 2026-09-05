import datetime as dt
import sys

import narwhals as nw
import polars as pl
import pytest

import tusk
from tusk.plotting import (
    SchemaDiagram,
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
        ("revenue %", f"revenue{FIGURE_SPACE}_"),
        ("", "_"),
    ],
)
def test_column_name_is_made_parseable(name, expected):
    assert render_column_name(name) == expected


@pytest.mark.parametrize(
    "unsafe",
    [
        ":",
        ";",
        "#",
        "'",
        "|",
        "/",
        "\\",
        "<",
        "=",
        "+",
        "&",
        "!",
        "?",
        "@",
        "$",
        "~",
        "^",
        "`",
        "\t",
        "{",
        "}",
        ">",
        "%",
        '"',
    ],
)
def test_every_reported_unsafe_character_is_replaced_in_column_names(unsafe):
    # Each of these was independently confirmed, by rendering through
    # mermaidx, to make an attribute name unparseable.
    assert unsafe not in render_column_name(f"a{unsafe}b")


def test_table_name_is_quoted():
    # Quoting is what lets a table name contain a space, which an attribute
    # name cannot.
    assert render_table_name("order items") == '"order items"'


def test_a_quote_in_a_table_name_is_dropped():
    # An embedded quote would close the entity name early and break the whole
    # diagram, not just this one label.
    assert render_table_name('a"b') == '"ab"'


def test_a_percent_in_a_table_name_is_replaced():
    # Reported broken even though the entity name is quoted.
    assert render_table_name("a%b") == '"a_b"'


def test_a_newline_in_a_table_name_is_replaced():
    assert render_table_name("a\nb") == '"a_b"'


def test_an_empty_table_name_falls_back_to_a_placeholder():
    # `""` quotes to `""`, which Mermaid also rejects.
    assert render_table_name("") == '"_"'


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
    source = SchemaDiagram.from_database(two_table_db, columns=True).source
    assert source.startswith("erDiagram\n")
    assert '"customers" 1 to 0+ "orders" : ""' in source
    # A table with no relationships still has to be drawn.
    assert '"regions"' in source


def test_every_entity_is_drawn_exactly_once(two_table_db):
    # Mermaid does not care what order entities and edges appear in, so this
    # counts rather than positions them: a table drawn twice, or dropped, is
    # the defect worth catching.
    source = SchemaDiagram.from_database(two_table_db, columns=True).source
    for table in ("customers", "orders", "regions"):
        assert source.count(f'"{table}" {{') == 1


def test_columns_true_lists_every_column_with_markers(two_table_db):
    source = SchemaDiagram.from_database(two_table_db, columns=True).source
    assert "Int64 id PK" in source
    assert "Int64 customer_id FK" in source
    assert 'Datetime[us] signed_up_at "row creation time"' in source
    assert "Float64 amount" in source


def test_columns_false_omits_every_attribute(two_table_db):
    source = SchemaDiagram.from_database(two_table_db, columns=False).source
    assert "Float64" not in source
    assert "amount" not in source
    assert '"customers" 1 to 0+ "orders" : ""' in source


def test_columns_structural_keeps_only_keys_and_the_time_index(two_table_db):
    source = SchemaDiagram.from_database(two_table_db, columns="structural").source
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
    assert "Int64 id PK, FK" in SchemaDiagram.from_database(db, columns=True).source


def test_a_table_without_a_primary_key_has_no_marker():
    with pytest.warns(tusk.exceptions.MissingPrimaryKeyWarning):
        db = tusk.Database("d").add_table("t", pl.LazyFrame({"a": [1]}))
    source = SchemaDiagram.from_database(db, columns=True).source
    assert "Int64 a" in source
    assert "PK" not in source


def test_an_unknown_columns_value_is_rejected(two_table_db):
    with pytest.raises(ValueError, match="structural"):
        SchemaDiagram.from_database(two_table_db, columns="all")


@pytest.mark.parametrize("columns", [0, 1])
def test_an_int_that_equals_a_bool_is_rejected(two_table_db, columns):
    # `0 == False` and `1 == True`, so a membership check using `==` would
    # silently accept these; only an identity check tells them apart.
    with pytest.raises(ValueError, match="structural"):
        SchemaDiagram.from_database(two_table_db, columns=columns)


def test_generated_source_parses(two_table_db):
    # The golden-string tests above compare text and would happily accept
    # source Mermaid cannot parse. This is the test that catches an escaping
    # regression, so it renders for real.
    mermaidx = pytest.importorskip("mermaidx")
    mermaidx.render(
        SchemaDiagram.from_database(two_table_db, columns=True).source
    ).svg()


def test_hostile_names_still_parse():
    db = tusk.Database("d").add_table(
        'order "items"',
        pl.LazyFrame({"id": [1], "unit price": [1.0], "2024 total": [2.0]}),
        primary_key="id",
    )
    mermaidx = pytest.importorskip("mermaidx")
    mermaidx.render(SchemaDiagram.from_database(db, columns=True).source).svg()


def test_a_percent_sign_in_a_column_name_still_parses():
    # The bug the finding opened with: an ordinary CSV header broke the whole
    # diagram because the foreign-key edge label passed through no sanitising
    # at all, and render_column_name only handled spaces and leading digits.
    db = tusk.Database("d").add_table(
        "t",
        pl.LazyFrame({"id": [1], "revenue %": [1.0]}),
        primary_key="id",
    )
    mermaidx = pytest.importorskip("mermaidx")
    mermaidx.render(SchemaDiagram.from_database(db, columns=True).source).svg()


@pytest.fixture
def parent_child_db():
    """A parent and a child linked by a foreign key named after the column."""

    def _make(foreign_key):
        return (
            tusk.Database("d")
            .add_table("customers", pl.LazyFrame({"id": [1]}), primary_key="id")
            .add_table(
                "orders",
                pl.LazyFrame({"id": [1], foreign_key: [1]}),
                primary_key="id",
            )
            .add_relationship(
                parent="customers", child="orders", foreign_key=foreign_key
            )
        )

    return _make


def test_a_foreign_key_named_with_a_space_still_parses(parent_child_db):
    mermaidx = pytest.importorskip("mermaidx")
    db = parent_child_db("unit price")
    mermaidx.render(SchemaDiagram.from_database(db, columns=True).source).svg()


def test_a_foreign_key_containing_a_percent_sign_still_parses(parent_child_db):
    mermaidx = pytest.importorskip("mermaidx")
    db = parent_child_db("cust%id")
    mermaidx.render(SchemaDiagram.from_database(db, columns=True).source).svg()


def test_a_table_name_containing_a_percent_sign_still_parses():
    mermaidx = pytest.importorskip("mermaidx")
    db = tusk.Database("d").add_table(
        "cust%omers",
        pl.LazyFrame({"id": [1]}),
        primary_key="id",
    )
    mermaidx.render(SchemaDiagram.from_database(db, columns=True).source).svg()


def test_an_empty_table_name_still_parses():
    mermaidx = pytest.importorskip("mermaidx")
    db = tusk.Database("d").add_table("", pl.LazyFrame({"id": [1]}), primary_key="id")
    mermaidx.render(SchemaDiagram.from_database(db, columns=True).source).svg()


def test_an_empty_column_name_still_parses():
    mermaidx = pytest.importorskip("mermaidx")
    db = tusk.Database("d").add_table(
        "t",
        pl.LazyFrame({"id": [1], "": [1.0]}),
        primary_key="id",
    )
    mermaidx.render(SchemaDiagram.from_database(db, columns=True).source).svg()


def test_a_foreign_key_names_the_table_it_points_at(parent_child_db):
    source = SchemaDiagram.from_database(
        parent_child_db("customer_id"), columns=True
    ).source
    assert 'customer_id FK "-> customers"' in source
    mermaidx = pytest.importorskip("mermaidx")
    mermaidx.render(source).svg()


def test_a_foreign_key_with_a_space_still_marks_its_attribute(parent_child_db):
    # The attribute slot cannot be quoted and uses U+2007, which renders as a
    # space but is not one to the parser.
    source = SchemaDiagram.from_database(
        parent_child_db("unit price"), columns=True
    ).source
    assert f"unit{FIGURE_SPACE}price FK" in source


def test_a_foreign_key_pointing_at_two_tables_names_both():
    db = (
        tusk.Database("d")
        .add_table("customers", pl.LazyFrame({"id": [1]}), primary_key="id")
        .add_table("vendors", pl.LazyFrame({"id": [1]}), primary_key="id")
        .add_table(
            "orders",
            pl.LazyFrame({"id": [1], "party_id": [1]}),
            primary_key="id",
        )
        .add_relationship(parent="customers", child="orders", foreign_key="party_id")
        .add_relationship(parent="vendors", child="orders", foreign_key="party_id")
    )
    source = SchemaDiagram.from_database(db, columns=True).source
    assert 'party_id FK "-> customers, vendors"' in source
    mermaidx = pytest.importorskip("mermaidx")
    mermaidx.render(source).svg()


def test_a_child_keyed_by_the_link_can_hold_only_one_row():
    # profile's primary key IS the foreign key, so it is unique there and one
    # customer matches at most one profile. Both keys are declared, so the
    # narrower cardinality costs no query.
    db = (
        tusk.Database("d")
        .add_table("customers", pl.LazyFrame({"id": [1]}), primary_key="id")
        .add_table(
            "profile", pl.LazyFrame({"customer_id": [1]}), primary_key="customer_id"
        )
        .add_relationship(
            parent="customers", child="profile", foreign_key="customer_id"
        )
    )
    source = SchemaDiagram.from_database(db, columns=True).source
    assert '"customers" 1 to zero or one "profile" : ""' in source
    mermaidx = pytest.importorskip("mermaidx")
    mermaidx.render(source).svg()


def test_a_punctuated_column_name_keeps_its_punctuation(parent_child_db):
    # Brackets, parens and commas parse in the attribute slot. They were
    # stripped only while the edge label shared this cleaner.
    db = tusk.Database("d").add_table(
        "t",
        pl.LazyFrame({"id": [1], "count(*)": [1], "a,b": [1], "x[0]": [1]}),
        primary_key="id",
    )
    source = SchemaDiagram.from_database(db, columns=True).source
    for name in ("count(*)", "a,b", "x[0]"):
        assert name in source
    mermaidx = pytest.importorskip("mermaidx")
    mermaidx.render(source).svg()


def test_a_quote_in_a_parent_table_name_is_dropped_from_the_comment():
    # A literal quote would close the comment early and break the diagram.
    db = (
        tusk.Database("d")
        .add_table('a"b', pl.LazyFrame({"id": [1]}), primary_key="id")
        .add_table(
            "orders",
            pl.LazyFrame({"id": [1], "parent_id": [1]}),
            primary_key="id",
        )
        .add_relationship(parent='a"b', child="orders", foreign_key="parent_id")
    )
    source = SchemaDiagram.from_database(db, columns=True).source
    assert 'parent_id FK "-> ab"' in source
    mermaidx = pytest.importorskip("mermaidx")
    mermaidx.render(source).svg()


@pytest.fixture
def one_table_diagram():
    """A diagram of the smallest database that draws anything."""
    db = tusk.Database("d").add_table(
        "t",
        pl.LazyFrame({"id": [1]}),
        primary_key="id",
    )
    return SchemaDiagram.from_database(db)


def test_str_is_the_source(one_table_diagram):
    assert str(one_table_diagram) == one_table_diagram.source


def test_markdown_repr_is_a_mermaid_block(one_table_diagram):
    # Jupyter, GitHub and the docs site all render a fenced mermaid block.
    rendered = one_table_diagram._repr_markdown_()
    assert rendered.startswith("```mermaid\n")
    assert rendered.endswith("```")
    assert one_table_diagram.source in rendered


@pytest.mark.parametrize("suffix", [".mmd", ".md"])
def test_saving_source_needs_no_renderer(
    tmp_path, suffix, monkeypatch, one_table_diagram
):
    # Hiding mermaidx proves the text formats never reach for it.
    monkeypatch.setitem(sys.modules, "mermaidx", None)
    path = tmp_path / f"schema{suffix}"
    one_table_diagram.save(path)
    assert path.read_text(encoding="utf-8") == one_table_diagram.source


@pytest.mark.parametrize("suffix", [".svg", ".png", ".pdf"])
def test_saving_an_image_writes_a_file(tmp_path, suffix, one_table_diagram):
    pytest.importorskip("mermaidx")
    path = tmp_path / f"schema{suffix}"
    one_table_diagram.save(path)
    assert path.stat().st_size > 0


def test_an_unsupported_suffix_is_rejected(tmp_path, one_table_diagram):
    with pytest.raises(ValueError, match=".mmd"):
        one_table_diagram.save(tmp_path / "schema.gif")


def test_a_missing_renderer_names_the_extra(tmp_path, monkeypatch, one_table_diagram):
    monkeypatch.setitem(sys.modules, "mermaidx", None)
    with pytest.raises(ImportError, match=r"tusk-ml\[plot\]"):
        one_table_diagram.save(tmp_path / "schema.svg")


def test_plot_returns_a_diagram_of_the_database(db):
    diagram = db.plot()
    assert isinstance(diagram, SchemaDiagram)
    assert '"customers" 1 to 0+ "sessions" : ""' in diagram.source
    assert '"sessions" 1 to 0+ "transactions" : ""' in diagram.source


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
