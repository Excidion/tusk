import narwhals as nw
import polars as pl

from tusk._codes import code_expression


def test_code_expression_gives_the_category_position():
    table = nw.from_native(
        pl.DataFrame(
            {"x": ["b", "a", None]},
            schema={"x": pl.Enum(["a", "b"])},
        ),
    )
    result = table.select(code_expression("x", ["a", "b"]))
    assert result["x"].to_list() == [1.0, 0.0, None]
    assert result.schema["x"] == nw.Float64
