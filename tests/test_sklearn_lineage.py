from tusk.sklearn._lineage import make_sentinels

COLUMNS = ["age", "MODE__transactions__category", "COUNT__transactions"]


def test_every_column_gets_a_distinct_sentinel():
    sentinels = make_sentinels(COLUMNS)
    assert len(set(sentinels.mapping.values())) == len(COLUMNS)


def test_sentinels_are_fixed_width_so_prefixes_cannot_collide():
    sentinels = make_sentinels([f"c{i}" for i in range(11)])
    lengths = {len(name) for name in sentinels.mapping.values()}
    assert len(lengths) == 1


def test_a_one_hot_name_resolves_to_its_source_column():
    sentinels = make_sentinels(COLUMNS)
    encoded = f"oh__{sentinels.mapping['MODE__transactions__category']}_a"
    assert sentinels.sources(encoded) == ["MODE__transactions__category"]


def test_a_multi_input_name_resolves_to_every_source():
    sentinels = make_sentinels(COLUMNS)
    encoded = f"{sentinels.mapping['age']} {sentinels.mapping['COUNT__transactions']}"
    assert sentinels.sources(encoded) == ["age", "COUNT__transactions"]


def test_an_opaque_name_resolves_to_nothing():
    assert make_sentinels(COLUMNS).sources("pca0") == []


def test_restore_puts_real_names_back():
    sentinels = make_sentinels(COLUMNS)
    encoded = f"oh__{sentinels.mapping['MODE__transactions__category']}_a"
    assert sentinels.restore(encoded) == "oh__MODE__transactions__category_a"
