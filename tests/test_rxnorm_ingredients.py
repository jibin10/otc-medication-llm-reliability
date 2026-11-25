from hrp.ingredient_agreement import mixed_literal_agreement
from hrp.rxnorm_ingredients import ingredient_identity_options, resolve_ingredients


def test_ingredient_resolution():
    concepts = {
        "100": ("SBD", "Example brand"),
        "200": ("SCD", "Example drug"),
        "300": ("SCDC", "Example component"),
        "400": ("PIN", "Example ingredient salt"),
        "500": ("IN", "Example ingredient"),
    }

    relationships = {
        "100": {"200"},
        "200": {"300"},
        "300": {"400"},
        "400": {"500"},
    }

    base, precise, unresolved = resolve_ingredients("100", concepts, relationships)

    assert base == ["500"]
    assert precise == ["400"]
    assert unresolved == 0


def test_substance_codes_keep_precise_forms_and_require_distinct_ingredient_branches():
    concepts = {"a": ("IN", "Base"), "b": ("PIN", "Salt"), "c": ("IN", "Other")}
    options = ingredient_identity_options(
        ["a", "c"],
        ["b"],
        concepts,
        {"b": {"a"}},
        {"a": {"AAAAAAAAA1"}, "b": {"BBBBBBBBB2"}, "c": {"CCCCCCCCC3"}},
    )
    assert options == [["AAAAAAAAA1", "BBBBBBBBB2"], ["CCCCCCCCC3"]]
    assert mixed_literal_agreement(frozenset({"BBBBBBBBB2", "CCCCCCCCC3"}), options)
    assert not mixed_literal_agreement(frozenset({"AAAAAAAAA1", "BBBBBBBBB2"}), options)
