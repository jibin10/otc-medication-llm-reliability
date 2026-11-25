from hrp.ingredient_agreement import compare_ingredient_names, normalized_name


def test_name_normalization():
    assert normalized_name("  Acetaminophen  ") == "acetaminophen"


def test_exact_base_ingredient():
    result = compare_ingredient_names(
        frozenset({"acetaminophen"}),
        frozenset({"acetaminophen"}),
        frozenset(),
    )
    assert result == "exact_base_names"


def test_precise_ingredient():
    result = compare_ingredient_names(
        frozenset({"diphenhydramine hydrochloride"}),
        frozenset({"diphenhydramine"}),
        frozenset({"diphenhydramine hydrochloride"}),
    )
    assert result == "exact_precise_names"


def test_unmatched_names_require_review():
    result = compare_ingredient_names(
        frozenset({"ingredient a"}),
        frozenset({"ingredient b"}),
        frozenset(),
    )
    assert result == "review_no_exact_name_overlap"


import json

from hrp.ingredient_agreement import (
    add_fda_ingredient_identities,
    extract_label_ingredients,
    integrate_product,
    mixed_literal_agreement,
)


def source_identity_product(ndc, ingredients, uniis):
    row = evidence_product(ingredients)
    row.update(
        product_ndc=ndc,
        support_tier="FDA_ONLY",
        ingredient_origin="ndc_structured",
        ingredient_eligible="True",
        formulation_eligible="True",
        strength_issues_json="[]",
        source_metadata_json=json.dumps([{"openfda": {"unii": uniis}}]),
    )
    return row


def test_equal_fda_uniis_resolve_different_submitted_ingredient_names(tmp_path):
    rows = [
        source_identity_product("a", [{"name": "HERB A", "strength": "5 mg/g"}], ["AAAAAAAAA1"]),
        source_identity_product(
            "b", [{"name": "HERB A WHOLE", "strength": "5 mg/g"}], ["AAAAAAAAA1"]
        ),
    ]
    summary = add_fda_ingredient_identities(rows, tmp_path)
    assert summary["comparison_eligible"] == 2
    assert {json.loads(row["active_ingredients_json"])[0]["identity"] for row in rows} == {
        "fda_unii:AAAAAAAAA1"
    }
    assert json.loads(rows[0]["active_ingredients_json"])[0]["name"] == "HERB A"


def test_unordered_fda_unii_arrays_are_validated_without_positional_pairing(tmp_path):
    rows = [
        source_identity_product("a", [{"name": "ALPHA", "strength": "5 mg/g"}], ["AAAAAAAAA1"]),
        source_identity_product("b", [{"name": "BETA", "strength": "5 mg/g"}], ["BBBBBBBBB2"]),
        source_identity_product(
            "c",
            [{"name": "ALPHA", "strength": "5 mg/g"}, {"name": "BETA", "strength": "5 mg/g"}],
            ["BBBBBBBBB2", "AAAAAAAAA1"],
        ),
    ]
    add_fda_ingredient_identities(rows, tmp_path)
    ingredients = json.loads(rows[2]["active_ingredients_json"])
    assert [i["identity"] for i in ingredients] == ["fda_unii:AAAAAAAAA1", "fda_unii:BBBBBBBBB2"]
    assert rows[2]["comparison_eligible"] == "True"


def test_incomplete_chemical_identity_mapping_preserves_name_tasks_and_withholds_comparison(
    tmp_path,
):
    row = source_identity_product(
        "c",
        [{"name": "ALPHA", "strength": "5 mg/g"}, {"name": "BETA", "strength": "5 mg/g"}],
        ["BBBBBBBBB2"],
    )
    add_fda_ingredient_identities([row], tmp_path)
    assert row["ingredient_eligible"] == "True"
    assert row["comparison_eligible"] == "False"
    assert row["formulation_eligible"] == "False"


def test_complete_unordered_identity_set_does_not_invent_name_alignment(tmp_path):
    row = source_identity_product(
        "c",
        [{"name": "ALPHA", "strength": "5 mg/g"}, {"name": "BETA", "strength": "5 mg/g"}],
        ["BBBBBBBBB2", "AAAAAAAAA1"],
    )
    add_fda_ingredient_identities([row], tmp_path)
    assert row["comparison_eligible"] == "True"
    assert row["formulation_eligible"] == "False"
    assert set(json.loads(row["ingredient_identities_json"])) == {
        "fda_unii:AAAAAAAAA1",
        "fda_unii:BBBBBBBBB2",
    }
    assert [i["identity"] for i in json.loads(row["active_ingredients_json"])] == ["alpha", "beta"]


def test_identifier_paths_recover_supported_records_without_inventing_name_agreement(tmp_path):
    row = source_identity_product(
        "a", [{"name": "SOURCE NAME", "strength": "5 mg/g"}], ["AAAAAAAAA1"]
    )
    row.update(
        ingredient_eligible="False",
        support_tier="NAME_NORMALIZATION_UNRESOLVED",
        exclusion_reasons_json='["rxnorm_ingredient_names_unresolved"]',
    )
    result = add_fda_ingredient_identities(
        [row], tmp_path, {"a": {"unii_options_json": '[["AAAAAAAAA1"]]'}}
    )
    assert row["ingredient_eligible"] == "True"
    assert row["support_tier"] == "FDA_RXNORM_IDENTIFIER_PATH"
    assert row["rxnorm_identifier_status"] == "FDA_UNII_PATH_SUPPORTED"
    assert json.loads(row["active_ingredients_json"])[0]["name"] == "SOURCE NAME"
    assert result["ingredient_records_recovered_by_identifier_paths"] == 1


def test_conflicting_identifier_paths_do_not_silently_recover_a_record(tmp_path):
    row = source_identity_product(
        "a", [{"name": "SOURCE NAME", "strength": "5 mg/g"}], ["AAAAAAAAA1"]
    )
    row.update(
        ingredient_eligible="False",
        support_tier="NAME_NORMALIZATION_UNRESOLVED",
        exclusion_reasons_json='["rxnorm_ingredient_names_unresolved"]',
    )
    add_fda_ingredient_identities([row], tmp_path, {"a": {"unii_options_json": '[["BBBBBBBBB2"]]'}})
    assert row["ingredient_eligible"] == "False"
    assert row["comparison_eligible"] == "False"


def evidence_product(ingredients=None):
    return {
        "display_name": "Example",
        "dosage_form": "CREAM",
        "route": '["TOPICAL"]',
        "active_ingredients_json": json.dumps(
            ingredients or [{"name": "MENTHOL", "strength": "5 mg/g"}]
        ),
    }


def test_fda_only_support_is_eligible_without_rxnorm():
    result = integrate_product(evidence_product(), {}, {})
    assert result["ingredient_eligible"] == "True"
    assert result["formulation_eligible"] == "True"
    assert result["support_tier"] == "FDA_ONLY"


def test_unknown_strength_does_not_erase_known_ingredients():
    row = evidence_product([{"name": "MENTHOL", "strength": "unresolved"}])
    result = integrate_product(row, {}, {})
    assert result["ingredient_eligible"] == "True"
    assert result["formulation_eligible"] == "False"


def test_label_extraction_rejects_incomplete_lists_and_preserves_named_basis():
    names = ["POTASSIUM NITRATE", "SODIUM FLUORIDE"]
    text = [
        "Active ingredients Potassium nitrate 5% Sodium fluoride 0.25% (0.15% w/v fluoride ion)"
    ]
    parsed = extract_label_ingredients(text, names)
    assert parsed == [
        {"name": "potassium nitrate", "strength": "5%"},
        {"name": "sodium fluoride", "strength": "0.25%"},
    ]
    assert not extract_label_ingredients(text, ["POTASSIUM NITRATE"])
    row = evidence_product(
        [
            {"name": "POTASSIUM NITRATE", "strength": "50 mg/g"},
            {"name": "SODIUM FLUORIDE", "strength": "1.15 mg/g"},
        ]
    )
    result = integrate_product(row, {"active_ingredient_json": json.dumps(text)}, {})
    assert result["ingredient_eligible"] == "True"
    assert result["formulation_eligible"] == "False"
    assert "label_strength_basis_unresolved" in result["strength_issues_json"]
    assert "1.15 mg/g" in result["original_ingredients_json"]


def test_mixed_precise_and_base_names_require_bijective_literal_coverage():
    names = frozenset({"carboxymethylcellulose sodium", "glycerin"})
    options = [["carboxymethylcellulose", "carboxymethylcellulose sodium"], ["glycerin"]]
    assert mixed_literal_agreement(names, options)
    assert not mixed_literal_agreement(names, [["carboxymethylcellulose"], ["glycerin"]])
    assert not mixed_literal_agreement(names, options + [["menthol"]])


def test_conflicting_sources_are_not_verified_by_agreement_elsewhere():
    mapping = {
        "ingredient_status": "resolved_path",
        "in_names_json": '["lidocaine"]',
        "pin_names_json": "[]",
    }
    result = integrate_product(evidence_product(), {}, mapping)
    assert result["ingredient_eligible"] == "False"
    assert result["support_tier"] == "NAME_NORMALIZATION_UNRESOLVED"
