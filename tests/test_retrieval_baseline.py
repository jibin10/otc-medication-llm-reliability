import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from hrp.retrieval_baseline import (
    build_catalog,
    evaluate_cases,
    load_evaluation_cases,
    predict,
    predict_from_evidence,
    retrieve_evidence,
)


def product(
    ndc="00001-001",
    name="Example",
    ingredient="MENTHOL",
    strength="5 mg/1",
    form="TABLET",
    route="ORAL",
):
    return {
        "product_ndc": ndc,
        "display_name": name,
        "brand_name": name,
        "generic_name": ingredient,
        "dosage_form": form,
        "route": json.dumps([route]),
        "active_ingredients_json": json.dumps([{"name": ingredient, "strength": strength}]),
    }


def catalog_for(products, missing=(), conflicts=()):
    labels, drugs, mappings = [], [], []
    for row in products:
        ndc = row["product_ndc"]
        labels.append(
            {
                "product_ndc": ndc,
                "link_status": "matched",
                "label_id": "label-" + ndc,
                "label_set_id": "set-" + ndc,
                "active_ingredient_json": '["Source text"]',
            }
        )
        if ndc in missing:
            continue
        ingredients = json.loads(row["active_ingredients_json"])
        names = [item["name"] for item in ingredients]
        if ndc in conflicts:
            names = ["OTHER INGREDIENT"]
        drugs.append({"product_ndc": ndc, "rxnorm_status": "matched", "rxnorm_rxcui": ndc})
        mappings.append(
            {
                "product_ndc": ndc,
                "ingredient_status": "resolved_path",
                "drug_rxcui": ndc,
                "in_names_json": json.dumps(names),
                "in_rxcuis_json": json.dumps(["id-" + name.lower() for name in names]),
            }
        )
    return build_catalog(products, labels, drugs, mappings)


def reference(row):
    return {
        "product_ndc": row["product_ndc"],
        "name": row["display_name"],
        "active_ingredients": json.loads(row["active_ingredients_json"]),
        "dosage_form": row["dosage_form"],
        "route": json.loads(row["route"]),
    }


def case(request, answer, case_id="case", task="product_identification", partition="train"):
    return {
        "case_id": case_id,
        "partition": partition,
        "task": task,
        "input_json": json.dumps(request),
        "reference_json": json.dumps(answer),
        "family_group_ids_json": '["family"]',
    }


def evaluate(cases, catalog):
    evidence = {row["case_id"]: retrieve_evidence(row["input_json"], catalog) for row in cases}
    predictions = {key: predict_from_evidence(value) for key, value in evidence.items()}
    return evaluate_cases(cases, predictions, evidence)


def test_unique_ndc_and_source_identifiers():
    catalog = catalog_for([product()])
    result = predict({"name": "  EXAMPLE "}, catalog)["products"][0]
    assert result["status"] == "EXACT_NDC_IDENTIFIED"
    assert result["product_ndc"] == "00001-001"
    assert result["ingredients"] == [
        {"identity": "menthol", "source_names": ["MENTHOL"], "rxnorm_ids": ["id-menthol"]}
    ]
    evidence = retrieve_evidence({"product_ndc": "00001-001"}, catalog)
    assert evidence["products"][0]["candidates"][0]["fda_label"]["label_id"] == "label-00001-001"


def test_multiple_ndcs_do_not_become_an_arbitrary_product():
    catalog = catalog_for([product(), product("00002-001")])
    result = predict({"name": "Example"}, catalog)["products"][0]
    assert result["status"] == "FORMULATION_IDENTIFIED"
    assert result["product_ndc"] is None
    assert result["candidate_ndcs"] == ["00001-001", "00002-001"]
    assert result["ingredient_status"] == "SUPPORTED"


@pytest.mark.parametrize(
    "change",
    [
        {"ingredient": "LIDOCAINE"},
        {"strength": "10 mg/1"},
        {"form": "CAPSULE"},
        {"route": "TOPICAL"},
    ],
)
def test_same_name_with_different_formulations_is_ambiguous(change):
    catalog = catalog_for([product(), product("00002-001", **change)])
    result = predict({"name": "Example"}, catalog)["products"][0]
    assert result["status"] == "INSUFFICIENT_INFORMATION"
    assert result["ingredients"] == []
    assert len(result["candidate_ndcs"]) == 2


@pytest.mark.parametrize(
    "details",
    [
        {"native_strengths": ["5 MG / 1"]},
        {"dosage_form": "tablet"},
        {"route": ["oral"]},
    ],
)
def test_supplied_details_can_distinguish_products(details):
    catalog = catalog_for(
        [product(), product("00002-001", strength="10 mg/1", form="CAPSULE", route="TOPICAL")]
    )
    result = predict({"name": "Example", **details}, catalog)["products"][0]
    assert result["product_ndc"] == "00001-001"


def test_explicit_ndc_and_conflicting_description():
    catalog = catalog_for([product(), product("00002-001", ingredient="LIDOCAINE")])
    result = predict({"product_ndc": "00002-001"}, catalog)["products"][0]
    assert result["product_ndc"] == "00002-001"
    for details in ({"name": "Different"}, {"native_strengths": ["500 mg/1"]}):
        result = predict({"product_ndc": "00002-001", **details}, catalog)["products"][0]
        assert result["status"] == "SOURCE_CONFLICT"
        assert result["ingredients"] == []


def test_no_match_and_missing_name():
    catalog = catalog_for([product()])
    for request in ({"name": "No Such Product"}, {"product_ndc": "00000-000"}):
        assert predict(request, catalog)["products"][0]["status"] == "NO_MATCH"
    assert predict({}, catalog)["products"][0]["status"] == "INSUFFICIENT_INFORMATION"


def test_missing_mapping_does_not_remove_a_different_plausible_product():
    catalog = catalog_for(
        [product(), product("00002-001", ingredient="LIDOCAINE")], missing=["00002-001"]
    )
    result = predict({"name": "Example"}, catalog)["products"][0]
    assert result["status"] == "INSUFFICIENT_INFORMATION"
    assert result["candidate_ndcs"] == ["00001-001", "00002-001"]


def test_missing_mapping_keeps_fda_supported_ingredients_and_known_identity():
    catalog = catalog_for([product()], missing=["00001-001"])
    result = predict({"name": "Example"}, catalog)["products"][0]
    assert result["status"] == "EXACT_NDC_IDENTIFIED"
    assert result["ingredient_status"] == "SUPPORTED"
    assert result["ingredients"][0]["identity"] == "menthol"
    assert result["ingredients"][0]["rxnorm_ids"] == []


def test_same_formulation_with_fda_only_alternative_has_supported_ingredients():
    catalog = catalog_for([product(), product("00002-001")], missing=["00002-001"])
    result = predict({"name": "Example"}, catalog)["products"][0]
    assert result["status"] == "FORMULATION_IDENTIFIED"
    assert result["ingredient_status"] == "SUPPORTED"


def test_unknown_metadata_does_not_remove_a_candidate():
    unknown = product("00002-001")
    unknown["dosage_form"] = ""
    catalog = catalog_for([product(), unknown])
    result = predict({"name": "Example", "dosage_form": "TABLET"}, catalog)["products"][0]
    assert result["status"] == "INSUFFICIENT_INFORMATION"
    assert len(result["candidate_ndcs"]) == 2


def test_conflicting_ingredient_or_product_mapping():
    catalog = catalog_for([product()], conflicts=["00001-001"])
    assert predict({"name": "Example"}, catalog)["products"][0]["status"] == "SOURCE_CONFLICT"
    catalog = catalog_for([product()])
    catalog["products"]["00001-001"]["rxnorm_ingredients"]["drug_rxcui"] = "different"
    assert predict({"name": "Example"}, catalog)["products"][0]["status"] == "SOURCE_CONFLICT"


def test_missing_label_does_not_hide_a_catalog_product():
    catalog = build_catalog([product()], [], [], [])
    result = predict({"name": "Example"}, catalog)["products"][0]
    assert result["status"] == "EXACT_NDC_IDENTIFIED"
    assert result["ingredient_status"] == "SUPPORTED"


@pytest.mark.parametrize(
    "ingredient,expected,shared",
    [
        ("MENTHOL", "shared", ["menthol"]),
        ("LIDOCAINE", "disjoint", []),
        ("MENTHOL HYDROCHLORIDE", "disjoint", []),
    ],
)
def test_shared_disjoint_and_distinct_salts(ingredient, expected, shared):
    catalog = catalog_for([product(), product("00002-001", "Second", ingredient)])
    result = predict({"product_a": {"name": "Example"}, "product_b": {"name": "Second"}}, catalog)
    assert result["comparison"]["ingredient_overlap"] == expected
    assert result["comparison"]["shared_active_ingredient_identities"] == shared


def test_partial_overlap_uses_complete_exact_ingredient_sets():
    a, b = product(), product("00002-001", "Second")
    b["active_ingredients_json"] = json.dumps(
        [
            {"name": "MENTHOL", "strength": "5 mg/1"},
            {"name": "LIDOCAINE", "strength": "4 mg/1"},
        ]
    )
    result = predict(
        {"product_a": {"name": "Example"}, "product_b": {"name": "Second"}}, catalog_for([a, b])
    )
    assert result["comparison"]["shared_active_ingredient_identities"] == ["menthol"]
    assert result["comparison"]["same_active_ingredient_set"] is False


def test_unlabelled_strengths_do_not_identify_swapped_ingredient_strengths():
    a, b = product(), product("00002-001")
    a["active_ingredients_json"] = json.dumps(
        [
            {"name": "MENTHOL", "strength": "5 mg/1"},
            {"name": "LIDOCAINE", "strength": "4 mg/1"},
        ]
    )
    b["active_ingredients_json"] = json.dumps(
        [
            {"name": "MENTHOL", "strength": "4 mg/1"},
            {"name": "LIDOCAINE", "strength": "5 mg/1"},
        ]
    )
    result = predict(
        {"name": "Example", "native_strengths": ["4 mg/1", "5 mg/1"]}, catalog_for([a, b])
    )
    assert result["products"][0]["status"] == "INSUFFICIENT_INFORMATION"


def test_precise_salt_identity_does_not_collapse_to_shared_base():
    catalog = catalog_for(
        [product(ingredient="LIDOCAINE"), product("00002-001", "Second", "LIDOCAINE HYDROCHLORIDE")]
    )
    mapping = catalog["products"]["00002-001"]["rxnorm_ingredients"]
    mapping.update(
        in_names=["lidocaine"],
        in_rxcuis=["base-id"],
        pin_names=["lidocaine hydrochloride"],
        pin_rxcuis=["salt-id"],
    )
    result = predict({"product_a": {"name": "Example"}, "product_b": {"name": "Second"}}, catalog)
    assert result["comparison"]["ingredient_overlap"] == "disjoint"
    assert result["products"][1]["ingredients"][0]["rxnorm_ids"] == ["salt-id"]


def test_quantity_conversion_requires_matching_dimensions_and_stated_basis():
    catalog = catalog_for([product(strength="5 mg/mL")])
    for strength in ["0.5%", "5 mg/g"]:
        result = predict({"name": "Example", "native_strengths": [strength]}, catalog)
        assert result["products"][0]["status"] == "NO_MATCH"
    result = predict({"name": "Example", "native_strengths": ["25 mg/5 mL"]}, catalog)
    assert result["products"][0]["status"] == "EXACT_NDC_IDENTIFIED"


def test_both_comparison_products_must_resolve_independently():
    catalog = catalog_for(
        [product(), product("00002-001", ingredient="LIDOCAINE"), product("00003-001", "Second")]
    )
    result = predict({"product_a": {"name": "Example"}, "product_b": {"name": "Second"}}, catalog)
    assert result["products"][0]["status"] == "INSUFFICIENT_INFORMATION"
    assert result["products"][1]["status"] == "EXACT_NDC_IDENTIFIED"
    assert result["comparison"]["status"] == "UNRESOLVED"
    assert result["comparison"]["ingredient_overlap"] is None


@pytest.mark.parametrize(
    "field",
    ["task", "case_type", "partition", "reference_json", "provenance_json", "active_ingredients"],
)
def test_prediction_rejects_reference_and_metadata_fields(field):
    catalog = catalog_for([product()])
    with pytest.raises(ValueError, match="description fields"):
        predict({"name": "Example", field: "answer hint"}, catalog)


def test_reference_changes_cannot_change_predictions_or_evidence():
    catalog = catalog_for([product()])
    row = case(
        {"name": "Example"}, {"answer_status": "IDENTIFIED", "product": reference(product())}
    )
    before = predict(row["input_json"], catalog)
    evidence = retrieve_evidence(row["input_json"], catalog)
    row["reference_json"] = row["provenance_json"] = "poison answer"
    row["task"] = row["case_type"] = "unrelated"
    assert predict(row["input_json"], catalog) == before
    assert predict_from_evidence(evidence) == before
    assert "poison answer" not in json.dumps(evidence)


def test_test_rows_are_skipped_before_json_decoding(tmp_path: Path):
    row = case({"name": "Example"}, {"answer_status": "INSUFFICIENT_INFORMATION"})
    frozen = {key: "invalid JSON" for key in row}
    frozen["partition"] = "test"
    path = tmp_path / "cases.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerows([row, frozen])
    assert load_evaluation_cases(path) == [row]
    with pytest.raises(ValueError, match="Only train and validation"):
        load_evaluation_cases(path, ("test",))
    with pytest.raises(ValueError, match="Test evaluation"):
        evaluate_cases([frozen], {}, {})
    with pytest.raises(ValueError, match="separately"):
        evaluate_cases([row, {**row, "partition": "validation"}], {}, {})


def test_unresolved_cases_stay_in_metrics_and_exact_ndc_is_separate():
    a, b = product(), product("00002-001")
    catalog = catalog_for([a, b])
    rows = [
        case({"name": "Example"}, {"answer_status": "IDENTIFIED", "product": reference(a)}),
        case(
            {"name": "Missing"},
            {"answer_status": "IDENTIFIED", "product": reference(a)},
            case_id="missing",
        ),
    ]
    result = evaluate(rows, catalog)["product_resolution"]
    assert result["exact_ndc_accuracy_when_justified"]["value"] is None
    assert result["reference_ndc_not_established_by_input"]["numerator"] == 2
    assert result["formulation_accuracy_all_eligible"]["value"] == 0.5
    ingredients = result["ingredient_identification"]
    assert ingredients["set_exact_match_all_eligible"]["denominator"] == 2
    assert ingredients["recall"]["value"] == 0.5
    assert result["incorrect_abstention_against_supported_reference"]["numerator"] == 1


def test_correct_abstention_has_no_single_ingredient_target():
    catalog = catalog_for([product(), product("00002-001", ingredient="LIDOCAINE")])
    row = case(
        {"name": "Example"},
        {"answer_status": "INSUFFICIENT_INFORMATION"},
        task="ambiguity_detection",
    )
    metrics = evaluate([row], catalog)["product_resolution"]
    assert metrics["correct_abstention"]["value"] == 1
    assert metrics["ingredient_identification"]["set_exact_match_all_eligible"]["value"] is None


def test_arbitrary_ndc_is_flagged_even_when_ingredients_are_correct():
    a = product()
    catalog = catalog_for([a, product("00002-001")])
    row = case({"name": "Example"}, {"answer_status": "IDENTIFIED", "product": reference(a)})
    evidence = {"case": retrieve_evidence(row["input_json"], catalog)}
    predictions = {"case": predict_from_evidence(evidence["case"])}
    predictions["case"]["products"][0].update(
        status="EXACT_NDC_IDENTIFIED", product_ndc=a["product_ndc"]
    )
    metrics = evaluate_cases([row], predictions, evidence)["product_resolution"]
    assert metrics["exact_ndc_accuracy_when_justified"]["value"] is None
    assert metrics["unjustified_exact_ndc_predictions"]["value"] == 1
    assert metrics["ingredient_identification"]["set_exact_match_all_eligible"]["value"] == 1


def test_fda_supported_prediction_is_not_called_unsupported_when_rxnorm_is_missing():
    a = product()
    catalog = catalog_for([a], missing=[a["product_ndc"]])
    row = case({"name": "Example"}, {"answer_status": "IDENTIFIED", "product": reference(a)})
    evidence = {"case": retrieve_evidence(row["input_json"], catalog)}
    predictions = {"case": predict_from_evidence(evidence["case"])}
    predictions["case"]["products"][0]["ingredients"] = [
        {"identity": "menthol", "source_names": ["MENTHOL"], "rxnorm_ids": []}
    ]
    metrics = evaluate_cases([row], predictions, evidence)["product_resolution"]
    unsupported = metrics["ingredient_identification"][
        "unsupported_ingredients_against_retrieved_evidence"
    ]
    assert unsupported["numerator"] == 0
    assert unsupported["denominator"] == 1


def test_comparison_metrics_keep_unresolved_and_absent_classes():
    a, b = product(), product("00002-001", "Second")
    incomplete = {**b, "active_ingredients_json": "[]"}
    catalog = catalog_for([a, incomplete], missing=[b["product_ndc"]])
    row = case(
        {"product_a": {"name": "Example"}, "product_b": {"name": "Second"}},
        {
            "answer_status": "SUPPORTED",
            "product_a": reference(a),
            "product_b": reference(b),
            "ingredient_overlap": "shared",
            "shared_active_ingredient_identities": ["menthol"],
        },
        task="ingredient_comparison",
    )
    result = evaluate([row], catalog)["ingredient_comparison"]
    assert result["classification_accuracy_all_cases"]["value"] == 0
    assert result["unresolved_comparisons"]["value"] == 1
    assert result["missed_overlaps_including_unresolved"]["value"] == 1
    assert result["false_overlap_predictions"]["value"] is None
    assert result["shared_ingredient_metrics"]["precision"]["value"] is None
    assert result["shared_ingredient_metrics"]["recall"]["value"] == 0


def test_evaluator_counts_unsupported_predictions_and_source_differences():
    a = product()
    catalog = catalog_for([a])
    answer = reference(a)
    answer["active_ingredients"][0]["strength"] = "0.5%"
    row = case({"name": "Example"}, {"answer_status": "IDENTIFIED", "product": answer})
    evidence = {"case": retrieve_evidence(row["input_json"], catalog)}
    predictions = {"case": deepcopy(predict_from_evidence(evidence["case"]))}
    predictions["case"]["products"][0]["ingredients"].append(
        {"identity": "invented", "source_names": ["invented"], "rxnorm_ids": []}
    )
    metrics = evaluate_cases([row], predictions, evidence)["product_resolution"]
    assert metrics["source_formulation_differs_from_reference"]["numerator"] == 1
    assert metrics["formulation_accuracy_all_eligible"]["value"] == 0
    ingredients = metrics["ingredient_identification"]
    assert ingredients["unsupported_ingredients_against_reference"]["numerator"] == 1
    assert ingredients["unsupported_ingredients_against_retrieved_evidence"]["numerator"] == 1
    assert ingredients["precision"]["value"] == 0.5


def test_catalog_and_predictions_are_deterministic_without_mutation():
    rows = [product(), product("00002-001")]
    catalog = catalog_for(rows)
    original = deepcopy(catalog)
    request = {"name": "Example"}
    assert predict(request, catalog) == predict(request, catalog_for(rows[::-1]))
    assert catalog == original


def test_exact_reference_denominator_does_not_disappear_after_retrieval_failure():
    a = product()
    answer = reference(a)
    answer["formulation_supported"] = True
    row = case({"name": "Missing"}, {"answer_status": "IDENTIFIED", "product": answer})
    metrics = evaluate([row], catalog_for([a]))["product_resolution"]
    assert metrics["exact_ndc_accuracy_when_justified"] == {
        "numerator": 0,
        "denominator": 1,
        "value": 0.0,
    }
    assert metrics["ingredient_identification"]["set_exact_match_all_eligible"]["denominator"] == 1


def test_unresolved_formulation_can_have_a_known_ndc_and_supported_ingredients():
    a = product(strength="unresolved")
    a["formulation_eligible"] = "False"
    a["ingredient_eligible"] = "True"
    a["strength_issues_json"] = '["unresolved_strength_or_form"]'
    answer = reference(a)
    answer["active_ingredients"] = [{"name": "MENTHOL"}]
    answer["formulation_supported"] = False
    catalog = catalog_for([a], missing=[a["product_ndc"]])
    request = {"name": "Example", "product_ndc": a["product_ndc"]}
    predicted = predict(request, catalog)["products"][0]
    assert predicted["status"] == "EXACT_NDC_IDENTIFIED"
    assert predicted["ingredient_status"] == "SUPPORTED"
    assert predicted["formulation"] is None
    metrics = evaluate([case(request, {"answer_status": "IDENTIFIED", "product": answer})], catalog)
    assert metrics["product_resolution"]["formulation_accuracy_all_eligible"]["denominator"] == 0
    assert (
        metrics["product_resolution"]["ingredient_identification"]["set_exact_match_all_eligible"][
            "value"
        ]
        == 1
    )


def test_uncertain_strength_does_not_remove_a_plausible_alternative():
    a, b = product(), product("00002-001", ingredient="LIDOCAINE", strength="10 mg/1")
    b["formulation_eligible"] = "False"
    result = predict({"name": "Example", "native_strengths": ["5 mg/1"]}, catalog_for([a, b]))
    assert result["products"][0]["status"] == "INSUFFICIENT_INFORMATION"
    assert len(result["products"][0]["candidate_ndcs"]) == 2


def test_source_unii_aliases_share_ingredients_without_a_product_rxnorm_mapping():
    a = product(ingredient="HERB A")
    b = product("00002-001", "Other", "HERB A WHOLE")
    for row in (a, b):
        ingredients = json.loads(row["active_ingredients_json"])
        ingredients[0]["identity"] = "fda_unii:AAAAAAAAA1"
        row["active_ingredients_json"] = json.dumps(ingredients)
        row["ingredient_identities_json"] = '["fda_unii:AAAAAAAAA1"]'
        row["comparison_eligible"] = "True"
    catalog = catalog_for([a, b], missing=[a["product_ndc"], b["product_ndc"]])
    result = predict(
        {
            "product_a": {"product_ndc": a["product_ndc"]},
            "product_b": {"product_ndc": b["product_ndc"]},
        },
        catalog,
    )
    assert result["comparison"]["ingredient_overlap"] == "shared"
    assert result["comparison"]["same_active_ingredient_set"] is True


def test_unpaired_identifier_set_preserves_names_without_inventing_name_links():
    a = product()
    a["active_ingredients_json"] = (
        '[{"name":"ALPHA","strength":"5 mg/g"},{"name":"BETA","strength":"5 mg/g"}]'
    )
    a["ingredient_identities_json"] = '["fda_unii:AAAAAAAAA1","fda_unii:BBBBBBBBB2"]'
    a["comparison_eligible"] = "True"
    a["formulation_eligible"] = "False"
    result = predict(
        {"product_ndc": a["product_ndc"]}, catalog_for([a], missing=[a["product_ndc"]])
    )["products"][0]
    assert {i["identity"] for i in result["ingredients"]} == {
        "fda_unii:AAAAAAAAA1",
        "fda_unii:BBBBBBBBB2",
    }
    assert all(i["source_names"] == [] for i in result["ingredients"])
    assert result["reported_ingredient_names"] == ["ALPHA", "BETA"]
    assert result["formulation"] is None


def test_missing_chemical_identifier_never_becomes_disjoint_reference_evidence():
    a, b = product(), product("00002-001", "Other", "LIDOCAINE")
    a["comparison_eligible"] = "False"
    result = predict(
        {
            "product_a": {"product_ndc": a["product_ndc"]},
            "product_b": {"product_ndc": b["product_ndc"]},
        },
        catalog_for([a, b]),
    )
    assert result["products"][0]["ingredient_status"] == "SUPPORTED"
    assert result["comparison"]["status"] == "UNRESOLVED"
    assert result["comparison"]["ingredient_overlap"] is None


def test_full_evaluation_never_loads_the_sealed_test_partition(tmp_path, monkeypatch):
    from hrp import retrieval_baseline as module
    from hrp.data_loading import file_sha256

    directory = tmp_path / "data/processed/final_v5"
    reports = tmp_path / "results/final_v5/part03"
    directory.mkdir(parents=True)
    reports.mkdir(parents=True)
    a = product()
    for part in ("train", "validation"):
        row = case(
            {"name": "Example"},
            {"answer_status": "IDENTIFIED", "product": reference(a)},
            partition=part,
        )
        with (directory / (part + ".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    (directory / "sealed_test.csv").write_text(
        "invalid test data must never be decoded", encoding="utf-8"
    )
    seal = {
        "files": {p.name: file_sha256(p) for p in directory.iterdir()},
        "source_catalog_sha256": "catalog-hash",
    }
    (reports / "freeze_manifest.json").write_text(json.dumps(seal), encoding="utf-8")
    catalog = catalog_for([a])
    catalog["source_hashes"] = {"data/interim/final_v5/catalog_evidence.csv": "catalog-hash"}
    monkeypatch.setattr(module, "load_catalog", lambda _: catalog)
    actual_load = module.load_evaluation_cases
    paths = []

    def checked_load(path, partitions):
        paths.append(path.name)
        assert path.name in {"train.csv", "validation.csv"}
        return actual_load(path, partitions)

    monkeypatch.setattr(module, "load_evaluation_cases", checked_load)
    summary = module.run_evaluation(tmp_path)
    assert paths == ["train.csv", "validation.csv"]
    assert summary["test_predictions_generated"] == 0
    assert not summary["test_input_json_decoded"]
    assert not summary["test_reference_json_decoded"]
