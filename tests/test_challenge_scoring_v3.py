"""Synthetic regressions for the scoring-only amendment; no dataset reads."""

import copy
from pathlib import Path

import pytest
from test_challenge_v1 import candidate, comparison_payload, payload, record

from hrp import challenge_evaluation as original
from hrp import challenge_scoring_v3 as corrected


@pytest.fixture
def resolver():
    return original.IdentityResolver(
        [
            {"name": "alpha", "uniis": ["0000000001"]},
            {"name": "alpha synonym", "uniis": ["0000000001"]},
            {"name": "beta", "uniis": ["0000000002"]},
            {"name": "alpha hydrochloride", "uniis": ["0000000003"]},
        ]
    )


def strip_joint_and_versions(result):
    """Everything except the amended joint score must compare equal to contract 2."""
    result = copy.deepcopy(result)
    result.pop("scoring_contract_version", None)
    result.pop("benchmark_contract_version", None)
    summaries = [result["overall"], *result["by_role"].values(), *result["by_category"].values()]
    for summary in summaries:
        summary.pop("joint_case_accuracy")
        summary["counts"].pop("joint_correct_cases", None)
    return result


@pytest.mark.parametrize("defect", ["strength", "dosage_form", "route", "missing", "salt_identity"])
def test_wrong_or_missing_formulation_exposes_original_bug_and_fails_joint(resolver, defect):
    evidence = payload(candidate())
    reference = original.predict_common_evidence(evidence, resolver)
    assert reference["products"][0]["formulation"] is not None
    prediction = copy.deepcopy(reference)
    product = prediction["products"][0]
    if defect == "strength":
        product["formulation"]["active_ingredients"][0]["strength"] = "2 mg/1"
    elif defect == "dosage_form":
        product["formulation"]["dosage_form"] = "CAPSULE"
    elif defect == "route":
        product["formulation"]["route"] = ["TOPICAL"]
    elif defect == "salt_identity":
        product["formulation"]["active_ingredients"][0]["name"] = "alpha hydrochloride"
    else:
        product["formulation"] = None
    cases, predictions, payloads = [record("v", reference)], {"v": prediction}, {"v": evidence}
    old = original.score_predictions(cases, predictions, payloads, resolver)
    new = corrected.score_predictions(cases, predictions, payloads, resolver)
    # These assertions demonstrate the actual old defect without modifying/xfailing the old scorer.
    assert old["overall"]["joint_case_accuracy"]["value"] == 1.0
    assert old["overall"]["full_formulation_accuracy"]["value"] == 0.0
    assert new["overall"]["joint_case_accuracy"] == {"numerator": 0, "denominator": 1, "value": 0.0}
    assert new["overall"]["ingredient_set_accuracy"]["value"] == 1.0
    assert new["overall"]["unique_ndc_accuracy"]["value"] == 1.0
    assert new["overall"]["counts"]["valid_outputs"] == 1
    assert strip_joint_and_versions(new) == strip_joint_and_versions(old)


@pytest.mark.parametrize("equivalent_normalization", [False, True])
def test_fully_correct_and_normalization_equivalent_predictions_retain_scores(
    resolver, equivalent_normalization
):
    evidence = payload(candidate())
    reference = original.predict_common_evidence(evidence, resolver)
    prediction = copy.deepcopy(reference)
    if equivalent_normalization:
        prediction["products"][0]["formulation"] = {
            "active_ingredients": [{"name": "alpha synonym", "strength": "1000 mcg/1"}],
            "dosage_form": "tablet",
            "route": ["oral", "ORAL"],
        }
    args = ([record("v", reference)], {"v": prediction}, {"v": evidence}, resolver)
    old, new = original.score_predictions(*args), corrected.score_predictions(*args)
    assert new.pop("scoring_contract_version") == corrected.SCORING_CONTRACT_VERSION
    assert new.pop("benchmark_contract_version") == original.VERSION
    assert new == old
    assert new["overall"]["joint_case_accuracy"]["value"] == 1.0


@pytest.mark.parametrize("prediction_formulation", ["missing", "wrong", "unchanged"])
def test_unavailable_reference_formulation_does_not_penalize_joint(
    resolver, prediction_formulation
):
    evidence = payload(candidate())
    reference = original.predict_common_evidence(evidence, resolver)
    prediction = copy.deepcopy(reference)
    reference["products"][0]["formulation"] = None
    if prediction_formulation == "missing":
        prediction["products"][0]["formulation"] = None
    elif prediction_formulation == "wrong":
        prediction["products"][0]["formulation"]["dosage_form"] = "CAPSULE"
    args = ([record("v", reference)], {"v": prediction}, {"v": evidence}, resolver)
    old, new = original.score_predictions(*args), corrected.score_predictions(*args)
    assert new["overall"]["joint_case_accuracy"]["value"] == 1.0
    assert new["overall"]["full_formulation_accuracy"] == {
        "numerator": 0,
        "denominator": 0,
        "value": None,
    }
    assert strip_joint_and_versions(new) == strip_joint_and_versions(old)


@pytest.mark.parametrize("mode", ["missing", "invalid_json", "invalid_nested_output"])
def test_missing_and_invalid_predictions_keep_original_failures_and_denominators(resolver, mode):
    evidence = comparison_payload(candidate(), candidate("2-2", name="beta"))
    reference = original.predict_common_evidence(evidence, resolver)
    if mode == "missing":
        predictions = {}
    elif mode == "invalid_json":
        predictions = {"v": "{invalid"}
    else:
        prediction = copy.deepcopy(reference)
        prediction["products"][0]["formulation"]["route"] = []
        predictions = {"v": prediction}
    args = ([record("v", reference)], predictions, {"v": evidence}, resolver)
    old, new = original.score_predictions(*args), corrected.score_predictions(*args)
    assert new["overall"]["joint_case_accuracy"] == {"numerator": 0, "denominator": 1, "value": 0.0}
    assert new["overall"]["counts"]["invalid_outputs"] == 1
    assert new["overall"]["ingredient_set_accuracy"]["denominator"] == 2
    assert new["overall"]["full_formulation_accuracy"]["denominator"] == 2
    assert new["overall"]["comparison_accuracy"]["denominator"] == 1
    assert new["format_errors"] == old["format_errors"]
    assert strip_joint_and_versions(new) == strip_joint_and_versions(old)


@pytest.mark.parametrize("wrong_products", [(0,), (1,), (0, 1)])
def test_comparison_loses_only_one_case_credit_when_either_or_both_formulations_fail(
    resolver, wrong_products
):
    evidence = comparison_payload(candidate(), candidate("2-2", name="beta"))
    reference = original.predict_common_evidence(evidence, resolver)
    prediction = copy.deepcopy(reference)
    for index in wrong_products:
        prediction["products"][index]["formulation"]["route"] = ["TOPICAL"]
    args = ([record("v", reference)], {"v": prediction}, {"v": evidence}, resolver)
    old, new = original.score_predictions(*args), corrected.score_predictions(*args)
    assert new["overall"]["counts"]["joint_correct_cases"] == 0
    assert new["overall"]["comparison_accuracy"]["value"] == 1.0
    assert new["overall"]["full_formulation_accuracy"]["numerator"] == 2 - len(wrong_products)
    assert strip_joint_and_versions(new) == strip_joint_and_versions(old)


def test_overall_role_and_category_joint_aggregates_are_correct(resolver):
    evidence = payload(candidate())
    reference = original.predict_common_evidence(evidence, resolver)
    cases = [
        {**record("v1", reference, "role_a"), "category": "category_a"},
        {**record("v2", reference, "role_a"), "category": "category_b"},
        {**record("v3", reference, "role_b"), "category": "category_a"},
    ]
    wrong = copy.deepcopy(reference)
    wrong["products"][0]["formulation"]["dosage_form"] = "CAPSULE"
    predictions = {"v1": wrong, "v2": reference, "v3": wrong}
    payloads = {r["case_id"]: evidence for r in cases}
    old = original.score_predictions(cases, predictions, payloads, resolver)
    new = corrected.score_predictions(cases, predictions, payloads, resolver)
    assert new["overall"]["joint_case_accuracy"] == {
        "numerator": 1,
        "denominator": 3,
        "value": 1 / 3,
    }
    assert new["by_role"]["role_a"]["joint_case_accuracy"]["value"] == 0.5
    assert new["by_role"]["role_b"]["joint_case_accuracy"]["value"] == 0.0
    assert new["by_category"]["category_a"]["joint_case_accuracy"]["value"] == 0.0
    assert new["by_category"]["category_b"]["joint_case_accuracy"]["value"] == 1.0
    assert strip_joint_and_versions(new) == strip_joint_and_versions(old)


@pytest.mark.parametrize("defect", ["ingredient", "status", "comparison"])
def test_other_scoring_components_are_not_overridden(resolver, defect):
    evidence = comparison_payload(candidate(), candidate("2-2", name="beta"))
    reference = original.predict_common_evidence(evidence, resolver)
    prediction = copy.deepcopy(reference)
    if defect == "ingredient":
        prediction["products"][0]["active_ingredient_names"] = ["beta"]
    elif defect == "status":
        prediction["products"][0]["status"] = "FORMULATION_IDENTIFIED"
        prediction["products"][0]["product_ndc"] = None
    else:
        prediction["comparison"]["ingredient_overlap"] = "shared"
    args = ([record("v", reference)], {"v": prediction}, {"v": evidence}, resolver)
    old, new = original.score_predictions(*args), corrected.score_predictions(*args)
    assert old["overall"]["joint_case_accuracy"]["value"] == 0.0
    assert new["overall"]["joint_case_accuracy"]["value"] == 0.0
    assert strip_joint_and_versions(new) == strip_joint_and_versions(old)


def test_abstention_and_empty_input_behavior_are_unchanged(resolver):
    evidence = payload(candidate(), candidate("2-2", name="beta"))
    reference = original.predict_common_evidence(evidence, resolver)
    args = ([record("v", reference, "ambiguity")], {"v": reference}, {"v": evidence}, resolver)
    old, new = original.score_predictions(*args), corrected.score_predictions(*args)
    assert new["overall"]["abstention_accuracy"]["value"] == 1.0
    assert new["overall"]["joint_case_accuracy"]["value"] == 1.0
    assert strip_joint_and_versions(new) == strip_joint_and_versions(old)
    empty = corrected.score_predictions([], {}, {}, resolver)
    assert empty["overall"]["joint_case_accuracy"]["value"] is None


def test_amendment_performs_no_file_access_and_preserves_inputs_and_frozen_function(
    resolver, monkeypatch
):
    evidence = payload(candidate())
    reference = original.predict_common_evidence(evidence, resolver)
    prediction = copy.deepcopy(reference)
    prediction["products"][0]["formulation"] = None
    cases, predictions, payloads = [record("v", reference)], {"v": prediction}, {"v": evidence}
    before = copy.deepcopy((cases, predictions, payloads))
    frozen_function = original.score_predictions
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("Scorer attempted file access"))
    result = corrected.score_predictions(cases, predictions, payloads, resolver)
    assert (cases, predictions, payloads) == before
    assert original.score_predictions is frozen_function
    original.reject_reserved_fields(result)
    with pytest.raises(ValueError, match="validation-only"):
        corrected.score_predictions(
            [{**cases[0], "partition": "test"}], predictions, payloads, resolver
        )
    with pytest.raises(ValueError, match="outside the frozen"):
        corrected.score_predictions(
            cases, {**predictions, "unknown": prediction}, payloads, resolver
        )
