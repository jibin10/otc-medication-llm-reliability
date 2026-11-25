"""Score the completed original Gemma pilot against frozen final_v5 validation labels."""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter

import stage06_original_gemma_pilot as pilot
from hrp.challenge_evaluation import SUPPORTED, IdentityResolver
from hrp.data_cleaning import formulation_key
from hrp.retrieval_baseline import _metric
from jsonschema import Draft202012Validator

VALIDATION = pilot.REPO / "data/processed/final_v5/validation.csv"
CATALOG = pilot.REPO / "data/interim/final_v5/catalog_evidence.csv"
PREPARATION = pilot.REPO / "results/part05/validation_pilot_preparation.json"
REPORT_NAME = "factual_validation_metrics.v1.json"
UNII = re.compile(r"fda_unii:([A-Z0-9]{10})")


def catalog_resolver(path):
    """Project the existing source catalog's literal name/UNII links, without labels."""
    links = set()
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            for item in json.loads(row["active_ingredients_json"]):
                match = UNII.fullmatch(item.get("identity") or "")
                if match and item.get("name"):
                    links.add((item["name"], match[1]))
    return IdentityResolver({"name": name, "uniis": [unii]} for name, unii in sorted(links))


def reference_identity(value):
    match = UNII.fullmatch(value or "")
    return "unii:" + match[1] if match else None


def form_key(form, identify):
    if not form:
        return None
    ingredients = []
    for item in form["active_ingredients"]:
        identity = identify(item)
        if not identity:
            return None
        ingredients.append({**item, "identity": identity})
    return formulation_key(ingredients, form["dosage_form"], form["route"])


def score_case(answer, prediction, resolver):
    counts = Counter()
    if "product_a" in answer:
        targets = [answer["product_a"], answer["product_b"]]
    elif "product" in answer:
        targets = [answer["product"]]
    elif answer["answer_status"] in {"INSUFFICIENT_INFORMATION", "NO_MATCH", "SOURCE_CONFLICT"}:
        targets = [None]
    else:
        raise pilot.PilotError("Unrecognized final_v5 reference structure")

    for target, product in zip(targets, prediction["products"], strict=True):
        counts["product_requests"] += 1
        actual, unknown = resolver.map_output(product["active_ingredient_names"])
        counts["unknown_ingredient_names"] += len(unknown)
        if target is None:
            counts["abstention_eligible"] += 1
            counts["abstention_correct"] += product["status"] == answer["answer_status"]
            counts["ingredient_fp"] += len(actual)
            continue

        counts["supported_requests"] += 1
        counts["inappropriate_abstentions"] += product["status"] not in SUPPORTED
        if target.get("product_ndc") is not None:
            counts["ndc_eligible"] += 1
            counts["ndc_correct"] += product["product_ndc"] == target["product_ndc"]

        expected = {reference_identity(value) for value in target.get("ingredient_identities", [])}
        if expected and None not in expected:
            counts["ingredient_eligible"] += 1
            counts["ingredient_exact"] += product["status"] in SUPPORTED and expected == actual
            counts["ingredient_tp"] += len(expected & actual)
            counts["ingredient_fp"] += len(actual - expected)
            counts["ingredient_fn"] += len(expected - actual)
        else:
            counts["ingredient_reference_missing_or_unmapped"] += 1

        if not target.get("formulation_supported"):
            counts["formulation_reference_withheld"] += 1
            continue
        expected_form = form_key(target, lambda item: reference_identity(item.get("identity")))
        if expected_form is None:
            counts["formulation_reference_unrepresentable"] += 1
            continue
        actual_form = form_key(product["formulation"], lambda item: resolver.identify(item["name"]))
        counts["formulation_eligible"] += 1
        counts["formulation_correct"] += actual_form == expected_form

    if len(targets) == 2:
        comparison = prediction["comparison"]
        expected_shared = {reference_identity(value) for value in
                           answer.get("shared_active_ingredient_identities", [])}
        overlap = answer.get("ingredient_overlap")
        same = answer.get("same_active_ingredient_set")
        if (overlap in {"shared", "disjoint"} and type(same) is bool
                and "shared_active_ingredient_identities" in answer and None not in expected_shared):
            actual_shared, unknown = resolver.map_output(comparison["shared_active_ingredient_names"])
            overlap_correct = comparison["ingredient_overlap"] == overlap
            same_correct = comparison["same_active_ingredient_set"] == same
            counts["comparison_eligible"] += 1
            counts["overlap_correct"] += overlap_correct
            counts["same_set_correct"] += same_correct
            counts["comparison_correct"] += (overlap_correct and same_correct and not unknown
                                             and actual_shared == expected_shared)
        else:
            counts["comparison_reference_missing_or_unmapped"] += 1
    return counts


def summarize(counts):
    tp, fp, fn = (counts["ingredient_" + key] for key in ("tp", "fp", "fn"))
    return {
        "counts": dict(counts),
        "full_formulation_accuracy": _metric(counts["formulation_correct"],
                                             counts["formulation_eligible"]),
        "ingredient_set_accuracy": _metric(counts["ingredient_exact"], counts["ingredient_eligible"]),
        "ingredient_precision": _metric(tp, tp + fp),
        "ingredient_recall": _metric(tp, tp + fn),
        "ingredient_f1": _metric(2 * tp, 2 * tp + fp + fn),
        "overlap_label_accuracy": _metric(counts["overlap_correct"], counts["comparison_eligible"]),
        "same_set_accuracy": _metric(counts["same_set_correct"], counts["comparison_eligible"]),
        "full_comparison_accuracy": _metric(counts["comparison_correct"], counts["comparison_eligible"]),
        "abstention_accuracy": _metric(counts["abstention_correct"], counts["abstention_eligible"]),
        "inappropriate_abstention_rate": _metric(counts["inappropriate_abstentions"],
                                                 counts["supported_requests"]),
        "unique_ndc_accuracy": _metric(counts["ndc_correct"], counts["ndc_eligible"]),
    }


def run():
    requests, schema, metadata = pilot.check_input(pilot.INPUT)
    configuration = pilot.run_configuration(requests, metadata, *pilot.response_policy(None, None))
    _, saved = pilot.read_run_state(pilot.FENCE_OUTPUT, requests, configuration)
    parsed_path = pilot.FENCE_OUTPUT / "validated_outputs.PRIVATE.jsonl"
    predictions = {}
    validator = Draft202012Validator(schema)
    for line in parsed_path.read_bytes().splitlines():
        row = pilot.strict_json(line)
        case_id = row["case_id"]
        if case_id not in saved or case_id in predictions:
            raise pilot.PilotError("Unexpected saved pilot prediction membership")
        value, error = pilot.validate_attempt(dict(saved[case_id]), validator,
                                              requests[case_id]["evidence"], pilot.FENCE_POLICY)
        if error or saved[case_id]["error"] or value != row["output"]:
            raise pilot.PilotError("Saved normalized prediction differs from the response ledger")
        predictions[case_id] = value
    if len(predictions) != 12 or predictions.keys() != requests.keys() or saved.keys() != requests.keys():
        raise pilot.PilotError("Expected all 12 completed pilot predictions")

    preparation = pilot.strict_json(PREPARATION.read_bytes())
    validation_hash = pilot.file_sha256(VALIDATION)
    catalog_hash = pilot.file_sha256(CATALOG)
    if (validation_hash != preparation["validation_csv_sha256"]
            or catalog_hash != preparation["source_hashes"][CATALOG.relative_to(pilot.REPO).as_posix()]):
        raise pilot.PilotError("Pilot validation labels or source catalog differ from preparation")
    resolver = catalog_resolver(CATALOG)
    counts, by_task, seen, groups = Counter(), {}, set(), set()
    with VALIDATION.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            case_id = row["case_id"]
            if case_id not in predictions:
                continue
            if case_id in seen or row["partition"] != "validation":
                raise pilot.PilotError("Pilot reference membership is not a unique validation selection")
            seen.add(case_id)
            groups.update(pilot.strict_json(row["family_group_ids_json"]))
            reference = pilot.strict_json(row["reference_json"]) if row["reference_json"] else None
            if reference is None:
                continue
            local = score_case(reference, predictions[case_id], resolver)
            local["cases_scored"] = 1
            counts.update(local)
            by_task.setdefault(row["task"], Counter()).update(local)

    result = {
        "evaluation_version": "final_v5_gemma_pilot_v1",
        "reference_provenance": "final_v5 validation labels; source-consistency only",
        "identity_mapping": "Literal source-catalog FDA name/UNII links; conflicts remain unresolved",
        "scoring_policy": "final_v5 eligibility and labels with frozen Stage 5C set, quantity, and comparison rules where applicable",
        "cases_requested": len(requests),
        "cases_missing_reference": len(requests) - counts["cases_scored"],
        "validation_family_groups": len(groups),
        "configuration": configuration,
        "inputs_sha256": {"validation_csv": validation_hash, "source_catalog": catalog_hash,
                          "normalized_predictions": pilot.file_sha256(parsed_path)},
        "overall": summarize(counts),
        "by_task": {task: summarize(value) for task, value in sorted(by_task.items())},
    }
    pilot.atomic_write(pilot.FENCE_OUTPUT / REPORT_NAME, json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    try:
        result = run()
    except Exception as exc:
        print(f"Pilot scoring failed. {type(exc).__name__}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result["overall"], indent=2))
