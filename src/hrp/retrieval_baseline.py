"""Deterministic source retrieval and separate train and validation evaluation."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from hrp import FINAL_BENCHMARK_VERSION
from hrp.data_cleaning import (
    formulation_key,
    ingredient_identity,
    normalize_name,
    quantity_key,
    route_signature,
)
from hrp.data_loading import file_sha256
from hrp.ingredient_agreement import (
    compare_ingredient_names,
    fda_ingredient_names,
    mixed_literal_agreement,
    rxnorm_names,
)

RESOLVED = {"EXACT_NDC_IDENTIFIED", "FORMULATION_IDENTIFIED"}
PARTITIONS = {"train", "validation"}
SOURCE_FILES = (
    "catalog_evidence.csv",
    "otc_ndc_label_links.csv",
    "otc_rxnorm_product_links.csv",
    "otc_rxnorm_ingredient_links.csv",
)
PRODUCT_FIELDS = {"name", "dosage_form", "route", "native_strengths", "product_ndc"}


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _strength_key(value: str) -> tuple:
    return quantity_key(value) or (normalize_name(value).replace(" ", ""), "literal")


def _formulation(product: dict) -> tuple | None:
    return formulation_key(product["active_ingredients"], product["dosage_form"], product["route"])


def _ingredient_support(product: dict) -> tuple[str, list[dict]]:
    mapping, drug = product["rxnorm_ingredients"], product["rxnorm_product"]
    if product.get("ingredient_eligible") == "False":
        return (
            "SOURCE_CONFLICT"
            if product.get("support_tier") in {"SOURCE_CONFLICT", "NAME_NORMALIZATION_UNRESOLVED"}
            else "UNAVAILABLE"
        ), []
    mapped = (
        drug.get("rxnorm_status") == "matched"
        and mapping.get("ingredient_status") == "resolved_path"
    )
    if mapped and (
        not drug.get("rxnorm_rxcui") or mapping.get("drug_rxcui") != drug["rxnorm_rxcui"]
    ):
        return "SOURCE_CONFLICT", []
    names = fda_ingredient_names(json.dumps(product["active_ingredients"]))
    if not names:
        return "UNAVAILABLE", []
    if mapped:
        base = rxnorm_names(json.dumps(mapping["in_names"]))
        precise = rxnorm_names(json.dumps(mapping["pin_names"]))
        agreement = compare_ingredient_names(names, base, precise)
        identifier_supported = product.get(
            "rxnorm_identifier_status"
        ) == "FDA_UNII_PATH_SUPPORTED" and mixed_literal_agreement(
            frozenset(
                value.removeprefix("fda_unii:") for value in product["ingredient_identities"]
            ),
            mapping["unii_options"],
        )
        if (
            not agreement.startswith("exact_")
            and not mixed_literal_agreement(names, mapping["identity_options"])
            and not identifier_supported
        ):
            return "SOURCE_CONFLICT", []
    identifiers = defaultdict(set)
    if mapped:
        for kind in ("in", "pin"):
            source_names, ids = mapping[kind + "_names"], mapping[kind + "_rxcuis"]
            if len(source_names) != len(ids):
                return "SOURCE_CONFLICT", []
            for name, identifier in zip(source_names, ids, strict=True):
                if identifier:
                    identifiers[normalize_name(name)].add(identifier)
    grouped = {}
    for item in product["active_ingredients"]:
        key = ingredient_identity(item)
        saved = grouped.setdefault(
            key, {"identity": key, "source_names": set(), "rxnorm_ids": set()}
        )
        saved["source_names"].add(item["name"])
        saved["rxnorm_ids"].update(identifiers[normalize_name(item["name"])])
    ingredients = [
        {
            "identity": key,
            "source_names": sorted(grouped.get(key, {}).get("source_names", [])),
            "rxnorm_ids": sorted(grouped.get(key, {}).get("rxnorm_ids", [])),
        }
        for key in product["ingredient_identities"]
    ]
    return "SUPPORTED", ingredients


def build_catalog(
    products: list[dict], labels: list[dict], drugs: list[dict], mappings: list[dict]
) -> dict:
    """Index the full catalog before checking optional source mappings."""
    links = []
    for rows in (labels, drugs, mappings):
        indexed = {row["product_ndc"]: row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError("Duplicate NDC in source links")
        links.append(indexed)
    catalog = {"products": {}, "by_name": defaultdict(set)}
    for row in products:
        ndc = row["product_ndc"]
        if not ndc or ndc in catalog["products"]:
            raise ValueError("Missing or duplicate catalog NDC")
        label, drug, mapping = (link.get(ndc, {}) for link in links)
        product = {
            "product_ndc": ndc,
            "name": row["display_name"],
            "brand_name": row.get("brand_name", ""),
            "generic_name": row.get("generic_name", ""),
            "dosage_form": row["dosage_form"],
            "route": json.loads(row["route"] or "[]"),
            "active_ingredients": json.loads(row["active_ingredients_json"] or "[]"),
            "ingredient_identities": json.loads(row.get("ingredient_identities_json") or "[]")
            or sorted(
                {
                    ingredient_identity(item)
                    for item in json.loads(row["active_ingredients_json"] or "[]") or []
                }
            ),
            "catalog_source_rows": row.get("source_rows", ""),
            "ingredient_eligible": row.get("ingredient_eligible"),
            "comparison_eligible": row.get("comparison_eligible", "True"),
            "identity_status": row.get("identity_status", "SYNTHETIC_LITERAL_IDENTITY"),
            "rxnorm_identifier_status": row.get("rxnorm_identifier_status", ""),
            "identity_provenance": json.loads(row.get("identity_provenance_json") or "{}"),
            "formulation_eligible": row.get("formulation_eligible", "True"),
            "support_tier": row.get("support_tier", "FDA_ONLY"),
            "exclusion_reasons": json.loads(row.get("exclusion_reasons_json") or "[]"),
            "strength_issues": json.loads(row.get("strength_issues_json") or "[]"),
            "original_ingredients": json.loads(
                row.get("original_ingredients_json") or row["active_ingredients_json"] or "[]"
            )
            or [],
            "label_extracted_ingredients": json.loads(
                row.get("label_extracted_ingredients_json") or "[]"
            ),
            "fda_label": {
                key: label.get(key, "")
                for key in ("link_status", "label_id", "label_set_id", "effective_time")
            },
            "rxnorm_product": {
                key: drug.get(key, "")
                for key in ("rxnorm_status", "rxnorm_rxcui", "rxnorm_tty", "rxnorm_drug_name")
            },
            "rxnorm_ingredients": {
                **{key: mapping.get(key, "") for key in ("ingredient_status", "drug_rxcui")},
                **{
                    key: json.loads(mapping.get(key + "_json") or "[]")
                    for key in (
                        "in_names",
                        "in_rxcuis",
                        "pin_names",
                        "pin_rxcuis",
                        "identity_options",
                        "unii_options",
                    )
                },
            },
        }
        product["fda_label"]["active_ingredient_text"] = json.loads(
            label.get("active_ingredient_json") or "[]"
        )
        catalog["products"][ndc] = product
        for name in (product["name"], product["brand_name"], product["generic_name"]):
            if key := normalize_name(name):
                catalog["by_name"][key].add(ndc)
    return catalog


def load_catalog(project_root: Path) -> dict:
    interim = project_root / "data/interim" / FINAL_BENCHMARK_VERSION
    catalog = build_catalog(*(_read_csv(interim / name) for name in SOURCE_FILES))
    catalog["source_hashes"] = {
        f"data/interim/{FINAL_BENCHMARK_VERSION}/" + name: file_sha256(interim / name)
        for name in SOURCE_FILES
    }
    return catalog


def _descriptions(input_json: str | dict) -> list[dict]:
    request = json.loads(input_json) if isinstance(input_json, str) else input_json
    if not isinstance(request, dict):
        raise TypeError("Prediction requires an input_json object")
    if "product_a" in request or "product_b" in request:
        if set(request) != {"product_a", "product_b"}:
            raise ValueError("Comparison requires exactly two product descriptions")
        descriptions = [request["product_a"], request["product_b"]]
    else:
        descriptions = [request]
    for description in descriptions:
        if not isinstance(description, dict) or set(description) - PRODUCT_FIELDS:
            raise ValueError("Only product description fields are permitted in input_json")
        for key, value in description.items():
            if key in {"route", "native_strengths"}:
                if (
                    not isinstance(value, list)
                    or not value
                    or not all(isinstance(item, str) and item.strip() for item in value)
                ):
                    raise ValueError(f"Expected a nonempty list of strings for {key}")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"Expected nonempty text for {key}")
    return descriptions


def _detail_match(description: dict, product: dict) -> bool:
    if "conflicting_ndc_rows" in product.get("exclusion_reasons", []):
        return True
    for key in ("dosage_form", "route", "native_strengths"):
        if key not in description:
            continue
        if key in {"dosage_form", "route"} and "rxnorm_route_or_form_conflict" in product.get(
            "strength_issues", []
        ):
            continue
        if key == "native_strengths" and product.get("formulation_eligible") == "False":
            continue
        if key == "dosage_form":
            wanted, known = normalize_name(description[key]), normalize_name(product[key])
        elif key == "route":
            wanted, known = route_signature(description[key]), route_signature(product[key])
        else:
            wanted = sorted(_strength_key(value) for value in description[key])
            items = product["active_ingredients"]
            known = (
                sorted(_strength_key(item["strength"]) for item in items)
                if items and all(item.get("strength") for item in items)
                else None
            )
        # Unknown metadata cannot eliminate another plausible product
        if known and wanted != known:
            return False
    return True


def retrieve_evidence(input_json: str | dict, catalog: dict) -> dict:
    """Return source candidates for reuse by every compared system."""
    evidence = {"catalog_ndcs": len(catalog["products"]), "products": []}
    for description in _descriptions(input_json):
        name = normalize_name(description.get("name"))
        ndc = description.get("product_ndc", "").strip()
        if ndc:
            ids = {ndc} if ndc in catalog["products"] else set()
        else:
            ids = catalog["by_name"].get(name, set())
        products = [catalog["products"][key] for key in sorted(ids)]
        matches = [product for product in products if _detail_match(description, product)]
        reason = "matched_source_candidates"
        if not name and not ndc:
            reason = "name_or_ndc_required"
        elif not products:
            reason = "identifier_not_in_catalog"
        elif not matches:
            reason = "description_conflicts_with_catalog"
        if (
            ndc
            and products
            and name
            and name
            not in {
                normalize_name(products[0][key]) for key in ("name", "brand_name", "generic_name")
            }
        ):
            matches, reason = [], "description_conflicts_with_catalog"
        evidence["products"].append(
            {
                "description": description,
                "identifier_candidate_count": len(products),
                "identifier_candidates": products,
                "candidates": matches,
                "reason": reason,
                "explicit_ndc_conflict": bool(ndc and products and not matches),
            }
        )
    return evidence


def resolve_product(evidence: dict) -> dict:
    """Resolve one description using only its retrieved source candidates."""
    candidates = evidence["candidates"]
    result = {
        "status": "INSUFFICIENT_INFORMATION",
        "product_ndc": None,
        "candidate_ndcs": [product["product_ndc"] for product in candidates],
        "formulation": None,
        "ingredient_status": "UNAVAILABLE",
        "ingredients": [],
        "reported_ingredient_names": [],
        "comparison_eligible": False,
        "source_identifiers": [],
        "reason": evidence["reason"],
    }
    if not candidates:
        if evidence["explicit_ndc_conflict"]:
            result["status"] = "SOURCE_CONFLICT"
        elif evidence["reason"] != "name_or_ndc_required":
            result["status"] = "NO_MATCH"
        return result
    formulations = {
        _formulation(product) if product["formulation_eligible"] == "True" else None
        for product in candidates
    }
    if len(candidates) > 1 and (None in formulations or len(formulations) != 1):
        result["reason"] = "different_or_incomplete_formulations_remain"
        return result
    support = [_ingredient_support(product) for product in candidates]
    if any(status == "SOURCE_CONFLICT" for status, _ in support):
        result.update(status="SOURCE_CONFLICT", reason="fda_rxnorm_identity_conflict")
        return result
    result["status"] = "EXACT_NDC_IDENTIFIED" if len(candidates) == 1 else "FORMULATION_IDENTIFIED"
    result["comparison_eligible"] = all(
        p.get("comparison_eligible", "True") == "True" for p in candidates
    )
    if len(candidates) == 1:
        result["product_ndc"] = candidates[0]["product_ndc"]
    result["formulation"] = (
        {key: candidates[0][key] for key in ("active_ingredients", "dosage_form", "route")}
        if None not in formulations
        else None
    )
    result["source_identifiers"] = [
        {
            "product_ndc": product["product_ndc"],
            "label_id": product["fda_label"]["label_id"],
            "label_set_id": product["fda_label"]["label_set_id"],
            "rxnorm_product_rxcui": product["rxnorm_product"]["rxnorm_rxcui"],
        }
        for product in candidates
    ]
    if all(status == "SUPPORTED" for status, _ in support):
        ingredients = {}
        for _, items in support:
            for item in items:
                saved = ingredients.setdefault(
                    item["identity"],
                    {"identity": item["identity"], "source_names": set(), "rxnorm_ids": set()},
                )
                saved["source_names"].update(item["source_names"])
                saved["rxnorm_ids"].update(item["rxnorm_ids"])
        result["ingredients"] = [
            {
                "identity": name,
                "source_names": sorted(item["source_names"]),
                "rxnorm_ids": sorted(item["rxnorm_ids"]),
            }
            for name, item in sorted(ingredients.items())
        ]
        result["ingredient_status"] = "SUPPORTED"
        result["reported_ingredient_names"] = sorted(
            {item["name"] for product in candidates for item in product["active_ingredients"]}
        )
    result["reason"] = (
        "unique_catalog_ndc" if len(candidates) == 1 else "same_formulation_multiple_ndcs"
    )
    return result


def predict_from_evidence(evidence: dict) -> dict:
    products = [resolve_product(product) for product in evidence["products"]]
    comparison = None
    if len(products) == 2:
        comparison = {
            "status": "UNRESOLVED",
            "ingredient_overlap": None,
            "shared_active_ingredient_identities": [],
            "same_active_ingredient_set": None,
        }
        if all(
            product["ingredient_status"] == "SUPPORTED" and product["comparison_eligible"]
            for product in products
        ):
            a, b = ({item["identity"] for item in product["ingredients"]} for product in products)
            comparison = {
                "status": "SUPPORTED",
                "ingredient_overlap": "shared" if a & b else "disjoint",
                "shared_active_ingredient_identities": sorted(a & b),
                "same_active_ingredient_set": a == b,
            }
    return {"products": products, "comparison": comparison}


def predict(input_json: str | dict, catalog: dict) -> dict:
    """Accept input_json alone, without benchmark metadata or reference answers."""
    return predict_from_evidence(retrieve_evidence(input_json, catalog))


def load_evaluation_cases(path: Path, partitions: tuple = ("train", "validation")) -> list[dict]:
    if not partitions or set(partitions) - PARTITIONS:
        raise ValueError("Only train and validation cases may be evaluated")
    cases = []
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            # Skip frozen rows before decoding any input or reference JSON
            if row["partition"] not in partitions:
                continue
            cases.append(
                {
                    key: row[key]
                    for key in (
                        "case_id",
                        "partition",
                        "task",
                        "input_json",
                        "reference_json",
                        "family_group_ids_json",
                    )
                }
            )
    return cases


def _metric(numerator: int, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def _set_metrics(pairs: list[tuple[set, set]]) -> dict:
    tp = sum(len(predicted & expected) for predicted, expected in pairs)
    fp = sum(len(predicted - expected) for predicted, expected in pairs)
    fn = sum(len(expected - predicted) for predicted, expected in pairs)
    return {
        "true_positive_ingredients": tp,
        "false_positive_ingredients": fp,
        "false_negative_ingredients": fn,
        "precision": _metric(tp, tp + fp),
        "recall": _metric(tp, tp + fn),
        "f1": _metric(2 * tp, 2 * tp + fp + fn),
    }


def _product_metrics(records: list[dict]) -> dict:
    eligible = [row for row in records if row["reference"] is not None]
    ambiguous = [row for row in records if row["reference"] is None]
    resolved = [row for row in eligible if row["prediction"]["status"] in RESOLVED]
    exact = [row for row in eligible if row["exact_justified"]]
    formulation_eligible = [
        row for row in eligible if row["reference"].get("formulation_supported", True)
    ]
    formulation_resolved = [
        row for row in formulation_eligible if row["prediction"]["formulation"] is not None
    ]
    ingredient_resolved = [
        row for row in eligible if row["prediction"]["ingredient_status"] == "SUPPORTED"
    ]
    chemical_targets = [
        row for row in eligible if row["reference"].get("comparison_eligible", True)
    ]
    name_only_targets = [
        row for row in eligible if not row["reference"].get("comparison_eligible", True)
    ]
    statuses = Counter(row["prediction"]["status"] for row in records)
    return {
        "product_requests": len(records),
        "status_counts": {
            status: _metric(statuses[status], len(records))
            for status in sorted(
                RESOLVED | {"INSUFFICIENT_INFORMATION", "NO_MATCH", "SOURCE_CONFLICT"}
            )
        },
        "exact_ndc_accuracy_when_justified": _metric(
            sum(
                row["prediction"]["product_ndc"] == row["reference"]["product_ndc"] for row in exact
            ),
            len(exact),
        ),
        "reference_ndc_not_established_by_input": _metric(
            len(eligible) - len(exact), len(eligible)
        ),
        "unjustified_exact_ndc_predictions": _metric(
            sum(
                row["prediction"]["product_ndc"] is not None and not row["exact_justified"]
                for row in records
            ),
            len(records) - len(exact),
        ),
        "formulation_accuracy_all_eligible": _metric(
            sum(row["formulation_correct"] for row in formulation_eligible),
            len(formulation_eligible),
        ),
        "formulation_accuracy_among_resolved": _metric(
            sum(row["formulation_correct"] for row in formulation_resolved),
            len(formulation_resolved),
        ),
        "formulation_reference_coverage": _metric(len(formulation_eligible), len(eligible)),
        "formulation_resolution_coverage": _metric(
            len(formulation_resolved), len(formulation_eligible)
        ),
        "source_limited_formulation_requests": _metric(
            len(eligible) - len(formulation_eligible), len(eligible)
        ),
        "correct_abstention": _metric(
            sum(row["prediction"]["status"] == "INSUFFICIENT_INFORMATION" for row in ambiguous),
            len(ambiguous),
        ),
        "incorrect_abstention_against_supported_reference": _metric(
            len(eligible) - len(resolved), len(eligible)
        ),
        "unresolved_product_matches": _metric(
            sum(
                row["prediction"]["status"] == "INSUFFICIENT_INFORMATION"
                and bool(row["evidence"]["candidates"])
                for row in records
            ),
            len(records),
        ),
        "formulation_only_exact_ndc_unresolved": _metric(
            statuses["FORMULATION_IDENTIFIED"], len(records)
        ),
        "no_match": _metric(statuses["NO_MATCH"], len(records)),
        "resolution_coverage": _metric(sum(statuses[s] for s in RESOLVED), len(records)),
        "source_formulation_differs_from_reference": _metric(
            sum(row["source_reference_difference"] for row in formulation_eligible),
            len(formulation_eligible),
        ),
        "ingredient_identification": {
            "eligible_product_requests": len(eligible),
            "chemical_identifier_reference_coverage": _metric(len(chemical_targets), len(eligible)),
            "chemical_identifier_set_accuracy_all_covered": _metric(
                sum(row["ingredient_correct"] for row in chemical_targets), len(chemical_targets)
            ),
            "source_name_only_reference_requests": _metric(len(name_only_targets), len(eligible)),
            "source_name_only_set_accuracy": _metric(
                sum(row["ingredient_correct"] for row in name_only_targets), len(name_only_targets)
            ),
            "set_exact_match_all_eligible": _metric(
                sum(row["ingredient_correct"] for row in eligible), len(eligible)
            ),
            "set_exact_match_among_supported_predictions": _metric(
                sum(row["ingredient_correct"] for row in ingredient_resolved),
                len(ingredient_resolved),
            ),
            "set_exact_match_among_resolved_products": _metric(
                sum(row["ingredient_correct"] for row in resolved), len(resolved)
            ),
            "ingredient_abstention_against_supported_reference": _metric(
                len(eligible) - len(ingredient_resolved), len(eligible)
            ),
            "coverage": _metric(len(ingredient_resolved), len(eligible)),
            **_set_metrics(
                [(row["predicted_ingredients"], row["expected_ingredients"]) for row in eligible]
            ),
            "unsupported_ingredients_against_reference": _metric(
                sum(
                    len(row["predicted_ingredients"] - row["expected_ingredients"])
                    for row in eligible
                ),
                sum(len(row["predicted_ingredients"]) for row in eligible),
            ),
            "unsupported_ingredients_against_retrieved_evidence": _metric(
                sum(row["unsupported_by_evidence"] for row in records),
                sum(len(row["predicted_ingredients"]) for row in records),
            ),
        },
    }


def _comparison_metrics(records: list[dict]) -> dict:
    matrix = {
        expected: dict.fromkeys(("shared", "disjoint", "unresolved"), 0)
        for expected in ("shared", "disjoint")
    }
    for row in records:
        matrix[row["expected_class"]][row["predicted_class"]] += 1
    shared = sum(matrix["shared"].values())
    disjoint = sum(matrix["disjoint"].values())
    unresolved = sum(row["predicted_class"] == "unresolved" for row in records)
    true_shared = matrix["shared"]["shared"]
    false_shared = matrix["disjoint"]["shared"]
    missed_shared = shared - true_shared
    set_comparisons = [row for row in records if row.get("expected_same") is not None]
    return {
        "cases": len(records),
        "confusion_matrix": matrix,
        "classification_accuracy_all_cases": _metric(
            matrix["shared"]["shared"] + matrix["disjoint"]["disjoint"], len(records)
        ),
        "overlap_precision": _metric(true_shared, true_shared + false_shared),
        "overlap_recall": _metric(true_shared, shared),
        "overlap_f1": _metric(2 * true_shared, 2 * true_shared + false_shared + missed_shared),
        "same_ingredient_set_accuracy": _metric(
            sum(row.get("predicted_same") == row["expected_same"] for row in set_comparisons),
            len(set_comparisons),
        ),
        "false_overlap_predictions": _metric(matrix["disjoint"]["shared"], disjoint),
        "missed_overlaps_including_unresolved": _metric(
            shared - matrix["shared"]["shared"], shared
        ),
        "explicit_false_disjoint_predictions": _metric(matrix["shared"]["disjoint"], shared),
        "unresolved_comparisons": _metric(unresolved, len(records)),
        "shared_ingredient_metrics": _set_metrics(
            [(row["predicted_shared"], row["expected_shared"]) for row in records]
        ),
    }


def evaluate_cases(cases: list[dict], predictions: dict, evidence: dict) -> dict:
    """Read references only here, after all predictions have been produced."""
    if any(row["partition"] not in PARTITIONS for row in cases):
        raise ValueError("Test evaluation is forbidden in Part 4")
    if len({row["partition"] for row in cases}) > 1:
        raise ValueError("Evaluate training and validation separately")
    products, comparisons, errors, case_results = [], [], [], []
    for case in cases:
        case_id = case["case_id"]
        answer = json.loads(case["reference_json"])
        prediction, source = predictions[case_id], evidence[case_id]
        if answer["answer_status"] == "INSUFFICIENT_INFORMATION":
            references = [None]
        elif "product" in answer:
            references = [answer["product"]]
        else:
            references = [answer["product_a"], answer["product_b"]]
        case_rows = []
        for side, (reference, predicted, retrieved) in enumerate(
            zip(references, prediction["products"], source["products"], strict=True)
        ):
            candidates = retrieved["candidates"]
            expected = (
                set(
                    reference.get("ingredient_identities")
                    or [ingredient_identity(item) for item in reference["active_ingredients"]]
                )
                if reference
                else set()
            )
            ingredients = {item["identity"] for item in predicted["ingredients"]}
            native = next(
                (
                    product
                    for product in retrieved["identifier_candidates"]
                    if reference and product["product_ndc"] == reference["product_ndc"]
                ),
                None,
            )
            source_difference = False
            if reference and native and reference.get("formulation_supported", True):
                source_difference = _formulation(native) != _formulation(reference)
            supported_sets = [set(product["ingredient_identities"]) for product in candidates]
            supported = set.intersection(*supported_sets) if supported_sets else set()
            row = {
                "task": case["task"],
                "reference": reference,
                "prediction": predicted,
                "evidence": retrieved,
                "exact_justified": bool(
                    reference
                    and (
                        reference["product_ndc"] is not None
                        if "formulation_supported" in reference
                        else len(candidates) == 1
                        and candidates[0]["product_ndc"] == reference["product_ndc"]
                    )
                ),
                "formulation_correct": bool(
                    reference
                    and predicted["formulation"] is not None
                    and predicted["status"] in RESOLVED
                    and _formulation(predicted["formulation"]) == _formulation(reference)
                ),
                "source_reference_difference": source_difference,
                "ingredient_correct": bool(
                    reference
                    and ingredients == expected
                    and predicted["ingredient_status"] == "SUPPORTED"
                ),
                "predicted_ingredients": ingredients,
                "expected_ingredients": expected,
                "unsupported_by_evidence": len(ingredients - supported),
            }
            products.append(row)
            case_rows.append(row)
            if (
                reference
                and (
                    not row["ingredient_correct"]
                    or (
                        reference.get("formulation_supported", True)
                        and not row["formulation_correct"]
                    )
                    or (reference["product_ndc"] is not None and not row["exact_justified"])
                    or (
                        row["exact_justified"]
                        and predicted["product_ndc"] != reference["product_ndc"]
                    )
                )
            ) or (not reference and predicted["status"] != "INSUFFICIENT_INFORMATION"):
                errors.append(
                    {
                        "case_id": case_id,
                        "task": case["task"],
                        "product_side": side + 1,
                        "input": retrieved["description"],
                        "status": predicted["status"],
                        "reason": predicted["reason"],
                        "candidate_count": len(candidates),
                        "candidate_ndcs_sample": predicted["candidate_ndcs"][:5],
                        "reference_ndc": reference["product_ndc"] if reference else None,
                        "exact_ndc_justified": row["exact_justified"],
                        "expected_ingredients": sorted(expected),
                        "predicted_ingredients": sorted(ingredients),
                        "ingredient_correct": row["ingredient_correct"],
                        "formulation_correct": row["formulation_correct"],
                        "ingredient_status": predicted["ingredient_status"],
                        "source_formulation": {
                            key: native[key]
                            for key in ("active_ingredients", "dosage_form", "route")
                        }
                        if source_difference
                        else None,
                    }
                )
        if len(references) == 2:
            comparison = prediction["comparison"]
            comparisons.append(
                {
                    "task": case["task"],
                    "predicted_class": comparison["ingredient_overlap"]
                    if comparison["status"] == "SUPPORTED"
                    else "unresolved",
                    "expected_class": answer["ingredient_overlap"],
                    "predicted_shared": set(comparison["shared_active_ingredient_identities"]),
                    "expected_shared": set(answer["shared_active_ingredient_identities"]),
                    "expected_same": answer.get("same_active_ingredient_set"),
                    "predicted_same": comparison["same_active_ingredient_set"],
                }
            )
        correct = all(
            (
                r["ingredient_correct"]
                and (
                    not r["reference"].get("formulation_supported", True)
                    or r["formulation_correct"]
                )
                and (
                    not r["exact_justified"]
                    or r["prediction"]["product_ndc"] == r["reference"]["product_ndc"]
                )
            )
            if r["reference"]
            else r["prediction"]["status"] == "INSUFFICIENT_INFORMATION"
            for r in case_rows
        )
        if len(references) == 2:
            correct = (
                correct and comparisons[-1]["expected_class"] == comparisons[-1]["predicted_class"]
            )
        case_results.append(
            {
                "case_id": case_id,
                "task": case["task"],
                "correct": correct,
                "families": json.loads(case["family_group_ids_json"]),
            }
        )
    examples = []
    seen = set()
    for error in sorted(
        errors,
        key=lambda row: (row["ingredient_correct"], row["formulation_correct"], row["case_id"]),
    ):
        key = (error["status"], error["input"].get("name"))
        if key not in seen and len(examples) < 8:
            examples.append(error)
            seen.add(key)
    tasks = {}
    for task in sorted({row["task"] for row in cases}):
        metrics = _product_metrics([row for row in products if row["task"] == task])
        tasks[task] = {
            "cases": sum(row["task"] == task for row in cases),
            **{
                key: metrics[key]
                for key in (
                    "product_requests",
                    "exact_ndc_accuracy_when_justified",
                    "reference_ndc_not_established_by_input",
                    "formulation_accuracy_all_eligible",
                    "correct_abstention",
                    "incorrect_abstention_against_supported_reference",
                    "resolution_coverage",
                )
            },
            "ingredient_identification": {
                key: metrics["ingredient_identification"][key]
                for key in (
                    "set_exact_match_all_eligible",
                    "precision",
                    "recall",
                    "f1",
                )
            },
        }
    families = defaultdict(list)
    for result in case_results:
        for family in result["families"]:
            families[family].append(result["correct"])
    return {
        "cases": len(cases),
        "family_groups": len(
            {group for row in cases for group in json.loads(row["family_group_ids_json"])}
        ),
        "cases_by_task": dict(sorted(Counter(row["task"] for row in cases).items())),
        "product_resolution": _product_metrics(products),
        "ingredient_comparison": _comparison_metrics(comparisons),
        "by_task": tasks,
        "issue_counts": dict(sorted(Counter(error["reason"] for error in errors).items())),
        "failure_examples": examples,
        "joint_case_accuracy": _metric(sum(r["correct"] for r in case_results), len(case_results)),
        "family_macro": {
            "families": len(families),
            "mean_joint_case_accuracy": sum(sum(v) / len(v) for v in families.values())
            / len(families)
            if families
            else None,
            "perfect_families": _metric(sum(all(v) for v in families.values()), len(families)),
            "note": "Equal weight per partition group. Cross-group comparisons contribute to both groups. Repeated products are not independent trials.",
        },
        "source_limited_failures": _metric(
            sum(
                r["reference"] is not None and r["prediction"]["ingredient_status"] != "SUPPORTED"
                for r in products
            ),
            len(products),
        ),
    }


def run_evaluation(project_root: Path) -> dict:
    """Evaluate development partitions after verifying the frozen artifact hashes."""
    directory = project_root / "data/processed" / FINAL_BENCHMARK_VERSION
    reports = project_root / "results" / FINAL_BENCHMARK_VERSION / "part04"
    reports.mkdir(parents=True, exist_ok=True)
    seal_path = project_root / "results" / FINAL_BENCHMARK_VERSION / "part03/freeze_manifest.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    for name, digest in seal["files"].items():
        if file_sha256(directory / name) != digest:
            raise ValueError("Frozen benchmark artifact changed " + name)
    catalog = load_catalog(project_root)
    if (
        catalog["source_hashes"][f"data/interim/{FINAL_BENCHMARK_VERSION}/catalog_evidence.csv"]
        != seal["source_catalog_sha256"]
    ):
        raise ValueError("Catalog changed after benchmark freeze")
    support = Counter(_ingredient_support(product)[0] for product in catalog["products"].values())
    summary = {
        "method": "deterministic_retrieval_v2",
        "catalog_ndcs": len(catalog["products"]),
        "catalog_ingredient_support": {
            status: _metric(support[status], len(catalog["products"]))
            for status in ("SUPPORTED", "UNAVAILABLE", "SOURCE_CONFLICT")
        },
        "source_hashes": catalog["source_hashes"],
        "freeze_manifest_sha256": file_sha256(seal_path),
        "retrieval_code_sha256": file_sha256(Path(__file__)),
        "prediction_inputs": ["input_json"],
        "evidence_contract": {
            "retrieval": "Complete catalog exact normalized display, brand or generic name or explicit NDC. "
            "Apply supplied form, route and dimensionally justified strength constraints. "
            "Uncertain metadata never removes plausible alternatives.",
            "source_fields": [
                "NDC",
                "names",
                "native FDA ingredients",
                "original ingredient quantities",
                "form",
                "route",
                "FDA label identifiers and active text",
                "parsed label quantities",
                "RxNorm product and ingredient names and identifiers",
                "literal IN and PIN paths",
                "source eligibility issues and confidence tiers",
            ],
            "fair_comparison": "Original and fine-tuned Gemma must use retrieve_evidence unchanged. "
            "No references, provenance, review decisions, partitions or case metadata enter prediction.",
            "ingredient_identity": "Complete FDA source names remain usable without RxNorm. FDA UNII "
            "sets establish source identifier overlap. RxNorm MTHSPL substance codes corroborate identifiers "
            "on valid IN or PIN paths even when names differ. No salt, moiety, density or clinical equivalence is assumed.",
        },
        "metric_notes": [
            "Product requests include both comparison endpoints and single requests.",
            "Ingredient accuracy includes every supported reference request and retains unresolved predictions as errors.",
            (
                "Ingredient targets use FDA substance identifiers when supported and declared source-name sets otherwise. "
                "Chemical comparisons require a complete supported identifier set. Unordered sets never invent name-to-dose alignment."
            ),
            "Exact NDC denominators follow the frozen reference resolution level, independently of prediction success.",
            "Formulation metrics exclude references with unresolved source quantities and report this source-limited denominator explicitly.",
            "Formulation-only references have no arbitrary reference NDC.",
            "Unresolved comparisons remain in overlap accuracy and recall denominators. Zero denominators remain null.",
            "Family macro scores give equal weight to conservative partition groups. Repeated products are not independent trials.",
        ],
        "limitations": [
            "Source-supported references and retrieval share submitted FDA evidence. High agreement is not independent clinical validation.",
            "Partition groups are conservative algorithmic related-name groups rather than human-confirmed marketing families.",
            "Source disagreements and unresolved strength bases remain in the catalog and source-quality ledger.",
            "Exact retrieval does not guess misspellings or partial names.",
            "Ingredient overlap does not establish clinical harm.",
        ],
        "test_predictions_generated": 0,
        "test_reference_json_decoded": False,
        "test_input_json_decoded": False,
    }
    for partition in ("train", "validation"):
        cases = load_evaluation_cases(directory / (partition + ".csv"), (partition,))
        cached = {}
        evidence = {}
        predictions = {}
        prediction_path = directory / (partition + "_retrieval_predictions.jsonl")
        with prediction_path.open("w", encoding="utf-8") as stream:
            for case in cases:
                sources = []
                for description in _descriptions(case["input_json"]):
                    key = json.dumps(description, sort_keys=True)
                    if key not in cached:
                        cached[key] = retrieve_evidence(description, catalog)["products"][0]
                    sources.append(cached[key])
                source = {"catalog_ndcs": len(catalog["products"]), "products": sources}
                case_id = case["case_id"]
                evidence[case_id] = source
                predictions[case_id] = predict_from_evidence(source)
                stream.write(
                    json.dumps(
                        {"case_id": case_id, "prediction": predictions[case_id]}, ensure_ascii=False
                    )
                    + "\n"
                )
        summary[partition] = evaluate_cases(cases, predictions, evidence)
        summary[partition]["prediction_file_sha256"] = file_sha256(prediction_path)
        summary[partition]["case_file_sha256"] = file_sha256(directory / (partition + ".csv"))
        print(partition, summary[partition]["cases"], "cases evaluated", flush=True)
        del cases, predictions, evidence, cached
    for name, digest in seal["files"].items():
        if file_sha256(directory / name) != digest:
            raise ValueError("Evaluation changed a frozen artifact " + name)
    (reports / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary
