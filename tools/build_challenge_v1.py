"""Build/freeze the source-reviewed validation overlay; never read the sealed test.

From the repository root, first run the targeted tests with --junitxml pointing
to results/challenge_v1/checks.junit.xml, then run this script with --repo . .
Rerunning a frozen build verifies it instead of rewriting it. --verify verifies
the manifest and every input/output hash. All generated data remain local.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import revised_scope_validation_audit as prior
from rapidfuzz import fuzz, process

from hrp import challenge_evaluation as contract
from hrp.data_cleaning import normalize_name, route_signature
from hrp.retrieval_baseline import retrieve_evidence
from hrp.rxnorm_ingredients import ALLOWED_EDGES, CONCEPT_TYPES, resolve_ingredients

SEED = 42
EXPECTED_REVIEW = {
    "PRIMARY_CANDIDATE": 17,
    "CONTROL_CANDIDATE": 14,
    "AMBIGUITY_CANDIDATE": 4,
    "SUPPLEMENTARY": 14,
    "EXCLUDE_PRIMARY": 13,
    "HOLD_SOURCE_OR_IDENTITY": 38,
}
POLICY = {
    "version": "challenge_v1_selection_1",
    "seed": SEED,
    "target_different_ingredient_cases": 120,
    "target_controls": 80,
    "target_ambiguity": 40,
    "minimum_different_ingredient_cases": 50,
    "minimum_different_groups": 25,
    "minimum_different_without_ndc": 25,
    "minimum_without_ndc_groups": 10,
    "minimum_controls": 30,
    "minimum_ambiguity": 10,
    "selection": "Keep source-eligible accepted candidates first; then maximize unused partition "
    "groups, prioritize NDC-free descriptions and rotate source categories with seed-42 "
    "ties. No model outcomes are used. Target sizes are computational budgets, not power claims.",
    "relationship": "Accepted PRIMARY/CONTROL relationships, or source-supported identical meaningful "
    "names, shared meaningful submitted brand base, same-labeler distinctive prefix "
    "variants, or non-generic name ratio >=92. These establish source-name relationships, "
    "not demonstrated clinical confusion or verified marketing ownership.",
    "scope": "Conventional source-labelled OTC categories lead; topical/dental/sunscreen and other "
    "source-resolved categories remain explicit strata. Homeopathic records stay secondary.",
    "evidence": "All input-derived candidates, complete active-ingredient text, original paired "
    "RxNorm records and identical source payloads for both Gemma checkpoints and the new "
    "deterministic comparator. No candidate or label truncation; no gold/review/split fields.",
    "unresolved": "Never score unresolved identities, incomplete source composition or disagreements "
    "as unique formulations. Preserve them in the decision ledger and diagnostics.",
    "leakage": "Keep final_v5 immutable. Quarantine training partition groups sharing meaningful "
    "submitted brand bases/anchors or ratio>=92 source-name relations with selected validation "
    "products. Stage 6 must enforce the frozen training allowlist.",
}


def digest(value):
    return hashlib.sha256(contract.canonical(value).encode("utf-8")).hexdigest()


def write_json(path, value):
    contract.reject_reserved_fields(value)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, values):
    with path.open("w", encoding="utf-8", newline="") as stream:
        for value in values:
            contract.reject_reserved_fields(value)
            stream.write(contract.canonical(value) + "\n")


def write_csv(path, rows, columns):
    if any(c.casefold().startswith("ai_") for c in columns):
        raise ValueError("Prohibited column prefix")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_review(inputs, validation_rows):
    path = inputs.repo / "results/revised_scope_audit/validation_review_queue.REVIEWED.csv"
    reviewed = list(inputs.csv(path))
    pending_path = inputs.repo / "results/revised_scope_audit/validation_review_queue.PRIVATE.csv"
    pending = {r["case_id"]: r for r in inputs.csv(pending_path)}
    counts = Counter(r["review_verdict"] for r in reviewed)
    expected_total = sum(EXPECTED_REVIEW.values())
    if counts != Counter(EXPECTED_REVIEW) or len(reviewed) != expected_total:
        raise ValueError("Authoritative review counts do not match the accepted 100-case review")
    lookup = {r["case_id"]: r for r in validation_rows}
    if len({r["case_id"] for r in reviewed}) != expected_total or set(pending) != {
        r["case_id"] for r in reviewed
    }:
        raise ValueError("Reviewed and original pending case memberships differ")
    for row in reviewed:
        if (
            row["review_status"] != "REVIEWED"
            or row["partition"] != "validation"
            or row["reviewer_1_verdict"] != row["review_verdict"]
        ):
            raise ValueError("Incomplete or inconsistent authoritative review")
        case = lookup.get(row["case_id"])
        if not case or json.loads(row["input_json"]) != case["input"]:
            raise ValueError("Reviewed case input differs from frozen validation input")
        if set(json.loads(row["product_ndcs_json"])) != set(case["ndcs"]):
            raise ValueError("Reviewed source membership differs from validation provenance")
        if json.loads(pending[row["case_id"]]["input_json"]) != case["input"]:
            raise ValueError("Original pending input changed")
    return {r["case_id"]: r for r in reviewed}, dict(sorted(counts.items())), path


def load_rxnorm(inputs, progress):
    releases = sorted((inputs.source_dir / "raw/rxnorm").glob("*/rrf/RXNCONSO.RRF"))
    if len(releases) != 1:
        raise ValueError("Exactly one original full RxNorm release is required")
    conso = inputs.register(releases[0])
    rel = inputs.register(conso.with_name("RXNREL.RRF"))
    concepts, substance_codes, synonyms = {}, defaultdict(set), defaultdict(set)
    with inputs.open(conso, encoding="utf-8") as stream:
        for line in stream:
            f = line.rstrip("\r\n").split("|")
            if len(f) != 19:
                raise ValueError("Invalid original RXNCONSO row")
            if f[11] == "RXNORM" and f[16] == "N" and f[12] in CONCEPT_TYPES:
                concepts[f[0]] = (f[12], f[14])
            if (
                f[11] == "MTHSPL"
                and f[12] == "SU"
                and f[16] == "N"
                and re.fullmatch(r"[A-Z0-9]{10}", f[13])
            ):
                substance_codes[f[0]].add(f[13])
    with inputs.open(conso, encoding="utf-8") as stream:
        for line in stream:
            f = line.rstrip("\r\n").split("|")
            if (
                f[0] in concepts
                and concepts[f[0]][0] in {"IN", "PIN"}
                and f[11] == "RXNORM"
                and f[1] == "ENG"
                and f[16] == "N"
            ):
                synonyms[f[0]].add(f[14])
    relationships = defaultdict(set)
    with inputs.open(rel, encoding="utf-8") as stream:
        for line in stream:
            f = line.rstrip("\r\n").split("|")
            if len(f) != 17:
                raise ValueError("Invalid original RXNREL row")
            if f[10] != "RXNORM" or f[14] not in ("", "N") or f[2] != "CUI" or f[6] != "CUI":
                continue
            a, b = concepts.get(f[0]), concepts.get(f[4])
            if a and b and (b[0], f[7], a[0]) in ALLOWED_EDGES:
                relationships[f[4]].add(f[0])
    entries = []
    for rxcui, names in sorted(synonyms.items()):
        uniis = sorted(substance_codes[rxcui])
        if uniis:
            for name in sorted(names):
                entries.append(
                    {
                        "name": name,
                        "uniis": uniis,
                        "source": "RxNorm_MTHSPL_SU",
                        "rxcui": rxcui,
                        "term_type": concepts[rxcui][0],
                    }
                )
    for row in inputs.csv(inputs.repo / "data/interim/final_v5/fda_ingredient_identifiers.csv"):
        entries.append(
            {
                "name": row["ingredient_name"],
                "uniis": [row["unii"]],
                "source": "FDA_singleton_name_unii_crosswalk",
                "supporting_source_ndcs": json.loads(row["source_ndcs_json"]),
            }
        )
    progress(f"Verified RxNorm source graph and {len(entries)} source name/identity records")
    return contract.IdentityResolver(entries), entries, concepts, substance_codes, relationships


def original_sources(inputs, ndcs, labels, progress):
    targets = {labels[n].get("label_id") for n in ndcs if n in labels and labels[n].get("label_id")}
    label_records, ndc_records = {}, defaultdict(list)
    for path in prior.source_archives(inputs.source_dir, "label"):
        for row, member in prior.fda_records(inputs, path):
            if row.get("id") not in targets:
                continue
            label_records[row["id"]] = {
                "id": row["id"],
                "set_id": row.get("set_id", ""),
                "version": str(row.get("version", "")),
                "effective_time": row.get("effective_time", ""),
                "active_ingredient": row.get("active_ingredient") or [],
                "openfda": {
                    key: (row.get("openfda") or {}).get(key, [])
                    for key in (
                        "brand_name",
                        "generic_name",
                        "substance_name",
                        "unii",
                        "product_ndc",
                        "route",
                        "rxcui",
                    )
                },
                "archive": str(path.relative_to(inputs.source_dir)),
                "member": member,
            }
        progress(f"Cross-checked original SPL shard {path.name}")
    for path in prior.source_archives(inputs.source_dir, "ndc"):
        for row, member in prior.fda_records(inputs, path):
            if row.get("product_ndc") not in ndcs:
                continue
            ndc_records[row["product_ndc"]].append(
                {
                    **{
                        key: row.get(key)
                        for key in (
                            "product_ndc",
                            "product_id",
                            "spl_id",
                            "brand_name",
                            "generic_name",
                            "active_ingredients",
                            "dosage_form",
                            "route",
                            "labeler_name",
                            "marketing_category",
                        )
                    },
                    "openfda": {
                        key: (row.get("openfda") or {}).get(key, [])
                        for key in ("unii", "spl_set_id")
                    },
                    "archive": str(path.relative_to(inputs.source_dir)),
                    "member": member,
                }
            )
        progress("Cross-checked original FDA NDC archive")
    return label_records, ndc_records


def source_candidate(ndc, catalog, metadata, labels, originals, rxnorm):
    product, meta, cached = catalog["products"][ndc], metadata[ndc], labels.get(ndc, {})
    raw_labels, raw_ndcs = originals
    label = raw_labels.get(cached.get("label_id"))
    ndc_options = [r for r in raw_ndcs[ndc] if r["spl_id"] == cached.get("label_id")]
    original_ingredients = json.loads(meta["original_ingredients_json"] or "[]") or []
    original = next(
        (
            r
            for r in ndc_options
            if prior.structured_signature(r.get("active_ingredients") or [])
            == prior.structured_signature(original_ingredients)
        ),
        None,
    )
    if original is None and len(raw_ndcs[ndc]) == 1:
        original = raw_ndcs[ndc][0]  # Native NDC existence is distinct from a unique SPL linkage.
    version_ok = bool(
        original
        and label
        and original.get("spl_id") == cached.get("label_id")
        and prior.structured_signature(original.get("active_ingredients") or [])
        == prior.structured_signature(original_ingredients)
        and label["set_id"] == cached.get("label_set_id")
        and label["version"] == cached.get("label_version")
        and label["effective_time"] == cached.get("effective_time")
        and label["active_ingredient"] == json.loads(cached.get("active_ingredient_json") or "[]")
    )
    native_names = {normalize_name(i["name"]) for i in product["active_ingredients"]}
    original_names = {
        normalize_name(i["name"]) for i in (original or {}).get("active_ingredients") or []
    }
    name_ok = bool(
        original
        and normalize_name(product["name"])
        in {
            normalize_name(original.get("brand_name")),
            normalize_name(original.get("generic_name")),
        }
    )
    origin = meta.get("ingredient_origin", "ndc_structured")
    native_ok = native_names == original_names or (
        not original_names and origin != "ndc_structured"
    )
    form_route_ok = bool(
        original
        and normalize_name(original.get("dosage_form")) == normalize_name(product["dosage_form"])
        and route_signature(original.get("route")) == route_signature(product["route"])
    )
    mapping, drug = product["rxnorm_ingredients"], product["rxnorm_product"]
    concepts, codes, relationships = rxnorm
    paired, pair_ok, path_ok = [], True, True
    for kind, tty in (("in", "IN"), ("pin", "PIN")):
        names, ids = mapping[kind + "_names"], mapping[kind + "_rxcuis"]
        if len(names) != len(ids):
            pair_ok = False
            continue
        for name, identifier in zip(names, ids, strict=True):
            preferred = concepts.get(identifier)
            pair_ok &= preferred == (tty, name)
            paired.append(
                {
                    "name": name,
                    "rxcui": identifier,
                    "term_type": tty,
                    "uniis": sorted(codes.get(identifier, set())),
                }
            )
    if drug.get("rxnorm_status") == "matched":
        cui = drug.get("rxnorm_rxcui")
        if cui not in concepts:
            path_ok = False
        else:
            base, precise, _ = resolve_ingredients(cui, concepts, relationships)
            path_ok = set(base) == set(mapping["in_rxcuis"]) and set(precise) == set(
                mapping["pin_rxcuis"]
            )
    candidate = {
        "product_ndc": ndc,
        "display_name": product["name"],
        "brand_name": product["brand_name"],
        "generic_name": product["generic_name"],
        "dosage_form": product["dosage_form"],
        "route": product["route"] or [],
        "fda_active_ingredients": [
            {"name": i["name"], "strength": i.get("strength", "")}
            for i in product["active_ingredients"]
        ],
        "spl": {
            "label_id": cached.get("label_id", ""),
            "set_id": cached.get("label_set_id", ""),
            "version": cached.get("label_version", ""),
            "effective_time": cached.get("effective_time", ""),
            "active_ingredient_text": (label or {}).get("active_ingredient", []),
            "structured_substance_names": (label or {})
            .get("openfda", {})
            .get("substance_name", []),
        },
        "rxnorm": {
            "product_rxcui": drug.get("rxnorm_rxcui", ""),
            "product_name": concepts.get(drug.get("rxnorm_rxcui"), ("", ""))[1],
            "ingredient_pairs": sorted(paired, key=lambda p: (p["term_type"], p["rxcui"])),
            "cached_pairs_confirmed": bool(pair_ok and path_ok),
        },
        "provenance": {
            "original_versions_confirmed": version_ok,
            "raw_ndc_name_matches_catalog": bool(name_ok and native_ok),
            "raw_form_route_confirmed": form_route_ok,
            "ingredient_origin": origin,
            "product_uniis": sorted((original or {}).get("openfda", {}).get("unii", [])),
            "ndc_spl_id": (original or {}).get("spl_id", ""),
            "original_ndc_archive": (original or {}).get("archive", ""),
            "original_label_archive": (label or {}).get("archive", ""),
            "original_ndc_selected_fields_sha256": digest(original) if original else None,
            "original_label_selected_fields_sha256": digest(label) if label else None,
            "dailymed_set_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid="
            + cached.get("label_set_id", ""),
            "version_note": "DailyMed/openFDA SPL are renderings of the same submission; "
            "use the recorded exact label version, not a newer web rendering.",
        },
    }
    contract.assert_source_only(candidate)
    return candidate


def submitted_bases(meta):
    return {
        prior.name_key(r.get("brand_name_base") or "")
        for r in json.loads(meta.get("source_metadata_json") or "[]")
    }


def source_relationship(row, evidence, metadata, qualities, screen, review=None):
    if row["task"] != "ingredient_comparison":
        return {"supported": False, "class": "not_a_comparison", "basis": []}
    a, b = row["input"]["product_a"]["name"], row["input"]["product_b"]["name"]
    relation = screen.relation(a, b)
    if review and review["review_verdict"] in {"PRIMARY_CANDIDATE", "CONTROL_CANDIDATE"}:
        return {**relation, "supported": True, "basis": ["accepted_review_relationship"]}
    sides = [[p["product_ndc"] for p in item["candidates"]] for item in evidence["products"]]
    if not all(sides):
        return {**relation, "supported": False, "basis": ["no_source_candidates"]}
    all_ndcs = sides[0] + sides[1]
    meaningful = (
        set(prior.name_key(a).split()) | set(prior.name_key(b).split())
    ) & screen.distinctive
    generic = any(qualities[n]["generic_like_name"] for n in all_ndcs)
    bases = [
        {
            base
            for n in ns
            for base in submitted_bases(metadata[n])
            if set(base.split()) & screen.distinctive
        }
        for ns in sides
    ]
    labelers = [
        {
            normalize_name(r.get("labeler_name", ""))
            for n in ns
            for r in json.loads(metadata[n].get("source_metadata_json") or "[]")
        }
        for ns in sides
    ]
    supported, basis = False, []
    if not generic and meaningful:
        if relation["class"] == "identical_normalized_name":
            supported, basis = True, ["identical_meaningful_submitted_names"]
        elif bases[0] & bases[1]:
            supported, basis = True, ["shared_meaningful_submitted_brand_base"]
        elif (
            relation["shared_distinctive_tokens"]
            and labelers[0] & labelers[1] - {""}
            and prior.name_key(a).split()[0] == prior.name_key(b).split()[0]
        ):
            supported, basis = True, ["same_labeler_distinctive_prefix_variant"]
        elif fuzz.ratio(prior.name_key(a), prior.name_key(b)) >= 92:
            supported, basis = True, ["highly_similar_non_generic_submitted_names"]
    return {
        **relation,
        "supported": supported,
        "basis": basis or ["relationship_not_source_established"],
    }


def disposition(row, reviewed, payload, reference, assessments, relation, category):
    verdict = reviewed.get("review_verdict") if reviewed else "NOT_PREVIOUSLY_REVIEWED"
    source_reasons = sorted({reason for n in row["ndcs"] for reason in assessments[n]["reasons"]})
    if verdict == "EXCLUDE_PRIMARY":
        return "EXCLUDED", "accepted_review_exclusion", "none", source_reasons
    if source_reasons:
        return "UNRESOLVED", "documented_source_or_identity_limit", "none", source_reasons
    if any(p["status"] == "SOURCE_CONFLICT" for p in reference["products"]):
        return (
            "UNRESOLVED",
            "input_depends_on_disputed_strength",
            "none",
            ["supplied_strengths_disagree_with_same_version_active_text"],
        )
    if reviewed and verdict == "SUPPLEMENTARY":
        return "ACCEPTED", "accepted_review_supplementary", "supplementary", []
    comparison = reference["comparison"]
    any_ndc = any(item["request"].get("product_ndc") for item in payload["products"])
    if row["task"] == "ingredient_comparison":
        if not relation["supported"]:
            return "EXCLUDED", "name_relationship_not_established", "none", []
        if comparison["same_active_ingredient_set"] is None:
            return "UNRESOLVED", "comparison_not_answerable", "none", []
        if category == "homeopathic":
            return "ACCEPTED", "homeopathic_secondary_scope", "supplementary", []
        if comparison["same_active_ingredient_set"]:
            role = "same_ingredient_control"
        else:
            role = "different_with_ndc" if any_ndc else "different_without_ndc"
        if verdict == "CONTROL_CANDIDATE" and not comparison["same_active_ingredient_set"]:
            return "UNRESOLVED", "new_source_findings_contradict_accepted_control", "none", []
        if verdict == "PRIMARY_CANDIDATE" and comparison["same_active_ingredient_set"]:
            return "UNRESOLVED", "new_source_findings_contradict_accepted_primary", "none", []
        return "ACCEPTED", "source_reviewed_comparison", role, []
    if row["case_type"] == "name_only_ambiguous":
        if reference["products"][0]["status"] == "INSUFFICIENT_INFORMATION":
            return "ACCEPTED", "source_supported_ingredient_ambiguity", "ambiguity", []
        return "ACCEPTED", "ingredient_set_resolved_without_unique_formulation", "supplementary", []
    if reviewed:
        return "ACCEPTED", "source_reviewed_supplementary", "supplementary", []
    return "EXCLUDED", "outside_focused_challenge_scope", "none", []


def select_cases(records):
    eligible = [r for r in records if r["automated_disposition"] == "ACCEPTED"]
    rng = random.Random(SEED)
    ties = {r["case_id"]: rng.random() for r in sorted(eligible, key=lambda r: r["case_id"])}
    selected, used, groups, categories = [], set(), set(), Counter()

    def add(row):
        selected.append(row)
        used.add(row["case_id"])
        groups.update(row["families"])
        categories[row["category"]] += 1

    for row in sorted(eligible, key=lambda r: r["case_id"]):
        if row["previous_review_verdict"] not in {
            "NOT_PREVIOUSLY_REVIEWED",
            "HOLD_SOURCE_OR_IDENTITY",
        }:
            add(row)
    for roles, target in [
        (
            {"different_without_ndc", "different_with_ndc"},
            POLICY["target_different_ingredient_cases"],
        ),
        ({"same_ingredient_control"}, POLICY["target_controls"]),
        ({"ambiguity"}, POLICY["target_ambiguity"]),
    ]:
        while sum(r["scoring_role"] in roles for r in selected) < target:
            available = [
                r for r in eligible if r["case_id"] not in used and r["scoring_role"] in roles
            ]
            if not available:
                break
            row = min(
                available,
                key=lambda r: (
                    bool(set(r["families"]) & groups),
                    r["contains_explicit_ndc"],
                    categories[r["category"]],
                    ties[r["case_id"]],
                    r["case_id"],
                ),
            )
            add(row)
    return sorted(selected, key=lambda r: r["case_id"])


def training_quarantine(train, selected, memberships, metadata, screen):
    train_ndcs = {n for n, m in memberships.items() if m["partition"] == "train"}
    val_ndcs = {n for r in selected for n in r["ndcs"]}
    train_names, bases, anchors = defaultdict(set), defaultdict(set), defaultdict(set)
    for ndc in sorted(train_ndcs):
        train_names[prior.name_key(metadata[ndc]["display_name"])].add(ndc)
        for base in submitted_bases(metadata[ndc]):
            if set(base.split()) & screen.distinctive:
                bases[base].add(ndc)
                first = base.split()[0] if base else ""
                if first in screen.distinctive:
                    anchors[first].add(ndc)
    names = sorted(train_names)
    blocked_ndcs, links = set(), []
    for ndc in sorted(val_ndcs):
        name = prior.name_key(metadata[ndc]["display_name"])
        related = set()
        for other, score, _ in process.extract_iter(
            name, names, scorer=fuzz.ratio, score_cutoff=92
        ):
            relation = screen.relation(name, other)
            if relation["class"] in prior.PRIMARY_RELATIONS:
                related.update(train_names[other])
        for base in submitted_bases(metadata[ndc]):
            if set(base.split()) & screen.distinctive:
                related.update(bases[base])
                if base and base.split()[0] in screen.distinctive:
                    related.update(anchors[base.split()[0]])
        if related:
            links.append({"validation_ndc": ndc, "training_source_ndcs": sorted(related)})
            blocked_ndcs.update(related)
    blocked_groups = {memberships[n]["family"] for n in blocked_ndcs}
    selected_groups = {g for r in selected for g in r["families"]}
    train_groups = {m["family"] for m in memberships.values() if m["partition"] == "train"}
    if selected_groups & train_groups or train_ndcs & val_ndcs:
        raise ValueError("Frozen train/validation membership leakage")
    allowed_ids = {r["case_id"] for r in train if not set(r["families"]) & blocked_groups}
    filtered = contract.filter_training_rows(train, allowed_ids, blocked_groups)
    selected_inputs = {contract.semantic_input_key(r["input"]) for r in selected}
    if any(contract.semantic_input_key(r["input"]) in selected_inputs for r in filtered):
        raise ValueError("Identical challenge input in permitted training rows")
    return {
        "blocked_training_groups": sorted(blocked_groups),
        "screened_source_links": links,
        "allowed_training_cases": len(filtered),
        "excluded_training_cases": len(train) - len(filtered),
        "original_training_cases": len(train),
        "training_allowlist_required": True,
        "policy": POLICY["leakage"],
        "scope": "Train/validation only; real-brand completeness not claimed",
    }, filtered


def quality_checks(selected, original, review_counts, test_info, common_scores, assessments):
    different = [
        r for r in selected if r["scoring_role"] in {"different_without_ndc", "different_with_ndc"}
    ]
    free = [r for r in different if not r["contains_explicit_ndc"]]
    counts = Counter(r["scoring_role"] for r in selected)
    return {
        "authoritative_review_reconciled": review_counts == EXPECTED_REVIEW,
        "all_cases_validation_only": all(r["partition"] == "validation" for r in selected),
        "all_selected_source_supported": all(
            assessments[n]["status"] == "SUPPORTED" for r in selected for n in r["ndcs"]
        ),
        "minimum_different_ingredient_cases": len(different)
        >= POLICY["minimum_different_ingredient_cases"],
        "minimum_different_groups": len({g for r in different for g in r["families"]})
        >= POLICY["minimum_different_groups"],
        "minimum_without_ndc_cases": len(free) >= POLICY["minimum_different_without_ndc"],
        "minimum_without_ndc_groups": len({g for r in free for g in r["families"]})
        >= POLICY["minimum_without_ndc_groups"],
        "minimum_controls": counts["same_ingredient_control"] >= POLICY["minimum_controls"],
        "minimum_ambiguity": counts["ambiguity"] >= POLICY["minimum_ambiguity"],
        "no_review_exclusions_promoted": not any(
            r["previous_review_verdict"] == "EXCLUDE_PRIMARY" for r in selected
        ),
        "source_comparator_contract_self_consistent": common_scores["overall"][
            "joint_case_accuracy"
        ]["value"]
        == 1.0,
        "all_targeted_tests_passed": test_info["failures"] == 0
        and test_info["errors"] == 0
        and test_info["tests"] >= 30,
        "original_inputs_unchanged": original,
    }


def read_test_report(path):
    tree = ET.parse(path)
    suites = list(tree.getroot().iter("testsuite"))
    return {
        key: sum(int(s.attrib.get(key, 0)) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def verify_freeze(output, *, check_inputs=True):
    manifest_path = output / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "FROZEN":
        raise ValueError("Challenge has not passed the freeze gate")
    for name, expected in manifest["outputs"].items():
        path = output / name
        if path.resolve().parent != output.resolve() or prior.sha256_file(path) != expected:
            raise ValueError("Frozen challenge output hash mismatch")
    if check_inputs:
        for item in manifest["inputs"].values():
            if prior.sha256_file(Path(item["path"])) != item["sha256"]:
                raise ValueError("Frozen challenge input hash mismatch")
    return manifest


def render_report(summary):
    selection, recon = summary["selected"], summary["review_reconciliation"]
    failed = [k for k, passed in summary["quality_checks"].items() if not passed]
    return f"""# Stage 5C challenge_v1 source review and finalization

Decision: **{summary["stage6_decision"]}**. Benchmark status: **{summary["status"]}**.
This is a validation-development overlay, source-reviewed and not independently clinically verified.
No additional human review is requested or required by this implementation.

## Accepted review and immutable boundaries

The authoritative input is `validation_review_queue.REVIEWED.csv`, with exactly
17 PRIMARY_CANDIDATE, 14 CONTROL_CANDIDATE, 4 AMBIGUITY_CANDIDATE, 14 SUPPLEMENTARY,
13 EXCLUDE_PRIMARY and 38 HOLD_SOURCE_OR_IDENTITY verdicts. All 100 REVIEWED markers and both
verdict fields were checked. The accepted input is preserved byte-for-byte in
`accepted_review_snapshot.PRIVATE.csv`; both original reviewed and pending CSVs remain unchanged.
No verdict was invented or overwritten. The per-case decision ledger records the original verdict
separately from automated eligibility, reasons and membership. Reconciliation:
`{prior.canonical(recon)}`.

Only existing train/validation cases and original public-source FDA/RxNorm records were used.
The sealed test was never opened, decoded, hashed or evaluated. No final_v5 files, historical
baseline results, source downloads or original review files were modified. No Gemma inference,
training, Runpod operation, source download or new clinical verification was performed.

## Automated source review

Original source paths and actual SHA-256 hashes are in `evidence_manifest.json` and the freeze
manifest. The exact SPL ID, set ID, version, effective date and active-ingredient text are
matched to the NDC SPL ID and cached source. Raw NDC product names and ingredient records are
cross-checked. FDA Drug Labeling/openFDA and DailyMed may display the same manufacturer SPL;
the saved DailyMed set links are provenance links, not independently fetched clinical evidence.
The local exact-version SPL is authoritative for version matching; no newer webpage is substituted.

The original RxNorm RXNCONSO and RXNREL release is read and hashed. Cached IN/PIN name/RxCUI
pairs and ingredient paths are checked against it. Names and identifiers stay paired. UNII
links come from original MTHSPL SU atoms and the source-only singleton FDA crosswalk, never
from validation references. Same-concept RxNorm aliases are allowed only when the UNII link
is unique. Salts, PIN and IN are not collapsed. Ambiguous, absent or conflicting mappings
remain unresolved. Percentages retain explicit or unspecified basis; no density conversion,
active-moiety equivalence or unsupported strength denominator is inferred.

Structured and amount-anchored active-text ingredient identities are compared. Missing names,
unrecognized text, uncovered quantities or unsupported mappings are unresolved, not automatic
proof of an incorrect label. Strength checks distinguish matches, comparable-basis differences
and unresolved basis. Ingredient-set support is distinct from full formulation support.
USP/NF grades, explicitly stated teaspoon/tablepoon mL denominators and fluoride-ion basis
annotations are recognized as source notation. No spoon volume is assumed, no fluoride-ion
strength is substituted for sodium fluoride, and unspecified percentage basis or RxNorm
strength/form/route disagreement withholds full-formulation support. Missing SPL linkage is
unresolved provenance, not a demonstrated contradiction. Pre-delivery automated corrections
are documented in `build_corrections.json` when present; no selection count gate was relaxed.
Every validation case has a source-linked row in `review_decisions.PRIVATE.csv`; every validation
product has source observations in `source_review.PRIVATE.jsonl`. Newly found contradictions
are separate in `contradictions.PRIVATE.jsonl`, retaining the original accepted verdict.
The known Spearmint conflict is retained in diagnostics and cannot supply a unique formulation.

## Systematic challenge expansion and composition

The original 35 primary/control/ambiguity candidates were not assumed to provide a sufficient
primary denominator. All validation cases were screened under the frozen policy, without
model outputs or expected model performance. Accepted source-eligible reviewed candidates
are retained first; remaining slots prioritize unused groups, NDC-free descriptions and
category diversity, with seed 42. Inputs are unchanged: exact NDCs are neither stripped nor
added. True ingredient ambiguity remains an abstention target; a shared ingredient set with
uncertain strengths is not mislabeled as a unique formulation.

Selected cases: **{selection["cases"]}**, across **{selection["groups"]}** conservative partition groups.
Roles: `{prior.canonical(selection["roles"])}`.
Categories: `{prior.canonical(selection["categories"])}`.
NDC-free different-ingredient cases: **{selection["different_without_ndc"]}**, across
**{selection["different_without_ndc_groups"]}** groups. These form the headline formulation-name
distinction analysis; exact-identifier comparisons are reported separately. Conventional oral,
dental, topical and sunscreen strata are reported separately; homeopathic cases are secondary.
Source-name relationships are operationally supported by the accepted review or explicit source
name/brand rules; this is not a measured real-world medication-error or clinical-confusion rate.

Automated dispositions across all validation rows: `{prior.canonical(summary["pool_dispositions"])}`.
Eligible but unselected cases remain in the ledger with an explicit sampling reason. Unresolved
and excluded cases are never silently removed, promoted to gold or treated as model failures.
Originally held cases may enter scoring only with documented newly resolved source evidence,
while their original HOLD verdict remains unchanged. Significant unresolved source cases do
not prevent freezing a well-supported overlay when they are transparently retained outside
the high-confidence denominators. Selection budgets and minimum count/group gates are feasibility
criteria, not a statistical power calculation or a guarantee that Gemma can outperform retrieval.

## Common evidence, output and scoring contract

`common_evidence.PRIVATE.jsonl` supplies every input-derived candidate and full active text,
without the previous eight-candidate/1,200-character cap. Identical serialized payloads and
messages are used for untuned Gemma, tuned Gemma and the new deterministic source comparator.
No review decisions, expected answers, partitions, family IDs or baseline predictions enter
prompts. Source UNII and paired RxNorm records are factual evidence, so this is explicitly an
evidence-grounded extraction/distinction task. Source-complete deterministic performance may
remain perfect; the historical baseline is retained as a source-consistency ceiling.

`output_schema.json` and the executable nested parser require actual JSON types, all required
fields, supported status/identifier logic, valid source identifiers and consistent comparison
nullability. Duplicate keys, nonfinite values, extra fields, malformed nested output and
ungrounded identifiers are invalid outputs. Unknown ingredient names count as unsupported
additions and missing expected ingredients, without closest-reference repair.

`scoring_specifications.json` freezes case, product-request, ingredient, comparison, abstention,
unique-NDC and complete-formulation denominators. Missing/invalid model outputs stay in their
eligible denominators. Main metrics and category/identifier strata remain separate. The
deterministic challenge comparator is recomputed from common evidence and stored under this
overlay only; its source-consistency score is not independent truth or a Gemma result.

## Leakage safeguards and Stage 6

Train and validation NDCs, group IDs, case IDs and identical inputs are checked. Potential
cross-split source-brand and near-name relations produce a conservative training-group
quarantine. `training_allowlist.PRIVATE.csv` and `training_quarantine.PRIVATE.jsonl` must be
enforced for QLoRA; original train.csv remains unchanged. Counts:
`{prior.canonical(summary["training_counts"])}`.
The training helper rejects non-training rows or an allowlist that contradicts the quarantine.
This bounds the stated automated leakage rules; it does not establish universal real-brand
independence. Uncertainty analyses must cluster by conservative group and examine category
and repeated-product sensitivity. This remains development validation, not a new sealed test.

Quality gates: `{prior.canonical(summary["quality_checks"])}`.
Unpassed gates/blockers: `{prior.canonical(failed)}`.
Targeted test evidence: `{prior.canonical(summary["tests"])}`.
All consumed input hashes are rechecked unchanged. Frozen output hashes and policy hashes are
recorded in `freeze_manifest.json`; rerunning a frozen build verifies rather than overwrites it.

Stage 6 decision concerns benchmark/software readiness. GPU compatibility, model checkpoint
revision, tokenizer, quantization, maximum context, deterministic decoding and the actual QLoRA
environment must still be recorded and exercised by the Stage 6 load/micro-run. No GPU readiness
or performance result is claimed here. Use both checkpoints with this same contract and the
provided training allowlist. Do not tune on challenge validation answers. Keep unresolved
diagnostics out of high-confidence accuracy denominators and report them separately.
"""


def build(repo, source_dir=None, progress=print):
    repo = Path(repo).resolve()
    source_dir = Path(source_dir or repo.parent / "data").resolve()
    output = repo / "results/challenge_v1"
    if output.resolve() != output:
        raise ValueError("Output directory redirect is prohibited")
    if (output / "freeze_manifest.json").exists():
        manifest = json.loads((output / "freeze_manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] == "FROZEN":
            verify_freeze(output)
            progress("Frozen challenge verified; nothing rewritten")
            return manifest
    output.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(32 * 1024 * 1024)
    inputs = prior.Inputs(repo, source_dir)
    if (output / "build_corrections.json").exists():
        inputs.register(output / "build_corrections.json")
    for name in (
        "tools/build_challenge_v1.py",
        "tools/revised_scope_validation_audit.py",
        "src/hrp/challenge_evaluation.py",
        "src/hrp/retrieval_baseline.py",
        "src/hrp/evidence_adapter.py",
        "src/hrp/data_cleaning.py",
        "src/hrp/data_loading.py",
        "src/hrp/rxnorm_ingredients.py",
        "src/hrp/ingredient_agreement.py",
        "pyproject.toml",
    ):
        inputs.register(repo / name)
    baseline = inputs.json(repo / "results/final_v5/part04/validation_summary.json")
    cases, memberships = prior.load_cases(inputs)
    reviewed, review_counts, review_path = load_review(inputs, cases["validation"])
    catalog, metadata, labels = prior.load_sources(inputs, memberships)
    for name, expected in baseline["source_hashes"].items():
        if inputs.hashes[name]["sha256"] != expected:
            raise ValueError("Source changed from the historical Part 4 input")
    for part in ("train", "validation"):
        if (
            inputs.hashes[f"data/processed/final_v5/{part}.csv"]["sha256"]
            != baseline[part]["case_file_sha256"]
        ):
            raise ValueError("Case file changed from the historical Part 4 input")
    progress(
        "Reconciled authoritative 17/14/4/14/13/38 review; frozen source and case hashes match"
    )
    resolver, entries, concepts, codes, graph = load_rxnorm(inputs, progress)
    val_ndcs = {n for n, m in memberships.items() if m["partition"] == "validation"}
    originals = original_sources(inputs, val_ndcs, labels, progress)
    candidates = {
        n: source_candidate(n, catalog, metadata, labels, originals, (concepts, codes, graph))
        for n in sorted(val_ndcs)
    }
    assessments = {n: contract.assess_candidate(p, resolver) for n, p in candidates.items()}
    screen = prior.NameScreen(metadata)
    qualities = {n: prior.product_quality(catalog["products"][n], metadata[n]) for n in memberships}
    records, payloads = [], {}
    source_review = [
        {
            "product_ndc": n,
            "source_payload_sha256": digest(candidates[n]),
            "source_payload": candidates[n],
            "source_assessment": assessments[n],
        }
        for n in sorted(val_ndcs)
    ]
    write_jsonl(output / "source_review.PRIVATE.jsonl", source_review)
    progress(
        f"Source product assessments: {dict(Counter(a['status'] for a in assessments.values()))}"
    )
    for row in cases["validation"]:
        evidence = retrieve_evidence(row["input"], catalog)
        matched = {p["product_ndc"] for item in evidence["products"] for p in item["candidates"]}
        if matched != set(row["ndcs"]) or not matched <= val_ndcs:
            raise ValueError("Query evidence exceeds validation provenance")
        payload = {
            "contract_version": contract.VERSION,
            "source_scope": "exact-version FDA SPL/NDC and pinned RxNorm",
            "products": [
                {
                    "request": item["description"],
                    "candidate_count": len(item["candidates"]),
                    "candidates": [candidates[p["product_ndc"]] for p in item["candidates"]],
                }
                for item in evidence["products"]
            ],
        }
        contract.assert_source_only(payload)
        reference = contract.predict_common_evidence(payload, resolver)
        relation = source_relationship(
            row, evidence, metadata, qualities, screen, reviewed.get(row["case_id"])
        )
        categories = {qualities[n]["category"] for n in row["ndcs"]}
        category = next(iter(categories)) if len(categories) == 1 else "mixed_categories"
        reviewed_row = reviewed.get(row["case_id"])
        status, reason, role, source_reasons = disposition(
            row, reviewed_row, payload, reference, assessments, relation, category
        )
        record = {
            "case_id": row["case_id"],
            "partition": "validation",
            "input": row["input"],
            "task": row["task"],
            "case_type": row["case_type"],
            "ndcs": row["ndcs"],
            "families": row["families"],
            "category": category,
            "scoring_role": role,
            "name_relationship": relation,
            "automated_disposition": status,
            "reason": reason,
            "source_reasons": source_reasons,
            "additional_source_findings": sorted(
                {
                    issue
                    for n in row["ndcs"]
                    for issue in assessments[n]["nonblocking_source_findings"]
                }
            ),
            "hold_resolution_basis": "Exact-version original sources, complete label ingredient "
            "recognition and unique source-only identity mapping now agree; original HOLD retained."
            if reviewed_row
            and reviewed_row["review_verdict"] == "HOLD_SOURCE_OR_IDENTITY"
            and status == "ACCEPTED"
            else "",
            "previous_review_verdict": reviewed_row["review_verdict"]
            if reviewed_row
            else "NOT_PREVIOUSLY_REVIEWED",
            "contains_explicit_ndc": any(
                item["request"].get("product_ndc") for item in payload["products"]
            ),
            "common_evidence_sha256": digest(payload),
            "reference": reference,
            "source_reviewed": status == "ACCEPTED",
            "independently_clinically_verified": False,
        }
        records.append(record)
        # Payloads share source object references in memory; only selected evidence is serialized later.
        payloads[row["case_id"]] = payload
    selected = select_cases(records)
    selected_ids = {r["case_id"] for r in selected}
    quarantine, training = training_quarantine(
        cases["train"], selected, memberships, metadata, screen
    )
    predictions = {
        r["case_id"]: contract.predict_common_evidence(payloads[r["case_id"]], resolver)
        for r in selected
    }
    scores = contract.score_predictions(selected, predictions, payloads, resolver)
    test_path = inputs.register(output / "checks.junit.xml")
    test_info = read_test_report(test_path)
    unchanged = inputs.verify_unchanged()
    checks = quality_checks(selected, unchanged, review_counts, test_info, scores, assessments)
    status = "FROZEN" if all(checks.values()) else "BLOCKED"
    reconcile = defaultdict(Counter)
    for row in records:
        if row["previous_review_verdict"] != "NOT_PREVIOUSLY_REVIEWED":
            outcome = (
                "selected_" + row["scoring_role"]
                if row["case_id"] in selected_ids
                else "eligible_not_selected"
                if row["automated_disposition"] == "ACCEPTED"
                else row["automated_disposition"].lower()
            )
            reconcile[row["previous_review_verdict"]][outcome] += 1
    summary = {
        "benchmark": "challenge_v1",
        "contract_version": contract.VERSION,
        "seed": SEED,
        "status": status,
        "stage6_decision": "GO" if status == "FROZEN" else "NO-GO",
        "accepted_review_counts": review_counts,
        "review_reconciliation": {k: dict(sorted(v.items())) for k, v in sorted(reconcile.items())},
        "pool_dispositions": dict(Counter(r["automated_disposition"] for r in records)),
        "source_product_assessment_counts": dict(
            Counter(a["status"] for a in assessments.values())
        ),
        "selected": {
            "cases": len(selected),
            "groups": len({g for r in selected for g in r["families"]}),
            "roles": dict(sorted(Counter(r["scoring_role"] for r in selected).items())),
            "categories": dict(sorted(Counter(r["category"] for r in selected).items())),
            "different_without_ndc": sum(
                r["scoring_role"] == "different_without_ndc" for r in selected
            ),
            "different_without_ndc_groups": len(
                {
                    g
                    for r in selected
                    if r["scoring_role"] == "different_without_ndc"
                    for g in r["families"]
                }
            ),
        },
        "training_counts": {k: v for k, v in quarantine.items() if k.endswith("cases")},
        "quality_checks": checks,
        "tests": test_info,
        "safeguards": {
            "sealed_test_opened": False,
            "sealed_test_hashed": False,
            "historical_benchmark_or_baseline_modified": False,
            "review_decisions_overwritten": False,
            "model_inference_or_training_run": False,
            "all_input_hashes_unchanged": unchanged,
        },
        "runtime": {
            "python": sys.version,
            **{name: version(name) for name in ("pandas", "rapidfuzz", "ijson")},
        },
    }
    # Every row, including unselected/excluded/unresolved cases, is accounted for explicitly.
    ledger = [
        {
            "case_id": r["case_id"],
            "partition": "validation",
            "review_verdict": r["previous_review_verdict"],
            "automated_disposition": r["automated_disposition"],
            "reason": r["reason"],
            "source_reasons_json": prior.canonical(r["source_reasons"]),
            "selected": r["case_id"] in selected_ids,
            "additional_source_findings_json": prior.canonical(r["additional_source_findings"]),
            "hold_resolution_basis": r["hold_resolution_basis"],
            "selection_reason": "selected_frozen_policy"
            if r["case_id"] in selected_ids
            else "eligible_pool_not_selected_fixed_budget"
            if r["automated_disposition"] == "ACCEPTED"
            else r["reason"],
            "scoring_role": r["scoring_role"],
            "category": r["category"],
            "product_ndcs_json": prior.canonical(r["ndcs"]),
            "family_group_ids_json": prior.canonical(r["families"]),
            "common_evidence_sha256": r["common_evidence_sha256"],
            "accepted_verdict_preserved": True,
        }
        for r in records
    ]
    write_csv(output / "review_decisions.PRIVATE.csv", ledger, list(ledger[0]))
    write_jsonl(output / "benchmark.PRIVATE.jsonl", selected)
    write_jsonl(
        output / "common_evidence.PRIVATE.jsonl",
        [
            {
                "case_id": r["case_id"],
                "payload_sha256": r["common_evidence_sha256"],
                "payload": payloads[r["case_id"]],
                "messages": contract.build_messages(payloads[r["case_id"]]),
            }
            for r in selected
        ],
    )
    write_jsonl(
        output / "deterministic_predictions.PRIVATE.jsonl",
        [{"case_id": r["case_id"], "prediction": predictions[r["case_id"]]} for r in selected],
    )
    write_jsonl(output / "ingredient_identity_mapping.PRIVATE.jsonl", entries)
    write_jsonl(output / "training_quarantine.PRIVATE.jsonl", [quarantine])
    write_csv(
        output / "training_allowlist.PRIVATE.csv",
        [
            {
                "case_id": r["case_id"],
                "partition": "train",
                "family_group_ids_json": prior.canonical(r["families"]),
            }
            for r in training
        ],
        ["case_id", "partition", "family_group_ids_json"],
    )
    contradictions = [
        {
            "case_id": r["case_id"],
            "review_verdict": r["previous_review_verdict"],
            "automated_disposition": r["automated_disposition"],
            "source_reasons": r["source_reasons"],
            "additional_source_findings": r["additional_source_findings"],
            "product_ndcs": r["ndcs"],
            "evidence_manifest_file": "source_review.PRIVATE.jsonl",
            "unique_correct_formulation_assigned": False,
        }
        for r in records
        if r["source_reasons"]
        or r["additional_source_findings"]
        or r["reason"].startswith("new_source_findings")
    ]
    write_jsonl(output / "contradictions.PRIVATE.jsonl", contradictions)
    write_jsonl(
        output / "diagnostic_cases.PRIVATE.jsonl",
        [
            r
            for r in records
            if r["previous_review_verdict"] != "NOT_PREVIOUSLY_REVIEWED"
            and r["case_id"] not in selected_ids
        ],
    )
    with inputs.open(review_path, "rb") as stream:
        (output / "accepted_review_snapshot.PRIVATE.csv").write_bytes(stream.read())
    write_json(output / "selection_policy.json", POLICY)
    write_json(output / "output_schema.json", contract.output_schema())
    write_json(
        output / "scoring_specifications.json",
        {
            "contract_version": contract.VERSION,
            "fixed_denominators": scores["overall"]["counts"],
            "fixed_denominators_by_role": {k: v["counts"] for k, v in scores["by_role"].items()},
            "identity_policy": "Source-only unique UNII/name and exact RxNorm concept aliases. Preserve IN/PIN and salts; unknowns count as unsupported additions.",
            "invalid_output_policy": "All missing or malformed outputs remain in every applicable fixed denominator; never count as correct abstention.",
            "formulation_policy": "Separate complete native+label strength/form/route target support from ingredient-set support. Unspecified percent basis is not converted.",
            "comparison_policy": "Exact canonical ingredient-set equality, overlap and shared-set extraction; no clinical equivalence.",
            "primary_role": "different_without_ndc",
            "secondary_roles": sorted(
                {r["scoring_role"] for r in selected} - {"different_without_ndc"}
            ),
            "reporting": "Report each role/category separately, invalid output rate, unknown ingredients, and group-clustered uncertainty; development validation only.",
        },
    )
    write_json(output / "deterministic_metrics.json", scores)
    write_json(output / "summary.json", summary)
    write_json(
        output / "evidence_manifest.json",
        {
            "inputs": dict(sorted(inputs.hashes.items())),
            "policy_sha256": digest(POLICY),
            "evidence_policy": POLICY["evidence"],
            "identity_contract": contract.VERSION,
            "membership_sha256": digest([r["case_id"] for r in selected]),
            "reference_sha256": digest([[r["case_id"], r["reference"]] for r in selected]),
            "training_allowlist_sha256": prior.sha256_file(
                output / "training_allowlist.PRIVATE.csv"
            ),
            "source_support_is_independent_clinical_verification": False,
        },
    )
    (output / "STAGE5C_FINAL_REPORT.md").write_text(render_report(summary), encoding="utf-8")
    output_names = [
        "accepted_review_snapshot.PRIVATE.csv",
        "benchmark.PRIVATE.jsonl",
        "common_evidence.PRIVATE.jsonl",
        "contradictions.PRIVATE.jsonl",
        "deterministic_predictions.PRIVATE.jsonl",
        "diagnostic_cases.PRIVATE.jsonl",
        "ingredient_identity_mapping.PRIVATE.jsonl",
        "review_decisions.PRIVATE.csv",
        "source_review.PRIVATE.jsonl",
        "training_allowlist.PRIVATE.csv",
        "training_quarantine.PRIVATE.jsonl",
        "selection_policy.json",
        "output_schema.json",
        "scoring_specifications.json",
        "deterministic_metrics.json",
        "summary.json",
        "evidence_manifest.json",
        "STAGE5C_FINAL_REPORT.md",
        "checks.junit.xml",
    ]
    if (output / "build_corrections.json").exists():
        output_names.append("build_corrections.json")
    manifest = {
        "benchmark": "challenge_v1",
        "contract_version": contract.VERSION,
        "status": status,
        "stage6_decision": summary["stage6_decision"],
        "seed": SEED,
        "policy_sha256": digest(POLICY),
        "quality_checks": checks,
        "inputs": dict(sorted(inputs.hashes.items())),
        "outputs": {name: prior.sha256_file(output / name) for name in output_names},
    }
    write_json(output / "freeze_manifest.json", manifest)
    progress(
        f"Stage 5C {status}: {len(selected)} source-reviewed cases; Stage 6 {summary['stage6_decision']}"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify_freeze(args.repo.resolve() / "results/challenge_v1")
        print(f"Verified {result['status']} challenge, all input/output SHA-256 hashes match")
    else:
        build(args.repo, args.source_dir)


if __name__ == "__main__":
    main()
