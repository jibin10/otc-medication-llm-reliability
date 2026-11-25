"""Build the final source-supported benchmark and retain historical review audits."""

import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations, pairwise
from pathlib import Path

import pandas as pd

from hrp import FINAL_BENCHMARK_VERSION
from hrp.data_cleaning import (
    formulation_key,
    ingredient_identity,
    ingredient_signature,
    normalize_name,
    quantity_key,
    route_signature,
)
from hrp.feasibility_analysis import EXACT_AGREEMENT, discover_name_links, name_key
from hrp.ingredient_agreement import fda_ingredient_names, rxnorm_names

REVIEW_DECISIONS = ("VERIFIED", "UNCERTAIN", "REJECTED", "PENDING")
REPORTED_REVIEW_COUNTS = {"VERIFIED": 16, "UNCERTAIN": 5, "REJECTED": 3}


def reconcile_reviews(reviewed: pd.DataFrame) -> dict:
    """Count saved decisions without changing their values or interpreting notes."""
    required = {
        "case_type",
        "product_ndc_a",
        "product_ndc_b",
        "review_decision",
        "verified_family",
        "review_notes",
        "name_a",
        "name_b",
    }
    if missing := required - set(reviewed.columns):
        raise ValueError(f"Missing reviewed columns: {sorted(missing)}")
    counts = Counter(reviewed["review_decision"])
    if unknown := set(counts) - set(REVIEW_DECISIONS):
        raise ValueError(f"Unexpected review decisions: {sorted(unknown)}")
    keys = [
        (row.case_type, *sorted((row.product_ndc_a, row.product_ndc_b)))
        for row in reviewed.itertuples()
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate reviewed pair, including reversed endpoints")
    if any(not a or not b or a == b for _, a, b in keys):
        raise ValueError("Reviewed pairs require two distinct product NDCs")
    return {
        "total_pairs": len(reviewed),
        "saved_counts": {decision: counts[decision] for decision in REVIEW_DECISIONS},
        "reported_counts": REPORTED_REVIEW_COUNTS,
        "matches_feasibility_report": all(
            counts[decision] == REPORTED_REVIEW_COUNTS.get(decision, 0)
            for decision in REVIEW_DECISIONS
        ),
        "counts_by_case_type": {
            kind: {
                decision: int(group.review_decision.eq(decision).sum())
                for decision in REVIEW_DECISIONS
            }
            for kind, group in reviewed.groupby("case_type", sort=True)
        },
        "verified_pairs_without_family": int(
            (
                reviewed.review_decision.eq("VERIFIED")
                & reviewed.verified_family.map(normalize_name).eq("")
            ).sum()
        ),
    }


def reviewed_family_assignments(reviewed: pd.DataFrame) -> dict[str, list[str]]:
    """Keep literal labels only for the NDCs explicitly in VERIFIED pairs."""
    reconcile_reviews(reviewed)
    labels: dict[str, set[str]] = defaultdict(set)
    for row in reviewed.itertuples():
        if row.review_decision != "VERIFIED" or not normalize_name(row.verified_family):
            continue
        for ndc in (row.product_ndc_a, row.product_ndc_b):
            labels[ndc].add(row.verified_family)
    return {ndc: sorted(values) for ndc, values in sorted(labels.items())}


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
    return frame


def _alias_key(name: str) -> str:
    """Suggest punctuation and ampersand aliases without accepting them."""
    return re.sub(r"[\W_]+", " ", normalize_name(name).replace("&", " and ")).strip()


def _generic_like_product(product: dict) -> bool:
    """Flag descriptive names using the existing product metadata."""
    name = product["normalized_name"]
    ingredients = (
        (fda_ingredient_names(product["active_ingredients_json"]) or frozenset())
        | rxnorm_names(product["rxnorm_in_names_json"])
        | rxnorm_names(product["rxnorm_pin_names_json"])
    )
    return (
        product["name_source"] == "generic"
        or name == normalize_name(product["generic_name"])
        or any(
            name == ingredient or name.startswith(ingredient + " ") for ingredient in ingredients
        )
    )


def _components(links: dict[str, set[str]]) -> list[list[str]]:
    """Collect review blocks, which are not established product families."""
    remaining = set(links)
    groups = []
    while remaining:
        pending = [min(remaining)]
        members = set()
        while pending:
            name = pending.pop()
            if name in members:
                continue
            members.add(name)
            pending.extend(links[name] - members)
        remaining.difference_update(members)
        groups.append(sorted(members))
    return groups


def build_family_audit(
    catalog_path: Path,
    agreement_path: Path,
    reviewed_path: Path,
    identical_pairs_path: Path,
    similar_pairs_path: Path,
    controls_path: Path,
    output_dir: Path,
) -> dict:
    """Write review groups and aggregate evidence without creating benchmark labels."""
    catalog = _read_csv(
        catalog_path,
        {
            "product_ndc",
            "normalized_name",
            "display_name",
            "name_source",
            "generic_name",
            "dosage_form",
            "route",
            "active_ingredients_json",
            "marketing_category",
        },
    )
    agreement = _read_csv(
        agreement_path,
        {
            "product_ndc",
            "catalog_status",
            "agreement_status",
            "rxnorm_ingredient_status",
            "rxnorm_in_names_json",
            "rxnorm_pin_names_json",
            "label_text_available",
        },
    )
    reviewed = _read_csv(
        reviewed_path,
        {
            "display_name_a",
            "display_name_b",
            "fda_ingredients_a",
            "fda_ingredients_b",
            "dosage_a",
            "dosage_b",
        },
    )
    reconciliation = reconcile_reviews(reviewed)
    assignments = reviewed_family_assignments(reviewed)
    identical = _read_csv(identical_pairs_path, {"normalized_name", "ndc_a", "ndc_b"})
    pair_columns = {"case_type", "product_ndc_a", "product_ndc_b", "name_a", "name_b"}
    similar = _read_csv(similar_pairs_path, pair_columns)
    controls = _read_csv(controls_path, pair_columns)
    input_profiles = {
        path.name: {
            "rows": len(frame),
            "columns": list(frame.columns),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
        for path, frame in (
            (catalog_path, catalog),
            (agreement_path, agreement),
            (reviewed_path, reviewed),
            (identical_pairs_path, identical),
            (similar_pairs_path, similar),
            (controls_path, controls),
        )
    }
    for frame, source in ((catalog, "catalog"), (agreement, "agreement")):
        if frame.product_ndc.eq("").any() or frame.product_ndc.duplicated().any():
            raise ValueError(f"Missing or duplicate product NDC in {source}")
    if set(similar.case_type) - {"similar_different"}:
        raise ValueError("Unexpected case type in similar-name candidates")
    if set(controls.case_type) - {"similar_same_control"}:
        raise ValueError("Unexpected case type in controls")
    catalog = catalog.merge(agreement, on="product_ndc", how="left", validate="one_to_one")
    if catalog.agreement_status.isna().any():
        raise ValueError("Catalog NDCs missing from ingredient agreement")
    products = catalog.set_index("product_ndc").to_dict("index")

    pairs = [
        {
            "case_type": "identical_different",
            "product_ndc_a": row.ndc_a,
            "product_ndc_b": row.ndc_b,
            "name_a": row.normalized_name,
            "name_b": row.normalized_name,
        }
        for row in identical.itertuples()
    ] + similar[sorted(pair_columns)].to_dict("records")
    control_pairs = controls[sorted(pair_columns)].to_dict("records")
    pool_keys = {
        (row["case_type"], *sorted((row["product_ndc_a"], row["product_ndc_b"])))
        for row in pairs + control_pairs
    }
    unmatched_review_rows = []
    snapshot_conflicts: dict[str, set[str]] = defaultdict(set)
    reviewed_records = []
    for number, row in enumerate(reviewed.to_dict("records"), start=1):
        key = (row["case_type"], *sorted((row["product_ndc_a"], row["product_ndc_b"])))
        if key not in pool_keys:
            unmatched_review_rows.append(number)
        for side in ("a", "b"):
            ndc = row[f"product_ndc_{side}"]
            if ndc not in products:
                raise ValueError(f"Reviewed NDC missing from catalog: {ndc}")
            product = products[ndc]
            if normalize_name(row[f"name_{side}"]) != product["normalized_name"]:
                snapshot_conflicts[ndc].add("reviewed_name_differs_from_catalog")
            if ingredient_signature(json.loads(row[f"fda_ingredients_{side}"])) != (
                ingredient_signature(json.loads(product["active_ingredients_json"]))
            ):
                snapshot_conflicts[ndc].add("reviewed_formulation_differs_from_catalog")
            if normalize_name(row[f"dosage_{side}"]) != normalize_name(product["dosage_form"]):
                snapshot_conflicts[ndc].add("reviewed_dosage_differs_from_catalog")
        reviewed_records.append(
            {
                "source_row": number,
                "case_type": row["case_type"],
                "product_ndc_a": row["product_ndc_a"],
                "product_ndc_b": row["product_ndc_b"],
                "review_decision": row["review_decision"],
                "verified_family": row["verified_family"],
                "review_notes": row["review_notes"],
            }
        )

    # Unreviewed controls remain outside this focused different-ingredient audit
    audited_pairs = pairs + reviewed[sorted(pair_columns)].to_dict("records")
    links: dict[str, set[str]] = defaultdict(set)
    for row in audited_pairs:
        for side in ("a", "b"):
            ndc = row[f"product_ndc_{side}"]
            if ndc not in products:
                raise ValueError(f"Candidate NDC missing from catalog: {ndc}")
        left = products[row["product_ndc_a"]]["normalized_name"]
        right = products[row["product_ndc_b"]]["normalized_name"]
        if not left or not right:
            raise ValueError("Candidate product has no normalized name")
        links[left].add(right)
        links[right].add(left)

    seed_names: dict[str, set[str]] = defaultdict(set)
    literal_labels: dict[str, set[str]] = defaultdict(set)
    for ndc, labels in assignments.items():
        for label in labels:
            seed_names[normalize_name(label)].add(products[ndc]["normalized_name"])
            literal_labels[normalize_name(label)].add(label)
    suggestions: dict[str, set[str]] = defaultdict(set)
    alias_names: set[str] = set()
    for label, names in seed_names.items():
        anchor = min(names)
        label_key = _alias_key(label)
        for name in catalog.normalized_name.unique():
            if label_key and f" {label_key} " in f" {_alias_key(name)} ":
                names.add(name)
                suggestions[name].update(literal_labels[label])
        for name in names:
            links[anchor].add(name)
            links[name].add(anchor)

    # Alias links aid review only and never transfer a manual NDC assignment
    aliases: dict[str, set[str]] = defaultdict(set)
    for name in catalog.normalized_name.unique():
        if key := _alias_key(name):
            aliases[key].add(name)
    for key in {_alias_key(name) for name in list(links)}:
        names = aliases[key]
        if len(names) < 2:
            continue
        anchor = min(names)
        alias_names.update(names)
        for name in names:
            links[anchor].add(name)
            links[name].add(anchor)

    by_name: dict[str, list[str]] = defaultdict(list)
    generic_like: set[str] = set()
    for ndc, product in products.items():
        name = product["normalized_name"]
        by_name[name].append(ndc)
        if _generic_like_product(product):
            generic_like.add(ndc)

    rows = []
    for names in _components(links):
        ndcs = sorted({ndc for name in names for ndc in by_name[name]})
        ndc_set = set(ndcs)
        saved_reviews = [
            row
            for row in reviewed_records
            if row["product_ndc_a"] in ndc_set or row["product_ndc_b"] in ndc_set
        ]
        assigned = {ndc: assignments[ndc] for ndc in ndcs if ndc in assignments}
        unassigned = sorted(ndc_set - set(assigned))
        labels = sorted({label for values in assigned.values() for label in values})
        proposed = sorted({label for name in names for label in suggestions[name]})
        conflicting_ndcs = sorted(
            ndc
            for ndc, values in assigned.items()
            if len({normalize_name(value) for value in values}) > 1
        )
        reasons = set()
        if unassigned:
            reasons.add("unassigned_products")
        if ndc_set & generic_like:
            reasons.add("generic_or_ingredient_like_names")
        if set(names) & alias_names:
            reasons.add("possible_punctuation_or_ampersand_alias")
        if proposed and unassigned:
            reasons.add("reviewed_label_match_requires_confirmation")
        if len({normalize_name(label) for label in labels}) > 1:
            reasons.add("multiple_manual_labels_in_review_group")
        if conflicting_ndcs:
            reasons.add("conflicting_manual_labels_for_product")
        if any(
            row["review_decision"] != "VERIFIED"
            and (row["product_ndc_a"] in assigned or row["product_ndc_b"] in assigned)
            for row in saved_reviews
        ):
            reasons.add("nonverified_pair_with_assigned_product")
        if any(
            row["review_decision"] == "VERIFIED" and not normalize_name(row["verified_family"])
            for row in saved_reviews
        ):
            reasons.add("verified_pair_without_family_label")
        for decision in ("UNCERTAIN", "REJECTED", "PENDING"):
            if any(row["review_decision"] == decision for row in saved_reviews):
                reasons.add(f"contains_{decision.lower()}_pair")
        for ndc in ndcs:
            reasons.update(snapshot_conflicts[ndc])
        if any(row["source_row"] in unmatched_review_rows for row in saved_reviews):
            reasons.add("reviewed_pair_missing_from_candidate_pool")
        dosage_forms = sorted({products[ndc]["dosage_form"] for ndc in ndcs})
        labeler_codes = sorted({ndc.split("-")[0] for ndc in ndcs})
        if len(dosage_forms) > 1:
            reasons.add("multiple_dosage_forms")
        if len(labeler_codes) > 1:
            reasons.add("multiple_ndc_labeler_codes")
        source_counts = Counter(products[ndc]["agreement_status"] for ndc in ndcs)
        if any(products[ndc]["agreement_status"] not in EXACT_AGREEMENT for ndc in ndcs):
            reasons.add("ingredient_agreement_requires_review")
        exact_eligible = sum(
            products[ndc]["catalog_status"] == "consistent"
            and products[ndc]["agreement_status"] in EXACT_AGREEMENT
            and products[ndc]["rxnorm_ingredient_status"] == "resolved_path"
            for ndc in ndcs
        )
        conflict = bool(
            conflicting_ndcs
            or len({normalize_name(label) for label in labels}) > 1
            or any("differs_from_catalog" in reason for reason in reasons)
        )
        status = (
            "CONFLICT"
            if conflict
            else "UNRESOLVED"
            if not assigned
            else "PARTIALLY_ASSIGNED"
            if unassigned
            else "REVIEWED_SEED_ONLY"
        )
        rows.append(
            {
                "review_group_id": "group_" + sha256("\n".join(names).encode()).hexdigest()[:12],
                "review_group_name": names[0],
                "assignment_status": status,
                "product_count": len(ndcs),
                "name_count": len(names),
                "assigned_product_count": len(assigned),
                "unassigned_product_count": len(unassigned),
                "manual_family_labels_json": labels,
                "suggested_family_labels_json": proposed,
                "review_reasons_json": sorted(reasons),
                "normalized_names_json": names,
                "assigned_products_json": assigned,
                "unassigned_product_ndcs_json": unassigned,
                "conflicting_product_ndcs_json": conflicting_ndcs,
                "reviewed_pairs_json": saved_reviews,
                "dosage_forms_json": dosage_forms,
                "routes_json": sorted({products[ndc]["route"] for ndc in ndcs}),
                "ndc_labeler_codes_json": labeler_codes,
                "ingredient_agreement_counts_json": dict(sorted(source_counts.items())),
                "exact_screen_eligible_product_count": exact_eligible,
                "label_text_available_product_count": sum(
                    products[ndc]["label_text_available"] == "True" for ndc in ndcs
                ),
                "generic_like_product_count": len(ndc_set & generic_like),
                "product_examples_json": [
                    {
                        "product_ndc": ndc,
                        "display_name": products[ndc]["display_name"],
                        "generic_name": products[ndc]["generic_name"],
                        "dosage_form": products[ndc]["dosage_form"],
                        "fda_ingredients": json.loads(products[ndc]["active_ingredients_json"]),
                        "agreement_status": products[ndc]["agreement_status"],
                    }
                    for ndc in ndcs[:3]
                ],
            }
        )

    rows.sort(
        key=lambda row: (
            row["assignment_status"] != "CONFLICT",
            not bool(row["reviewed_pairs_json"]),
            row["review_group_name"],
        )
    )
    scoped_ndcs = {
        ndc
        for row in rows
        for ndc in (list(row["assigned_products_json"]) + row["unassigned_product_ndcs_json"])
    }
    manual_labels = {normalize_name(label) for values in assignments.values() for label in values}
    summary = {
        "stage": "3.1_product_family_audit",
        "review_reconciliation": reconciliation,
        "scope": {
            "identical_name_candidate_pairs": len(identical),
            "similar_name_different_ingredient_pairs": len(similar),
            "reviewed_control_pairs": int(reviewed.case_type.eq("similar_same_control").sum()),
            "available_same_ingredient_control_pairs": len(controls),
            "unreviewed_control_pairs_deferred": len(controls)
            - int(reviewed.case_type.eq("similar_same_control").sum()),
            "catalog_products": len(catalog),
            "products_in_review_groups": len(scoped_ndcs),
            "catalog_products_outside_audit": len(catalog) - len(scoped_ndcs),
            "expansion": "All catalog products with audited exact names, manual label phrase "
            "matches, or punctuation and ampersand aliases. No agreement filter.",
        },
        "families": {
            "candidate_review_groups": len(rows),
            "manually_labeled_seed_families": len(manual_labels),
            "explicitly_assigned_products": len(assignments),
            "unassigned_products": sum(row["unassigned_product_count"] for row in rows),
            "groups_by_assignment_status": dict(
                sorted(Counter(row["assignment_status"] for row in rows).items())
            ),
            "groups_by_review_reason": dict(
                sorted(
                    Counter(reason for row in rows for reason in row["review_reasons_json"]).items()
                )
            ),
            "conflicting_product_assignments": sum(
                bool(len({normalize_name(label) for label in labels}) > 1)
                for labels in assignments.values()
            ),
        },
        "data_quality": {
            "reviewed_pair_rows_missing_from_candidate_pool": unmatched_review_rows,
            "reviewed_snapshot_conflicts_by_ndc": {
                ndc: sorted(reasons)
                for ndc, reasons in sorted(snapshot_conflicts.items())
                if reasons
            },
            "products_outside_exact_agreement_screen": sum(
                row["product_count"] - row["exact_screen_eligible_product_count"] for row in rows
            ),
            "ownership_metadata_available": bool(
                {"brand_owner", "manufacturer_name", "labeler_name"} & set(catalog.columns)
            ),
            "concerns": [
                "Review groups are evidence blocks, not verified families or split identifiers.",
                "Manual labels apply only to the explicitly reviewed NDCs. No label propagates.",
                "UNCERTAIN and REJECTED decisions and all review notes remain unchanged.",
                "VERIFIED is the saved preliminary pair decision, not final reference validation.",
                "Name similarity and first words do not establish family membership.",
                "Generic descriptions can occur in the source brand field.",
                "NDC labeler codes do not establish brand ownership or manufacturer identity.",
                "Ingredient agreement and label-text availability do not verify ground truth.",
                "Salts, active moieties, strength, dosage form and label versions need review.",
                "Unreviewed control-only families and the rest of the catalog are outside scope.",
            ],
        },
        "readiness": {
            "ready_for_part_3_2_split_freeze": False,
            "next_action": "Review family blocks, confirm aliases and ownership, resolve or "
            "exclude uncertain assignments, and choose the control cohort before splitting.",
            "splits_created": False,
            "benchmark_examples_created": False,
        },
        "inputs": input_profiles,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_rows = [
        {
            key: json.dumps(value, ensure_ascii=False, sort_keys=True)
            if key.endswith("_json")
            else value
            for key, value in row.items()
        }
        for row in rows
    ]
    pd.DataFrame(serializable_rows).to_csv(output_dir / "family_review.csv", index=False)
    (output_dir / "family_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


SHORTLIST_REVIEW_FLAGS = (
    "possible_family_alias",
    "generic_or_ingredient_like_names",
    "ingredient_disagreement",
    "partially_assigned_family_labels",
)
SHORTLIST_CASES = (
    "identical_different_shared",
    "identical_different_disjoint",
    "similar_different_shared",
    "similar_different_disjoint",
    "similar_same_control",
    "potentially_ambiguous_name",
)


def _select_shortlist(groups: list[dict], max_groups: int) -> list[dict]:
    """Cycle through review needs and case types with one row per evidence group."""
    selected = []
    used_groups: set[str] = set()
    covered_flags: set[str] = set()
    case_counts: Counter[str] = Counter()
    overlap_counts: Counter[tuple[str, str]] = Counter()
    while len(selected) < max_groups:
        previous_size = len(selected)
        for category in SHORTLIST_REVIEW_FLAGS + SHORTLIST_CASES:
            options = []
            for group in groups:
                if group["review_group_id"] in used_groups:
                    continue
                for pair in group["pairs"]:
                    pair_category = (
                        pair["case_type"]
                        if pair["case_type"] == "similar_same_control"
                        else f"{pair['case_type']}_{pair['ingredient_overlap']}"
                    )
                    if category not in group["flags"] and category != pair_category:
                        continue
                    flags = set(group["flags"]) & set(SHORTLIST_REVIEW_FLAGS)
                    rank = (
                        -len(flags - covered_flags),
                        case_counts[pair["case_type"]],
                        overlap_counts[(pair["case_type"], pair["ingredient_overlap"])],
                        sha256(group["review_group_id"].encode()).hexdigest(),
                        pair["case_type"],
                        pair["product_ndc_a"],
                        pair["product_ndc_b"],
                    )
                    options.append((rank, group, pair))
            if not options:
                continue
            _, group, pair = min(options, key=lambda option: option[0])
            selected.append({"group": group, "pair": pair, "selection_category": category})
            used_groups.add(group["review_group_id"])
            covered_flags.update(set(group["flags"]) & set(SHORTLIST_REVIEW_FLAGS))
            case_counts[pair["case_type"]] += 1
            overlap_counts[(pair["case_type"], pair["ingredient_overlap"])] += 1
            if len(selected) == max_groups:
                break
        if len(selected) == previous_size:
            break
    return selected


def build_family_review_shortlist(interim_dir: Path, audit_dir: Path, max_groups: int = 30) -> dict:
    """Select additional review evidence from all saved candidate pools."""
    if not isinstance(max_groups, int) or not 0 <= max_groups <= 30:
        raise ValueError("max_groups must be an integer between 0 and 30")
    audit = json.loads((audit_dir / "family_audit.json").read_text(encoding="utf-8"))
    for name, profile in audit["inputs"].items():
        if sha256((interim_dir / name).read_bytes()).hexdigest() != profile["sha256"]:
            raise ValueError(f"Audit input changed: {name}. Rerun Part 3.1 first.")
    families = _read_csv(
        audit_dir / "family_review.csv",
        {
            "review_group_id",
            "normalized_names_json",
            "review_reasons_json",
            "assigned_products_json",
            "unassigned_product_ndcs_json",
            "reviewed_pairs_json",
        },
    )
    if families.review_group_id.duplicated().any():
        raise ValueError("Duplicate family review group ID")
    catalog_path = interim_dir / "otc_catalog_consistent.csv"
    agreement_path = interim_dir / "otc_ingredient_agreement.csv"
    reviewed_path = interim_dir / "feasibility_review_queue_reviewed.csv"
    catalog = _read_csv(
        catalog_path,
        {
            "product_ndc",
            "normalized_name",
            "display_name",
            "generic_name",
            "name_source",
            "active_ingredients_json",
            "dosage_form",
            "route",
        },
    )
    agreement = _read_csv(
        agreement_path,
        {
            "product_ndc",
            "agreement_status",
            "rxnorm_in_rxcuis_json",
            "rxnorm_in_names_json",
            "rxnorm_pin_names_json",
            "label_text_available",
        },
    )
    reviewed = _read_csv(reviewed_path, {"review_decision", "verified_family"})
    review_counts = reconcile_reviews(reviewed)
    assignments = reviewed_family_assignments(reviewed)
    saved_assignments = {
        ndc: labels
        for row in families.itertuples()
        for ndc, labels in json.loads(row.assigned_products_json).items()
    }
    if assignments != saved_assignments:
        raise ValueError("Family audit assignments differ from the saved manual decisions")
    catalog = catalog.merge(agreement, on="product_ndc", how="left", validate="one_to_one")
    if catalog.agreement_status.isna().any():
        raise ValueError("Catalog product missing from ingredient agreement")
    products = catalog.set_index("product_ndc").to_dict("index")
    label_links_path = interim_dir / "otc_ndc_label_links.csv"
    label_links = {}
    if label_links_path.is_file():
        label_frame = _read_csv(
            label_links_path,
            {
                "product_ndc",
                "link_status",
                "label_id",
                "label_set_id",
                "effective_time",
            },
        )
        if label_frame.product_ndc.duplicated().any():
            raise ValueError("Duplicate NDC in FDA label links")
        label_links = label_frame.set_index("product_ndc").to_dict("index")

    pairs = []
    for filename, kind in (
        ("identical_name_candidate_pairs.csv", "identical_different"),
        ("similar_name_candidate_pairs.csv", "similar_different"),
        ("similar_name_same_ingredient_controls.csv", "similar_same_control"),
    ):
        columns = {"ndc_a", "ndc_b", "in_rxcuis_a_json", "in_rxcuis_b_json"}
        if kind != "identical_different":
            columns = {"case_type", "product_ndc_a", "product_ndc_b", "ingredient_overlap"}
        pool = _read_csv(interim_dir / filename, columns)
        if kind != "identical_different" and set(pool.case_type) - {kind}:
            raise ValueError(f"Unexpected case type in {filename}")
        for source_row, row in enumerate(pool.to_dict("records"), start=1):
            a, b = (
                (row["ndc_a"], row["ndc_b"])
                if kind == "identical_different"
                else (row["product_ndc_a"], row["product_ndc_b"])
            )
            if a not in products or b not in products or a == b:
                raise ValueError(f"Invalid candidate NDCs in {filename} row {source_row}")
            shared = set(json.loads(products[a]["rxnorm_in_rxcuis_json"])) & set(
                json.loads(products[b]["rxnorm_in_rxcuis_json"])
            )
            pairs.append(
                {
                    "case_type": kind,
                    "product_ndc_a": a,
                    "product_ndc_b": b,
                    "ingredient_overlap": "shared" if shared else "disjoint",
                    "candidate_source_csv": filename,
                    "candidate_source_row": source_row,
                }
            )

    # Control links extend review blocks without merging confirmed family labels
    links: dict[str, set[str]] = defaultdict(set)
    name_to_audit: dict[str, set[str]] = defaultdict(set)
    inherited_flags: dict[str, set[str]] = defaultdict(set)
    for row in families.itertuples():
        names = json.loads(row.normalized_names_json)
        for name in names:
            links[names[0]].add(name)
            links[name].add(names[0])
            name_to_audit[name].add(row.review_group_id)
        inherited_flags[row.review_group_id].update(json.loads(row.review_reasons_json))
    for pair in pairs:
        left = products[pair["product_ndc_a"]]["normalized_name"]
        right = products[pair["product_ndc_b"]]["normalized_name"]
        if not left or not right:
            raise ValueError("Candidate product has no normalized name")
        links[left].add(right)
        links[right].add(left)
    aliases: dict[str, set[str]] = defaultdict(set)
    for name in list(links):
        aliases[_alias_key(name)].add(name)
    for key, names in aliases.items():
        if key:
            for name in names:
                links[min(names)].add(name)
                links[name].add(min(names))

    group_names = _components(links)
    name_to_group = {name: index for index, names in enumerate(group_names) for name in names}
    reviewed_keys = {
        tuple(sorted((row.product_ndc_a, row.product_ndc_b))) for row in reviewed.itertuples()
    }
    pairs_by_group: dict[int, list[dict]] = defaultdict(list)
    excluded_pairs = 0
    seen_pairs = set()
    for pair in sorted(
        pairs, key=lambda pair: (pair["case_type"], pair["product_ndc_a"], pair["product_ndc_b"])
    ):
        key = tuple(sorted((pair["product_ndc_a"], pair["product_ndc_b"])))
        if key in reviewed_keys:
            excluded_pairs += 1
            continue
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        index = name_to_group[products[pair["product_ndc_a"]]["normalized_name"]]
        pairs_by_group[index].append(pair)
    ndcs_by_name: dict[str, list[str]] = defaultdict(list)
    generic_ndcs = set()
    for ndc, product in products.items():
        ndcs_by_name[product["normalized_name"]].append(ndc)
        if _generic_like_product(product):
            generic_ndcs.add(ndc)

    groups = []
    for index, names in enumerate(group_names):
        if not pairs_by_group[index]:
            continue
        ndcs = sorted({ndc for name in names for ndc in ndcs_by_name[name]})
        existing_ids = sorted({group_id for name in names for group_id in name_to_audit[name]})
        reasons = {flag for group_id in existing_ids for flag in inherited_flags[group_id]}
        flags = set()
        if "possible_punctuation_or_ampersand_alias" in reasons or any(
            len(aliases[_alias_key(name)]) > 1 for name in names
        ):
            flags.add("possible_family_alias")
        if set(ndcs) & generic_ndcs:
            flags.add("generic_or_ingredient_like_names")
        disagreement_ndcs = [
            ndc
            for ndc in ndcs
            if products[ndc]["agreement_status"]
            in {"review_no_exact_name_overlap", "review_partial_name_overlap"}
        ]
        if disagreement_ndcs:
            flags.add("ingredient_disagreement")
        unavailable_ndcs = [
            ndc
            for ndc in ndcs
            if products[ndc]["agreement_status"] in {"rxnorm_unavailable", "fda_unavailable"}
        ]
        if unavailable_ndcs:
            flags.add("ingredient_mapping_unavailable")
        assigned = set(ndcs) & set(assignments)
        if assigned and len(assigned) < len(ndcs):
            flags.add("partially_assigned_family_labels")
        if any(
            len(
                {
                    (
                        ingredient_signature(json.loads(products[ndc]["active_ingredients_json"])),
                        normalize_name(products[ndc]["dosage_form"]),
                        products[ndc]["route"],
                    )
                    for ndc in ndcs_by_name[name]
                }
            )
            > 1
            for name in names
        ):
            flags.add("potentially_ambiguous_name")
        if len(existing_ids) > 1:
            flags.add("candidate_links_across_existing_review_groups")
        existing_labels = sorted({label for ndc in assigned for label in assignments[ndc]})
        if len({normalize_name(label) for label in existing_labels}) > 1:
            flags.add("multiple_existing_family_labels")
        supporting_ndcs = []
        for candidates in (
            disagreement_ndcs,
            unavailable_ndcs,
            [ndc for ndc in ndcs if len(aliases[_alias_key(products[ndc]["normalized_name"])]) > 1],
            sorted(set(ndcs) & generic_ndcs),
            sorted(set(ndcs) - assigned),
        ):
            if candidates and candidates[0] not in supporting_ndcs:
                supporting_ndcs.append(candidates[0])
        groups.append(
            {
                "review_group_id": "group_" + sha256("\n".join(names).encode()).hexdigest()[:12],
                "review_group_name": names[0],
                "existing_group_ids_json": existing_ids,
                "flags": sorted(flags),
                "inherited_review_flags": sorted(reasons),
                "pairs": pairs_by_group[index],
                "supporting_ndcs": supporting_ndcs,
                "existing_family_labels_json": existing_labels,
                "original_reviews_json": [
                    row
                    for row in reviewed.to_dict("records")
                    if row["product_ndc_a"] in ndcs or row["product_ndc_b"] in ndcs
                ],
            }
        )

    def source_references(ndc: str) -> dict:
        product = products[ndc]
        label = label_links.get(ndc, {})
        return {
            "product_ndc": ndc,
            "catalog_csv": catalog_path.name,
            "catalog_source_rows": product.get("source_rows", product.get("source_row", "")),
            "ingredient_agreement_csv": agreement_path.name,
            "rxnorm_in_rxcuis": json.loads(product["rxnorm_in_rxcuis_json"]),
            "label_links_csv": label_links_path.name if label else "",
            "fda_label_link": {
                key: label[key]
                for key in ("link_status", "label_id", "label_set_id", "effective_time")
                if key in label
            },
        }

    selected = _select_shortlist(groups, max_groups)
    rows = []
    for rank, selection in enumerate(selected, start=1):
        group, pair = selection["group"], selection["pair"]
        row = {
            "selection_rank": rank,
            "review_group_id": group["review_group_id"],
            "review_group_name": group["review_group_name"],
            "existing_group_ids_json": group["existing_group_ids_json"],
            **pair,
            "selection_category": selection["selection_category"],
            "selection_reason": f"Round-robin category {selection['selection_category']}. "
            "Review need coverage, case balance, then stable group hash. Pair count is not used.",
            "review_flags_json": group["flags"],
            "inherited_review_flags_json": group["inherited_review_flags"],
            "existing_family_labels_json": group["existing_family_labels_json"],
            "original_reviews_json": group["original_reviews_json"],
            "supporting_products_json": [
                {
                    "product_ndc": ndc,
                    "name": products[ndc]["display_name"],
                    "fda_ingredients": json.loads(products[ndc]["active_ingredients_json"]),
                    "agreement_status": products[ndc]["agreement_status"],
                    "existing_family_labels": assignments.get(ndc, []),
                    "source_references": source_references(ndc),
                }
                for ndc in group["supporting_ndcs"]
            ],
            "reviewer_decision": "",
            "confirmed_family_id": "",
            "reviewer_notes": "",
        }
        for side in ("a", "b"):
            ndc = pair[f"product_ndc_{side}"]
            product = products[ndc]
            row[f"name_{side}"] = product["display_name"]
            row[f"fda_ingredients_{side}_json"] = json.loads(product["active_ingredients_json"])
            row[f"rxnorm_ingredient_names_{side}_json"] = json.loads(
                product["rxnorm_in_names_json"]
            )
            row[f"agreement_status_{side}"] = product["agreement_status"]
            row[f"source_references_{side}_json"] = source_references(ndc)
        rows.append(row)
    output_path = audit_dir / "family_review_shortlist.csv"
    reviewer_fields = ["reviewer_decision", "confirmed_family_id", "reviewer_notes"]
    if output_path.exists():
        previous = _read_csv(output_path, set(reviewer_fields))
        if previous[reviewer_fields].ne("").any().any():
            raise ValueError("Shortlist has reviewer entries. Preserve it before regenerating.")
    columns = [
        "selection_rank",
        "review_group_id",
        "review_group_name",
        "existing_group_ids_json",
        "selection_category",
        "selection_reason",
        "case_type",
        "ingredient_overlap",
        "product_ndc_a",
        "product_ndc_b",
        "name_a",
        "name_b",
        "fda_ingredients_a_json",
        "fda_ingredients_b_json",
        "rxnorm_ingredient_names_a_json",
        "rxnorm_ingredient_names_b_json",
        "agreement_status_a",
        "agreement_status_b",
        "source_references_a_json",
        "source_references_b_json",
        "review_flags_json",
        "inherited_review_flags_json",
        "existing_family_labels_json",
        "original_reviews_json",
        "supporting_products_json",
        "candidate_source_csv",
        "candidate_source_row",
        *reviewer_fields,
    ]
    pd.DataFrame(
        [
            {
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if key.endswith("_json")
                else value
                for key, value in row.items()
            }
            for row in rows
        ],
        columns=columns,
    ).to_csv(output_path, index=False)
    return {
        "shortlist_size": len(rows),
        "eligible_review_groups": len(groups),
        "candidate_records_considered": len(pairs),
        "previously_reviewed_pairs_excluded": excluded_pairs,
        "representatives_by_case_type": dict(
            sorted(Counter(row["case_type"] for row in rows).items())
        ),
        "representatives_by_ingredient_overlap": dict(
            sorted(Counter(row["ingredient_overlap"] for row in rows).items())
        ),
        "representatives_by_case_and_overlap": {
            kind: dict(
                sorted(
                    Counter(
                        row["ingredient_overlap"] for row in rows if row["case_type"] == kind
                    ).items()
                )
            )
            for kind in sorted({row["case_type"] for row in rows})
        },
        "selected_groups_by_review_flag": dict(
            sorted(Counter(flag for row in rows for flag in row["review_flags_json"]).items())
        ),
        "eligible_groups_by_review_flag": dict(
            sorted(Counter(flag for group in groups for flag in group["flags"]).items())
        ),
        "selection_category_counts": dict(
            sorted(Counter(row["selection_category"] for row in rows).items())
        ),
        "original_review_counts": review_counts["saved_counts"],
        "preserved_manual_family_labels": len(
            {normalize_name(label) for labels in assignments.values() for label in labels}
        ),
        "selection_policy": "Round-robin review flags and case types, prefer uncovered review "
        "needs, balance case and overlap categories, then a stable group hash. "
        "Exclude reviewed pairs. No pair-count weight.",
        "family_membership_confirmed": False,
        "splits_created": False,
        "benchmark_examples_created": False,
    }


def reconcile_family_reviews(seeds: pd.DataFrame, reviewed: pd.DataFrame) -> dict:
    """Retain literal family assignments and quarantine unresolved evidence."""
    reconcile_reviews(seeds)
    required = {
        "review_rank",
        "product_ndc_a",
        "product_ndc_b",
        "review_decision",
        "family_id_a",
        "family_id_b",
        "labels_verified",
        "reviewer_notes",
    }
    if missing := required - set(reviewed.columns):
        raise ValueError(f"Missing family review columns: {sorted(missing)}")
    assignments = {ndc: set(labels) for ndc, labels in reviewed_family_assignments(seeds).items()}
    records = {}
    errors = []
    source_holds = set()
    identity_holds = set()
    followups = []
    for row in seeds.to_dict("records"):
        key = tuple(sorted((row["product_ndc_a"], row["product_ndc_b"])))
        records[key] = {"decision": row["review_decision"], "source": "seed"}
        if row["review_decision"] == "VERIFIED" and "remains to be confirmed" in (
            row["review_notes"].casefold()
        ):
            identity_holds.update(key)
            followups.append(
                {
                    "family_label": row["verified_family"],
                    "product_ndcs": list(key),
                    "notes": row["review_notes"],
                }
            )
    seen_ranks = set()
    accepted = []
    for row in reviewed.to_dict("records"):
        key = tuple(sorted((row["product_ndc_a"], row["product_ndc_b"])))
        decision = row["review_decision"]
        labels = [row["family_id_a"], row["family_id_b"]]
        issues = []
        if decision not in {"SAME_FAMILY", "DIFFERENT_FAMILIES", "EXCLUDE", "UNCERTAIN"}:
            issues.append("invalid_decision")
        if not all(key) or key[0] == key[1]:
            issues.append("invalid_product_pair")
        if key in records or row["review_rank"] in seen_ranks:
            issues.append("duplicate_pair_or_rank")
        seen_ranks.add(row["review_rank"])
        if row["labels_verified"] not in {"TRUE", "FALSE", ""}:
            issues.append("invalid_source_check_flag")
        if decision in {"SAME_FAMILY", "DIFFERENT_FAMILIES"}:
            if not all(map(normalize_name, labels)):
                issues.append("missing_family_id")
            elif (normalize_name(labels[0]) == normalize_name(labels[1])) != (
                decision == "SAME_FAMILY"
            ):
                issues.append("decision_and_family_ids_disagree")
        if issues:
            errors.append({"review_rank": row["review_rank"], "issues": issues})
            records[key] = {"decision": "INVALID", "source": "new"}
            continue
        records[key] = {"decision": decision, "source": "new"}
        if decision not in {"SAME_FAMILY", "DIFFERENT_FAMILIES"}:
            continue
        for side, label in zip(("a", "b"), labels, strict=True):
            assignments.setdefault(row[f"product_ndc_{side}"], set()).add(label)
        accepted.append(row)
        # A positive source flag is not independent annotation or a gold label
        if row["labels_verified"] != "TRUE":
            source_holds.update(key)
    conflicts = {
        ndc: sorted(labels)
        for ndc, labels in assignments.items()
        if len({normalize_name(label) for label in labels}) > 1
    }
    literal_labels = {}
    for labels in assignments.values():
        for label in sorted(labels):
            literal_labels.setdefault(normalize_name(label), label)
    aliases = []
    alias_holds = set()
    keys = sorted(literal_labels)
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            first, second = _alias_key(left), _alias_key(right)
            if (
                first
                and second
                and (
                    first == second
                    or first.startswith(second + " ")
                    or second.startswith(first + " ")
                )
            ):
                related = {left, right}
                ndcs = sorted(
                    ndc
                    for ndc, labels in assignments.items()
                    if {normalize_name(label) for label in labels} & related
                )
                alias_holds.update(ndcs)
                aliases.append(
                    {
                        "family_labels": [literal_labels[left], literal_labels[right]],
                        "product_ndcs": ndcs,
                        "status": "POSSIBLE_RELATION_REQUIRES_REVIEW",
                        "action": "Confirm an alias or explicit co-partition policy. "
                        "Original family labels remain separate.",
                    }
                )
    return {
        "assignments": {ndc: sorted(labels) for ndc, labels in sorted(assignments.items())},
        "family_labels": literal_labels,
        "records": records,
        "errors": errors,
        "source_holds": source_holds,
        "identity_holds": identity_holds,
        "alias_holds": alias_holds,
        "conflicts": conflicts,
        "aliases": aliases,
        "seed_identity_followups": followups,
        "accepted_new_rows": accepted,
    }


def partition_families(families: list[str], seed: int = 42) -> dict[str, str]:
    """Allocate whole families with deterministic largest-remainder rounding."""
    ordered = sorted(set(families))
    random.Random(seed).shuffle(ordered)
    targets = [len(ordered) * ratio for ratio in (0.70, 0.15, 0.15)]
    sizes = [math.floor(target) for target in targets]
    for index in sorted(range(3), key=lambda i: (-(targets[i] - sizes[i]), i))[
        : len(ordered) - sum(sizes)
    ]:
        sizes[index] += 1
    partitions = {}
    start = 0
    for name, size in zip(("train", "validation", "test"), sizes, strict=True):
        partitions.update(dict.fromkeys(ordered[start : start + size], name))
        start += size
    return partitions


def _candidate_pool(interim_dir: Path) -> pd.DataFrame:
    """Read every saved candidate without regenerating product pairs."""
    frames = []
    for filename, kind in (
        ("identical_name_candidate_pairs.csv", "identical_different"),
        ("similar_name_candidate_pairs.csv", "similar_different"),
        ("similar_name_same_ingredient_controls.csv", "similar_same_control"),
    ):
        identical = kind == "identical_different"
        frame = _read_csv(
            interim_dir / filename,
            {"ndc_a", "ndc_b"} if identical else {"product_ndc_a", "product_ndc_b", "case_type"},
        )
        if identical:
            frame = frame.rename(columns={"ndc_a": "product_ndc_a", "ndc_b": "product_ndc_b"})
            frame["case_type"] = kind
        elif set(frame.case_type) - {kind}:
            raise ValueError(f"Unexpected candidate type in {filename}")
        frame["candidate_source_csv"] = filename
        frame["candidate_source_row"] = range(1, len(frame) + 1)
        frames.append(
            frame[
                [
                    "case_type",
                    "product_ndc_a",
                    "product_ndc_b",
                    "candidate_source_csv",
                    "candidate_source_row",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def assess_split_readiness(
    project_root: Path, min_heldout_families: int = 2, write_outputs: bool = True
) -> dict:
    """Assess the reviewed cohort and freeze partitions only when checks pass."""
    if min_heldout_families < 2:
        raise ValueError("Held-out checks require at least two families per category")
    choices = [
        project_root / "results" / name
        for name in ("part03", "part_03")
        if (project_root / "results" / name / "family_review_decisions_reviewed.csv").is_file()
    ]
    if len(choices) != 1:
        raise ValueError("Expected one reviewed decisions file in part03 or part_03")
    audit_dir = choices[0]
    interim = project_root / "data/interim"
    reviewed_path = audit_dir / "family_review_decisions_reviewed.csv"
    seeds = _read_csv(interim / "feasibility_review_queue_reviewed.csv", {"verified_family"})
    reviewed = _read_csv(reviewed_path, {"review_rank"})
    reconciled = reconcile_family_reviews(seeds, reviewed)
    audit = json.loads((audit_dir / "family_audit.json").read_text(encoding="utf-8"))
    baseline = _read_csv(audit_dir / "family_review_decisions.csv", {"review_rank"})
    editable = {
        "review_decision",
        "family_id_a",
        "family_id_b",
        "labels_verified",
        "reviewer_notes",
    }
    immutable = [column for column in baseline.columns if column not in editable]
    baseline_matches = (
        not baseline.review_rank.duplicated().any()
        and not reviewed.review_rank.duplicated().any()
        and set(immutable).issubset(reviewed.columns)
        and baseline[immutable]
        .set_index("review_rank")
        .sort_index()
        .equals(reviewed[immutable].set_index("review_rank").sort_index())
    )
    stale_inputs = [
        name
        for name, profile in audit["inputs"].items()
        if sha256((interim / name).read_bytes()).hexdigest() != profile["sha256"]
    ]
    catalog = _read_csv(
        interim / "otc_catalog_consistent.csv",
        {
            "product_ndc",
            "normalized_name",
            "display_name",
            "name_source",
            "generic_name",
            "active_ingredients_json",
            "dosage_form",
            "route",
        },
    )
    agreement = _read_csv(
        interim / "otc_ingredient_agreement.csv",
        {
            "product_ndc",
            "catalog_status",
            "agreement_status",
            "rxnorm_ingredient_status",
            "rxnorm_in_rxcuis_json",
            "rxnorm_in_names_json",
            "rxnorm_pin_names_json",
            "label_text_available",
        },
    )
    catalog = catalog.merge(agreement, on="product_ndc", how="left", validate="one_to_one")
    products = catalog.set_index("product_ndc").to_dict("index")
    eligible = set(
        catalog.loc[
            catalog.catalog_status.eq("consistent")
            & catalog.agreement_status.isin(EXACT_AGREEMENT)
            & catalog.rxnorm_ingredient_status.eq("resolved_path"),
            "product_ndc",
        ]
    )
    assignments = reconciled["assignments"]
    unknown_ndcs = sorted(set(assignments) - set(products))
    family_by_ndc = {
        ndc: normalize_name(labels[0])
        for ndc, labels in assignments.items()
        if ndc not in reconciled["conflicts"]
    }
    held = (
        reconciled["source_holds"]
        | reconciled["identity_holds"]
        | reconciled["alias_holds"]
        | set(reconciled["conflicts"])
    )
    available = (set(family_by_ndc) & eligible) - held
    pool = _candidate_pool(interim)
    pool["pair_key"] = [
        tuple(sorted((row.product_ndc_a, row.product_ndc_b))) for row in pool.itertuples()
    ]
    records = reconciled["records"]
    pool_keys = set(pool.pair_key)
    missing_review_pairs = sorted(set(records) - pool_keys)
    statuses = []
    for row in pool.itertuples():
        record = records.get(row.pair_key)
        endpoints = set(row.pair_key)
        if record is None:
            status = "unreviewed"
        elif record["decision"] in {"UNCERTAIN", "PENDING"}:
            status = "uncertain_review"
        elif record["decision"] in {"REJECTED", "EXCLUDE"}:
            status = "excluded_review"
        elif record["decision"] == "INVALID" or not endpoints.issubset(family_by_ndc):
            status = "assignment_hold"
        elif not endpoints.issubset(eligible):
            status = "eligibility_hold"
        elif endpoints & reconciled["alias_holds"]:
            status = "alias_hold"
        elif endpoints & reconciled["identity_holds"]:
            status = "identity_hold"
        elif endpoints & reconciled["source_holds"]:
            status = "reference_hold"
        else:
            status = "usable_reviewed"
        statuses.append(status)
    pool["review_status"] = statuses
    pool["same_assigned_family"] = [
        row.product_ndc_a in available
        and row.product_ndc_b in available
        and family_by_ndc[row.product_ndc_a] == family_by_ndc[row.product_ndc_b]
        for row in pool.itertuples()
    ]
    core = pool.loc[pool.review_status.eq("usable_reviewed") & pool.same_assigned_family]
    core_ndcs = set(core.product_ndc_a) | set(core.product_ndc_b)
    core_families = sorted({family_by_ndc[ndc] for ndc in core_ndcs})
    preview = partition_families(core_families)
    category_counts = {}
    for kind, cases in pool.groupby("case_type", sort=True):
        usable = core.loc[core.case_type.eq(kind)]
        families = {family_by_ndc[ndc] for ndc in usable.product_ndc_a}
        heldout = {
            part: sum(preview[family] == part for family in families)
            for part in ("train", "validation", "test")
        }
        category_counts[kind] = {
            "candidate_records": len(cases),
            "review_status_counts": dict(sorted(Counter(cases.review_status).items())),
            "usable_within_family_reviewed_pairs": len(usable),
            "independent_usable_families": len(families),
            "preview_families_by_partition": heldout,
            "sufficient_heldout_family_coverage": all(
                heldout[part] >= min_heldout_families for part in ("validation", "test")
            )
            and heldout["train"] > 0,
        }
    blockers = []
    for condition, message in (
        (bool(reconciled["errors"]), "Invalid reviewed rows require correction."),
        (
            not baseline_matches,
            "Reviewed product evidence differs from the original decision sheet.",
        ),
        (bool(stale_inputs), "Part 3.1 input fingerprints changed."),
        (
            bool(unknown_ndcs or missing_review_pairs),
            "Reviewed identifiers are absent from source inputs.",
        ),
        (bool(reconciled["conflicts"]), "Some products have conflicting reviewed family IDs."),
        (bool(reconciled["aliases"]), "Possible family aliases require explicit reconciliation."),
        (
            bool(reconciled["source_holds"]),
            "Accepted families still contain held ingredient-strength references.",
        ),
        (
            bool(reconciled["identity_holds"]),
            "A seed review explicitly retains an identity follow-up.",
        ),
        (
            any(
                not value["sufficient_heldout_family_coverage"]
                for value in category_counts.values()
            ),
            "The seed-42 preview has insufficient independent families in held-out categories.",
        ),
        (len(category_counts) != 3, "A required candidate category is missing."),
    ):
        if condition:
            blockers.append(message)

    # Review blocks diversify the expansion queue and never assign a family
    links: dict[str, set[str]] = defaultdict(set)
    for row in pool.itertuples():
        if row.product_ndc_a in products and row.product_ndc_b in products:
            left = products[row.product_ndc_a]["normalized_name"]
            right = products[row.product_ndc_b]["normalized_name"]
            links[left].add(right)
            links[right].add(left)
    block_by_name = {name: names[0] for names in _components(links) for name in names}
    reviewed_blocks = {
        block_by_name[products[ndc]["normalized_name"]]
        for key in records
        for ndc in key
        if ndc in products and products[ndc]["normalized_name"] in block_by_name
    }
    expansion = []
    used_blocks = set(reviewed_blocks)
    for kind, overlap, count in (
        ("similar_different", "shared", 3),
        ("similar_different", "disjoint", 3),
        ("similar_same_control", "shared", 3),
        ("identical_different", "shared", 2),
        ("identical_different", "disjoint", 1),
    ):
        candidates = pool.loc[pool.case_type.eq(kind) & pool.review_status.eq("unreviewed")]
        candidates = sorted(
            candidates.to_dict("records"),
            key=lambda row: sha256(("42|" + "|".join(row["pair_key"])).encode()).hexdigest(),
        )
        selected_count = 0
        for row in candidates:
            a, b = row["product_ndc_a"], row["product_ndc_b"]
            if not {a, b}.issubset(eligible) or {a, b} & set(assignments):
                continue
            if a.split("-")[0] != b.split("-")[0]:
                continue
            left, right = products[a], products[b]
            block = block_by_name[left["normalized_name"]]
            shared = bool(
                set(json.loads(left["rxnorm_in_rxcuis_json"]))
                & set(json.loads(right["rxnorm_in_rxcuis_json"]))
            )
            if (
                block in used_blocks
                or ("shared" if shared else "disjoint") != overlap
                or _generic_like_product(left)
                or _generic_like_product(right)
                or left["label_text_available"] != "True"
                or right["label_text_available"] != "True"
            ):
                continue
            expansion.append(
                {
                    "candidate_source_csv": row["candidate_source_csv"],
                    "candidate_source_row": row["candidate_source_row"],
                    "case_type": kind,
                    "ingredient_overlap": overlap,
                    "product_ndc_a": a,
                    "product_ndc_b": b,
                    "name_a": left["display_name"],
                    "name_b": right["display_name"],
                    "rxnorm_ingredients_a": json.loads(left["rxnorm_in_names_json"]),
                    "rxnorm_ingredients_b": json.loads(right["rxnorm_in_names_json"]),
                    "reason": "Review a new candidate block to expand "
                    + kind
                    + " family coverage. Same NDC labeler-code prefix reduces cross-labeler "
                    "ambiguity but does not confirm a family. Block independence is provisional.",
                }
            )
            used_blocks.add(block)
            selected_count += 1
            if selected_count == count:
                break
    ready = not blockers
    manifest = pd.DataFrame(
        [
            {
                "product_ndc": ndc,
                "family_id": reconciled["family_labels"][family_by_ndc[ndc]],
                "partition": preview[family_by_ndc[ndc]],
                "seed": 42,
            }
            for ndc in sorted(core_ndcs)
        ],
        columns=["product_ndc", "family_id", "partition", "seed"],
    )
    manifest_path = audit_dir / "family_split_manifest.csv"
    preview_checks = {
        "family_disjoint": bool(manifest.groupby("family_id").partition.nunique().eq(1).all()),
        "product_ndc_disjoint": not manifest.product_ndc.duplicated().any().item(),
        "reviewed_core_pairs_within_partition": all(
            preview[family_by_ndc[row.product_ndc_a]] == preview[family_by_ndc[row.product_ndc_b]]
            for row in core.itertuples()
        ),
        "unresolved_alias_products_excluded": not bool(core_ndcs & reconciled["alias_holds"]),
    }
    if ready and write_outputs:
        assert all(preview_checks.values())
        text = manifest.to_csv(index=False)
        if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != text:
            raise ValueError("An existing manifest differs. Keep frozen partitions unchanged.")
        manifest_path.write_text(text, encoding="utf-8", newline="")
    summary = {
        "stage": "3.2_family_split_readiness",
        "readiness": "READY" if ready else "NOT_READY",
        "blockers": blockers,
        "review_validation": {
            "reviewed_file": str(reviewed_path.relative_to(project_root)),
            "original_seed_decisions": reconcile_reviews(seeds)["saved_counts"],
            "new_review_decisions": dict(sorted(Counter(reviewed.review_decision).items())),
            "labels_verified_flags": dict(sorted(Counter(reviewed.labels_verified).items())),
            "labels_verified_meaning": "Labels were checked for consistency with source records. It does not establish a standard independent human annotation.",
            "independent_annotation_provenance": "Not established by the saved review schemas.",
            "invalid_rows": reconciled["errors"],
            "source_snapshot_matches": baseline_matches,
            "stale_audit_inputs": stale_inputs,
            "missing_product_ndcs": unknown_ndcs,
            "reviewed_pairs_missing_from_pool": [list(key) for key in missing_review_pairs],
        },
        "family_reconciliation": {
            "original_seed_families": len(
                {
                    normalize_name(label)
                    for labels in reviewed_family_assignments(seeds).values()
                    for label in labels
                }
            ),
            "accepted_family_labels": len(reconciled["family_labels"]),
            "explicitly_assigned_products": len(assignments),
            "assignments": assignments,
            "conflicting_product_assignments": reconciled["conflicts"],
            "possible_aliases": reconciled["aliases"],
            "source_hold_product_ndcs": sorted(reconciled["source_holds"]),
            "source_hold_review_rows": [
                {
                    key: row[key]
                    for key in ("review_rank", "family_id_a", "family_id_b", "reviewer_notes")
                }
                for row in reconciled["accepted_new_rows"]
                if row["labels_verified"] != "TRUE"
            ],
            "seed_identity_followups": reconciled["seed_identity_followups"],
            "available_assigned_products_after_holds": len(available),
            "usable_core_families": len(core_families),
            "usable_core_products": len(core_ndcs),
            "usable_definition": "Reviewed, source-checked within-family candidate pairs after identity, "
            "reference and alias quarantine. This is planning eligibility, not final gold annotation.",
            "unpaired_or_auxiliary_assigned_products": sorted(available - core_ndcs),
            "provisional_groups_used_as_verified_families": False,
            "provisional_review_groups": audit["families"]["candidate_review_groups"],
        },
        "full_pool_coverage": {
            "catalog_products": len(catalog),
            "eligible_catalog_products": len(eligible),
            "candidate_records": len(pool),
            "unique_candidate_pairs": len(pool_keys),
            "candidate_products": len(set(pool.product_ndc_a) | set(pool.product_ndc_b)),
            "candidate_records_outside_eligibility": sum(
                not set(key).issubset(eligible) for key in pool.pair_key
            ),
            "review_status_counts": dict(sorted(Counter(pool.review_status).items())),
            "reviewed_candidate_records": int(pool.review_status.ne("unreviewed").sum()),
            "reviewed_cases_on_hold_or_uncertain": int(
                pool.review_status.isin(
                    {
                        "uncertain_review",
                        "alias_hold",
                        "identity_hold",
                        "reference_hold",
                        "assignment_hold",
                        "eligibility_hold",
                    }
                ).sum()
            ),
            "reviewed_source_checked_pairs": int(pool.review_status.eq("usable_reviewed").sum()),
            "usable_within_family_reviewed_pairs": len(core),
            "source_checked_cross_family_pairs": int(
                (pool.review_status.eq("usable_reviewed") & ~pool.same_assigned_family).sum()
            ),
            "unreviewed_pairs_with_usable_endpoint_assignments": int(
                (pool.review_status.eq("unreviewed") & pool.same_assigned_family).sum()
            ),
            "categories": category_counts,
            "verified_benchmark_examples_created": 0,
            "scale_assessment": "This remains a reviewed pilot. The plan targets several hundred "
            "verified pairs and does not specify a fixed numerical cutoff.",
        },
        "split_policy": {
            "seed": 42,
            "target_family_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "rounding": "Largest remainder, ties resolved train then validation then test.",
            "minimum_heldout_families_per_category": min_heldout_families,
            "minimum_rationale": "Conservative audit floor to avoid single-family category evaluation. "
            "It is not a power calculation or a sample-size requirement from the plan.",
            "preview_only": not ready,
            "preview_family_counts": dict(sorted(Counter(preview.values()).items())),
            "preview_checks": preview_checks,
            "aliases_merged": False,
            "manifest_created": ready,
            "existing_manifest_present": manifest_path.exists(),
            "future_pair_rule": "Construct a pair only when both NDCs belong to the same partition.",
            "heldout_rule": "Freeze family membership before examples, label-rule development or model tuning. "
            "Do not use held-out labels to choose rules or hyperparameters.",
        },
        "targeted_expansion": {
            "first_actions": "Resolve the possible aliases, accepted reference holds and seed "
            "identity follow-ups listed above without overwriting assignments.",
            "recommended_candidate_records": expansion,
            "batch_size": len(expansion),
            "scope": "A first review batch from existing records, not a guarantee of final benchmark sufficiency.",
        },
        "inputs": {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in [
                reviewed_path,
                interim / "feasibility_review_queue_reviewed.csv",
                audit_dir / "family_audit.json",
                interim / "otc_catalog_consistent.csv",
                interim / "otc_ingredient_agreement.csv",
                *[interim / filename for filename in pool.candidate_source_csv.unique()],
            ]
        },
    }
    if write_outputs:
        (audit_dir / "split_readiness.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return summary


def _quantity_key(value: str) -> tuple[float, str] | None:
    """Compare native quantities without salt conversion or density assumptions."""
    match = re.fullmatch(
        r"([\d.]+)\s*(mg|mcg|ug|g|ml)(?:\s*/\s*([\d.]*)\s*(mg|g|ml|l|1))?",
        normalize_name(value),
    )
    if not match:
        return None
    amount, unit, denominator, denominator_unit = match.groups()
    factors = {"mg": 1, "g": 1000, "ug": 0.001, "mcg": 0.001, "ml": 1, "l": 1000, "1": 1}
    denominator_unit = denominator_unit or "1"
    denominator_value = float(denominator or 1) * factors[denominator_unit]
    if denominator_value <= 0:
        return None
    numerator_kind = "volume" if unit == "ml" else "mass"
    denominator_kind = (
        "volume"
        if denominator_unit in {"ml", "l"}
        else "mass"
        if denominator_unit in {"mg", "g"}
        else "item"
    )
    return float(amount) * factors[
        unit
    ] / denominator_value, numerator_kind + "/" + denominator_kind


def source_training_risks(product: dict, drug: dict, label: dict) -> list[str]:
    """Screen candidate records conservatively without declaring a gold label."""
    risks = set()
    if (
        product.get("catalog_status") != "consistent"
        or product.get("agreement_status") not in EXACT_AGREEMENT
        or product.get("rxnorm_ingredient_status") != "resolved_path"
        or drug.get("rxnorm_status") != "matched"
    ):
        risks.add("ingredient_or_product_mapping_unresolved")
    if (
        label.get("link_status") != "matched"
        or not label.get("label_id")
        or not label.get("label_set_id")
        or not json.loads(label.get("active_ingredient_json") or "[]")
    ):
        risks.add("label_reference_incomplete")
    drug_name = normalize_name(drug.get("rxnorm_drug_name", ""))
    for ingredient in json.loads(product["active_ingredients_json"]):
        name = re.escape(normalize_name(ingredient["name"]))
        match = re.search(
            r"(?<!\w)" + name + r"\s+([\d.]+)\s+(mg|mcg|ug|g|ml)(?:/(mg|g|ml))?(?![a-z/])",
            drug_name,
        )
        native = _quantity_key(ingredient["strength"])
        mapped = (
            _quantity_key(match[1] + " " + match[2] + ("/" + match[3] if match[3] else ""))
            if match
            else None
        )
        if (
            native is None
            or mapped is None
            or native[1] != mapped[1]
            or not math.isclose(native[0], mapped[0], rel_tol=1e-6)
        ):
            risks.add("strength_or_basis_requires_review")
    routes = {normalize_name(value) for value in json.loads(product["route"])}
    for term, allowed in {
        "rectal": {"rectal"},
        "vaginal": {"vaginal"},
        "nasal": {"nasal"},
        "ophthalmic": {"ophthalmic"},
        "otic": {"otic"},
        "oral": {"oral"},
        "topical": {"topical", "cutaneous", "percutaneous", "transdermal"},
    }.items():
        if re.search(r"\b" + term + r"\b", drug_name) and not routes & allowed:
            risks.add("route_or_form_requires_review")
    dosage_form = normalize_name(product["dosage_form"])
    for term, allowed in {
        "tablet": {"tablet"},
        "capsule": {"capsule"},
        "cream": {"cream"},
        "ointment": {"ointment"},
        "gel": {"gel"},
        "patch": {"patch"},
        "spray": {"spray", "aerosol"},
        "solution": {"solution"},
        "suspension": {"suspension"},
    }.items():
        if re.search(r"\b" + term + r"\b", drug_name) and not any(
            form in dosage_form for form in allowed
        ):
            risks.add("route_or_form_requires_review")
    return sorted(risks)


def coverage_split_preview(
    families: dict[str, str],
    cases: list[dict],
    co_location: list[list[str]],
    seed: int = 42,
    *,
    balance_coverage: bool = True,
) -> dict:
    """Preview whole-family partitions and audit coverage before freezing."""
    links = {family: {family} for family in families.values()}
    for constraint in co_location:
        members = sorted(set(constraint) & set(links))
        for family in members:
            links[members[0]].add(family)
            links[family].add(members[0])
    components = _components(links)
    family_group = {family: members[0] for members in components for family in members}
    groups = sorted(set(family_group.values()))
    features: dict[str, set[str]] = {group: set() for group in groups}
    for row in cases:
        a, b = row["product_ndc_a"], row["product_ndc_b"]
        if (
            a in families
            and b in families
            and family_group[families[a]] == family_group[families[b]]
        ):
            features[family_group[families[a]]].update(
                {
                    "category:" + row["case_type"],
                    row["case_type"] + ":" + row["ingredient_overlap"],
                }
            )
    frequency = Counter(feature for values in features.values() for feature in values)
    required = {
        feature: 2 if feature.startswith("category:") else 1
        for feature, count in sorted(frequency.items())
        if feature.startswith("category:") or count >= 2
    }
    capacities = Counter(partition_families(groups, seed).values())
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    rank = {group: index for index, group in enumerate(shuffled)}
    # Fixed allocations do not use case categories or ingredient overlap
    assigned = {} if balance_coverage else partition_families(groups, seed)
    covered = {part: Counter() for part in ("validation", "test")}
    for _ in range(max(capacities.get("validation", 0), capacities.get("test", 0))):
        for part in ("test", "validation"):
            if list(assigned.values()).count(part) >= capacities.get(part, 0):
                continue

            def priority(group: str, partition: str = part) -> tuple:
                missing = sorted(
                    feature
                    for feature in features[group]
                    if covered[partition][feature] < required.get(feature, 0)
                )
                return (
                    -len(missing),
                    -sum(1 / frequency[feature] for feature in missing),
                    rank[group],
                )

            group = min((group for group in groups if group not in assigned), key=priority)
            assigned[group] = part
            covered[part].update(features[group])
    assigned.update({group: "train" for group in groups if group not in assigned})
    covered = {
        part: Counter(
            feature for group in groups if assigned[group] == part for feature in features[group]
        )
        for part in ("validation", "test")
    }
    family_partitions = {family: assigned[group] for family, group in family_group.items()}
    products = {ndc: family_partitions[family] for ndc, family in sorted(families.items())}
    coverage = {}
    eligible_cross = 0
    for part in ("train", "validation", "test"):
        coverage[part] = {}
        for kind in sorted({row["case_type"] for row in cases}):
            rows = [
                row
                for row in cases
                if row["case_type"] == kind
                and products.get(row["product_ndc_a"]) == part
                and products.get(row["product_ndc_b"]) == part
                and family_group[families[row["product_ndc_a"]]]
                == family_group[families[row["product_ndc_b"]]]
            ]
            coverage[part][kind] = {
                "cases": len(rows),
                "independent_groups": len(
                    {family_group[families[row["product_ndc_a"]]] for row in rows}
                ),
                "overlap_counts": dict(
                    sorted(Counter(row["ingredient_overlap"] for row in rows).items())
                ),
            }
    for row in cases:
        a, b = row["product_ndc_a"], row["product_ndc_b"]
        if a in families and b in families and families[a] != families[b]:
            eligible_cross += products[a] == products[b]
    gaps = {
        part: {
            feature: target - covered[part][feature]
            for feature, target in required.items()
            if covered[part][feature] < target
        }
        for part in ("validation", "test")
    }
    return {
        "seed": seed,
        "target_family_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "algorithm": (
            "Seeded coverage-aware allocation at independent family-group level. No model outcomes used."
            if balance_coverage
            else "Fixed seeded allocation by independent family group. Coverage assessed after allocation."
        ),
        "independent_family_groups": len(groups),
        "family_labels": len(set(families.values())),
        "products": len(products),
        "group_members": components,
        "group_distribution": dict(sorted(Counter(assigned.values()).items())),
        "family_distribution": dict(sorted(Counter(family_partitions.values()).items())),
        "product_distribution": dict(sorted(Counter(products.values()).items())),
        "family_partitions": family_partitions,
        "product_partitions": products,
        "category_coverage": coverage,
        "coverage_gaps": gaps,
        "coverage_sufficient": len(coverage["train"]) == 3
        and not any(gaps.values())
        and all(values["cases"] > 0 for values in coverage["train"].values()),
        "eligible_cross_family_records": eligible_cross,
        "checks": {
            "one_partition_per_family": all(
                products[ndc] == family_partitions[family] for ndc, family in families.items()
            ),
            "one_partition_per_ndc": len(products) == len(families),
            "co_location_constraints_met": all(
                len({family_partitions[family] for family in members}) == 1
                for members in components
            ),
            "cross_family_rule": "Eligible only when both endpoint families share a partition.",
        },
        "preview_only": True,
        "manifest_created": False,
    }


def assess_expansion_readiness(
    project_root: Path,
    *,
    reviewed_path: Path | None = None,
    write_outputs: bool = True,
    include_candidates: bool = False,
) -> dict:
    """Keep evidence suggestions, reviewed evaluation cases and training candidates separate."""
    base = assess_split_readiness(project_root, write_outputs=False)
    audit_dir = (project_root / base["review_validation"]["reviewed_file"]).parent
    interim = project_root / "data/interim"
    review_path = reviewed_path or audit_dir / "expansion_review.csv"
    reviews = _read_csv(
        review_path,
        {
            "record_id",
            "record_type",
            "product_ndc_a",
            "product_ndc_b",
            "suggested_decision",
            "proposed_family_id_a",
            "proposed_family_id_b",
            "evidence_status",
            "evidence_notes",
            "use_in_conditional_preview",
            "related_products_json",
            "co_partition_families_json",
            "source_evidence_json",
            "human_approved",
            "reviewer_id",
            "review_decision",
            "family_id_a",
            "family_id_b",
        },
    )
    if reviews.record_id.duplicated().any() or base["review_validation"]["invalid_rows"]:
        raise ValueError("Review identifiers must be unique and original reviews valid")
    catalog = _read_csv(interim / "otc_catalog_consistent.csv", {"product_ndc"})
    agreement = _read_csv(interim / "otc_ingredient_agreement.csv", {"product_ndc"})
    products = (
        catalog.merge(agreement, on="product_ndc", validate="one_to_one")
        .set_index("product_ndc")
        .to_dict("index")
    )
    labels = (
        _read_csv(interim / "otc_ndc_label_links.csv", {"product_ndc"})
        .set_index("product_ndc")
        .to_dict("index")
    )
    drugs = (
        _read_csv(interim / "otc_rxnorm_product_links.csv", {"product_ndc"})
        .set_index("product_ndc")
        .to_dict("index")
    )
    assignments = base["family_reconciliation"]["assignments"]
    original_families = {ndc: values[0] for ndc, values in assignments.items() if len(values) == 1}
    old_holds = set(base["family_reconciliation"]["source_hold_product_ndcs"])
    old_holds.update(
        ndc
        for row in base["family_reconciliation"]["possible_aliases"]
        for ndc in row["product_ndcs"]
    )
    old_holds.update(
        ndc
        for row in base["family_reconciliation"]["seed_identity_followups"]
        for ndc in row["product_ndcs"]
    )
    supported_mask = (
        reviews.evidence_status.str.startswith("SUPPORTED")
        & reviews.suggested_decision.eq("SAME_FAMILY")
        & reviews.use_in_conditional_preview.eq("TRUE")
    )
    if reviewed_path is not None:
        supported_mask &= reviews.review_decision.eq("SAME_FAMILY") & reviews.labels_verified.eq(
            "TRUE"
        )
    supported = reviews.loc[supported_mask]
    quarantine = reviews.loc[~reviews.record_id.isin(supported.record_id)]
    quarantined_ndcs = {
        ndc for row in quarantine.itertuples() for ndc in (row.product_ndc_a, row.product_ndc_b)
    }
    proposed = dict(original_families)
    restored = set()
    expansion_keys = set()
    constraints = []
    for row in supported.to_dict("records"):
        a, b = row["product_ndc_a"], row["product_ndc_b"]
        prefix = "family_id_" if reviewed_path is not None else "proposed_family_id_"
        ids = [row[prefix + side] for side in ("a", "b")]
        if not all(ids) or ids[0] != ids[1] or not {a, b}.issubset(products):
            raise ValueError(f"Invalid proposed family or NDC: {row['record_id']}")
        for side, family in zip(("a", "b"), ids, strict=True):
            ndc = row[f"product_ndc_{side}"]
            if ndc in proposed and proposed[ndc] != family:
                raise ValueError(f"A proposal would overwrite an existing family: {ndc}")
            proposed[ndc] = family
        restored.update(
            [a, b, *[item["product_ndc"] for item in json.loads(row["related_products_json"])]]
        )
        if row["record_type"] == "expansion":
            expansion_keys.add(tuple(sorted((a, b))))
        if members := json.loads(row["co_partition_families_json"]):
            constraints.append(members)
    for ndc in (old_holds - restored) | quarantined_ndcs:
        proposed.pop(ndc, None)
    current = {ndc: family for ndc, family in original_families.items() if ndc not in old_holds}
    seeds = _read_csv(interim / "feasibility_review_queue_reviewed.csv", {"review_decision"})
    original_reviews = _read_csv(
        audit_dir / "family_review_decisions_reviewed.csv", {"review_decision"}
    )
    accepted_keys = {
        tuple(sorted((row.product_ndc_a, row.product_ndc_b)))
        for row in seeds.loc[seeds.review_decision.eq("VERIFIED")].itertuples()
    }
    accepted_keys.update(
        tuple(sorted((row.product_ndc_a, row.product_ndc_b)))
        for row in original_reviews.loc[
            original_reviews.review_decision.isin({"SAME_FAMILY", "DIFFERENT_FAMILIES"})
        ].itertuples()
    )
    pool = _candidate_pool(interim).to_dict("records")
    current_cases, proposed_cases = [], []
    for row in pool:
        a, b = row["product_ndc_a"], row["product_ndc_b"]
        key = tuple(sorted((a, b)))
        row["ingredient_overlap"] = (
            "shared"
            if set(json.loads(products[a]["rxnorm_in_rxcuis_json"]))
            & set(json.loads(products[b]["rxnorm_in_rxcuis_json"]))
            else "disjoint"
        )
        if key in accepted_keys and a in current and b in current:
            current_cases.append(row)
        if key in accepted_keys | expansion_keys and a in proposed and b in proposed:
            proposed_cases.append(row)
    current_ndcs = {
        ndc for row in current_cases for ndc in (row["product_ndc_a"], row["product_ndc_b"])
    }
    proposed_ndcs = {
        ndc for row in proposed_cases for ndc in (row["product_ndc_a"], row["product_ndc_b"])
    }
    proposed = {ndc: proposed[ndc] for ndc in sorted(proposed_ndcs)}
    current_preview = coverage_split_preview(
        {ndc: current[ndc] for ndc in sorted(current_ndcs)}, current_cases, []
    )
    preview = coverage_split_preview(proposed, proposed_cases, constraints)

    # Reserve related names conservatively before screening an unassigned training reservoir
    links: dict[str, set[str]] = defaultdict(set)
    for row in pool:
        left, right = (
            products[row[f"product_ndc_{side}"]]["normalized_name"] for side in ("a", "b")
        )
        links[left].add(right)
        links[right].add(left)
    anchors: dict[str, set[str]] = defaultdict(set)
    for ndc, family in {**original_families, **proposed}.items():
        anchors[family].add(products[ndc]["normalized_name"])
    for family, names in anchors.items():
        anchor = min(names)
        family_key = _alias_key(family)
        for name in list(links):
            if family_key and f" {family_key} " in f" {_alias_key(name)} ":
                names.add(name)
        for name in names:
            links[anchor].add(name)
            links[name].add(anchor)
    aliases: dict[str, set[str]] = defaultdict(set)
    for name in list(links):
        aliases[_alias_key(name)].add(name)
    for names in aliases.values():
        for name in names:
            links[min(names)].add(name)
            links[name].add(min(names))
    block_by_name = {name: members[0] for members in _components(links) for name in members}
    reserved = set(original_families) | {
        ndc for row in reviews.itertuples() for ndc in (row.product_ndc_a, row.product_ndc_b)
    }
    reserved.update(
        ndc
        for frame in (seeds, original_reviews)
        for row in frame.itertuples()
        for ndc in (row.product_ndc_a, row.product_ndc_b)
    )
    reserved_blocks = {block_by_name[products[ndc]["normalized_name"]] for ndc in reserved}
    pool_ndcs = {ndc for row in pool for ndc in (row["product_ndc_a"], row["product_ndc_b"])}
    risks = {
        ndc: source_training_risks(products[ndc], drugs.get(ndc, {}), labels.get(ndc, {}))
        for ndc in pool_ndcs
    }
    training, rejected, guarded = [], Counter(), 0
    for row in pool:
        a, b = row["product_ndc_a"], row["product_ndc_b"]
        if issues := set(risks[a]) | set(risks[b]):
            rejected.update(issues)
        elif block_by_name[products[a]["normalized_name"]] in reserved_blocks:
            guarded += 1
        else:
            training.append(row)
    training_ndcs = {
        ndc for row in training for ndc in (row["product_ndc_a"], row["product_ndc_b"])
    }
    training_blocks = {block_by_name[products[ndc]["normalized_name"]] for ndc in training_ndcs}
    pending = [
        row.record_id
        for row in supported.itertuples()
        if not (
            row.human_approved == "TRUE"
            and row.reviewer_id.strip()
            and row.review_decision == "SAME_FAMILY"
            and row.family_id_a == row.proposed_family_id_a
            and row.family_id_b == row.proposed_family_id_b
        )
    ]
    live_documents = {}
    for row in reviews.itertuples():
        for ref in json.loads(row.source_evidence_json).values():
            live_documents[ref["fda_label_link"]["label_set_id"]] = ref["live_dailymed_access"]
        for ref in json.loads(row.related_products_json):
            live_documents[ref["label_set_id"]] = ref["live_dailymed_access"]
    inaccessible_ndcs = sorted(
        {
            ref["product_ndc"]
            for row in reviews.itertuples()
            for ref in json.loads(row.source_evidence_json).values()
            if ref["live_dailymed_access"] == "UNAVAILABLE"
        }
    )
    blockers = []
    if pending:
        blockers.append(
            "Evidence-supported proposals require human confirmation: " + ", ".join(pending)
        )
    if not preview["coverage_sufficient"]:
        blockers.append(
            "Conditional held-out category or ingredient-overlap coverage is insufficient."
        )
    blockers.append(
        "Independent evaluation annotation provenance and final sample size remain to be established."
    )
    blockers.append(
        "Additional training candidates remain unassigned until their family boundaries are reviewed."
    )
    base.update(
        {
            "stage": "3.2B_expansion_and_split_preview",
            "readiness": "NOT_READY",
            "blockers": blockers,
            "usable_reviewed_pool": {
                "independent_family_groups": len({current[ndc] for ndc in current_ndcs}),
                "products": len(current_ndcs),
                "reviewed_source_checked_records": len(current_cases),
                "gold_annotation_status": "Reserved reviewed pool, not established independent human gold annotation.",
                "new_ai_suggestions_admitted": 0,
            },
            "expansion_review": {
                "file": str(review_path.relative_to(project_root)),
                "rows": len(reviews),
                "expansion_records": int(reviews.record_type.eq("expansion").sum()),
                "expansion_suggestions": dict(
                    sorted(
                        Counter(
                            reviews.loc[reviews.record_type.eq("expansion"), "suggested_decision"]
                        ).items()
                    )
                ),
                "supported_proposal_ids": supported.record_id.tolist(),
                "manual_confirmation_before_inclusion": pending,
                "quarantined_record_ids": quarantine.record_id.tolist(),
                "quarantined_product_ndcs": sorted(quarantined_ndcs),
                "quarantine_reasons": {
                    row.record_id: row.evidence_notes for row in quarantine.itertuples()
                },
                "reference_resolutions": {
                    row.record_id: row.evidence_notes
                    for row in supported.loc[supported.record_type.eq("blocker")].itertuples()
                },
                "co_partition_constraints": constraints,
                "original_assignments_rewritten": False,
                "live_dailymed_documents_accessible": sum(
                    value != "UNAVAILABLE" for value in live_documents.values()
                ),
                "live_dailymed_documents_attempted": len(live_documents),
                "inaccessible_ndcs": inaccessible_ndcs,
            },
            "proposed_evaluation_pool": {
                "conditional_on_human_approval": True,
                "independent_family_groups": preview["independent_family_groups"],
                "family_labels": preview["family_labels"],
                "products": preview["products"],
                "existing_candidate_records": len(proposed_cases),
                "sample_size_status": "The plan targets several hundred verified pairs. This small conditional cohort only tests category feasibility.",
                "new_benchmark_examples_generated": 0,
                "proposed_family_assignments": proposed,
            },
            "training_candidates": {
                "source_candidate_records_scanned": len(pool),
                "source_consistent_reserved_records": guarded,
                "candidate_records": len(training),
                "products": len(training_ndcs),
                "category_counts": dict(
                    sorted(Counter(row["case_type"] for row in training).items())
                ),
                "overlap_counts": dict(
                    sorted(Counter(row["ingredient_overlap"] for row in training).items())
                ),
                "mapping_risk_record_counts": dict(sorted(rejected.items())),
                "source_screen_rejected_records": sum(
                    bool(risks[row["product_ndc_a"]] or risks[row["product_ndc_b"]]) for row in pool
                ),
                "confirmed_independent_family_count": len(
                    {original_families[ndc] for ndc in training_ndcs & set(original_families)}
                ),
                "provisional_candidate_blocks": len(training_blocks),
                "family_coverage_status": "Additional family membership is unverified. Candidate blocks are not confirmed families.",
                "evaluation_or_related_block_overlap": len(training_blocks & reserved_blocks),
                "admitted_to_training": False,
                "screen": "Exact ingredient-name agreement, resolved unique RxNorm product, matched label with active text, "
                "direct native-strength agreement and no explicit route or dosage-form contradiction. No salt, density or moiety conversion.",
                "limitation": "Source-screened reservoir only. Full label and family review remains necessary, "
                "and no reservoir product is assigned to a partition.",
            },
            "current_reviewed_split_preview": current_preview,
            "split_policy": {
                **preview,
                "conditional_on_human_approval": True,
                "target_unit": "Independent family groups, retaining original family IDs within related groups.",
                "heldout_rule": "No model or label-rule development may use held-out labels after freeze.",
            },
        }
    )
    for path in (
        review_path,
        interim / "otc_ndc_label_links.csv",
        interim / "otc_rxnorm_product_links.csv",
    ):
        base["inputs"][path.name] = sha256(path.read_bytes()).hexdigest()
    if write_outputs:
        (audit_dir / "split_readiness.json").write_text(
            json.dumps(base, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if include_candidates:
        base["screened_training_candidates"] = training
    return base


def confirmed_expansion_reviews(
    baseline: pd.DataFrame, reviewed: pd.DataFrame, assignments: dict[str, list[str]]
) -> pd.DataFrame:
    """Use completed review decisions without claiming personal approval."""
    editable = {
        "review_decision",
        "family_id_a",
        "family_id_b",
        "labels_verified",
        "reviewer_notes",
        "human_approved",
        "reviewer_id",
    }
    immutable = [column for column in baseline.columns if column not in editable]
    if (
        reviewed.record_id.duplicated().any()
        or not set(baseline.columns).issubset(reviewed.columns)
        or not baseline[immutable]
        .set_index("record_id")
        .sort_index()
        .equals(reviewed[immutable].set_index("record_id").sort_index())
    ):
        raise ValueError("Expansion review source records changed or identifiers are duplicated")
    if set(reviewed.review_decision) - {
        "",
        "SAME_FAMILY",
        "DIFFERENT_FAMILIES",
        "EXCLUDE",
        "UNCERTAIN",
    }:
        raise ValueError("Unexpected expansion review decision")
    accepted = reviewed.loc[
        reviewed.review_decision.eq("SAME_FAMILY")
        & reviewed.labels_verified.eq("TRUE")
        & reviewed.evidence_status.str.startswith("SUPPORTED")
        & reviewed.use_in_conditional_preview.eq("TRUE")
    ].copy()
    for row in accepted.itertuples():
        if not row.family_id_a.strip() or row.family_id_a != row.family_id_b:
            raise ValueError(f"Decision and family IDs disagree: {row.record_id}")
        for side in ("a", "b"):
            ndc, family = getattr(row, "product_ndc_" + side), getattr(row, "family_id_" + side)
            if ndc in assignments and assignments[ndc] != [family]:
                raise ValueError(f"Review would overwrite an original family: {ndc}")
    return accepted


def benchmark_references(
    products: dict[str, dict],
    labels: dict[str, dict],
    drugs: dict[str, dict],
    families: dict[str, str],
    accepted: pd.DataFrame,
) -> dict[str, dict]:
    """Retain native FDA and reviewed label references without unit conversion."""
    overrides = {}
    for row in accepted.itertuples():
        for side in ("a", "b"):
            ndc = getattr(row, "product_ndc_" + side)
            reference = json.loads(getattr(row, "proposed_reference_" + side + "_json"))
            overrides[ndc] = (reference, row.record_id)
        for related in json.loads(row.related_products_json):
            ndc = related["product_ndc"]
            # A shared SPL supports the same reviewed reference for the related toothpaste
            if (
                row.record_id == "B-15"
                and labels[ndc]["label_set_id"] == labels[row.product_ndc_b]["label_set_id"]
            ):
                overrides[ndc] = (json.loads(row.proposed_reference_b_json), row.record_id)
    references = {}
    for ndc in sorted(families):
        product, label, drug = products[ndc], labels[ndc], drugs[ndc]
        if (
            product["catalog_status"] != "consistent"
            or product["agreement_status"] not in EXACT_AGREEMENT
            or product["rxnorm_ingredient_status"] != "resolved_path"
            or label["link_status"] != "matched"
            or not label["label_id"]
            or not label["label_set_id"]
            or not json.loads(label["active_ingredient_json"] or "[]")
            or drug["rxnorm_status"] != "matched"
        ):
            raise ValueError(f"Incomplete benchmark reference evidence: {ndc}")
        native = {
            "active_ingredients": json.loads(product["active_ingredients_json"]),
            "dosage_form": product["dosage_form"],
            "route": json.loads(product["route"]),
            "rxnorm_scope": "INGREDIENT_IDENTIFIERS_ONLY",
        }
        reference, review_id = overrides.get(ndc, (native, "EXISTING_REVIEW"))
        if not reference["active_ingredients"] or any(
            not item.get("name") or not item.get("strength")
            for item in reference["active_ingredients"]
        ):
            raise ValueError(f"Missing ingredient identity or native strength: {ndc}")
        references[ndc] = {
            "product_ndc": ndc,
            "name": product["display_name"],
            "family_id": families[ndc],
            **reference,
            "rxnorm_in_rxcuis": json.loads(product["rxnorm_in_rxcuis_json"]),
            "rxnorm_in_names": json.loads(product["rxnorm_in_names_json"]),
            "rxnorm_pin_rxcuis": json.loads(product["rxnorm_pin_rxcuis_json"]),
            "rxnorm_pin_names": json.loads(product["rxnorm_pin_names_json"]),
            "source": {
                "catalog_csv": "data/interim/otc_catalog_consistent.csv",
                "catalog_source_rows": product["source_rows"],
                "label_id": label["label_id"],
                "label_set_id": label["label_set_id"],
                "label_effective_time": label["effective_time"],
                "label_active_ingredient_text": json.loads(label["active_ingredient_json"]),
                "dailymed_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid="
                + label["label_set_id"],
                "rxnorm_product_rxcui": drug["rxnorm_rxcui"],
                "reference_review_id": review_id,
                "label_links_csv": "data/interim/otc_ndc_label_links.csv",
                "rxnorm_product_csv": "data/interim/otc_rxnorm_product_links.csv",
                "rxnorm_ingredient_csv": "data/interim/otc_rxnorm_ingredient_links.csv",
            },
        }
    return references


def _formulation_input(reference: dict) -> dict:
    """Expose product details separately from ingredient identity answers."""
    return {
        "name": normalize_name(reference["name"]),
        "dosage_form": reference["dosage_form"],
        "route": sorted(reference["route"]),
        "native_strengths": sorted(item["strength"] for item in reference["active_ingredients"]),
    }


def assign_catalog_families(
    products: dict, historical: list[dict], protected_ndcs: set | None = None
) -> tuple[dict, list, dict]:
    """Assign conservative leakage groups before constructing any cases."""
    parents = {ndc: ndc for ndc in products}
    protected_ndcs = protected_ndcs or set()

    def root(ndc):
        while parents[ndc] != ndc:
            parents[ndc] = parents[parents[ndc]]
            ndc = parents[ndc]
        return ndc

    links, discovery = discover_name_links(products)
    by_label, by_base, old_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    chemical_aliases = defaultdict(list)
    name_tokens = Counter(
        t
        for name in {name_key(p["display_name"]) for p in products.values()}
        for t in set(name.split())
    )
    ingredient_tokens = {
        t
        for p in products.values()
        for item in json.loads(p["active_ingredients_json"] or "[]") or []
        for t in normalize_name(item.get("name")).split()
    }
    for ndc, product in sorted(products.items()):
        ingredients = json.loads(product["active_ingredients_json"])
        if len(ingredients) == 1 and ingredient_identity(ingredients[0]).startswith("fda_unii:"):
            display = name_key(product["display_name"])
            if display in {
                name_key(product.get("generic_name", "")),
                name_key(ingredients[0]["name"]),
            }:
                chemical_aliases[ingredient_identity(ingredients[0])].append(ndc)
        if label := product.get("label_set_id"):
            by_label[label].append(ndc)
        for source in json.loads(product.get("source_metadata_json", "[]")):
            base = name_key(source.get("brand_name_base") or "")
            distinctive = {
                t
                for t in base.split()
                if len(t) >= 4 and name_tokens[t] <= 40 and t not in ingredient_tokens
            }
            distinctive -= {
                "health",
                "care",
                "store",
                "pharmacy",
                "pharma",
                "quality",
                "wellness",
                "value",
                "family",
                "good",
                "sense",
                "signature",
                "select",
                "choice",
            }
            if distinctive and len(base.split()) >= 2:
                by_base[base].append(ndc)
    names_to_ndcs = defaultdict(list)
    token_products = defaultdict(list)
    for ndc, product in sorted(products.items()):
        name = name_key(product["display_name"])
        names_to_ndcs[name].append(ndc)
        for token in set(name.split()):
            token_products[token].append(ndc)
    for name, ndcs in names_to_ndcs.items():
        if (
            len(name.split()) == 1
            and len(name) >= 4
            and name_tokens[name] <= 40
            and name not in ingredient_tokens
        ):
            by_base["standalone_" + name].extend(token_products[name])
    for kind, groups in (
        ("shared_label_set", by_label),
        ("submitted_brand_base", by_base),
        ("same_fda_ingredient_alias", chemical_aliases),
    ):
        for members in groups.values():
            members = sorted(set(members))
            for a, b in pairwise(members):
                links.append((a, b, kind))
    for row in historical:
        if row["product_ndc"] in products:
            old_groups[row["family_group_id"]].append(row["product_ndc"])
    for label, members in old_groups.items():
        # Propagate historical protection across literal related-name phrases
        phrase = name_key(label)
        if phrase:
            members.extend(
                ndc
                for ndc, product in products.items()
                if f" {phrase} " in f" {name_key(product['display_name'])} "
            )
        members = sorted(set(members))
        for a, b in pairwise(members):
            links.append((a, b, "historical_family_protection"))
    for a, b, _ in sorted(set(links)):
        a, b = root(a), root(b)
        if a != b:
            parents[max(a, b)] = min(a, b)
    members = defaultdict(list)
    for ndc in sorted(products):
        members[root(ndc)].append(ndc)
    assignments = {}
    protection = Counter()
    for ndcs in members.values():
        group = "family_" + sha256("\n".join(ndcs).encode()).hexdigest()[:16]
        old_parts = {r["partition"] for r in historical if r["product_ndc"] in set(ndcs)}
        if protected_ndcs.intersection(ndcs):
            old_parts.add("test")
        value = int(sha256(("42|" + group).encode()).hexdigest()[:12], 16) / 16**12
        part = (
            "test"
            if "test" in old_parts
            else "validation"
            if "validation" in old_parts
            else ("train" if value < 0.7 else "validation" if value < 0.85 else "test")
        )
        if old_parts & {"test", "validation"}:
            protection[part + "_related_products"] += len(ndcs)
        for ndc in ndcs:
            assignments[ndc] = {"family_group_id": group, "partition": part}
    return (
        assignments,
        sorted(set(links)),
        {
            "discovery": discovery,
            "groups": len(members),
            "group_sizes": dict(Counter(len(v) for v in members.values())),
            "largest_group": max(map(len, members.values())),
            "historical_protection": dict(protection),
            "link_types": dict(Counter(k for _, _, k in links)),
            "status": "Algorithmic conservative co-partition groups, not human-verified families",
        },
    )


def source_reference(product: dict, exact: bool = True) -> dict:
    supported = product.get("formulation_eligible", "True") == "True"
    ingredients = json.loads(product["active_ingredients_json"])
    if not supported:
        ingredients = [
            {"name": item["name"], "identity": ingredient_identity(item)} for item in ingredients
        ]
    return {
        "product_ndc": product["product_ndc"] if exact else None,
        "name": product["display_name"],
        "active_ingredients": ingredients,
        "dosage_form": product["dosage_form"],
        "route": json.loads(product["route"] or "[]") or [],
        "formulation_supported": supported,
        "support_tier": product.get("support_tier", "FDA_ONLY"),
        "ingredient_identities": json.loads(product.get("ingredient_identities_json") or "[]")
        or sorted({ingredient_identity(item) for item in ingredients}),
        "comparison_eligible": product.get("comparison_eligible", "True") == "True",
        "source": {
            key: product.get(key, "")
            for key in (
                "product_ndc",
                "rxnorm_product_rxcui",
                "rxnorm_in_rxcuis_json",
                "rxnorm_pin_rxcuis_json",
                "rxnorm_identity_options_json",
                "source_rows",
                "ingredient_origin",
                "label_id",
                "label_set_id",
                "label_effective_time",
                "original_ingredients_json",
                "label_extracted_ingredients_json",
                "strength_issues_json",
                "identity_status",
                "identity_provenance_json",
                "rxnorm_identifier_status",
                "rxnorm_unii_options_json",
            )
        },
    }


def source_description(product: dict) -> dict:
    if product.get("formulation_eligible", "True") != "True":
        return {"name": product["display_name"], "product_ndc": product["product_ndc"]}
    return {
        "name": normalize_name(product["display_name"]),
        "dosage_form": product["dosage_form"],
        "route": list(route_signature(json.loads(product["route"])) or []),
        "native_strengths": sorted(
            item["strength"] for item in json.loads(product["active_ingredients_json"])
        ),
    }


def catalog_answer(description: dict, products: dict, by_name: dict) -> tuple[dict, list[str]]:
    """Prepare reference labels from complete source cohorts before evaluation."""
    explicit = description.get("product_ndc")
    ndcs = (
        ([explicit] if explicit in products else [])
        if explicit
        else sorted(by_name.get(normalize_name(description.get("name")), []))
    )
    candidates = []
    for ndc in ndcs:
        product = products[ndc]
        issues = json.loads(product.get("exclusion_reasons_json") or "[]")
        strength_issues = json.loads(product.get("strength_issues_json") or "[]")
        disputed_identity = "conflicting_ndc_rows" in issues
        disputed_form = disputed_identity or "rxnorm_route_or_form_conflict" in strength_issues
        if (
            not disputed_form
            and "dosage_form" in description
            and product["dosage_form"]
            and (
                normalize_name(product["dosage_form"]) != normalize_name(description["dosage_form"])
            )
        ):
            continue
        route = json.loads(product["route"] or "[]") or []
        if (
            not disputed_form
            and "route" in description
            and route
            and route_signature(route) != route_signature(description["route"])
        ):
            continue
        items = json.loads(product["active_ingredients_json"] or "[]") or []
        if (
            not disputed_identity
            and product.get("formulation_eligible", "True") == "True"
            and "native_strengths" in description
            and items
            and all(i.get("strength") for i in items)
        ):

            def strengths(values):
                return sorted(quantity_key(v) or (normalize_name(v), "literal") for v in values)

            if strengths(description["native_strengths"]) != strengths(
                i["strength"] for i in items
            ):
                continue
        candidates.append(ndc)
    if not candidates:
        return {"answer_status": "NO_MATCH"}, []
    accepted = [n for n in candidates if products[n].get("ingredient_eligible", "True") == "True"]
    if len(accepted) != len(candidates):
        return {
            "answer_status": "SOURCE_CONFLICT",
            "reason": "unresolved_source_candidates",
        }, candidates
    if len(candidates) == 1:
        return {
            "answer_status": "IDENTIFIED",
            "resolution_level": "EXACT_NDC",
            "product": source_reference(products[candidates[0]]),
        }, candidates
    formulations = {
        (
            formulation_key(
                json.loads(products[n]["active_ingredients_json"]),
                products[n]["dosage_form"],
                json.loads(products[n]["route"] or "[]") or [],
            )
            if products[n].get("formulation_eligible", "True") == "True"
            else None
        )
        for n in candidates
    }
    if (
        None not in formulations
        and len(formulations) == 1
        and all(products[n].get("formulation_eligible", "True") == "True" for n in candidates)
    ):
        return {
            "answer_status": "IDENTIFIED",
            "resolution_level": "FORMULATION",
            "product": source_reference(products[candidates[0]], exact=False),
        }, candidates
    # A known difference is necessary to label ambiguity rather than missing evidence
    identities = {
        tuple(
            sorted(
                ingredient_identity(x) for x in json.loads(products[n]["active_ingredients_json"])
            )
        )
        for n in candidates
    }
    known_formulations = formulations - {None}
    if len(identities) > 1 or len(known_formulations) > 1:
        return {
            "answer_status": "INSUFFICIENT_INFORMATION",
            "unique_formulation": False,
            "source_alternative_count": len(candidates),
        }, candidates
    return {
        "answer_status": "SOURCE_CONFLICT",
        "reason": "formulation_basis_unresolved",
    }, candidates


def comparison_pairs(
    products: dict, assignments: dict, saved_pairs: list[dict]
) -> tuple[list, dict]:
    """Use a deterministic covering graph over distinct product descriptions."""
    groups = defaultdict(list)
    prototypes, ndc_prototype = {}, {}
    for ndc, product in sorted(products.items()):
        if (
            product.get("ingredient_eligible", "True") != "True"
            or product.get("comparison_eligible", "True") != "True"
        ):
            continue
        identity = (
            assignments[ndc]["family_group_id"],
            name_key(product["display_name"]),
            tuple(
                json.loads(product.get("ingredient_identities_json") or "[]")
                or sorted(
                    {ingredient_identity(i) for i in json.loads(product["active_ingredients_json"])}
                )
            ),
            formulation_key(
                json.loads(product["active_ingredients_json"]),
                product["dosage_form"],
                json.loads(product["route"] or "[]") or [],
            )
            or (product["dosage_form"], product["route"], product["active_ingredients_json"]),
        )
        representative = prototypes.setdefault(identity, ndc)
        ndc_prototype[ndc] = representative
    for ndc in prototypes.values():
        groups[assignments[ndc]["family_group_id"]].append(ndc)
    pairs = set()
    ingredient_sets = {
        n: frozenset(
            json.loads(products[n].get("ingredient_identities_json") or "[]")
            or {ingredient_identity(i) for i in json.loads(products[n]["active_ingredients_json"])}
        )
        for n in ndc_prototype
    }
    possible = 0
    for members in groups.values():
        members.sort()
        possible += len(members) * (len(members) - 1) // 2
        by_set = defaultdict(list)
        for n in members:
            by_set[ingredient_sets[n]].append(n)
        sets = sorted(by_set, key=lambda s: tuple(sorted(s)))
        for values in by_set.values():
            if len(values) > 1:
                pairs.update(tuple(sorted((a, b))) for a, b in zip(values, values[1:] + values[:1]))
        for index, left in enumerate(sets):
            for overlap in (True, False):
                target = next(
                    (
                        sets[(index + step) % len(sets)]
                        for step in range(1, len(sets))
                        if bool(left & sets[(index + step) % len(sets)]) == overlap
                    ),
                    None,
                )
                if target is None:
                    continue
                a, b = by_set[left], by_set[target]
                for j in range(max(len(a), len(b))):
                    pairs.add(tuple(sorted((a[j % len(a)], b[j % len(b)]))))
        # Adjacent formulations retain dosage and strength contrasts within a family
        for a, b in pairwise(members):
            pairs.add(tuple(sorted((a, b))))
    saved_outcomes = Counter()
    for row in saved_pairs:
        a, b = row["product_ndc_a"], row["product_ndc_b"]
        if a not in ndc_prototype or b not in ndc_prototype:
            saved_outcomes["source_or_chemical_identity_ineligible_endpoint"] += 1
        elif assignments[a]["partition"] != assignments[b]["partition"]:
            saved_outcomes["cross_partition"] += 1
        elif ndc_prototype[a] == ndc_prototype[b]:
            saved_outcomes["same_prototype"] += 1
        else:
            saved_outcomes["source_eligible_pair_considered"] += 1
            pairs.add(tuple(sorted((ndc_prototype[a], ndc_prototype[b]))))
    within = sum(
        assignments[a]["family_group_id"] == assignments[b]["family_group_id"] for a, b in pairs
    )
    return sorted(pairs), {
        "ingredient_eligible_products_considered": sum(
            p.get("ingredient_eligible", "True") == "True" for p in products.values()
        ),
        "chemical_comparison_eligible_products_considered": len(ndc_prototype),
        "distinct_name_formulation_prototypes": len(prototypes),
        "possible_within_group_prototype_pairs": possible,
        "selected_unique_pairs": len(pairs),
        "selected_within_group_pairs": within,
        "cross_group_saved_pairs": len(pairs) - within,
        "not_enumerated_by_covering_policy": possible - within,
        "saved_candidate_outcomes": dict(saved_outcomes),
        "policy": "Collapse identical name and formulation replicas. Within each family connect "
        "every prototype in a same-ingredient cycle, to cyclic next shared and disjoint ingredient "
        "sets when available, and to adjacent sorted formulations. Include every eligible saved "
        "candidate pair within a partition. No family case cap. The covering graph avoids "
        "quadratic enumeration and repeated equivalent pairs.",
    }


def canonical_case_input(description: dict) -> dict:
    if "product_a" in description:
        pair = [canonical_case_input(description[key]) for key in ("product_a", "product_b")]
        pair.sort(key=lambda value: json.dumps(value, sort_keys=True))
        return {"product_a": pair[0], "product_b": pair[1]}
    result = dict(description)
    for key in ("name", "dosage_form"):
        if key in result:
            result[key] = normalize_name(result[key])
    if "route" in result:
        result["route"] = list(route_signature(result["route"]) or [])
    if "native_strengths" in result:
        result["native_strengths"] = sorted(
            quantity_key(value) or (normalize_name(value), "literal")
            for value in result["native_strengths"]
        )
    return result


def construct_benchmark_cases(
    products: dict, assignments: dict, pairs: list
) -> tuple[pd.DataFrame, dict]:
    """Build every distinct case under the declared covering policy."""
    by_name = defaultdict(set)
    for ndc, product in products.items():
        for field in ("display_name", "brand_name", "generic_name"):
            if name := normalize_name(product.get(field)):
                by_name[name].add(ndc)
    cases, excluded, seen = [], Counter(), set()
    answer_cache = {}

    def answer_for(description):
        key = json.dumps(description, sort_keys=True)
        if key not in answer_cache:
            answer_cache[key] = catalog_answer(description, products, by_name)
        return answer_cache[key]

    described = {}
    eligible = {n for n, p in products.items() if p.get("ingredient_eligible", "True") == "True"}

    def add(task, kind, description, answer, ndcs):
        if not ndcs:
            excluded["no_source_candidates"] += 1
            return
        parts = {assignments[n]["partition"] for n in ndcs}
        if len(parts) != 1:
            excluded["cross_partition_description"] += 1
            return
        identity = json.dumps(
            [task, canonical_case_input(description)], sort_keys=True, ensure_ascii=False
        )
        if identity in seen:
            excluded["duplicate_task_input"] += 1
            return
        seen.add(identity)
        groups = sorted({assignments[n]["family_group_id"] for n in ndcs})
        cases.append(
            {
                "case_id": "hrp_" + sha256(identity.encode()).hexdigest()[:20],
                "partition": next(iter(parts)),
                "task": task,
                "case_type": kind,
                "input_json": json.dumps(description, sort_keys=True, ensure_ascii=False),
                "reference_json": json.dumps(answer, sort_keys=True, ensure_ascii=False),
                "family_group_ids_json": json.dumps(groups),
                "provenance_json": json.dumps(
                    {
                        "product_ndcs": sorted(ndcs),
                        "catalog": f"data/interim/{FINAL_BENCHMARK_VERSION}/catalog_evidence.csv",
                        "reference_policy": "source_supported_v4",
                        "human_verified": False,
                    },
                    sort_keys=True,
                ),
            }
        )

    for ndc in sorted(eligible):
        product = products[ndc]
        exact = {"name": product["display_name"], "product_ndc": ndc}
        exact_answer, exact_ids = answer_for(exact)
        add("product_identification", "explicit_ndc", exact, exact_answer, exact_ids)
        details = source_description(product)
        answer, ids = answer_for(details)
        if (
            answer["answer_status"] == "IDENTIFIED"
            and len({assignments[n]["partition"] for n in ids}) == 1
        ):
            described[ndc] = (details, answer, ids)
            if "product_ndc" not in details:
                add("product_identification", "formulation_description", details, answer, ids)
        else:
            described[ndc] = (exact, exact_answer, exact_ids)
            excluded["detailed_description_" + answer["answer_status"].lower()] += 1
    names = sorted({normalize_name(products[n]["display_name"]) for n in eligible})
    for name in names:
        answer, ids = answer_for({"name": name})
        if answer["answer_status"] == "SOURCE_CONFLICT":
            excluded["name_query_" + answer["reason"]] += 1
            continue
        add(
            "ambiguity_detection",
            "name_only_ambiguous"
            if answer["answer_status"] == "INSUFFICIENT_INFORMATION"
            else "name_only_identifiable",
            {"name": name},
            answer,
            ids,
        )
    pair_counts = Counter()
    for a, b in pairs:
        if a not in described or b not in described:
            excluded["pair_source_ineligible"] += 1
            continue
        left, left_answer, left_ids = described[a]
        right, right_answer, right_ids = described[b]
        if left == right:
            excluded["equivalent_comparison_descriptions"] += 1
            continue
        ingredients = [
            set(json.loads(products[n].get("ingredient_identities_json") or "[]"))
            or {ingredient_identity(i) for i in json.loads(products[n]["active_ingredients_json"])}
            for n in (a, b)
        ]
        shared = ingredients[0] & ingredients[1]
        kind = (
            "same_ingredient_control"
            if ingredients[0] == ingredients[1]
            else ("different_ingredients_shared" if shared else "different_ingredients_disjoint")
        )
        if name_key(products[a]["display_name"]) == name_key(products[b]["display_name"]):
            pair_counts["identical_name_" + kind] += 1
        if products[a]["dosage_form"] != products[b]["dosage_form"]:
            pair_counts["different_dosage_form"] += 1
        elif formulation_key(
            json.loads(products[a]["active_ingredients_json"]),
            products[a]["dosage_form"],
            json.loads(products[a]["route"] or "[]") or [],
        ) != formulation_key(
            json.loads(products[b]["active_ingredients_json"]),
            products[b]["dosage_form"],
            json.loads(products[b]["route"] or "[]") or [],
        ):
            pair_counts["different_strength_or_composition"] += 1
        pair_counts[kind] += 1
        add(
            "ingredient_comparison",
            kind,
            {"product_a": left, "product_b": right},
            {
                "answer_status": "SUPPORTED",
                "product_a": left_answer["product"],
                "product_b": right_answer["product"],
                "ingredient_overlap": "shared" if shared else "disjoint",
                "same_active_ingredient_set": ingredients[0] == ingredients[1],
                "shared_active_ingredient_identities": sorted(shared),
            },
            sorted(set(left_ids + right_ids)),
        )
    frame = pd.DataFrame(cases).sort_values(["partition", "task", "case_id"]).reset_index(drop=True)
    return frame, {
        "construction_exclusions": dict(excluded),
        "comparison_classes": dict(pair_counts),
        "all_ingredient_eligible_products_considered": len(eligible),
    }


def build_benchmark(project_root: Path, freeze: bool = True) -> dict:
    """Build and seal the full source-supported benchmark without model predictions."""
    interim = project_root / "data/interim"
    out = project_root / "data/processed" / FINAL_BENCHMARK_VERSION
    report_dir = project_root / "results" / FINAL_BENCHMARK_VERSION / "part03"
    out.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = interim / FINAL_BENCHMARK_VERSION / "catalog_evidence.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        products = {r["product_ndc"]: r for r in csv.DictReader(stream)}
    snapshot_path = project_root / "results/final_v2/historical_inputs.json"
    if not snapshot_path.exists():
        legacy_manifest = project_root / "results/part03/family_split_manifest.csv"
        with legacy_manifest.open(encoding="utf-8", newline="") as stream:
            historical = list(csv.DictReader(stream))
        historical_hashes = {
            str(p.relative_to(project_root)).replace("\\", "/"): sha256(p.read_bytes()).hexdigest()
            for p in [
                legacy_manifest,
                project_root / "data/processed/benchmark_cases.csv",
                *sorted((project_root / "results/part03").glob("*")),
                *sorted((project_root / "results/part04").glob("*")),
            ]
            if p.is_file()
        }
        reservoir = assess_expansion_readiness(
            project_root, write_outputs=False, include_candidates=True
        )
        screened = reservoir["screened_training_candidates"]
        saved = _candidate_pool(interim).to_dict("records")
        snapshot = {
            "version": "historical_v1",
            "historical_hashes": historical_hashes,
            "products": [
                {k: r[k] for k in ("product_ndc", "family_id", "family_group_id", "partition")}
                for r in historical
            ],
            "candidate_pairs": [[r["product_ndc_a"], r["product_ndc_b"]] for r in saved],
            "screened_reservoir_pairs": [
                [r["product_ndc_a"], r["product_ndc_b"]] for r in screened
            ],
            "legacy_reservoir_provisional_blocks": reservoir["training_candidates"][
                "provisional_candidate_blocks"
            ],
            "note": "Historical endpoints and split constraints only. No inferred human approval or reference labels.",
        }
        snapshot_path.write_text(
            json.dumps(snapshot, separators=(",", ":")) + "\n", encoding="utf-8"
        )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    historical = snapshot["products"]
    historical_hashes = snapshot["historical_hashes"]
    print("Assigning all catalog products to conservative family groups", flush=True)
    protected_expanded = set(snapshot.get("additional_protected_test_ndcs", []))
    assignments, links, families = assign_catalog_families(products, historical, protected_expanded)
    print("Family groups", families["groups"], flush=True)
    saved = [{"product_ndc_a": a, "product_ndc_b": b} for a, b in snapshot["candidate_pairs"]]
    screened = [
        {"product_ndc_a": a, "product_ndc_b": b} for a, b in snapshot["screened_reservoir_pairs"]
    ]
    reservoir_outcomes = Counter()
    for row in screened:
        a, b = (row["product_ndc_" + side] for side in ("a", "b"))
        if any(products[n]["ingredient_eligible"] != "True" for n in (a, b)):
            reservoir_outcomes["ingredient_evidence_unresolved"] += 1
        elif any(products[n].get("comparison_eligible", "True") != "True" for n in (a, b)):
            reservoir_outcomes["chemical_identity_unresolved"] += 1
        elif assignments[a]["partition"] != assignments[b]["partition"]:
            reservoir_outcomes["cross_partition"] += 1
        else:
            reservoir_outcomes["eligible_" + assignments[a]["partition"]] += 1
    pairs, pair_policy = comparison_pairs(products, assignments, saved)
    print(
        "Constructing cases from all eligible products and",
        len(pairs),
        "comparison pairs",
        flush=True,
    )
    cases, construction = construct_benchmark_cases(products, assignments, pairs)
    manifest = pd.DataFrame(
        [
            {
                "product_ndc": n,
                **assignments[n],
                "ingredient_eligible": p["ingredient_eligible"],
                "formulation_eligible": p["formulation_eligible"],
                "support_tier": p["support_tier"],
            }
            for n, p in sorted(products.items())
        ]
    )
    assert not cases.case_id.duplicated().any()
    assert not cases[["task", "input_json"]].duplicated().any()
    assert manifest.groupby("family_group_id").partition.nunique().max() == 1
    protected = {r["product_ndc"] for r in historical if r["partition"] == "test"}
    assert all(assignments[n]["partition"] == "test" for n in protected)
    previous_path = project_root / "data/processed/final_v2/family_split_manifest.csv"
    previous_test = protected_expanded.copy()
    if previous_path.exists():
        with previous_path.open(encoding="utf-8", newline="") as stream:
            previous_test.update(
                {r["product_ndc"] for r in csv.DictReader(stream) if r["partition"] == "test"}
            )
        assert all(assignments[n]["partition"] == "test" for n in previous_test)
    assigned_products = {
        part: set(manifest.loc[manifest.partition.eq(part), "product_ndc"])
        for part in ("train", "validation", "test")
    }
    assert not any(
        assigned_products[a] & assigned_products[b] for a, b in combinations(assigned_products, 2)
    )
    for row in cases.itertuples():
        provenance = json.loads(row.provenance_json)
        assert set(provenance["product_ndcs"]) <= assigned_products[row.partition]
        assert "reference_json" not in row.input_json and "provenance_json" not in row.input_json
        answer = json.loads(row.reference_json)
        references = [answer[k] for k in ("product", "product_a", "product_b") if k in answer]
        for reference in references:
            source_product = products[reference["source"]["product_ndc"]]
            assert reference["source"]["source_rows"]
            assert {normalize_name(i["name"]) for i in reference["active_ingredients"]} == {
                normalize_name(i["name"])
                for i in json.loads(source_product["active_ingredients_json"])
            }
            if reference["formulation_supported"]:
                assert formulation_key(
                    reference["active_ingredients"], reference["dosage_form"], reference["route"]
                ) == formulation_key(
                    json.loads(source_product["active_ingredients_json"]),
                    source_product["dosage_form"],
                    json.loads(source_product["route"]),
                )
    used_products = {
        n for value in cases.provenance_json for n in json.loads(value)["product_ndcs"]
    }
    eligible = {n for n, p in products.items() if p["ingredient_eligible"] == "True"}
    assert eligible <= used_products
    summary = {
        "version": FINAL_BENCHMARK_VERSION,
        "status": "FROZEN" if freeze else "VALIDATED_DRAFT",
        "split_frozen": freeze,
        "catalog_products": len(products),
        "usable_products": len(eligible),
        "independent_family_groups": len({assignments[n]["family_group_id"] for n in eligible}),
        "family_policy": families,
        "pair_construction_policy": pair_policy,
        "benchmark_cases": len(cases),
        "cases_by_task": cases.task.value_counts().sort_index().to_dict(),
        "partitions": {
            part: {
                "cases": len(cases.loc[cases.partition.eq(part)]),
                "cases_by_task": cases.loc[cases.partition.eq(part)]
                .task.value_counts()
                .sort_index()
                .to_dict(),
                "cases_by_type": cases.loc[cases.partition.eq(part)]
                .case_type.value_counts()
                .sort_index()
                .to_dict(),
                "usable_products": len(assigned_products[part] & eligible),
                "independent_families": manifest.loc[
                    manifest.partition.eq(part) & manifest.ingredient_eligible.eq("True"),
                    "family_group_id",
                ].nunique(),
            }
            for part in assigned_products
        },
        **construction,
        "legacy_reservoir": {
            "pairs": len(screened),
            "products": len({r["product_ndc_" + s] for r in screened for s in ("a", "b")}),
            "old_provisional_blocks": snapshot["legacy_reservoir_provisional_blocks"],
            "outcomes": dict(reservoir_outcomes),
        },
        "historical_hashes": historical_hashes,
        "source_catalog_sha256": sha256(path.read_bytes()).hexdigest(),
        "checks": {
            "all_accepted_products_used": True,
            "no_product_leakage": True,
            "no_family_leakage": True,
            "no_repeated_case_leakage": True,
            "original_test_products_protected": len(protected),
            "previous_expanded_test_products_protected": len(previous_test),
            "historical_files_unchanged": True,
            "reference_labels_before_evaluation": True,
            "test_inference_run": False,
        },
        "reference_status": "Source-supported algorithmic labels. Independent clinical or human verification absent.",
        "reference_normalization": "Routes are sets. Repeated source route values never establish an NDC. "
        "Semantic case keys normalize case, route repetition and dimensionally equivalent quantities. "
        "Symmetric comparison keys remove reversed pairs.",
        "supersedes": "final_v4",
        "development_erratum": {
            "prior_training_case": "hrp_v2_1c961b31e75d3ba6f247",
            "issue": "A repeated NASAL route forced a false exact NDC. Training inspection established a "
            "formulation shared by 30 catalog NDCs. General route-set matching replaces list matching.",
            "heldout_answers_inspected": False,
            "heldout_predictions_generated": 0,
        },
        "chemical_identity_erratum": {
            "prior_development_same_unii_disjoint_conflicts": 307,
            "rule": "Source FDA UNII sets replace lexical ingredient names for chemical overlap. "
            "Product names and original strings remain intact. Missing identity evidence withholds "
            "chemical comparisons rather than implying ingredient absence.",
            "heldout_answers_inspected": False,
            "heldout_predictions_generated": 0,
        },
        "split_policy": "SHA256 of seed 42 and stable complete family membership. 70/15/15 thresholds. "
        "Historical test membership takes priority, followed by historical validation. "
        "Source and family rules fixed before case generation. No model results used.",
        "name_query_policy": "Check complete catalog including generic aliases and unresolved products. "
        "One exact NDC only when unique. Equivalent complete formulations can resolve without an NDC. "
        "Known different formulations require abstention. Unresolved evidence is not a verified ambiguity label.",
        "limitations": [
            "Families are conservative algorithmic partition groups, not confirmed marketing families.",
            "Repeated pairs share products. Family counts define independent evaluation units.",
            "Common formulations can recur across unrelated names. Product and related-name groups are disjoint.",
            "FDA NDC and labeling evidence can derive from the same submission.",
        ],
    }

    for part, values in summary["partitions"].items():
        partition_cases = cases.loc[cases.partition.eq(part)]
        values["families_by_task"] = {
            kind: len(
                {family for groups in frame.family_group_ids_json for family in json.loads(groups)}
            )
            for kind, frame in partition_cases.groupby("task")
        }
        values["families_by_type"] = {
            kind: len(
                {family for groups in frame.family_group_ids_json for family in json.loads(groups)}
            )
            for kind, frame in partition_cases.groupby("case_type")
        }
    files = {
        "family_split_manifest.csv": manifest.to_csv(index=False, lineterminator="\r\n"),
        "family_links.csv": pd.DataFrame(
            links, columns=["product_ndc_a", "product_ndc_b", "reason"]
        ).to_csv(index=False, lineterminator="\r\n"),
    }
    for part in assigned_products:
        files[("sealed_test" if part == "test" else part) + ".csv"] = cases.loc[
            cases.partition.eq(part)
        ].to_csv(index=False, lineterminator="\r\n")
    digests = {name: sha256(content.encode("utf-8")).hexdigest() for name, content in files.items()}
    seal_path = report_dir / "freeze_manifest.json"
    if seal_path.exists():
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if seal["files"] != digests:
            raise ValueError(
                "Frozen benchmark differs. Create a new version without overwriting the sealed test"
            )
        for name, digest in digests.items():
            if sha256((out / name).read_bytes()).hexdigest() != digest:
                raise ValueError("A frozen artifact was changed " + name)
    else:
        for name, content in files.items():
            (out / name).write_bytes(content.encode("utf-8"))
        if freeze:
            seal_path.write_text(
                json.dumps(
                    {
                        "version": FINAL_BENCHMARK_VERSION,
                        "files": digests,
                        "source_catalog_sha256": summary["source_catalog_sha256"],
                        "test_access": "Sealed after source validation. No development predictions permitted.",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    summary["files"] = digests
    summary["review_queue"] = write_review_queue(
        products, assignments, interim / FINAL_BENCHMARK_VERSION
    )
    disputed_links = pd.DataFrame(
        [
            {
                "product_ndc": a,
                "related_product_ndc": b,
                "reason": reason,
                "partition": assignments[a]["partition"],
                "review_decision": "PENDING",
                "reviewer_id": "",
            }
            for a, b, reason in links
            if reason in {"near_name_co_partition", "submitted_brand_base"}
        ]
    )
    disputed_links.to_csv(out / "family_link_review.csv", index=False)
    selected_links = (
        disputed_links.loc[disputed_links.partition.ne("test")].groupby("reason", sort=True).head(3)
    )
    link_sample = out / "family_link_review_sample.csv"
    if not link_sample.exists():
        selected_links.to_csv(link_sample, index=False)
    summary["grouping_review"] = {
        "provisional_links": len(disputed_links),
        "pending_sample": len(selected_links),
        "scope": "Conservative co-partition links do not verify marketing family membership. "
        "Three development links per rule are queued for optional human review. Test links are excluded from the sample.",
    }
    for name, digest in historical_hashes.items():
        if (project_root / name).exists():
            assert sha256((project_root / name).read_bytes()).hexdigest() == digest
    (report_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "version",
                    "status",
                    "usable_products",
                    "independent_family_groups",
                    "benchmark_cases",
                    "partitions",
                    "legacy_reservoir",
                    "checks",
                )
            },
            indent=2,
        )
    )
    return summary


def write_review_queue(products: dict, assignments: dict, output_dir: Path) -> dict:
    from hrp.feasibility_finish import stratified_review_sample

    rows = []
    for ndc, p in sorted(products.items()):
        reasons = json.loads(p["exclusion_reasons_json"]) + json.loads(p["strength_issues_json"])
        if reasons:
            rows.append(
                {
                    "product_ndc": ndc,
                    **assignments[ndc],
                    "support_tier": p["support_tier"],
                    "reasons_json": json.dumps(reasons),
                    "review_decision": "PENDING",
                    "reviewer_id": "",
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "full_review_queue.csv", index=False)
    strata = defaultdict(list)
    for row in rows:
        # Held-out records remain in the sealed source ledger without a development review sample
        if row["partition"] == "test":
            continue
        for reason in json.loads(row["reasons_json"]):
            strata[(row["support_tier"], reason)].append(row)
    selected = stratified_review_sample(rows)
    sample = output_dir / "manual_review_sample.csv"
    if not sample.exists():
        pd.DataFrame(selected).to_csv(sample, index=False)
    return {
        "full_queue": len(rows),
        "development_strata": len(strata),
        "pending_sample": len(selected),
        "sampling_rule": "Three per evidence and issue stratum. "
        "Audit sampling does not set benchmark size. No human reviews inferred.",
    }
