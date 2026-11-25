# Resolve RxNorm drug concepts to ingredient concepts

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# RXNREL describes the relationship of RXCUI2 to RXCUI1
ALLOWED_EDGES = {
    ("SCD", "consists_of", "SCDC"),
    ("SCD", "has_ingredients", "MIN"),
    ("SBD", "consists_of", "SCDC"),
    ("SBD", "tradename_of", "SCD"),
    ("GPCK", "contains", "SCD"),
    ("BPCK", "contains", "SCD"),
    ("BPCK", "contains", "SBD"),
    ("BPCK", "tradename_of", "GPCK"),
    ("SCDC", "has_ingredient", "IN"),
    ("SCDC", "has_precise_ingredient", "PIN"),
    ("MIN", "has_part", "IN"),
    ("MIN", "has_part", "PIN"),
    ("PIN", "form_of", "IN"),
}

CONCEPT_TYPES = {tty for edge in ALLOWED_EDGES for tty in (edge[0], edge[2])}


def resolve_ingredients(
    drug_rxcui: str,
    concepts: dict[str, tuple[str, str]],
    relationships: dict[str, set[str]],
) -> tuple[list[str], list[str], int]:

    # Follow ingredient relationships and track unresolved branches

    pending = [drug_rxcui]
    visited = set()
    base_ids = set()
    precise_ids = set()
    unresolved = 0

    while pending:
        current = pending.pop()

        if current in visited:
            continue

        visited.add(current)
        tty = concepts[current][0]

        if tty == "IN":
            base_ids.add(current)
            continue

        if tty == "PIN":
            precise_ids.add(current)

        next_ids = relationships.get(current, set())

        if not next_ids:
            unresolved += 1
        else:
            pending.extend(next_ids - visited)

    return sorted(base_ids), sorted(precise_ids), unresolved


def map_rxnorm_ingredients(
    product_links_path: Path,
    rxnorm_dir: Path,
    output_dir: Path,
    report_dir: Path,
) -> dict:

    # Map linked drug products to RxNorm ingredient concepts

    conso_path = rxnorm_dir / "RXNCONSO.RRF"
    rel_path = rxnorm_dir / "RXNREL.RRF"

    for path in (product_links_path, conso_path, rel_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    with product_links_path.open(encoding="utf-8", newline="") as file:
        products = list(csv.DictReader(file))

    # Load active RxNorm concepts needed for ingredient resolution
    concepts = {}
    substance_codes = defaultdict(set)

    with conso_path.open(encoding="utf-8") as file:
        for line in file:
            fields = line.rstrip("\r\n").split("|")

            if len(fields) != 19:
                raise ValueError("Invalid RXNCONSO layout")

            if fields[11] == "RXNORM" and fields[16] == "N" and fields[12] in CONCEPT_TYPES:
                concepts[fields[0]] = (
                    fields[12],
                    fields[14],
                )
            if (
                fields[11] == "MTHSPL"
                and fields[12] == "SU"
                and fields[16] == "N"
                and re.fullmatch(r"[A-Z0-9]{10}", fields[13])
            ):
                substance_codes[fields[0]].add(fields[13])

    # Build only the relationships relevant to ingredient mapping
    relationships = defaultdict(set)

    with rel_path.open(encoding="utf-8") as file:
        for line in file:
            fields = line.rstrip("\r\n").split("|")

            if len(fields) != 17:
                raise ValueError("Invalid RXNREL layout")

            # RxNorm relationships normally have an empty suppression field
            # Accept blank values and older records marked as not suppressed
            if (
                fields[10] != "RXNORM"
                or fields[14] not in ("", "N")
                or fields[2] != "CUI"
                or fields[6] != "CUI"
            ):
                continue

            first = concepts.get(fields[0])
            second = concepts.get(fields[4])

            if first is None or second is None:
                continue

            # The second concept has the named relationship to the first
            if (second[0], fields[7], first[0]) in ALLOWED_EDGES:
                relationships[fields[4]].add(fields[0])

    counts = Counter()
    unique_in = set()
    unique_pin = set()

    output_path = output_dir / "otc_rxnorm_ingredient_links.csv"

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "product_ndc",
                "catalog_status",
                "drug_rxcui",
                "product_link_status",
                "ingredient_status",
                "in_rxcuis_json",
                "in_names_json",
                "pin_rxcuis_json",
                "pin_names_json",
                "unresolved_branches",
                "identity_options_json",
                "unii_options_json",
            ]
        )

        for row in products:
            rxcui = row["rxnorm_rxcui"].strip()
            product_status = row["rxnorm_status"]
            base_ids, precise_ids, unresolved = [], [], 0

            if product_status != "matched":
                status = "no_unique_drug_concept"

            elif rxcui not in concepts:
                status = "drug_concept_not_found"

            else:
                base_ids, precise_ids, unresolved = resolve_ingredients(
                    rxcui, concepts, relationships
                )

                if base_ids:
                    status = "partial_path" if unresolved else "resolved_path"
                    counts["ingredient_mapped"] += 1

                    if row["catalog_status"] == "pending":
                        counts["pending_ingredient_mapped"] += 1

                elif precise_ids:
                    status = "precise_only"

                else:
                    status = "no_ingredient_path"

            counts[status] += 1
            unique_in.update(base_ids)
            unique_pin.update(precise_ids)

            writer.writerow(
                [
                    row["product_ndc"],
                    row["catalog_status"],
                    rxcui,
                    product_status,
                    status,
                    json.dumps(base_ids),
                    json.dumps([concepts[i][1] for i in base_ids]),
                    json.dumps(precise_ids),
                    json.dumps([concepts[i][1] for i in precise_ids]),
                    unresolved,
                    json.dumps(
                        ingredient_identity_options(base_ids, precise_ids, concepts, relationships)
                    ),
                    json.dumps(
                        ingredient_identity_options(
                            base_ids, precise_ids, concepts, relationships, substance_codes
                        )
                    ),
                ]
            )

    matched = sum(row["rxnorm_status"] == "matched" for row in products)
    total = len(products)

    statuses = [
        "no_unique_drug_concept",
        "drug_concept_not_found",
        "resolved_path",
        "partial_path",
        "precise_only",
        "no_ingredient_path",
    ]

    assert sum(counts[s] for s in statuses) == total

    summary = {
        "otc_ndcs": total,
        "matched_drug_ndcs": matched,
        "ingredient_mapped_ndcs": counts["ingredient_mapped"],
        "ingredient_mapping_pct_of_matched": round(100 * counts["ingredient_mapped"] / matched, 2)
        if matched
        else 0.0,
        "ingredient_mapping_pct_of_all": round(100 * counts["ingredient_mapped"] / total, 2)
        if total
        else 0.0,
        "resolved_paths": counts["resolved_path"],
        "partial_paths": counts["partial_path"],
        "precise_only": counts["precise_only"],
        "no_ingredient_path": counts["no_ingredient_path"],
        "drug_concept_not_found": counts["drug_concept_not_found"],
        "no_unique_drug_concept": counts["no_unique_drug_concept"],
        "pending_with_ingredient_paths": counts["pending_ingredient_mapped"],
        "distinct_in_rxcuis": len(unique_in),
        "distinct_pin_rxcuis": len(unique_pin),
    }

    summary_path = report_dir / "otc_rxnorm_ingredient_summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary


def ingredient_identity_options(
    base_ids, precise_ids, concepts, relationships, substance_codes=None
):
    options = {
        identifier: set(substance_codes.get(identifier, []))
        if substance_codes is not None
        else {concepts[identifier][1]}
        for identifier in base_ids
    }
    for precise in precise_ids:
        pending, visited, reachable = [precise], set(), set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            if concepts[current][0] == "IN":
                reachable.add(current)
            else:
                pending.extend(relationships.get(current, set()) - visited)
        if len(reachable) == 1:
            base = next(iter(reachable))
            if base in options:
                if substance_codes is None:
                    options[base].add(concepts[precise][1])
                else:
                    options[base].update(substance_codes.get(precise, []))
    return [sorted(options[key]) for key in sorted(options)]
